#!/usr/bin/env python3
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import click
import cv2
import numpy as np
from tqdm import tqdm

# IMPORTANT: torchvision registers ops used by some torchscript / detectron2 exports
import torch
import torchvision  # noqa: F401

# EgoBlur (gen2) imports (same as your notebook) :contentReference[oaicite:3]{index=3}
from gen2.script.predictor import EgoblurDetector, ClassID, PATCH_INSTANCES_FIELDS
from gen2.script.detectron2.export.torchscript_patch import patch_instances
from gen2.script.utils import get_device, get_image_tensor, scale_box
from gen2.script.constants import RESIZE_MIN_GEN2, RESIZE_MAX_GEN2


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def blur_soft(image: np.ndarray, box: List[int], strength: float = 0.5) -> None:
    """Your modified soft blur (kernel proportional to box size). :contentReference[oaicite:4]{index=4}"""
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return

    h, w = roi.shape[:2]
    k = max(3, int(min(h, w) * strength))
    if k % 2 == 0:
        k += 1
    roi_blur = cv2.GaussianBlur(roi, (k, k), sigmaX=0)
    image[y1:y2, x1:x2] = roi_blur


def blur_pixelate(image: np.ndarray, box: List[int], factor: int = 10) -> None:
    """Your pixelation option. :contentReference[oaicite:5]{index=5}"""
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    if h <= 1 or w <= 1:
        return
    ds_w = max(1, w // factor)
    ds_h = max(1, h // factor)
    roi_small = cv2.resize(roi, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
    roi_pix = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = roi_pix


class EgoBlurGen2:
    """Minimal, notebook-faithful wrapper. :contentReference[oaicite:6]{index=6}"""

    def __init__(
        self,
        face_model_path: Optional[str] = None,
        lp_model_path: Optional[str] = None,
        face_threshold: float = 0.2,
        lp_threshold: float = 0.2,
        nms_iou_threshold: float = 0.5,
        scale_factor_detections: float = 1.0,
        device: Optional[str] = None,
    ):
        self.device = device or get_device()
        self.scale_factor = scale_factor_detections

        self.face_detector = (
            EgoblurDetector(
                model_path=face_model_path,
                device=self.device,
                detection_class=ClassID.FACE,
                score_threshold=face_threshold,
                nms_iou_threshold=nms_iou_threshold,
                resize_aug={"min_size_test": RESIZE_MIN_GEN2, "max_size_test": RESIZE_MAX_GEN2},
            )
            if face_model_path
            else None
        )

        self.lp_detector = (
            EgoblurDetector(
                model_path=lp_model_path,
                device=self.device,
                detection_class=ClassID.LICENSE_PLATE,
                score_threshold=lp_threshold,
                nms_iou_threshold=nms_iou_threshold,
                resize_aug={"min_size_test": RESIZE_MIN_GEN2, "max_size_test": RESIZE_MAX_GEN2},
            )
            if lp_model_path
            else None
        )

    def detect(self, bgr_image: np.ndarray) -> List[List[int]]:
        image_tensor = get_image_tensor(bgr_image)
        detections: List[List[float]] = []

        with patch_instances(fields=PATCH_INSTANCES_FIELDS):
            if self.face_detector:
                res = self.face_detector.run(image_tensor)
                if res:
                    detections.extend(res[0])

            if self.lp_detector:
                res = self.lp_detector.run(image_tensor)
                if res:
                    detections.extend(res[0])

        h, w = bgr_image.shape[:2]
        final_boxes: List[List[int]] = []
        for box in detections:
            if self.scale_factor != 1.0:
                box = scale_box(box, w, h, self.scale_factor)
            x1, y1, x2, y2 = map(int, box)
            # clip
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                final_boxes.append([x1, y1, x2, y2])

        return final_boxes

    def anonymize(
        self,
        bgr_image: np.ndarray,
        blur_mode: str,
        blur_strength: float,
        pixelate_factor: int,
        return_boxes: bool = False,
    ):
        image = bgr_image.copy()
        boxes = self.detect(image)

        if blur_mode == "soft":
            for box in boxes:
                blur_soft(image, box, strength=blur_strength)
        elif blur_mode == "pixelate":
            for box in boxes:
                blur_pixelate(image, box, factor=pixelate_factor)
        else:
            raise ValueError(f"Unknown blur_mode={blur_mode}")

        return (image, boxes) if return_boxes else image


# ---- multiprocessing support (CPU-safe) ----
@dataclass
class WorkerConfig:
    face_model: str
    lp_model: str
    face_threshold: float
    lp_threshold: float
    nms_iou: float
    scale_factor: float
    device: str
    blur_mode: str
    blur_strength: float
    pixelate_factor: int
    save_boxes: bool


_G_EGOBLUR: Optional[EgoBlurGen2] = None
_G_CFG: Optional[WorkerConfig] = None


def _init_worker(cfg: WorkerConfig):
    global _G_EGOBLUR, _G_CFG
    _G_CFG = cfg
    _G_EGOBLUR = EgoBlurGen2(
        face_model_path=cfg.face_model,
        lp_model_path=cfg.lp_model,
        face_threshold=cfg.face_threshold,
        lp_threshold=cfg.lp_threshold,
        nms_iou_threshold=cfg.nms_iou,
        scale_factor_detections=cfg.scale_factor,
        device=cfg.device,
    )


def _process_one(args: Tuple[str, str]) -> Tuple[str, bool, str]:
    """(in_path, out_path) -> (name, ok, msg)"""
    in_path, out_path = args
    assert _G_EGOBLUR is not None and _G_CFG is not None

    p_in = Path(in_path)
    p_out = Path(out_path)

    img = cv2.imread(str(p_in), cv2.IMREAD_COLOR)
    if img is None:
        return (p_in.name, False, "cv2.imread failed")

    t0 = time.time()
    if _G_CFG.save_boxes:
        out, boxes = _G_EGOBLUR.anonymize(
            img,
            blur_mode=_G_CFG.blur_mode,
            blur_strength=_G_CFG.blur_strength,
            pixelate_factor=_G_CFG.pixelate_factor,
            return_boxes=True,
        )
    else:
        out = _G_EGOBLUR.anonymize(
            img,
            blur_mode=_G_CFG.blur_mode,
            blur_strength=_G_CFG.blur_strength,
            pixelate_factor=_G_CFG.pixelate_factor,
            return_boxes=False,
        )
        boxes = None
    dt = time.time() - t0

    p_out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p_out), out)
    if not ok:
        return (p_in.name, False, "cv2.imwrite failed")

    if boxes is not None:
        meta = {
            "input": str(p_in),
            "output": str(p_out),
            "boxes_xyxy": boxes,
            "seconds": dt,
        }
        with open(p_out.with_suffix(p_out.suffix + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return (p_in.name, True, f"{dt:.2f}s")


@click.command()
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--face-model", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--lp-model", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="auto", show_default=True, help="auto|cpu|cuda")
@click.option("--workers", default=4, show_default=True, help="CPU workers. For CUDA, use 1 unless you REALLY know why.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
@click.option("--suffix", default="_egoblur", show_default=True, help="Output filename suffix.")
@click.option("--out-ext", default=".jpg", show_default=True, help="Output extension (.jpg/.png).")
@click.option("--save-boxes", is_flag=True, help="Write sidecar JSON with boxes per image.")
# thresholds + detection params
@click.option("--face-threshold", default=0.3, show_default=True)
@click.option("--lp-threshold", default=0.4, show_default=True)
@click.option("--nms-iou", default=0.5, show_default=True)
@click.option("--scale-factor", default=0.9, show_default=True)
# blur params
@click.option("--blur", "blur_mode", type=click.Choice(["soft", "pixelate"]), default="soft", show_default=True)
@click.option("--blur-strength", default=0.5, show_default=True, help="Only for soft blur.")
@click.option("--pixelate-factor", default=10, show_default=True, help="Only for pixelate.")
def main(
    input_dir: Path,
    output_dir: Path,
    face_model: Path,
    lp_model: Path,
    device: str,
    workers: int,
    overwrite: bool,
    suffix: str,
    out_ext: str,
    save_boxes: bool,
    face_threshold: float,
    lp_threshold: float,
    nms_iou: float,
    scale_factor: float,
    blur_mode: str,
    blur_strength: float,
    pixelate_factor: int,
):
    # device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # IMPORTANT practical default: multiprocessing + one GPU usually hurts
    # if device == "cuda" and workers > 1:
    #     click.echo("[WARN] device=cuda with workers>1 will contend for the same GPU. For best throughput, use --workers 1.")
    #     workers = 1

    # find images
    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise click.ClickException(f"No images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, str]] = []
    for p in images:
        out_path = output_dir / f"{p.stem}{suffix}{out_ext}"
        if out_path.exists() and not overwrite:
            continue
        tasks.append((str(p), str(out_path)))

    if not tasks:
        click.echo("Nothing to do (all outputs exist).")
        return

    cfg = WorkerConfig(
        face_model=str(face_model),
        lp_model=str(lp_model),
        face_threshold=face_threshold,
        lp_threshold=lp_threshold,
        nms_iou=nms_iou,
        scale_factor=scale_factor,
        device=device,
        blur_mode=blur_mode,
        blur_strength=blur_strength,
        pixelate_factor=pixelate_factor,
        save_boxes=save_boxes,
    )

    t_all = time.time()

    # Single-worker path (simpler + best for GPU)
    if workers == 1:
        _init_worker(cfg)
        ok_n = 0
        for t in tqdm(tasks, desc="egoblur"):
            name, ok, msg = _process_one(t)
            if ok:
                ok_n += 1
            else:
                click.echo(f"FAIL {name}: {msg}", err=True)
        click.echo(f"Done: {ok_n}/{len(tasks)} in {time.time() - t_all:.2f}s")
        return

    # Multiprocessing path (best for CPU)
    # from concurrent.futures import ProcessPoolExecutor, as_completed

    # ok_n = 0
    # with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cfg,)) as ex:
    #     futs = [ex.submit(_process_one, t) for t in tasks]
    #     for fut in tqdm(as_completed(futs), total=len(futs), desc="egoblur"):
    #         name, ok, msg = fut.result()
    #         if ok:
    #             ok_n += 1
    #         else:
    #             click.echo(f"FAIL {name}: {msg}", err=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    
    ok_n = 0
    
    # CUDA + multiprocessing must use spawn to avoid forked CUDA contexts.
    ctx = mp.get_context("spawn")
    
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(cfg,),
    ) as ex:
        futs = [ex.submit(_process_one, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="egoblur"):
            name, ok, msg = fut.result()
            if ok:
                ok_n += 1
            else:
                click.echo(f"FAIL {name}: {msg}", err=True)
    
    click.echo(f"Done: {ok_n}/{len(tasks)} in {time.time() - t_all:.2f}s")

    # click.echo(f"Done: {ok_n}/{len(tasks)} in {time.time() - t_all:.2f}s")


if __name__ == "__main__":
    main()
