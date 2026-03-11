#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pytorch360convert as p360
import torch
import torch.nn.functional as F
from open_image_models import LicensePlateDetector
from open_image_models.detection.core.yolo_v9.inference import YoloV9ObjectDetector
from ultralytics import YOLO


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
SIDE_FACES = ("Front", "Right", "Back", "Left")
ALL_FACES = ("Front", "Right", "Back", "Left", "Up", "Down")


def iter_images(input_dir: Path) -> Sequence[Path]:
    return [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def bgr_to_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(device) / 255.0


def tensor_to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    image_rgb = (
        image_tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0
    ).astype(np.uint8)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def gaussian_kernel_1d(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    sigma = max(0.1, float(sigma))
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_gpu_bgr(
    image_bgr_u8: np.ndarray,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    img = (
        torch.from_numpy(image_bgr_u8)
        .to(device=device, dtype=dtype)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    kernel_1d = gaussian_kernel_1d(sigma=sigma, device=device, dtype=dtype)
    k = kernel_1d.numel()

    kernel_x = kernel_1d.view(1, 1, 1, k).repeat(3, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, k, 1).repeat(3, 1, 1, 1)

    pad = k // 2
    x = F.pad(img, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel_x, groups=3)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, kernel_y, groups=3)

    out = x.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return out


def apply_roi_blur_bgr(
    image_bgr_u8: np.ndarray,
    mask_f32: np.ndarray,
    sigma: float,
    blur_backend: str,
    device: torch.device,
    dtype: torch.dtype,
    roi_pad: int,
    roi_min_area: int,
    roi_mask_threshold: float,
) -> np.ndarray:
    out = image_bgr_u8.astype(np.float32).copy()
    bin_mask = (mask_f32 >= roi_mask_threshold).astype(np.uint8)
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )

    h, w = image_bgr_u8.shape[:2]
    for label_id in range(1, n_labels):
        x, y, rw, rh, area = stats[label_id]
        if area < roi_min_area:
            continue

        x1 = max(0, int(x) - roi_pad)
        y1 = max(0, int(y) - roi_pad)
        x2 = min(w, int(x + rw) + roi_pad)
        y2 = min(h, int(y + rh) + roi_pad)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = image_bgr_u8[y1:y2, x1:x2]
        if blur_backend == "gpu":
            pad = max(1, int(round(3.0 * max(0.1, float(sigma)))))
            if roi.shape[0] <= pad or roi.shape[1] <= pad:
                continue
            roi_blur = gaussian_blur_gpu_bgr(
                image_bgr_u8=roi,
                sigma=sigma,
                device=device,
                dtype=dtype,
            )
        else:
            roi_blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma)

        roi_mask = mask_f32[y1:y2, x1:x2][..., None].astype(np.float32)
        out[y1:y2, x1:x2] = (out[y1:y2, x1:x2] * (1.0 - roi_mask)) + (
            roi_blur.astype(np.float32) * roi_mask
        )

    return np.clip(out, 0, 255).astype(np.uint8)


def detect_faces_ultralytics(
    detector: YOLO,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> List[Tuple[int, int, int, int, float]]:
    pred = detector.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    if not pred:
        return []
    result = pred[0]
    if result.boxes is None or result.boxes.xyxy is None:
        return []

    h, w = image_bgr.shape[:2]
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    out: List[Tuple[int, int, int, int, float]] = []
    for box, score in zip(boxes, confs):
        x1 = max(0, min(int(round(float(box[0]))), w - 1))
        y1 = max(0, min(int(round(float(box[1]))), h - 1))
        x2 = max(0, min(int(round(float(box[2]))), w))
        y2 = max(0, min(int(round(float(box[3]))), h))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((x1, y1, x2, y2, float(score)))
    return out


def detect_lp_open_image(
    detector,
    image_bgr: np.ndarray,
) -> List[Tuple[int, int, int, int, float]]:
    out: List[Tuple[int, int, int, int, float]] = []
    detections = detector.predict(image_bgr)
    h, w = image_bgr.shape[:2]
    for det in detections:
        b = det.bounding_box
        x1 = max(0, min(int(b.x1), w - 1))
        y1 = max(0, min(int(b.y1), h - 1))
        x2 = max(0, min(int(b.x2), w))
        y2 = max(0, min(int(b.y2), h))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((x1, y1, x2, y2, float(det.confidence)))
    return out


def draw_label(
    image: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x0 = max(0, min(x, image.shape[1] - tw - 6))
    y0 = max(th + 6, min(y, image.shape[0] - 2))
    cv2.rectangle(
        image,
        (x0 - 2, y0 - th - baseline - 4),
        (x0 + tw + 4, y0 + 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x0, y0 - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def write_jpg(path: Path, image_bgr: np.ndarray, quality: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])


def _process_chunk(packed: dict) -> tuple[list, list]:
    """Worker function run in a subprocess. Loads models and processes image_paths."""
    image_paths = [Path(p) for p in packed["image_paths"]]
    output_dir = Path(packed["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    p360_device = select_device(packed["p360_device"])
    if packed["blur_backend"] == "auto":
        blur_backend = "gpu" if p360_device.type == "cuda" else "cpu"
    else:
        blur_backend = packed["blur_backend"]
    blur_dtype = torch.float16 if p360_device.type == "cuda" else torch.float32
    face_device = "cuda:0" if p360_device.type == "cuda" else "cpu"

    face_detector = YOLO(packed["face_model"])
    lp_model_path = Path(packed["lp_model"])
    if lp_model_path.exists():
        lp_detector = YoloV9ObjectDetector(
            model_path=str(lp_model_path),
            class_labels=["License Plate"],
            conf_thresh=packed["lp_conf"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    else:
        lp_detector = LicensePlateDetector(
            detection_model=packed["lp_model"],
            conf_thresh=packed["lp_conf"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    rows_summary: list = []
    rows_profile: list = []

    for image_path in image_paths:
        t0 = time.perf_counter()
        blur_path = output_dir / f"{image_path.stem}_blur.jpg"
        annot_path = output_dir / f"{image_path.stem}_annot.jpg"
        if (
            blur_path.exists()
            and (packed["output_mode"] == "blur_only" or annot_path.exists())
            and not packed["overwrite"]
        ):
            continue

        t = time.perf_counter()
        pano = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        t_read = time.perf_counter() - t
        if pano is None:
            print(f"skip unreadable: {image_path.name}", flush=True)
            continue

        pano_h, pano_w = pano.shape[:2]

        t = time.perf_counter()
        cube_dict = p360.e2c(
            bgr_to_tensor(pano, device=p360_device),
            face_w=packed["det_face_w"],
            mode="bilinear",
            cube_format="dict",
            channels_first=True,
        )
        t_e2c = time.perf_counter() - t

        all_dets: List[Tuple[str, int, int, int, int, float, str]] = []
        image_face_count = 0
        image_lp_count = 0

        t = time.perf_counter()
        for face_name in SIDE_FACES:
            face_bgr = tensor_to_bgr(cube_dict[face_name])
            face_boxes = detect_faces_ultralytics(
                detector=face_detector,
                image_bgr=face_bgr,
                conf=packed["face_conf"],
                iou=packed["face_iou"],
                imgsz=packed["face_imgsz"],
                device=face_device,
            )
            image_face_count += len(face_boxes)
            for x1, y1, x2, y2, conf in face_boxes:
                all_dets.append((face_name, x1, y1, x2, y2, conf, "F"))
        t_detect_face = time.perf_counter() - t

        t = time.perf_counter()
        for face_name in SIDE_FACES:
            face_bgr = tensor_to_bgr(cube_dict[face_name])
            lp_boxes = detect_lp_open_image(detector=lp_detector, image_bgr=face_bgr)
            image_lp_count += len(lp_boxes)
            for x1, y1, x2, y2, conf in lp_boxes:
                all_dets.append((face_name, x1, y1, x2, y2, conf, "LP"))
        t_detect_lp = time.perf_counter() - t

        annotated_pano = pano.copy() if packed["output_mode"] == "both" else None
        t_project = 0.0
        t_blur = 0.0

        if all_dets:
            t = time.perf_counter()
            n = len(all_dets)
            mask_cube: Dict[str, torch.Tensor] = {
                name: torch.zeros(
                    (n, packed["det_face_w"], packed["det_face_w"]),
                    dtype=torch.float32,
                    device=p360_device,
                )
                for name in ALL_FACES
            }
            for idx, (face_name, x1, y1, x2, y2, _conf, _typ) in enumerate(all_dets):
                mask_cube[face_name][idx, y1:y2, x1:x2] = 1.0

            projected = p360.c2e(
                mask_cube,
                h=pano_h,
                w=pano_w,
                mode="bilinear",
                cube_format="dict",
                channels_first=True,
            )

            mask_t = projected.max(dim=0).values
            mask = np.clip(mask_t.detach().cpu().numpy(), 0.0, 1.0)
            if packed["mask_threshold"] > 0:
                mask = (mask >= packed["mask_threshold"]).astype(np.float32)
            if packed["mask_feather"] > 0:
                mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=packed["mask_feather"])
                mask = np.clip(mask, 0.0, 1.0)

            if annotated_pano is not None:
                for idx, (_face_name, _x1, _y1, _x2, _y2, conf, typ) in enumerate(
                    all_dets
                ):
                    single_mask = projected[idx].detach().cpu().numpy()
                    bw = (single_mask >= packed["mask_threshold"]).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    contours = [
                        c
                        for c in contours
                        if cv2.contourArea(c) >= packed["min_contour_area"]
                    ]
                    if not contours:
                        continue
                    color = (24, 180, 24) if typ == "F" else (0, 0, 255)
                    cv2.drawContours(annotated_pano, contours, -1, color, 2)
                    main_contour = max(contours, key=cv2.contourArea)
                    m = cv2.moments(main_contour)
                    if m["m00"] > 0:
                        cx = int(m["m10"] / m["m00"])
                        cy = int(m["m01"] / m["m00"])
                    else:
                        cx, cy = map(int, main_contour[0][0])
                    draw_label(annotated_pano, f"{typ} {conf:.2f}", cx, cy, color)

            t_project = time.perf_counter() - t

            t = time.perf_counter()
            if packed["blur_scope"] == "roi":
                out_blur = apply_roi_blur_bgr(
                    image_bgr_u8=pano,
                    mask_f32=mask,
                    sigma=packed["blur_sigma"],
                    blur_backend=blur_backend,
                    device=p360_device,
                    dtype=blur_dtype,
                    roi_pad=packed["roi_pad"],
                    roi_min_area=packed["roi_min_area"],
                    roi_mask_threshold=packed["roi_mask_threshold"],
                )
            else:
                if blur_backend == "gpu":
                    blurred = gaussian_blur_gpu_bgr(
                        image_bgr_u8=pano,
                        sigma=packed["blur_sigma"],
                        device=p360_device,
                        dtype=blur_dtype,
                    )
                else:
                    blurred = cv2.GaussianBlur(
                        pano, (0, 0), sigmaX=packed["blur_sigma"]
                    )
                mask3 = mask[..., None]
                out_blur = (pano.astype(np.float32) * (1.0 - mask3)) + (
                    blurred.astype(np.float32) * mask3
                )
                out_blur = np.clip(out_blur, 0, 255).astype(np.uint8)
            t_blur = time.perf_counter() - t
        else:
            out_blur = pano

        t = time.perf_counter()
        if not write_jpg(blur_path, out_blur, quality=packed["jpg_quality"]):
            print(f"failed write: {blur_path}", flush=True)
            continue
        if annotated_pano is not None:
            if not write_jpg(annot_path, annotated_pano, quality=packed["jpg_quality"]):
                print(f"failed write: {annot_path}", flush=True)
                continue
        t_write = time.perf_counter() - t

        elapsed = time.perf_counter() - t0
        print(
            f"[w{packed['worker_id']}] ok {image_path.name} face={image_face_count} lp={image_lp_count} sec={elapsed:.2f}",
            flush=True,
        )
        rows_summary.append(
            [
                image_path.name,
                str(image_face_count),
                str(image_lp_count),
                str(image_face_count + image_lp_count),
                f"{elapsed:.6f}",
                str(blur_path),
                str(annot_path) if annotated_pano is not None else "",
            ]
        )
        rows_profile.append(
            [
                image_path.name,
                f"{t_read:.6f}",
                f"{t_e2c:.6f}",
                f"{t_detect_face:.6f}",
                f"{t_detect_lp:.6f}",
                f"{t_project:.6f}",
                f"{t_blur:.6f}",
                f"{t_write:.6f}",
                f"{elapsed:.6f}",
            ]
        )

    return rows_summary, rows_profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final face+license-plate blur pipeline (debug + production modes)"
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument(
        "--face-model",
        default="models/privacy_blur/face_yolov8n.pt",
        type=Path,
        help="Ultralytics face detector weights",
    )
    parser.add_argument(
        "--lp-model",
        default="models/privacy_blur/yolo-v9-s-608-license-plates-end2end.onnx",
        help="open-image-models LP model id OR local ONNX path",
    )

    parser.add_argument("--face-conf", default=0.4, type=float)
    parser.add_argument("--lp-conf", default=0.4, type=float)
    parser.add_argument("--face-iou", default=0.5, type=float)
    parser.add_argument("--face-imgsz", default=1024, type=int)

    parser.add_argument("--det-face-w", default=1024, type=int)
    parser.add_argument(
        "--p360-device",
        default="auto",
        help="Device for pytorch360convert (auto|cpu|cuda:0)",
    )

    parser.add_argument("--mask-threshold", default=0.35, type=float)
    parser.add_argument("--mask-feather", default=2.0, type=float)
    parser.add_argument("--blur-sigma", default=16.0, type=float)
    parser.add_argument(
        "--blur-scope",
        choices=["roi", "full"],
        default="roi",
        help="Blur only ROI regions or full panorama",
    )
    parser.add_argument(
        "--blur-backend",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Blur implementation backend",
    )
    parser.add_argument("--roi-pad", default=24, type=int)
    parser.add_argument("--roi-min-area", default=64, type=int)
    parser.add_argument("--roi-mask-threshold", default=0.01, type=float)

    parser.add_argument("--min-contour-area", default=20.0, type=float)
    parser.add_argument("--jpg-quality", default=85, type=int)
    parser.add_argument(
        "--output-mode",
        choices=["both", "blur_only"],
        default="blur_only",
        help="both = blur + annotated output; blur_only = production output",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        help="Number of parallel worker processes (each loads its own models)",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"input dir not found: {args.input_dir}")
    if not args.face_model.exists():
        raise SystemExit(f"face model not found: {args.face_model}")

    images = iter_images(args.input_dir)
    if not images:
        raise SystemExit(f"no images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "summary.csv"
    profile_csv = args.output_dir / "profile.csv"

    # Build a serialisable param dict shared across all workers.
    params = dict(
        output_dir=str(args.output_dir),
        face_model=str(args.face_model),
        lp_model=str(args.lp_model),
        face_conf=args.face_conf,
        lp_conf=args.lp_conf,
        face_iou=args.face_iou,
        face_imgsz=args.face_imgsz,
        det_face_w=args.det_face_w,
        p360_device=args.p360_device,
        blur_backend=args.blur_backend,
        blur_scope=args.blur_scope,
        blur_sigma=args.blur_sigma,
        mask_threshold=args.mask_threshold,
        mask_feather=args.mask_feather,
        roi_pad=args.roi_pad,
        roi_min_area=args.roi_min_area,
        roi_mask_threshold=args.roi_mask_threshold,
        min_contour_area=args.min_contour_area,
        jpg_quality=args.jpg_quality,
        output_mode=args.output_mode,
        overwrite=args.overwrite,
    )

    workers = max(1, args.workers)
    # Split images into chunks — one chunk per worker.
    chunks: list[list[str]] = [[] for _ in range(workers)]
    for i, img in enumerate(images):
        chunks[i % workers].append(str(img))

    work_items = [
        {**params, "image_paths": chunk, "worker_id": i}
        for i, chunk in enumerate(chunks)
        if chunk
    ]

    rows_summary: List[List[str]] = []
    rows_profile: List[List[str]] = []

    if len(work_items) == 1:
        # Single worker — run in-process, no subprocess overhead.
        rs, rp = _process_chunk(work_items[0])
        rows_summary.extend(rs)
        rows_profile.extend(rp)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(_process_chunk, item): item["worker_id"]
                for item in work_items
            }
            for fut in as_completed(futures):
                wid = futures[fut]
                try:
                    rs, rp = fut.result()
                    rows_summary.extend(rs)
                    rows_profile.extend(rp)
                except Exception as exc:
                    raise SystemExit(f"worker {wid} failed: {exc}") from exc

    rows_summary.sort(key=lambda r: r[0])
    rows_profile.sort(key=lambda r: r[0])

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image",
                "face_boxes",
                "lp_boxes",
                "total_boxes",
                "elapsed_sec",
                "blur_output",
                "annot_output",
            ]
        )
        w.writerows(rows_summary)

    with profile_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image",
                "read_sec",
                "e2c_sec",
                "detect_face_sec",
                "detect_lp_sec",
                "project_sec",
                "blur_sec",
                "write_sec",
                "total_sec",
            ]
        )
        w.writerows(rows_profile)

    processed = len(rows_summary)
    total_face = sum(int(float(r[1])) for r in rows_summary)
    total_lp = sum(int(float(r[2])) for r in rows_summary)
    total_elapsed = sum(float(r[4]) for r in rows_summary)
    profile_cols = [
        "read_sec",
        "e2c_sec",
        "detect_face_sec",
        "detect_lp_sec",
        "project_sec",
        "blur_sec",
        "write_sec",
    ]
    col_sums = [
        sum(float(r[i + 1]) for r in rows_profile) for i in range(len(profile_cols))
    ]

    print(f"processed_images={processed}")
    print(f"total_face_boxes={total_face}")
    print(f"total_lp_boxes={total_lp}")
    print(f"total_boxes={total_face + total_lp}")
    print(f"elapsed_sec={total_elapsed:.2f}")
    print(f"avg_sec_per_image={(total_elapsed / processed) if processed else 0.0:.4f}")
    for name, s in zip(profile_cols, col_sums):
        print(f"avg_{name}={(s / processed) if processed else 0.0:.4f}")
    print(f"workers={workers}")
    print(f"output_dir={args.output_dir}")
    print(f"summary_csv={summary_csv}")
    print(f"profile_csv={profile_csv}")


if __name__ == "__main__":
    main()
