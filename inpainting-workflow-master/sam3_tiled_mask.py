#!/usr/bin/env python3

import gc
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

DEFAULT_MODEL_PATH = "/workspace/ComfyUI/models/sam3"


def _pad_to_tile_grid(rgb_u8, rows, cols):
    h, w = rgb_u8.shape[:2]
    pad_right = (cols - (w % cols)) % cols
    pad_bottom = (rows - (h % rows)) % rows
    if pad_right == 0 and pad_bottom == 0:
        return rgb_u8, 0, 0
    return (
        np.pad(rgb_u8, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge"),
        pad_right,
        pad_bottom,
    )


def _tile_windows(width, height, rows, cols, overlap_ratio, overlap_x, overlap_y):
    tile_h = height // rows
    tile_w = width // cols
    overlap_h = min(tile_h // 2, int(tile_h * overlap_ratio) + overlap_y)
    overlap_w = min(tile_w // 2, int(tile_w * overlap_ratio) + overlap_x)
    if rows == 1:
        overlap_h = 0
    if cols == 1:
        overlap_w = 0

    windows = []
    for i in range(rows):
        for j in range(cols):
            y1 = i * tile_h - (overlap_h if i > 0 else 0)
            x1 = j * tile_w - (overlap_w if j > 0 else 0)
            y2 = y1 + tile_h + overlap_h
            x2 = x1 + tile_w + overlap_w
            if y2 > height:
                y2 = height
                y1 = y2 - tile_h - overlap_h
            if x2 > width:
                x2 = width
                x1 = x2 - tile_w - overlap_w
            windows.append((x1, y1, x2, y2))
    return windows


def _pad_mask_to_square(mask_u8):
    h, w = mask_u8.shape
    if h == w:
        return mask_u8
    side = max(h, w)
    if w >= h:
        pt = (side - h) // 2
        return np.pad(mask_u8, ((pt, side - h - pt), (0, 0)), mode="constant")
    pl = (side - w) // 2
    return np.pad(mask_u8, ((0, 0), (pl, side - w - pl)), mode="constant")


_MODEL = None
_PROCESSOR = None
_CFG = None


def _worker_init(cfg):
    global _MODEL, _PROCESSOR, _CFG
    _CFG = cfg
    _PROCESSOR = Sam3Processor.from_pretrained(cfg["model_path"])
    _MODEL = Sam3Model.from_pretrained(cfg["model_path"]).to(cfg["device"])
    _MODEL.eval()


def _infer_mask(tile_pil, prompt, threshold):
    h, w = tile_pil.size[1], tile_pil.size[0]
    inputs = _PROCESSOR(images=tile_pil, text=prompt.strip(), return_tensors="pt").to(
        _CFG["device"]
    )
    with torch.no_grad():
        outputs = _MODEL(**inputs)
    results = _PROCESSOR.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=_CFG["mask_threshold"],
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = results.get("masks")
    if masks is None or len(masks) == 0:
        return np.zeros((h, w), dtype=np.float32)
    return torch.stack(list(masks)).float().amax(dim=0).cpu().numpy()


def _process_one(task):
    in_path, out_path = Path(task[0]), Path(task[1])
    t0 = time.perf_counter()
    cfg = _CFG

    rgb_u8 = np.asarray(Image.open(in_path).convert("RGB"), dtype=np.uint8)
    if cfg["resize_width"] > 0 and cfg["resize_height"] > 0:
        rgb_u8 = np.asarray(
            Image.fromarray(rgb_u8).resize(
                (cfg["resize_width"], cfg["resize_height"]), LANCZOS
            ),
            dtype=np.uint8,
        )

    rgb_u8, pad_right, pad_bottom = _pad_to_tile_grid(
        rgb_u8, cfg["tile_rows"], cfg["tile_cols"]
    )
    h, w = rgb_u8.shape[:2]
    windows = _tile_windows(
        w,
        h,
        cfg["tile_rows"],
        cfg["tile_cols"],
        cfg["overlap_ratio"],
        cfg["overlap_x"],
        cfg["overlap_y"],
    )

    stitched = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in windows:
        tile_u8 = rgb_u8[y1:y2, x1:x2]
        tile_pad = max(0, int(cfg["tile_pad"]))
        if tile_pad > 0:
            tile_u8 = np.pad(
                tile_u8,
                ((tile_pad, tile_pad), (tile_pad, tile_pad), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        tile_pil = Image.fromarray(tile_u8, mode="RGB")
        sky = _infer_mask(tile_pil, cfg["sky_prompt"], cfg["sky_threshold"])
        glare = _infer_mask(tile_pil, cfg["glare_prompt"], cfg["glare_threshold"])

        if tile_pad > 0:
            sky = sky[tile_pad:-tile_pad, tile_pad:-tile_pad]
            glare = glare[tile_pad:-tile_pad, tile_pad:-tile_pad]

        tile_mask = np.maximum(sky, glare)
        stitched[y1:y2, x1:x2] = np.maximum(
            stitched[y1:y2, x1:x2], tile_mask[: y2 - y1, : x2 - x1]
        )

    if pad_bottom > 0:
        stitched = stitched[:-pad_bottom, :]
    if pad_right > 0:
        stitched = stitched[:, :-pad_right]

    stitched = (stitched >= cfg["mask_threshold"]).astype(np.uint8) * 255
    if cfg["square_output"]:
        stitched = _pad_mask_to_square(stitched)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stitched, mode="L").save(out_path)
    return {
        "input": str(in_path),
        "output": str(out_path),
        "elapsed_sec": time.perf_counter() - t0,
    }


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "-i",
    "--input-dir",
    "input_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "-o", "--output-dir", "output_dir", required=True, type=click.Path(path_type=Path)
)
@click.option("--pattern", default="*.jpg", show_default=True)
@click.option("--recursive/--no-recursive", default=True, show_default=True)
@click.option("--skip-existing/--no-skip-existing", default=True, show_default=True)
@click.option("-j", "--workers", default=1, show_default=True, type=int)
@click.option(
    "--model-path",
    default=DEFAULT_MODEL_PATH,
    show_default=True,
    type=str,
    help="HuggingFace hub ID (e.g. 'facebook/sam3') or local model directory.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "cuda", "cpu"]),
)
@click.option("--resize-width", default=2000, show_default=True, type=int)
@click.option("--resize-height", default=1000, show_default=True, type=int)
@click.option("--tile-rows", default=1, show_default=True, type=int)
@click.option("--tile-cols", default=2, show_default=True, type=int)
@click.option("--overlap", "overlap_ratio", default=0.0, show_default=True, type=float)
@click.option("--overlap-x", default=30, show_default=True, type=int)
@click.option("--overlap-y", default=0, show_default=True, type=int)
@click.option("--tile-pad", default=20, show_default=True, type=int)
@click.option("--sky-prompt", default="sky", show_default=True)
@click.option("--sky-threshold", default=0.5, show_default=True, type=float)
@click.option("--glare-prompt", default="sun glare", show_default=True)
@click.option("--glare-threshold", default=0.4, show_default=True, type=float)
@click.option("--mask-threshold", default=0.5, show_default=True, type=float)
@click.option("--square-output/--no-square-output", default=True, show_default=True)
def main(
    input_path,
    output_dir,
    pattern,
    recursive,
    skip_existing,
    workers,
    model_path,
    device,
    resize_width,
    resize_height,
    tile_rows,
    tile_cols,
    overlap_ratio,
    overlap_x,
    overlap_y,
    tile_pad,
    sky_prompt,
    sky_threshold,
    glare_prompt,
    glare_threshold,
    mask_threshold,
    square_output,
):
    t0 = time.perf_counter()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if input_path.is_file():
        in_files, in_root = [input_path], input_path.parent
    else:
        files = sorted(
            input_path.rglob(pattern) if recursive else input_path.glob(pattern)
        )
        in_files = [p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        in_root = input_path

    if not in_files:
        raise click.ClickException(f"No images found under: {input_path}")

    tasks = [
        (str(f), str(output_dir / f.relative_to(in_root).with_suffix(".png")))
        for f in in_files
        if not (
            skip_existing
            and (output_dir / f.relative_to(in_root).with_suffix(".png")).exists()
        )
    ]

    if not tasks:
        click.echo("No work to do (all outputs already exist).")
        return

    cfg = dict(
        model_path=model_path,
        device=device,
        resize_width=resize_width,
        resize_height=resize_height,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        overlap_ratio=overlap_ratio,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
        tile_pad=tile_pad,
        sky_prompt=sky_prompt,
        sky_threshold=sky_threshold,
        glare_prompt=glare_prompt,
        glare_threshold=glare_threshold,
        mask_threshold=mask_threshold,
        square_output=square_output,
    )

    click.echo(
        f"SAM3 tiled mask: input={len(in_files)} pending={len(tasks)} workers={workers} device={device}"
    )
    click.echo(
        f"resize={resize_width}x{resize_height} tiles={tile_rows}x{tile_cols} overlap=({overlap_x},{overlap_y}) tile_pad={tile_pad} sky='{sky_prompt}'({sky_threshold}) glare='{glare_prompt}'({glare_threshold})"
    )

    results, failures = [], []

    if workers == 1:
        _worker_init(cfg)
        for task in tasks:
            result = _process_one(task)
            results.append(result)
            click.echo(
                f"OK   {Path(result['input']).name} ({result['elapsed_sec']:.2f}s)"
            )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(cfg,),
            mp_context=ctx,
        ) as ex:
            futs = {ex.submit(_process_one, task): task for task in tasks}
            for fut in as_completed(futs):
                task = futs[fut]
                try:
                    result = fut.result()
                    results.append(result)
                    click.echo(
                        f"OK   {Path(result['input']).name} ({result['elapsed_sec']:.2f}s)"
                    )
                except Exception as exc:
                    failures.append(f"FAIL {Path(task[0]).name}: {exc}")
                    click.echo(failures[-1], err=True)

    avg = (sum(r["elapsed_sec"] for r in results) / len(results)) if results else 0.0
    click.echo(
        f"done={len(results)} fail={len(failures)} total={time.perf_counter() - t0:.2f}s avg={avg:.2f}s"
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if failures:
        raise click.ClickException(f"{len(failures)} image(s) failed")


if __name__ == "__main__":
    main()
