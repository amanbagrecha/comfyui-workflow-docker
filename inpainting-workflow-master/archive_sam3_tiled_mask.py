#!/usr/bin/env python3

import gc
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
import pytorch360convert as p360
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Sam3Model, Sam3Processor


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
DEFAULT_MODEL_PATH = "/workspace/ComfyUI/models/sam3"
PREDICT_FACES = ("Front", "Right", "Back", "Left", "Up")
ALL_FACES = ("Front", "Right", "Back", "Left", "Up", "Down")


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


def _infer_mask(tile_rgb_u8, prompt, threshold):
    h, w = tile_rgb_u8.shape[:2]
    tile_pil = Image.fromarray(tile_rgb_u8, mode="RGB")
    inputs = _PROCESSOR(images=tile_pil, text=prompt.strip(), return_tensors="pt").to(
        _CFG["device"]
    )
    with torch.no_grad():
        outputs = _MODEL(**inputs)

    result = _PROCESSOR.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=_CFG["mask_threshold"],
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = result.get("masks")
    if masks is None or len(masks) == 0:
        return np.zeros((h, w), dtype=np.float32)
    return torch.stack(list(masks)).float().amax(dim=0).cpu().numpy()


def _tensor_chw_to_rgb_u8(t):
    arr = t.detach().cpu().permute(1, 2, 0).float().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _dilate_mask(mask_np: np.ndarray, iterations: int):
    iterations = max(0, int(iterations))
    if iterations == 0:
        return mask_np
    t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        t = F.max_pool2d(t, kernel_size=3, stride=1, padding=1)
    return t.squeeze(0).squeeze(0).numpy()


def _process_one(task):
    in_path, out_path = Path(task[0]), Path(task[1])
    t0 = time.perf_counter()
    cfg = _CFG

    rgb_u8 = np.array(Image.open(in_path).convert("RGB"), dtype=np.uint8, copy=True)
    h, w = rgb_u8.shape[:2]

    equi_t = torch.from_numpy(rgb_u8).permute(2, 0, 1).float().to(cfg["device"]) / 255.0
    cube_dict = p360.e2c(
        equi_t,
        face_w=cfg["face_size"],
        mode="bilinear",
        cube_format="dict",
        channels_first=True,
    )

    mask_cube = {
        face: torch.zeros(
            (1, cfg["face_size"], cfg["face_size"]),
            dtype=torch.float32,
            device=cfg["device"],
        )
        for face in ALL_FACES
    }

    for face in PREDICT_FACES:
        face_rgb_u8 = _tensor_chw_to_rgb_u8(cube_dict[face])

        if cfg["tile_pad"] > 0:
            face_rgb_u8 = np.pad(
                face_rgb_u8,
                (
                    (cfg["tile_pad"], cfg["tile_pad"]),
                    (cfg["tile_pad"], cfg["tile_pad"]),
                    (0, 0),
                ),
                mode="reflect",
            )

        sky = _infer_mask(face_rgb_u8, cfg["sky_prompt"], cfg["sky_threshold"])
        glare = _infer_mask(face_rgb_u8, cfg["glare_prompt"], cfg["glare_threshold"])
        glare = _dilate_mask(glare, cfg["glare_dilation"])
        combined = np.maximum(sky, glare)

        if cfg["tile_pad"] > 0:
            p = cfg["tile_pad"]
            combined = combined[p:-p, p:-p]

        mask_cube[face] = torch.from_numpy(combined).to(cfg["device"]).unsqueeze(0)

    eq_mask_t = p360.c2e(
        mask_cube,
        h=h,
        w=w,
        mode="bilinear",
        cube_format="dict",
        channels_first=True,
    )

    eq_mask = eq_mask_t.squeeze(0).detach().cpu().numpy()
    eq_mask_u8 = np.clip(eq_mask * 255.0, 0, 255).astype(np.uint8)
    if cfg["square_output"]:
        eq_mask_u8 = _pad_mask_to_square(eq_mask_u8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(eq_mask_u8, mode="L").save(out_path)
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
@click.option("-j", "--workers", default=4, show_default=True, type=int)
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
@click.option("--resize-width", default=0, show_default=True, type=int)
@click.option("--resize-height", default=0, show_default=True, type=int)
@click.option("--tile-rows", default=1, show_default=True, type=int)
@click.option("--tile-cols", default=2, show_default=True, type=int)
@click.option("--overlap", "overlap_ratio", default=0.0, show_default=True, type=float)
@click.option("--overlap-x", default=30, show_default=True, type=int)
@click.option("--overlap-y", default=0, show_default=True, type=int)
@click.option("--tile-pad", default=20, show_default=True, type=int)
@click.option("--face-size", default=1000, show_default=True, type=int)
@click.option("--sky-prompt", default="sky", show_default=True)
@click.option("--sky-threshold", default=0.5, show_default=True, type=float)
@click.option("--glare-prompt", default="sunlight", show_default=True)
@click.option("--glare-threshold", default=0.4, show_default=True, type=float)
@click.option("--glare-dilation", default=3, show_default=True, type=int)
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
    face_size,
    sky_prompt,
    sky_threshold,
    glare_prompt,
    glare_threshold,
    glare_dilation,
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

    if resize_width > 0 or resize_height > 0:
        click.echo(
            f"NOTE: --resize-width/--resize-height are deprecated and ignored (requested {resize_width}x{resize_height})."
        )

    cfg = dict(
        model_path=model_path,
        device=device,
        face_size=face_size,
        tile_pad=max(0, int(tile_pad)),
        sky_prompt=sky_prompt,
        sky_threshold=sky_threshold,
        glare_prompt=glare_prompt,
        glare_threshold=glare_threshold,
        glare_dilation=max(0, int(glare_dilation)),
        mask_threshold=mask_threshold,
        square_output=square_output,
    )

    click.echo(
        f"SAM3 cube mask: input={len(in_files)} pending={len(tasks)} workers={workers} device={device} face_size={face_size}"
    )
    click.echo(
        f"tile-compat rows={tile_rows} cols={tile_cols} overlap=({overlap_x},{overlap_y}) overlap_ratio={overlap_ratio} pad={tile_pad} sky='{sky_prompt}'({sky_threshold}) glare='{glare_prompt}'({glare_threshold}) glare_dilation={max(0, int(glare_dilation))}"
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
