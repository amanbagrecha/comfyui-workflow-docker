#!/usr/bin/env python3

import time
from pathlib import Path

import click
import numpy as np
import pytorch360convert as p360
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

PREDICT_FACES = ("Front", "Right", "Back", "Left", "Up")
ALL_FACES = ("Front", "Right", "Back", "Left", "Up", "Down")


def tensor_chw_to_rgb_u8(t: torch.Tensor) -> np.ndarray:
    if t.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(t.shape)}")
    arr = t.detach().cpu().permute(1, 2, 0).float().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def infer_mask(
    processor, model, device, tile_rgb_u8, prompt, threshold, mask_threshold
):
    tile_pil = Image.fromarray(tile_rgb_u8, mode="RGB")
    inputs = processor(images=tile_pil, text=prompt.strip(), return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = result.get("masks")
    h, w = tile_rgb_u8.shape[:2]
    if masks is None or len(masks) == 0:
        return np.zeros((h, w), dtype=np.float32)
    return torch.stack(list(masks)).float().amax(dim=0).cpu().numpy()


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "-i", "--input-dir", required=True, type=click.Path(path_type=Path, exists=True)
)
@click.option("-o", "--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--pattern", default="*.jpg", show_default=True)
@click.option("--recursive/--no-recursive", default=False, show_default=True)
@click.option("--limit", default=20, show_default=True, type=int)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
@click.option(
    "--model-path",
    default="/data/comfyui-workflow-docker/repo/models/comfyui/sam3",
    show_default=True,
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "cuda", "cpu"]),
)
@click.option("--face-size", default=1000, show_default=True, type=int)
@click.option("--pad", default=20, show_default=True, type=int)
@click.option("--sky-prompt", default="sky", show_default=True)
@click.option("--sky-threshold", default=0.5, show_default=True, type=float)
@click.option("--glare-prompt", default="sunlight", show_default=True)
@click.option("--glare-threshold", default=0.4, show_default=True, type=float)
@click.option("--mask-threshold", default=0.5, show_default=True, type=float)
def main(
    input_dir,
    output_dir,
    pattern,
    recursive,
    limit,
    overwrite,
    model_path,
    device,
    face_size,
    pad,
    sky_prompt,
    sky_threshold,
    glare_prompt,
    glare_threshold,
    mask_threshold,
):
    t0 = time.perf_counter()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    files = sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    images = [p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if limit > 0:
        images = images[:limit]
    if not images:
        raise click.ClickException(f"No images found in {input_dir} matching {pattern}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processor = Sam3Processor.from_pretrained(model_path)
    model = Sam3Model.from_pretrained(model_path).to(device)
    model.eval()

    click.echo(
        f"equi2cube SAM3: images={len(images)} device={device} face_size={face_size} pad={pad} prompts=('{sky_prompt}','{glare_prompt}')"
    )

    done = 0
    for image_path in images:
        out_path = output_dir / f"{image_path.stem}.png"
        if out_path.exists() and not overwrite:
            continue

        t1 = time.perf_counter()
        rgb_u8 = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        h, w = rgb_u8.shape[:2]

        equi_t = torch.from_numpy(rgb_u8).permute(2, 0, 1).float().to(device) / 255.0
        cube_dict = p360.e2c(
            equi_t,
            face_w=face_size,
            mode="bilinear",
            cube_format="dict",
            channels_first=True,
        )

        mask_cube = {
            face: torch.zeros(
                (1, face_size, face_size), dtype=torch.float32, device=device
            )
            for face in ALL_FACES
        }

        for face in PREDICT_FACES:
            face_rgb_u8 = tensor_chw_to_rgb_u8(cube_dict[face])

            if pad > 0:
                face_rgb_u8 = np.pad(
                    face_rgb_u8,
                    ((pad, pad), (pad, pad), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            sky = infer_mask(
                processor,
                model,
                device,
                face_rgb_u8,
                sky_prompt,
                sky_threshold,
                mask_threshold,
            )
            glare = infer_mask(
                processor,
                model,
                device,
                face_rgb_u8,
                glare_prompt,
                glare_threshold,
                mask_threshold,
            )

            combined = np.maximum(sky, glare)

            if pad > 0:
                combined = combined[pad:-pad, pad:-pad]

            combined = (combined >= mask_threshold).astype(np.float32)
            mask_cube[face] = torch.from_numpy(combined).to(device).unsqueeze(0)

        # Keep Down face as zeros by design.
        eq_mask_t = p360.c2e(
            mask_cube,
            h=h,
            w=w,
            mode="bilinear",
            cube_format="dict",
            channels_first=True,
        )

        eq_mask = eq_mask_t.squeeze(0).detach().cpu().numpy()
        eq_mask_u8 = (eq_mask >= 0.5).astype(np.uint8) * 255
        Image.fromarray(eq_mask_u8, mode="L").save(out_path)

        done += 1
        click.echo(
            f"OK   {image_path.name} -> {out_path.name} ({time.perf_counter() - t1:.2f}s)"
        )

    click.echo(
        f"done={done} total_images={len(images)} elapsed={time.perf_counter() - t0:.2f}s"
    )


if __name__ == "__main__":
    main()
