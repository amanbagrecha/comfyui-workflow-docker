#!/usr/bin/env python3
"""
CLIPSeg-based sky classifier for panoramic images.

Produces a CSV with image_id, has_sky, max_prob, inference_ms.
has_sky is True when max_prob > 0.4 (max sigmoid probability over any pixel).

Uses letterbox padding to 352x352 (no squish, 2:1 aspect preserved).
Coverage is computed over the content region only (excludes black bars).
"""
import csv
import time
from pathlib import Path

import click
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
MODEL_ID = "CIDAS/clipseg-rd64-refined"
TEXT_LABEL = "sky"
MAX_PROB_THRESHOLD = 0.4


def letterbox_352(img: Image.Image):
    """Resize to fit 352x352 maintaining aspect ratio, pad with black. No squish."""
    w, h = img.size
    scale = 352 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (352, 352), (0, 0, 0))
    paste_x = (352 - new_w) // 2
    paste_y = (352 - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory of input images.",
)
@click.option(
    "--output-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path for output CSV (image_id, has_sky, max_prob, inference_ms).",
)
@click.option(
    "--model-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to CLIPSeg model cache directory (HuggingFace cache root).",
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    show_default=True,
    help="Torch device.",
)
def main(input_dir: Path, output_csv: Path, model_path: Path, device: str):
    """Classify panoramic images for sky presence using CLIPSeg."""
    t_start = time.perf_counter()

    click.echo(f"Device: {device}")
    click.echo(f"Loading CLIPSeg from {model_path} ...")
    t_load = time.perf_counter()
    model = CLIPSegForImageSegmentation.from_pretrained(
        MODEL_ID, cache_dir=str(model_path), local_files_only=True
    ).to(device).eval()
    processor = CLIPSegProcessor.from_pretrained(
        MODEL_ID, cache_dir=str(model_path), local_files_only=True
    )
    processor.image_processor.do_resize = False
    processor.image_processor.do_center_crop = False
    if device == "cuda":
        torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - t_load) * 1000
    click.echo(f"Model load time: {model_load_ms:.1f} ms")

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise click.ClickException(f"No images found in {input_dir}")

    click.echo(f"Classifying {len(image_paths)} images ...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "has_sky", "max_prob", "inference_ms"])
        writer.writeheader()

        for img_path in image_paths:
            img = letterbox_352(Image.open(img_path).convert("RGB"))
            inputs = processor(
                text=[TEXT_LABEL], images=[img],
                padding="max_length", return_tensors="pt",
            )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(**inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            inf_ms = (time.perf_counter() - t0) * 1000

            max_prob = torch.sigmoid(out.logits).max().item()
            has_sky = max_prob > MAX_PROB_THRESHOLD

            row = {
                "image_id": img_path.name,
                "has_sky": has_sky,
                "max_prob": round(max_prob, 4),
                "inference_ms": round(inf_ms, 2),
            }
            writer.writerow(row)
            rows.append(row)
            click.echo(
                f"  {'HAS_SKY' if has_sky else 'NO_SKY ':7}  "
                f"max_prob={max_prob:.4f}  {img_path.name}"
            )

    total_ms = (time.perf_counter() - t_start) * 1000
    has_count = sum(1 for r in rows if r["has_sky"])
    no_count = len(rows) - has_count

    click.echo(f"\nmodel_load_ms={model_load_ms:.1f}")
    click.echo(f"total_elapsed_ms={total_ms:.1f}")
    click.echo(f"count_sky_has={has_count}")
    click.echo(f"count_sky_no={no_count}")
    click.echo(f"output_csv={output_csv}")


if __name__ == "__main__":
    main()
