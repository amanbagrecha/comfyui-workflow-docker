import json
import time
import uuid
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import requests


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def timer(func):
    """Decorator that prints execution time after function completes."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"[TIMER] {func.__name__} completed in {elapsed_time:.2f}s")
        return result
    return wrapper


@click.command()
@click.option("--workflow-json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--mask", "mask_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("comfy_outputs"), show_default=True)
@click.option("--server", default="http://127.0.0.1:8188", show_default=True)
@click.option("--workers", default=3, show_default=True)
@click.option("--overwrite", is_flag=True)
@click.option("--image-node-id", default="349", show_default=True, help="LoadImage node for the main image")
@click.option("--mask-node-id", default="463", show_default=True, help="LoadImage node for the mask image")
@click.option("--poll-s", default=1.0, show_default=True)
@click.option("--timeout-s", default=1800, show_default=True)
def main(
    workflow_json: Path,
    input_dir: Path,
    mask_path: Path,
    output_dir: Path,
    server: str,
    workers: int,
    overwrite: bool,
    image_node_id: str,
    mask_node_id: str,
    poll_s: float,
    timeout_s: int,
):
    """Batch-run a ComfyUI workflow via HTTP API.

    This workflow expects:
      - main image at LoadImage node `image_node_id` (default 349)
      - mask image at LoadImage node `mask_node_id` (default 463)

    Mask is fixed (uploaded once); each input image is uploaded and injected into the workflow.
    """

    server = server.rstrip("/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read workflow JSON (either raw prompt dict OR wrapper with {"nodes": ...})
    wf_raw = json.loads(workflow_json.read_text(encoding="utf-8"))
    prompt_template = wf_raw["nodes"] if isinstance(wf_raw, dict) and isinstance(wf_raw.get("nodes"), dict) else wf_raw
    if not isinstance(prompt_template, dict):
        raise click.ClickException("Workflow JSON must be a dict or contain a top-level 'nodes' dict")

    # Validate node IDs exist
    if image_node_id not in prompt_template:
        raise click.ClickException(f"image-node-id {image_node_id} not found in workflow")
    if mask_node_id not in prompt_template:
        raise click.ClickException(f"mask-node-id {mask_node_id} not found in workflow")
    if prompt_template[image_node_id].get("class_type") != "LoadImage":
        raise click.ClickException(f"Node {image_node_id} is not a LoadImage node")
    if prompt_template[mask_node_id].get("class_type") != "LoadImage":
        raise click.ClickException(f"Node {mask_node_id} is not a LoadImage node")

    # List input images
    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise click.ClickException(f"No images found in: {input_dir}")

    # Helper: upload a local image file to ComfyUI
    def _upload(local_path: Path) -> str:
        with open(local_path, "rb") as f:
            r = requests.post(f"{server}/upload/image", files={"image": f})
        r.raise_for_status()
        j = r.json()
        return j.get("name") or local_path.name

    # Upload fixed mask ONCE
    uploaded_mask_name = _upload(mask_path)
    click.echo(f"Uploaded fixed mask: {mask_path.name} -> {uploaded_mask_name}")

    # Per-image worker
    @timer
    def _run_one(img_path: Path) -> str:
        out_file = output_dir / f"{img_path.stem}_comfyui.jpg"
        if out_file.exists() and not overwrite:
            return f"SKIP {img_path.name} (exists)"

        # Upload main image
        uploaded_img_name = _upload(img_path)

        # Clone the prompt dict (deep-ish copy) so threads don't fight
        prompt = json.loads(json.dumps(prompt_template))

        # Inject filenames into the two LoadImage nodes
        prompt[image_node_id].setdefault("inputs", {})["image"] = uploaded_img_name
        prompt[mask_node_id].setdefault("inputs", {})["image"] = uploaded_mask_name

        # Submit prompt
        client_id = str(uuid.uuid4())
        r = requests.post(f"{server}/prompt", json={"prompt": prompt, "client_id": client_id})
        r.raise_for_status()
        prompt_id = r.json()["prompt_id"]

        # Poll history until completed
        t0 = time.time()
        entry = None
        while True:
            h = requests.get(f"{server}/history/{prompt_id}")
            h.raise_for_status()
            hist = h.json()
            entry = hist.get(prompt_id)
            if entry and entry.get("status", {}).get("completed"):
                break
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Timed out waiting for prompt_id={prompt_id} for {img_path.name}")
            time.sleep(poll_s)

        # Download ALL output images from history (usually one SaveImage)
        outputs = entry.get("outputs", {}) or {}
        saved_any = False
        idx = 0
        for node_out in outputs.values():
            for img in (node_out.get("images", []) or []):
                url = (
                    f"{server}/view?filename={img['filename']}"
                    f"&subfolder={img['subfolder']}&type={img['type']}"
                )
                resp = requests.get(url, stream=True)
                resp.raise_for_status()

                target = out_file if idx == 0 else out_file.with_name(f"{out_file.stem}_{idx}{out_file.suffix}")
                with open(target, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                saved_any = True
                idx += 1

        if not saved_any:
            raise RuntimeError(f"No output images found for {img_path.name} (did SaveImage run?)")

        return f"OK   {img_path.name} -> {out_file.name}" if idx == 1 else f"OK   {img_path.name} -> {out_file.stem}_*.jpg"

    # Run in parallel
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, p): p for p in images}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                click.echo(fut.result())
            except Exception as e:
                click.echo(f"FAIL {p.name}: {e}", err=True)


if __name__ == "__main__":
    main()
