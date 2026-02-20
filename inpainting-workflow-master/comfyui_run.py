import json
import time
import uuid
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import requests


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
COMFY_INPUT_ROOT = Path("/workspace/ComfyUI/input")
COMFY_OUTPUT_ROOT = Path("/workspace/ComfyUI/output")


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
@click.option(
    "--workflow-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--mask",
    "mask_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("comfy_outputs"),
    show_default=True,
)
@click.option("--server", default="http://127.0.0.1:8188", show_default=True)
@click.option("--workers", default=3, show_default=True)
@click.option("--overwrite", is_flag=True)
@click.option(
    "--image-node-id",
    default="349",
    show_default=True,
    help="LoadImage node for the main image",
)
@click.option(
    "--mask-node-id",
    default="463",
    show_default=True,
    help="LoadImage node for the mask image",
)
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

    Mask and images are referenced directly from /workspace/ComfyUI/input.
    """

    server = server.rstrip("/")
    input_dir = input_dir.resolve()
    mask_path = mask_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        output_subdir = output_dir.relative_to(COMFY_OUTPUT_ROOT).as_posix()
    except ValueError as exc:
        raise click.ClickException(
            f"--output-dir must be inside {COMFY_OUTPUT_ROOT}: {output_dir}"
        ) from exc
    output_path_value = output_subdir if output_subdir else "."

    # Read workflow JSON (either raw prompt dict OR wrapper with {"nodes": ...})
    wf_raw = json.loads(workflow_json.read_text(encoding="utf-8"))
    prompt_template = (
        wf_raw["nodes"]
        if isinstance(wf_raw, dict) and isinstance(wf_raw.get("nodes"), dict)
        else wf_raw
    )
    if not isinstance(prompt_template, dict):
        raise click.ClickException(
            "Workflow JSON must be a dict or contain a top-level 'nodes' dict"
        )

    # Validate node IDs exist
    if image_node_id not in prompt_template:
        raise click.ClickException(
            f"image-node-id {image_node_id} not found in workflow"
        )
    if mask_node_id not in prompt_template:
        raise click.ClickException(f"mask-node-id {mask_node_id} not found in workflow")
    if prompt_template[image_node_id].get("class_type") != "LoadImage":
        raise click.ClickException(f"Node {image_node_id} is not a LoadImage node")
    if prompt_template[mask_node_id].get("class_type") != "LoadImage":
        raise click.ClickException(f"Node {mask_node_id} is not a LoadImage node")

    save_node_ids = [
        node_id
        for node_id, node in prompt_template.items()
        if isinstance(node, dict) and node.get("class_type") == "Image Save"
    ]
    if not save_node_ids:
        raise click.ClickException("No 'Image Save' node found in workflow")

    # List input images
    images = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )
    if not images:
        raise click.ClickException(f"No images found in: {input_dir}")

    def _to_comfy_input_name(local_path: Path) -> str:
        try:
            return local_path.resolve().relative_to(COMFY_INPUT_ROOT).as_posix()
        except ValueError as exc:
            raise click.ClickException(
                f"Path must be inside {COMFY_INPUT_ROOT}: {local_path}"
            ) from exc

    mask_input_name = _to_comfy_input_name(mask_path)
    click.echo(f"Using fixed mask from ComfyUI input: {mask_input_name}")

    # Per-image worker
    @timer
    def _run_one(img_path: Path) -> str:
        input_image_name = _to_comfy_input_name(img_path)

        # Clone the prompt dict (deep-ish copy) so threads don't fight
        prompt = json.loads(json.dumps(prompt_template))

        # Inject direct filenames into LoadImage nodes (no upload API copy)
        prompt[image_node_id].setdefault("inputs", {})["image"] = input_image_name
        prompt[mask_node_id].setdefault("inputs", {})["image"] = mask_input_name

        expected_outputs = []
        for idx, node_id in enumerate(save_node_ids):
            save_inputs = prompt[node_id].setdefault("inputs", {})
            save_prefix = (
                f"{img_path.stem}_comfyui"
                if idx == 0
                else f"{img_path.stem}_comfyui_{node_id}"
            )
            save_inputs["output_path"] = output_path_value
            save_inputs["filename_prefix"] = save_prefix
            save_inputs["overwrite_mode"] = "prefix_as_filename"

            extension = str(save_inputs.get("extension", "jpg")).lstrip(".") or "jpg"
            expected_outputs.append(output_dir / f"{save_prefix}.{extension}")

        if all(p.exists() for p in expected_outputs) and not overwrite:
            return f"SKIP {img_path.name} (exists)"

        # Submit prompt
        client_id = str(uuid.uuid4())
        r = requests.post(
            f"{server}/prompt", json={"prompt": prompt, "client_id": client_id}
        )
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
                raise TimeoutError(
                    f"Timed out waiting for prompt_id={prompt_id} for {img_path.name}"
                )
            time.sleep(poll_s)

        missing = [str(p) for p in expected_outputs if not p.exists()]
        if missing:
            raise RuntimeError(
                f"Prompt completed but expected output file(s) not found for {img_path.name}: {', '.join(missing)}"
            )

        return (
            f"OK   {img_path.name} -> {expected_outputs[0].name}"
            if len(expected_outputs) == 1
            else f"OK   {img_path.name} -> {img_path.stem}_comfyui_*"
        )

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
