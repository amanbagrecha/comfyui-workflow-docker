import json
import time
import uuid
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import click
import requests


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
COMFY_INPUT_ROOT = Path("/workspace/ComfyUI/input")
COMFY_OUTPUT_ROOT = Path("/workspace/ComfyUI/output")
IMAGE_NODE_TYPES = {"LoadImage", "Image Load"}
MASK_NODE_TYPES = {"LoadImage", "LoadImageMask", "Image Load"}
SAVE_NODE_SUFFIX_BY_ID = {
    "54": "comfyui_newsky",
    "59": "comfyui_carremoved",
}


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
    default="48",
    show_default=True,
    help="Main image node (LoadImage or WAS Image Load)",
)
@click.option(
    "--mask-node-id",
    default="34",
    show_default=True,
    help="Static mask node (LoadImage / LoadImageMask / WAS Image Load)",
)
@click.option(
    "--sam3-mask-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing per-image SAM3 mask PNGs",
)
@click.option(
    "--sam3-mask-node-id",
    default="",
    help="Per-image SAM3 mask node ID (LoadImage / LoadImageMask / WAS Image Load)",
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
    sam3_mask_dir: Optional[Path],
    sam3_mask_node_id: str,
    poll_s: float,
    timeout_s: int,
):
    """Batch-run a ComfyUI workflow via HTTP API.

    This workflow expects:
      - main image at node `image_node_id`
      - static mask image at node `mask_node_id`
      - optional per-image SAM3 mask at node `sam3_mask_node_id`

    Supports standard LoadImage nodes and WAS "Image Load" nodes.
    """

    server = server.rstrip("/")
    input_dir = input_dir.resolve()
    mask_path = mask_path.resolve()
    sam3_mask_dir = sam3_mask_dir.resolve() if sam3_mask_dir else None
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(sam3_mask_dir) != bool(sam3_mask_node_id):
        raise click.ClickException(
            "--sam3-mask-dir and --sam3-mask-node-id must be used together"
        )

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

    def _validate_node_type(node_id: str, allowed_types: set[str], label: str):
        if node_id not in prompt_template:
            raise click.ClickException(
                f"{label} node-id {node_id} not found in workflow"
            )
        class_type = str(prompt_template[node_id].get("class_type", ""))
        if class_type not in allowed_types:
            allowed = ", ".join(sorted(allowed_types))
            raise click.ClickException(
                f"Node {node_id} ({label}) has unsupported class_type '{class_type}'. "
                f"Expected one of: {allowed}"
            )

    _validate_node_type(image_node_id, IMAGE_NODE_TYPES, "image")
    _validate_node_type(mask_node_id, MASK_NODE_TYPES, "mask")
    if sam3_mask_node_id:
        _validate_node_type(sam3_mask_node_id, MASK_NODE_TYPES, "sam3-mask")

    save_node_ids = sorted(
        [
            node_id
            for node_id, node in prompt_template.items()
            if isinstance(node, dict) and node.get("class_type") == "Image Save"
        ],
        key=lambda node_id: int(node_id) if str(node_id).isdigit() else str(node_id),
    )
    if not save_node_ids:
        raise click.ClickException("No 'Image Save' node found in workflow")

    def _save_suffix(node_id: str, idx: int) -> str:
        if node_id in SAVE_NODE_SUFFIX_BY_ID:
            return SAVE_NODE_SUFFIX_BY_ID[node_id]
        if idx == 0:
            return "comfyui"
        return f"comfyui_{node_id}"

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

    def _set_image_path(
        prompt: dict, node_id: str, file_path: Path, label: str
    ) -> None:
        node = prompt[node_id]
        class_type = str(node.get("class_type", ""))
        inputs = node.setdefault("inputs", {})

        if class_type in {"LoadImage", "LoadImageMask"}:
            inputs["image"] = _to_comfy_input_name(file_path)
            return

        if class_type == "Image Load":
            inputs["image_path"] = str(file_path)
            inputs.setdefault("RGBA", "false")
            inputs.setdefault("filename_text_extension", "true")
            return

        raise click.ClickException(
            f"Unsupported class_type '{class_type}' for {label} node {node_id}"
        )

    click.echo(
        f"main_image_node={image_node_id} class={prompt_template[image_node_id].get('class_type')}"
    )
    click.echo(
        f"fixed_mask_node={mask_node_id} class={prompt_template[mask_node_id].get('class_type')} path={mask_path}"
    )
    if sam3_mask_dir:
        click.echo(
            f"sam3_mask_node={sam3_mask_node_id} class={prompt_template[sam3_mask_node_id].get('class_type')} dir={sam3_mask_dir}"
        )

    # Per-image worker
    @timer
    def _run_one(img_path: Path) -> str:
        # Clone the prompt dict (deep-ish copy) so threads don't fight
        prompt = json.loads(json.dumps(prompt_template))

        _set_image_path(prompt, image_node_id, img_path, "main image")
        _set_image_path(prompt, mask_node_id, mask_path, "fixed mask")

        if sam3_mask_dir and sam3_mask_node_id:
            rel = img_path.relative_to(input_dir)
            sam3_mask_path = sam3_mask_dir / rel.with_suffix(".png")
            if not sam3_mask_path.exists():
                raise FileNotFoundError(
                    f"SAM3 mask not found for {img_path.name}: {sam3_mask_path}"
                )
            _set_image_path(prompt, sam3_mask_node_id, sam3_mask_path, "sam3 mask")

        expected_outputs = []
        for idx, node_id in enumerate(save_node_ids):
            save_inputs = prompt[node_id].setdefault("inputs", {})
            save_prefix = f"{img_path.stem}_{_save_suffix(node_id, idx)}"
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

        if len(expected_outputs) == 1:
            return f"OK   {img_path.name} -> {expected_outputs[0].name}"
        out_names = ", ".join(p.name for p in expected_outputs)
        return f"OK   {img_path.name} -> {out_names}"

    # Run in parallel
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, p): p for p in images}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                click.echo(fut.result())
            except Exception as e:
                message = f"FAIL {p.name}: {e}"
                click.echo(message, err=True)
                failures.append(message)

    if failures:
        raise click.ClickException(
            f"{len(failures)} of {len(images)} image(s) failed in ComfyUI stage"
        )


if __name__ == "__main__":
    main()
