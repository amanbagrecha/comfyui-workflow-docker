#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def parse_strict(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def image_exts(include_bmp: bool = False) -> set[str]:
    exts = set(BASE_IMAGE_EXTS)
    if include_bmp:
        exts.add(".bmp")
    return exts


def count_images(path: Path, include_bmp: bool = False) -> int:
    if not path.exists():
        return 0
    exts = image_exts(include_bmp=include_bmp)
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def wait_http(args: argparse.Namespace) -> int:
    deadline = time.time() + args.timeout
    last_error = None

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=5) as response:
                if response.status == 200:
                    print("ComfyUI API ready")
                    return 0
        except Exception as exc:  # pragma: no cover - readiness polling
            last_error = exc
        time.sleep(args.poll)

    print(
        f"ERROR: ComfyUI API not ready within {args.timeout}s. Last error: {last_error}"
    )
    return 1


def stage_images(args: argparse.Namespace) -> int:
    src = args.src
    dst = args.dst
    strict_hardlink = parse_strict(args.strict_hardlink)
    dst.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix.lower() in BASE_IMAGE_EXTS]
    for image in images:
        target = dst / image.name
        if target.exists():
            src_stat = image.stat()
            dst_stat = target.stat()
            same_inode = src_stat.st_dev == dst_stat.st_dev and src_stat.st_ino == dst_stat.st_ino
            if same_inode:
                continue
            if strict_hardlink:
                raise SystemExit(
                    "Hardlink requirement failed (STRICT_HARDLINK=1): "
                    f"existing target is not a hardlink: {target}"
                )
            shutil.copy2(image, target)
            continue

        try:
            target.hardlink_to(image)
        except OSError as exc:
            if strict_hardlink:
                raise SystemExit(
                    f"Hardlink failed (STRICT_HARDLINK=1): {image} -> {target}: {exc}"
                )
            shutil.copy2(image, target)

    print(f"staged_images={len(images)}")
    return 0


def count_images_cmd(args: argparse.Namespace) -> int:
    print(count_images(args.path, include_bmp=args.include_bmp))
    return 0


def coerce_scalar(value: str):
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_kv_list(items: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"Invalid key=value item: {item}")
        parsed[key] = coerce_scalar(value)
    return parsed


def append_event(args: argparse.Namespace) -> int:
    record: dict[str, object] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "level": args.level,
        "run_type": args.run_type,
        "run_id": args.run_id,
        "event": args.event,
        "status": args.status,
    }

    optional_fields = {
        "script": args.script,
        "parent_run_id": args.parent_run_id,
        "stage": args.stage,
        "gpu_id": args.gpu_id,
        "shard_index": args.shard_index,
        "container_name": args.container_name,
        "batch_name": args.batch_name,
        "elapsed_sec": args.elapsed_sec,
        "exit_code": args.exit_code,
        "command": args.command,
        "error": args.error,
        "reason": args.reason,
        "child_run_id": args.child_run_id,
    }
    for key, value in optional_fields.items():
        if value is not None and value != "":
            record[key] = value

    params = parse_kv_list(args.param)
    metrics = parse_kv_list(args.metric)
    paths = parse_kv_list(args.path)
    if params:
        record["params"] = params
    if metrics:
        record["metrics"] = metrics
    if paths:
        record["paths"] = paths

    args.file.parent.mkdir(parents=True, exist_ok=True)
    with args.file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True))
        fh.write("\n")
    return 0


def split_shards(args: argparse.Namespace) -> int:
    src = args.src.resolve()
    shards_root = args.shards_root.resolve()
    manifest_path = args.manifest_json.resolve()
    count_path = args.count_json.resolve()
    strict_hardlink = parse_strict(args.strict_hardlink)
    exts = image_exts(include_bmp=True)

    files = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise SystemExit(f"No image files found in {src}")

    for i in range(args.num_gpus):
        shard_dir = shards_root / f"gpu{i}"
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    counts = {str(i): 0 for i in range(args.num_gpus)}

    start = 0
    base = len(files) // args.num_gpus
    remainder = len(files) % args.num_gpus

    for shard_i in range(args.num_gpus):
        shard_size = base + (1 if shard_i < remainder else 0)
        shard_dir = shards_root / f"gpu{shard_i}"
        shard_files = files[start : start + shard_size]
        start += shard_size

        for src_file in shard_files:
            stem = src_file.stem
            suffix = src_file.suffix.lower()
            dst = shard_dir / f"{stem}{suffix}"
            dup_idx = 1
            while dst.exists():
                dst = shard_dir / f"{stem}__dup{dup_idx:03d}{suffix}"
                dup_idx += 1

            try:
                dst.hardlink_to(src_file)
            except OSError as exc:
                if strict_hardlink:
                    raise SystemExit(
                        f"Hardlink failed (STRICT_HARDLINK=1): {src_file} -> {dst}: {exc}"
                    )
                shutil.copy2(src_file, dst)

            manifest.append(
                {
                    "src": str(src_file),
                    "dst": str(dst),
                    "shard_index": shard_i,
                    "dst_name": dst.name,
                }
            )
            counts[str(shard_i)] += 1

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    count_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")

    print(f"input_images={len(files)}")
    for i in range(args.num_gpus):
        print(f"shard_gpu_index_{i}_count={counts[str(i)]}")
    print(f"manifest={manifest_path}")
    return 0


def link_flat(args: argparse.Namespace) -> int:
    src = args.src
    dst = args.dst
    strict_hardlink = parse_strict(args.strict_hardlink)
    dst.mkdir(parents=True, exist_ok=True)

    for file in src.iterdir():
        if not file.is_file():
            continue
        target = dst / file.name
        try:
            if target.exists():
                target.unlink()
            target.hardlink_to(file)
        except OSError as exc:
            if strict_hardlink:
                raise SystemExit(
                    f"Hardlink failed (STRICT_HARDLINK=1): {file} -> {target}: {exc}"
                )
            shutil.copy2(file, target)

    print(f"final_output_written={dst}")
    return 0


def link_tree(args: argparse.Namespace) -> int:
    src = args.src
    dst = args.dst
    strict_hardlink = parse_strict(args.strict_hardlink)
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        return 0

    for file in src.rglob("*"):
        if not file.is_file():
            continue
        target = dst / file.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        try:
            target.hardlink_to(file)
        except OSError as exc:
            if strict_hardlink:
                raise SystemExit(
                    f"Hardlink failed (STRICT_HARDLINK=1): {file} -> {target}: {exc}"
                )
            shutil.copy2(file, target)

    return 0


def report_counts(args: argparse.Namespace) -> int:
    print(f"count_input={count_images(args.input_dir)}")
    print(f"count_sam3_mask={count_images(args.sam3_dir)}")
    print(f"count_inpainting={count_images(args.inpainting_dir)}")
    print(f"count_postprocess={count_images(args.postprocess_dir)}")
    print(f"count_egoblur={count_images(args.egoblur_dir)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline helper utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    wait_parser = subparsers.add_parser("wait-http")
    wait_parser.add_argument("--url", required=True)
    wait_parser.add_argument("--timeout", type=int, required=True)
    wait_parser.add_argument("--poll", type=float, required=True)
    wait_parser.set_defaults(func=wait_http)

    stage_parser = subparsers.add_parser("stage-images")
    stage_parser.add_argument("--src", type=Path, required=True)
    stage_parser.add_argument("--dst", type=Path, required=True)
    stage_parser.add_argument("--strict-hardlink", default="1")
    stage_parser.set_defaults(func=stage_images)

    count_parser = subparsers.add_parser("count-images")
    count_parser.add_argument("--path", type=Path, required=True)
    count_parser.add_argument("--include-bmp", action="store_true")
    count_parser.set_defaults(func=count_images_cmd)

    event_parser = subparsers.add_parser("append-event")
    event_parser.add_argument("--file", type=Path, required=True)
    event_parser.add_argument("--run-type", required=True)
    event_parser.add_argument("--run-id", required=True)
    event_parser.add_argument("--event", required=True)
    event_parser.add_argument("--status", required=True)
    event_parser.add_argument("--level", default="INFO")
    event_parser.add_argument("--script")
    event_parser.add_argument("--parent-run-id")
    event_parser.add_argument("--stage")
    event_parser.add_argument("--gpu-id")
    event_parser.add_argument("--shard-index", type=int)
    event_parser.add_argument("--container-name")
    event_parser.add_argument("--batch-name")
    event_parser.add_argument("--elapsed-sec", type=float)
    event_parser.add_argument("--exit-code", type=int)
    event_parser.add_argument("--command")
    event_parser.add_argument("--error")
    event_parser.add_argument("--reason")
    event_parser.add_argument("--child-run-id")
    event_parser.add_argument("--param", action="append", default=[])
    event_parser.add_argument("--metric", action="append", default=[])
    event_parser.add_argument("--path", action="append", default=[])
    event_parser.set_defaults(func=append_event)

    shard_parser = subparsers.add_parser("split-shards")
    shard_parser.add_argument("--src", type=Path, required=True)
    shard_parser.add_argument("--shards-root", type=Path, required=True)
    shard_parser.add_argument("--num-gpus", type=int, required=True)
    shard_parser.add_argument("--manifest-json", type=Path, required=True)
    shard_parser.add_argument("--count-json", type=Path, required=True)
    shard_parser.add_argument("--strict-hardlink", default="1")
    shard_parser.set_defaults(func=split_shards)

    link_parser = subparsers.add_parser("link-flat")
    link_parser.add_argument("--src", type=Path, required=True)
    link_parser.add_argument("--dst", type=Path, required=True)
    link_parser.add_argument("--strict-hardlink", default="1")
    link_parser.set_defaults(func=link_flat)

    link_tree_parser = subparsers.add_parser("link-tree")
    link_tree_parser.add_argument("--src", type=Path, required=True)
    link_tree_parser.add_argument("--dst", type=Path, required=True)
    link_tree_parser.add_argument("--strict-hardlink", default="1")
    link_tree_parser.set_defaults(func=link_tree)

    report_parser = subparsers.add_parser("report-counts")
    report_parser.add_argument("--input-dir", type=Path, required=True)
    report_parser.add_argument("--sam3-dir", type=Path, required=True)
    report_parser.add_argument("--inpainting-dir", type=Path, required=True)
    report_parser.add_argument("--postprocess-dir", type=Path, required=True)
    report_parser.add_argument("--egoblur-dir", type=Path, required=True)
    report_parser.set_defaults(func=report_counts)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
