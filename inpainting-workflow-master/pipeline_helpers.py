#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
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

    link_parser = subparsers.add_parser("link-flat")
    link_parser.add_argument("--src", type=Path, required=True)
    link_parser.add_argument("--dst", type=Path, required=True)
    link_parser.add_argument("--strict-hardlink", default="1")
    link_parser.set_defaults(func=link_flat)

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
