#!/usr/bin/env python3
"""Generate a JSON manifest for a local tar file.

Records each file's name, byte size, and exact byte offset within the tar so
individual files can be fetched from S3 without downloading the whole archive:

    aws s3api get-object --bucket BUCKET --key path/run.tar \
        --range "bytes=<tar_offset>-<tar_offset+size-1>" out.jpg

Usage:
    uv run python scripts/generate_manifest.py <tar_path> <run_id> [--out <path>]

Called automatically by orchestrate.sh after tar creation.
"""
import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def build_manifest(tar_path: Path, run_id: str) -> dict:
    files = []
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if m.isfile():
                files.append({
                    "name": m.name,
                    "size": m.size,
                    "tar_offset": m.offset_data,
                })
    return {
        "run_id": run_id,
        "file_count": len(files),
        "tar_bytes": tar_path.stat().st_size,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tar_path", help="Path to the local tar file")
    parser.add_argument("run_id", help="Run ID written into the manifest")
    parser.add_argument("--out", help="Output path (default: <tar_path without .tar>.manifest.json)")
    args = parser.parse_args()

    tar_path = Path(args.tar_path)
    out_path = Path(args.out) if args.out else tar_path.with_suffix(".manifest.json")

    manifest = build_manifest(tar_path, args.run_id)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest['file_count']} files, {manifest['tar_bytes']:,} bytes → {out_path}")


if __name__ == "__main__":
    main()
