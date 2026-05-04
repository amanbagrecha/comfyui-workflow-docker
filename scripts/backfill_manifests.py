#!/usr/bin/env python3
"""Backfill manifests for tar files already on S3 that have no .manifest.json.

Reads tar headers via targeted S3 range requests (512 bytes per file entry) —
never downloads full tar files. Marks generated manifests with "backfilled": true.

Requires boto3 (not in pyproject.toml — run via):
    uv run --with boto3 python scripts/backfill_manifests.py \\
        --bucket aipanoexport-batch2 \\
        --prefixes batch-03 batch-04 batch-05 batch-06 batch-07 batch-08 batch-09 batch-10 \\
        --profile s3

    # Dry run first to see what's missing:
    uv run --with boto3 python scripts/backfill_manifests.py \\
        --bucket aipanoexport-batch2 --prefixes batch-03 --profile s3 --dry-run
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone

import boto3

BLOCK = 512  # tar block size in bytes


def _parse_header(block: bytes) -> dict | None:
    """Parse one 512-byte tar header. Returns None on end-of-archive padding."""
    if len(block) < BLOCK or block == b"\x00" * BLOCK:
        return None
    try:
        name = block[:100].rstrip(b"\x00").decode("utf-8", errors="replace")
        # GNU/POSIX prefix field for long paths
        prefix = block[345:500].rstrip(b"\x00").decode("utf-8", errors="replace")
        if prefix:
            name = f"{prefix}/{name}"
        size_field = block[124:136].rstrip(b"\x00 ").decode("ascii", errors="replace")
        size = int(size_field, 8) if size_field.strip() else 0
        typeflag = chr(block[156]) if block[156] != 0 else "0"
        return {"name": name, "size": size, "typeflag": typeflag}
    except Exception as e:
        print(f"  Header parse error at block: {e}", file=sys.stderr)
        return None


def _data_blocks(size: int) -> int:
    return math.ceil(size / BLOCK) if size > 0 else 0


def _range_read(s3, bucket: str, key: str, start: int, length: int) -> bytes:
    resp = s3.get_object(
        Bucket=bucket, Key=key, Range=f"bytes={start}-{start + length - 1}"
    )
    return resp["Body"].read()


def scan_tar_headers(s3, bucket: str, key: str) -> list[dict]:
    """Walk tar headers via 512-byte S3 range reads. Returns regular-file list only."""
    files: list[dict] = []
    offset = 0
    pending_long_name: str | None = None

    while True:
        try:
            block = _range_read(s3, bucket, key, offset, BLOCK)
        except Exception as e:
            print(f"  S3 read error at offset {offset}: {e}", file=sys.stderr)
            break

        if len(block) < BLOCK or block == b"\x00" * BLOCK:
            break

        hdr = _parse_header(block)
        if hdr is None:
            break

        data_offset = offset + BLOCK
        skip = _data_blocks(hdr["size"]) * BLOCK

        if hdr["typeflag"] == "L":
            # GNU long name: data block(s) contain the real filename for the next entry
            try:
                long_name_data = _range_read(s3, bucket, key, data_offset, max(skip, BLOCK))
                pending_long_name = long_name_data.rstrip(b"\x00").decode("utf-8", errors="replace")
            except Exception:
                pending_long_name = None
        elif hdr["typeflag"] in ("0", "\x00", ""):
            name = pending_long_name if pending_long_name else hdr["name"]
            pending_long_name = None
            files.append({"name": name, "size": hdr["size"], "tar_offset": data_offset})
        else:
            pending_long_name = None

        offset = data_offset + skip

    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--prefixes", nargs="+", required=True,
        help="S3 key prefixes to scan (e.g. batch-03 batch-04)"
    )
    parser.add_argument("--profile", default="s3", help="AWS profile (default: s3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be backfilled, don't upload anything")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after this many tars (0 = no limit)")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile)
    s3 = session.client("s3")

    tar_keys: list[tuple[str, int]] = []
    manifest_keys: set[str] = set()

    print(f"Scanning s3://{args.bucket} prefixes: {args.prefixes}")
    for prefix in args.prefixes:
        clean_prefix = prefix.rstrip("/") + "/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=args.bucket, Prefix=clean_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".tar"):
                    tar_keys.append((key, obj["Size"]))
                elif key.endswith(".manifest.json"):
                    manifest_keys.add(key[: -len(".manifest.json")] + ".tar")

    to_backfill = [(k, s) for k, s in tar_keys if k not in manifest_keys]
    print(
        f"Found {len(tar_keys)} tar files — "
        f"{len(manifest_keys)} already have manifests — "
        f"{len(to_backfill)} to backfill"
    )

    if not to_backfill:
        print("Nothing to do.")
        return

    done = 0
    for i, (tar_key, tar_size) in enumerate(to_backfill, 1):
        run_id = tar_key.rsplit("/", 1)[-1].removesuffix(".tar")
        manifest_key = tar_key[: -len(".tar")] + ".manifest.json"
        print(f"[{i}/{len(to_backfill)}] {tar_key} ({tar_size:,} bytes)", end="", flush=True)

        if args.dry_run:
            print(" [dry-run]")
            continue

        files = scan_tar_headers(s3, args.bucket, tar_key)
        manifest = {
            "run_id": run_id,
            "file_count": len(files),
            "tar_bytes": tar_size,
            "backfilled": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
        }
        body = json.dumps(manifest, indent=2).encode()
        s3.put_object(
            Bucket=args.bucket,
            Key=manifest_key,
            Body=body,
            ContentType="application/json",
        )
        print(f" → {len(files)} files indexed, manifest uploaded")
        done += 1
        if args.limit and done >= args.limit:
            print(f"Reached --limit {args.limit}, stopping.")
            break

    if not args.dry_run:
        print(f"\nDone. Backfilled {done} manifest(s).")


if __name__ == "__main__":
    main()
