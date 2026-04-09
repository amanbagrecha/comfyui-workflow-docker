#!/usr/bin/env -S uv run --with boto3 --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3"]
# ///
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3


def list_s3_objects(bucket: str, prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def download_file(args_tuple):
    bucket, key, dest_dir, rank, total = args_tuple
    filename = os.path.basename(key)
    dest_path = os.path.join(dest_dir, filename)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, dest_path)
    print(f"[{rank}/{total}] Downloaded: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Parallel S3 image downloader")
    parser.add_argument("--bucket", default="pano-bkp", help="S3 bucket name")
    parser.add_argument(
        "--prefix",
        nargs="+",
        required=True,
        help="S3 prefix(es)/path(s). Pass as space-separated values.",
    )
    parser.add_argument(
        "--dest", default="/workspace/imgs", help="Local destination root"
    )
    parser.add_argument(
        "--every", type=int, default=3, help="Download every Nth file (default: 3)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="Parallel download workers (default: 50)",
    )
    args = parser.parse_args()

    for prefix in args.prefix:
        parts = prefix.rstrip("/").split("/")
        if len(parts) >= 2:
            folder_name = parts[-2] + "_" + parts[-1]
        else:
            folder_name = parts[-1]
        dest_dir = os.path.join(args.dest, folder_name)
        os.makedirs(dest_dir, exist_ok=True)

        print(f"Listing s3://{args.bucket}/{prefix} ...")
        keys = list_s3_objects(args.bucket, prefix)
        print(f"Found {len(keys)} objects")

        every_nth = keys[:: args.every]
        print(f"Downloading every {args.every}th file: {len(every_nth)} files")

        tasks = [
            (args.bucket, key, dest_dir, i + 1, len(every_nth))
            for i, key in enumerate(every_nth)
        ]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_file, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    task = futures[future]
                    print(f"ERROR downloading {task[1]}: {e}")

        print(f"\nDone. Files saved to {dest_dir}")


if __name__ == "__main__":
    main()
