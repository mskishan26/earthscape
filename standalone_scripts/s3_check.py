import argparse
import sys
from pathlib import Path

import boto3
import botocore
import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_bucket_and_prefix(config_path: Path, bucket: str | None, prefix: str | None) -> tuple[str, str]:
    config = load_config(config_path)
    data_cfg = config.get("data", {})

    resolved_bucket = bucket or data_cfg.get("bucket")
    resolved_prefix = prefix if prefix is not None else data_cfg.get("base_prefix", "")

    if not resolved_bucket:
        raise ValueError("No bucket provided. Set data.bucket in config.yaml or pass --bucket.")

    resolved_prefix = resolved_prefix.lstrip("/")
    return resolved_bucket, resolved_prefix


def check_bucket_access(s3_client, bucket: str) -> None:
    s3_client.head_bucket(Bucket=bucket)


def list_prefix(s3_client, bucket: str, prefix: str, max_items: int) -> tuple[list[str], list[str]]:
    paginator = s3_client.get_paginator("list_objects_v2")
    operation_parameters = {
        "Bucket": bucket,
        "Prefix": prefix,
        "Delimiter": "/",
        "PaginationConfig": {"MaxItems": max_items},
    }

    directories: list[str] = []
    files: list[str] = []

    for page in paginator.paginate(**operation_parameters):
        for entry in page.get("CommonPrefixes", []):
            value = entry.get("Prefix")
            if value:
                directories.append(value)

        for entry in page.get("Contents", []):
            key = entry.get("Key")
            if not key or key == prefix:
                continue
            files.append(key)

    return directories, files


def print_section(title: str, values: list[str]) -> None:
    print(f"\n{title} ({len(values)})")
    if not values:
        print("  <none>")
        return
    for value in values:
        print(f"  {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--max-items", type=int, default=200)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        bucket, prefix = resolve_bucket_and_prefix(config_path, args.bucket, args.prefix)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    session_kwargs = {}
    if args.profile:
        session_kwargs["profile_name"] = args.profile

    session = boto3.Session(**session_kwargs)
    s3_client_kwargs = {}
    if args.region:
        s3_client_kwargs["region_name"] = args.region
    s3_client = session.client("s3", **s3_client_kwargs)

    print("Checking S3 connection...")
    print(f"Bucket: {bucket}")
    print(f"Base prefix: {prefix or '<bucket root>'}")

    try:
        check_bucket_access(s3_client, bucket)
    except botocore.exceptions.NoCredentialsError:
        print("AWS credentials not found.", file=sys.stderr)
        return 2
    except botocore.exceptions.ClientError as exc:
        error = exc.response.get("Error", {})
        code = error.get("Code", "Unknown")
        message = error.get("Message", str(exc))
        print(f"Failed to access bucket: {code} - {message}", file=sys.stderr)
        return 3
    except botocore.exceptions.BotoCoreError as exc:
        print(f"Failed to connect to S3: {exc}", file=sys.stderr)
        return 4

    print("S3 connection OK.")

    try:
        root_dirs, root_files = list_prefix(s3_client, bucket, "", args.max_items)
        prefix_dirs, prefix_files = list_prefix(s3_client, bucket, prefix, args.max_items)
    except botocore.exceptions.ClientError as exc:
        error = exc.response.get("Error", {})
        code = error.get("Code", "Unknown")
        message = error.get("Message", str(exc))
        print(f"Failed to list objects: {code} - {message}", file=sys.stderr)
        return 5
    except botocore.exceptions.BotoCoreError as exc:
        print(f"Failed during listing: {exc}", file=sys.stderr)
        return 6

    print_section("Bucket root directories", root_dirs)
    print_section("Bucket root files", root_files)
    print_section(f"Directories under prefix '{prefix or '/'}'", prefix_dirs)
    print_section(f"Files under prefix '{prefix or '/'}'", prefix_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
