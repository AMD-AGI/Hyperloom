"""Verify a local artifact without relying on shell-output parsing."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_artifact(path: Path, expected_sha256: str, expected_size: int) -> bool:
    try:
        size = path.stat().st_size
        actual_sha256 = sha256_file(path)
    except OSError as exc:
        print(f"ERROR: cannot read artifact: {path}", file=sys.stderr)
        print(f"       {exc}", file=sys.stderr)
        return False

    expected_sha256 = expected_sha256.upper()
    print(f"Artifact: {path}")
    print(f"Size:     {size} bytes (expected {expected_size})")
    print(f"SHA256:   {actual_sha256}")
    if size != expected_size:
        print("ERROR: artifact size does not match", file=sys.stderr)
        return False
    if actual_sha256 != expected_sha256:
        print(f"ERROR: expected SHA256 {expected_sha256}", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("expected_sha256")
    parser.add_argument("expected_size", type=int)
    args = parser.parse_args()
    return 0 if verify_artifact(
        args.path.resolve(), args.expected_sha256, args.expected_size
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
