# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep pytest timing seeds stable and publish only complete, disjoint shard timings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate duration nodeid: {key!r}")
        result[key] = value
    return result


def merge_durations(artifacts: Path, total_shards: int) -> dict[str, float]:
    """Validate the artifact boundary before reconstructing the shared timing DB."""
    if type(total_shards) is not int or total_shards < 1:
        raise ValueError("total_shards must be a positive integer")
    expected = {artifacts / f"durations-shard{shard}" for shard in range(1, total_shards + 1)}
    actual = set(artifacts.iterdir())
    if actual != expected:
        missing = sorted(path.name for path in expected - actual)
        unexpected = sorted(path.name for path in actual - expected)
        raise ValueError(f"Duration artifacts are incomplete: missing={missing}, unexpected={unexpected}")

    merged: dict[str, float] = {}
    for shard in range(1, total_shards + 1):
        directory = artifacts / f"durations-shard{shard}"
        path = directory / f".test_durations.shard{shard}"
        if set(directory.iterdir()) != {path} or not path.is_file():
            raise ValueError(f"Expected exactly one duration file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
        if not isinstance(data, dict) or not data:
            raise ValueError(f"Expected a nonempty duration mapping: {path}")
        for nodeid, seconds in data.items():
            if not nodeid.strip():
                raise ValueError(f"Empty duration nodeid in {path}")
            if (
                type(seconds) not in (int, float)
                or seconds < 0
                or seconds > sys.float_info.max
                or not math.isfinite(seconds)
            ):
                raise ValueError(f"Invalid duration for {nodeid!r} in {path}: {seconds!r}")
            if nodeid in merged:
                raise ValueError(f"Duplicate duration nodeid across shards: {nodeid!r}")
            merged[nodeid] = seconds
    return merged


def main(argv: list[str] | None = None) -> int:
    """Merge downloaded artifacts without importing pytest or project dependencies."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--artifacts", type=Path, default=Path("durations"))
    parser.add_argument("--output", type=Path, default=Path(".test_durations"))
    args = parser.parse_args(argv)
    try:
        config = tomllib.loads(args.config.read_text(encoding="utf-8"))
        total = config["tool"]["hyperloom"]["tests_coverage"]["total_shards"]
        merged = merge_durations(args.artifacts, total)
        args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"::error::Cannot publish test durations: {error}", file=sys.stderr)
        return 1
    print(f"Merged {total} shards -> {len(merged)} test timings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    import pytest

    @pytest.hookimpl(tryfirst=True)
    def pytest_configure(config: pytest.Config) -> None:
        marker = Path(f"{config.option.durations_path}.complete")
        if hasattr(config, "workerinput"):
            # A replacement worker must read the same seed as the original workers.
            config.option.store_durations = False
        else:
            marker.unlink(missing_ok=True)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(session: pytest.Session) -> None:
        config = session.config
        if config.option.store_durations and not hasattr(config, "workerinput"):
            # pytest-split's normal-priority hook has now replaced the seed.
            Path(f"{config.option.durations_path}.complete").touch()
