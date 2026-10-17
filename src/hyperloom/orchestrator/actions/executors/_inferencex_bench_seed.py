# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Give ``run_grid``'s warmup round a different prompt seed than its measured round.

The throughput client seeds ``random`` with ``--seed`` (default 0) and InferenceX never passes it, so two rounds with
one config send identical prompts and the discarded warmup primes the server's prefix cache with the measured round's
prompts. This module patches ``benchmark_lib.sh`` to forward ``$INFERENCEX_BENCH_SEED`` as ``--seed`` and names the
seed the warmup round uses; the measured round keeps the client default so its numbers stay comparable with history.
"""

from __future__ import annotations

import logging
import tempfile
from functools import partial
from pathlib import Path

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx

from ._bench_seed_anchor import BENCH_SEED_ENV, BENCH_SEED_LEGACY, BENCH_SEED_PATCHED, BENCH_SEED_SENTINEL
from ._inferencex_patcher import _apply_line_replacement_atomic, _ensure_patched, _resolve_benchmark_lib_paths
from ._patch_sentinel import file_contains_sentinel

log = logging.getLogger(__name__)

WARMUP_ROUND_SEED = 1
WARMUP_SEED_UNPATCHED_WARNING = "run_grid_warmup_seed_unpatched"

_LOCK_PATH = str(Path(tempfile.gettempdir()) / "hyperloom_benchmark_lib_seed_patcher.lock")


def _is_bench_seed_patched(src: Path) -> bool:
    """Return whether ``benchmark_lib.sh`` already forwards ``$INFERENCEX_BENCH_SEED`` as ``--seed``."""
    return file_contains_sentinel(src, BENCH_SEED_SENTINEL, log, "_inferencex_bench_seed")


def ensure_benchmark_lib_seed_patched(
    inferencex_path: Path | str | None = None,
) -> bool:
    """Ensure ``run_benchmark_serving`` passes ``--seed "$INFERENCEX_BENCH_SEED"`` to the client when that env is set."""
    return _ensure_patched(
        _resolve_benchmark_lib_paths(inferencex_path),
        _is_bench_seed_patched,
        partial(
            _apply_line_replacement_atomic,
            legacy=BENCH_SEED_LEGACY,
            patched_line=BENCH_SEED_PATCHED,
            missing_msg=(
                "_inferencex_bench_seed: expected endpoint branch of run_benchmark_serving not "
                "found in %s; upstream layout may have changed. A warmup round will replay "
                "the measured round's prompts."
            ),
            success_msg="_inferencex_bench_seed: added $INFERENCEX_BENCH_SEED pass-through to %s",
        ),
        _LOCK_PATH,
        empty_msg=(
            "_inferencex_bench_seed: no InferenceX root discovered "
            "(checked $INFERENCEX_PATH, $MAGPIE_PATH/InferenceX) or "
            "benchmark_lib.sh missing — skipping bench-seed patch (fine for "
            "tests and dry-runs without a real InferenceX tree)"
        ),
        failure_msg=("_inferencex_bench_seed: failed to patch %s; other discovered roots will still be attempted"),
    )


def benchmark_lib_seed_patched_in(root: Path | str) -> bool:
    """Return whether the one checkout at ``root`` forwards ``$INFERENCEX_BENCH_SEED``, whatever other roots do."""
    lib = Path(root).joinpath("benchmarks", "benchmark_lib.sh")
    return lib.is_file() and _is_bench_seed_patched(lib)


def warmup_seed_warnings(warmup_cfg_path: Path) -> list[str]:
    """Record which seeds the two rounds use, or that the warmup cannot be reseeded.

    Without the patch the warmup replays the measured round's prompts and primes the server's prefix cache with them,
    so the result must say so rather than look like a clean warmup.
    """
    bench = (yaml.safe_load(warmup_cfg_path.read_text(encoding="utf-8")) or {}).get("benchmark") or {}
    if is_magpie_agentx(bench):
        return []
    root = str(bench.get("inferencex_path") or "").strip() or None
    # ``ensure`` succeeds when any discovered checkout is patched; the round runs the configured one.
    if not ensure_benchmark_lib_seed_patched(root) or (root and not benchmark_lib_seed_patched_in(root)):
        log.warning(
            "grid_runner: %s is not forwarded by benchmark_lib.sh; the warmup round replays the measured round's prompts",
            BENCH_SEED_ENV,
        )
        return [WARMUP_SEED_UNPATCHED_WARNING]
    return [f"warmup_round_seed:{WARMUP_ROUND_SEED}", "measured_round_seed:client_default"]
