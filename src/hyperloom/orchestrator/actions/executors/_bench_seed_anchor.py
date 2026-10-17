# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``benchmark_lib.sh`` text the bench-seed patch matches and writes.

Kept apart from ``_inferencex_bench_seed`` so the anchor contract in ``_inferencex_patcher`` can name it without the
patch logic, which imports the patcher's helpers, closing an import cycle.
"""

from __future__ import annotations

BENCH_SEED_ENV = "INFERENCEX_BENCH_SEED"

BENCH_SEED_LEGACY = '    if [[ -n "$endpoint" ]]; then'
BENCH_SEED_PATCHED = (
    f'    if [[ -n "${{{BENCH_SEED_ENV}:-}}" ]]; then\n'
    f'        benchmark_cmd+=(--seed "${BENCH_SEED_ENV}")\n'
    "    fi\n\n" + BENCH_SEED_LEGACY
)
BENCH_SEED_SENTINEL = BENCH_SEED_ENV
