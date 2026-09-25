###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Golden equivalence lock: the analysis.json path reproduces the md-scrape path.

``golden_moe_decode_baseline.json`` was captured from the old md-scrape path on a
representative MoE decode compute trace (GEMMs + Tensile ``Cijk_`` symbols) before the
migration deleted that path. This test runs the new ``analysis.json`` reader over
the rendered fixture and asserts the kept structural fields -- the shape, metric,
category and identity fields a downstream consumer reads -- match that baseline
kernel-for-kernel. Source-resolution fields are intentionally NOT locked: they
move from HL's deleted cascade to TraceLens' resolver (design §5) and depend on
the live source tree, so they are exercised by ``test_analysis_json_reader``
instead. The resolver is stubbed here so the structural check is host-independent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _analysis_json as aj
import _kernel_source as ks
import tracelens_analysis as tla
from TraceLens.TraceUtils.kernel_source import ResolveResult

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Fields that must survive the md->json ingest change byte-for-byte; a drift here
# is the silent field-shape regression the golden gate exists to catch.
_STRUCTURAL = [
    "name",
    "duration_us",
    "call_count",
    "shapes",
    "library",
    "tracelens_category",
    "kernel_category",
    "efficiency_percent",
    "efficiency_peak_value",
    "efficiency_peak_unit",
    "flops_per_byte",
    "device_kernel_names",
]


def test_analysis_json_path_matches_md_scrape_baseline(monkeypatch):
    monkeypatch.setattr(
        ks,
        "resolve_kernel_source",
        lambda *a, **k: ResolveResult(location=None, patchable=False, method="unresolved"),
    )
    analysis_json = _FIXTURES / "golden_moe_decode_analysis.json"
    baseline = json.loads((_FIXTURES / "golden_moe_decode_baseline.json").read_text())

    rows = aj.load_report_tasks(analysis_json)
    finalized = tla._finalize_candidates(rows, perf_report_csv_dir=None, framework="")

    produced = {c.get("device_kernel_name") or c.get("name"): c for c in finalized}
    assert set(produced) == set(baseline), (
        f"kernel set drift: new-only={sorted(set(produced) - set(baseline))} "
        f"old-only={sorted(set(baseline) - set(produced))}"
    )
    problems = []
    for key, want in baseline.items():
        got = produced[key]
        for field in _STRUCTURAL:
            if got.get(field) != want.get(field):
                problems.append(f"{key[:40]}.{field}: baseline={want.get(field)!r} new={got.get(field)!r}")
    assert not problems, "structural drift from the md-scrape baseline:\n" + "\n".join(problems)
