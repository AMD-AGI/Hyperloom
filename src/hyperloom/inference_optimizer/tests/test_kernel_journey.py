# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the additive ``kernel_journey`` breakdown section.

Covers the recorder substreams (discovery / dispatch / backend_result / e2e),
their assembly into the kernel-major view, and the guarantee that the section
stays absent (so historical breakdowns are byte-for-byte unchanged) when no
substream was recorded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from hyperloom.inference_optimizer.breakdown.recorder import (
    assemble_parts,
    instrument,
)


def _init_git_repo(path: Path) -> str:
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "commit", "--allow-empty", "-q", "-m", "init"],
    ):
        subprocess.run(argv, cwd=path, check=True, capture_output=True)
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _distribution_version() -> str:
    """The version forge's provenance entry must now report.

    Resolved independently of ``instrument``'s own probe so the assertion is an
    oracle rather than a tautology.
    """
    from importlib.metadata import version

    return version("hyperloom-inference_optimizer")


def test_kernel_journey_absent_without_substreams(tmp_path: Path) -> None:
    # Only an unrelated section recorded -> kernel_journey must not appear.
    instrument.record_phase_event(
        tmp_path,
        action="profile",
        entry={"task_id": "t1", "status": "succeeded"},
    )
    out = assemble_parts(tmp_path)
    assert "kernel_journey" not in out


def test_kernel_journey_composes_full_lifecycle(tmp_path: Path) -> None:
    instrument.record_kernel_discovery(
        tmp_path,
        source="tracelens",
        status="success",
        hot_kernels=[
            {
                "kernel_id": "k001",
                "name": "moe",
                "gpu_pct": 42.0,
                "bottleneck": "memory",
                "reusable_native_kernel": True,
                "recommended_backends": ["geak"],
            },
            {"kernel_id": "k002", "name": "ln", "gpu_pct": 7.5},
        ],
        scan={"splitter_mode": "auto", "candidates_path": str(tmp_path / "c.json")},
    )
    instrument.record_kernel_dispatch(
        tmp_path,
        kernel_id="k001",
        dispatched=True,
        backends=["geak"],
        orchestration_commit="abc1234",
    )
    instrument.record_kernel_dispatch(
        tmp_path,
        kernel_id="k002",
        dispatched=False,
        skip_reason="non_reusable_kernel",
    )
    instrument.record_kernel_backend_result(
        tmp_path,
        {
            "kernel_id": "k001",
            "run_id": "r1",
            "attempts": [
                {
                    "attempt_id": "a1",
                    "backend": "geak",
                    "status": "succeeded",
                    "decision": "KEEP",
                    "micro_speedup": 1.8,
                    "compile_passed": True,
                    "correctness_passed": "pass",
                    "duration_sec": 120.5,
                },
                {
                    "attempt_id": "a2",
                    "backend": "claude",
                    "status": "failed",
                    "decision": "FAILED",
                    "error": "compile err",
                },
            ],
        },
    )
    instrument.record_kernel_e2e(
        tmp_path,
        kernel_id="k001",
        integrated=True,
        e2e_gain_pct=3.2,
        validated=True,
        decision="KEEP",
        patch_path="patches/k001.patch",
        target_file="moe.py",
    )

    out = assemble_parts(tmp_path)
    kj = out["kernel_journey"]

    # Raw substreams are popped, never leaking into the envelope.
    for raw in ("kernel_discovery", "kernel_dispatch", "kernel_backend_result", "kernel_e2e"):
        assert raw not in out

    assert len(kj["discovery_runs"]) == 1
    assert kj["discovery_runs"][0]["hot_kernel_count"] == 2

    # Sorted by gpu_pct desc.
    assert [k["kernel_id"] for k in kj["kernels"]] == ["k001", "k002"]

    k001 = kj["kernels"][0]
    assert k001["outcome"] == "adopted"
    assert k001["dispatch"]["dispatched"] is True
    assert k001["dispatch"]["orchestration_commit"] == "abc1234"
    assert len(k001["backend_attempts"]) == 2
    assert k001["backend_attempts"][0]["correctness_passed"] is True
    assert k001["backend_attempts"][0]["duration_sec"] == 120.5
    assert k001["e2e"]["e2e_gain_pct"] == 3.2

    k002 = kj["kernels"][1]
    assert k002["outcome"] == "skipped"
    assert k002["dispatch"]["skip_reason"] == "non_reusable_kernel"
    assert k002["backend_attempts"] == []


def test_kernel_journey_keep_requires_final_validation(tmp_path: Path) -> None:
    instrument.record_kernel_dispatch(
        tmp_path,
        kernel_id="k-provisional",
        dispatched=True,
        backends=["geak"],
    )
    instrument.record_kernel_backend_result(
        tmp_path,
        {
            "kernel_id": "k-provisional",
            "run_id": "r1",
            "attempts": [
                {
                    "attempt_id": "a1",
                    "backend": "geak",
                    "status": "succeeded",
                    "decision": "KEEP",
                }
            ],
        },
        route_strategy="geak_internal",
    )
    instrument.record_kernel_e2e(
        tmp_path,
        kernel_id="k-provisional",
        integrated=True,
        validated=False,
        decision="KEEP",
        route_strategy="geak_internal",
    )

    entry = assemble_parts(tmp_path)["kernel_journey"]["kernels"][0]
    assert entry["outcome"] != "adopted"


def test_kernel_backend_result_keeps_retries_across_runs(tmp_path: Path) -> None:
    # Same kernel/backend, two runs -> two distinct attempts.
    for run in ("r1", "r2"):
        instrument.record_kernel_backend_result(
            tmp_path,
            {
                "kernel_id": "k001",
                "run_id": run,
                "attempts": [
                    {"attempt_id": "", "backend": "geak", "status": "failed", "decision": "FAILED"},
                ],
            },
        )
    out = assemble_parts(tmp_path)
    attempts = out["kernel_journey"]["kernels"][0]["backend_attempts"]
    assert len(attempts) == 2


def test_kernel_backend_result_records_pre_dispatch_failure(tmp_path: Path) -> None:
    # Empty attempts + failed status -> a synthetic FAILED marker in kernel_journey.
    instrument.record_kernel_backend_result(
        tmp_path,
        {
            "kernel_id": "k001",
            "run_id": "r1",
            "attempts": [],
            "status": "failed",
            "error_class": "non_reusable_kernel",
            "error": "empty kernel shape",
            "backend": "geak",
        },
    )
    out = assemble_parts(tmp_path)
    attempts = out["kernel_journey"]["kernels"][0]["backend_attempts"]
    assert len(attempts) == 1
    att = attempts[0]
    assert att["decision"] == "FAILED"
    assert att["pre_dispatch_failure"] is True
    assert att["error_class"] == "non_reusable_kernel"
    assert att["backend"] == "geak"
    # With an attempt present the kernel reads as "attempted", not "skipped".
    assert out["kernel_journey"]["kernels"][0]["outcome"] == "attempted"


def test_backend_attempt_maps_kernel_agent_field_names(tmp_path: Path) -> None:
    # The recorder maps kernel-agent's elapsed_s / created_at / error_type and
    # the kernel-level best speedup onto the journey attempt + entry.
    instrument.record_kernel_backend_result(
        tmp_path,
        {
            "kernel_id": "k001",
            "run_id": "r1",
            "verification": {"micro_speedup": 1.42, "best_attempt_id": "a1"},
            "attempts": [
                {
                    "attempt_id": "a1",
                    "backend": "geak",
                    "status": "completed",
                    "elapsed_s": 87.5,
                    "created_at": "2026-06-12T00:00:00Z",
                },
                {
                    "attempt_id": "a2",
                    "backend": "claude",
                    "status": "timeout",
                    "error_type": "timeout",
                    "elapsed_s": 12.0,
                },
            ],
        },
    )
    out = assemble_parts(tmp_path)
    entry = out["kernel_journey"]["kernels"][0]
    a1, a2 = entry["backend_attempts"]
    assert a1["duration_sec"] == 87.5
    assert a1["ts"] == "2026-06-12T00:00:00Z"
    assert a1["micro_speedup"] == 1.42
    assert a2["error_class"] == "timeout"
    assert entry["micro_speedup"] == 1.42


def test_versions_map_composed_at_top_level(tmp_path: Path) -> None:
    # Discovery + backend recording feed the top-level versions map (one object
    # per tool, keyed by tool name).
    instrument.record_kernel_discovery(
        tmp_path,
        source="tracelens",
        status="success",
        hot_kernels=[{"kernel_id": "k1", "name": "moe", "gpu_pct": 10.0}],
        scan={},
    )
    instrument.record_kernel_backend_result(
        tmp_path,
        {
            "kernel_id": "k1",
            "run_id": "r1",
            "attempts": [
                {"attempt_id": "a1", "backend": "geak", "status": "completed"},
            ],
        },
    )
    out = assemble_parts(tmp_path)
    versions = out["versions"]
    assert isinstance(versions, dict)
    assert set(versions) >= {"tracelens", "geak"}
    assert versions["geak"]["tool"] == "geak"
    # No inline tool metadata in the per-element shapes.
    assert "tool" not in out["kernel_journey"]["discovery_runs"][0]
    assert "tool" not in out["kernel_journey"]["kernels"][0]["backend_attempts"][0]


def test_forge_backend_mints_versions_entry(tmp_path: Path) -> None:
    # A forge attempt keeps backend="forge" in the journey and mints a distinct
    # versions["forge"] provenance entry.
    sha = _init_git_repo(tmp_path)
    instrument.record_kernel_discovery(
        tmp_path,
        source="tracelens",
        status="success",
        hot_kernels=[{"kernel_id": "k1", "name": "moe", "gpu_pct": 10.0}],
        scan={},
    )
    instrument.record_kernel_backend_result(
        tmp_path,
        {
            "kernel_id": "k1",
            "run_id": "r1",
            "attempts": [
                {
                    "attempt_id": "a1",
                    "backend": "forge",
                    "status": "completed",
                    "metadata": {"root_dir": str(tmp_path)},
                },
            ],
        },
    )
    out = assemble_parts(tmp_path)
    atts = out["kernel_journey"]["kernels"][0]["backend_attempts"]
    assert atts[0]["backend"] == "forge"
    versions = out["versions"]
    assert versions["forge"]["tool"] == "forge"
    # KernelForge ships inside this distribution, so its version IS Hyperloom's;
    # there is no separate checkout left to ``git rev-parse``. The producer-supplied
    # root_dir still yields a commit, so provenance keeps both halves.
    assert versions["forge"]["version"] == _distribution_version()
    assert versions["forge"]["version"] != sha
    assert versions["forge"]["commit"] == sha


def test_geak_provenance_resolves_geak_root_env_without_explicit_root(tmp_path: Path, monkeypatch) -> None:
    # With no producer-supplied root, ``geak`` falls back to $GEAK_ROOT so
    # versions["geak"] records that repo's SHA.
    geak_root = tmp_path / "GEAK"
    geak_root.mkdir()
    geak_sha = _init_git_repo(geak_root)
    monkeypatch.setenv("GEAK_ROOT", str(geak_root))

    meta = instrument._tool_metadata("geak")

    assert meta["tool"] == "geak"
    assert meta["root_dir"] == str(geak_root)
    assert meta["commit"] == geak_sha
    assert meta["version"] == geak_sha


def test_discovery_run_carries_duration(tmp_path: Path) -> None:
    instrument.record_kernel_discovery(
        tmp_path,
        source="tracelens",
        status="success",
        hot_kernels=[{"kernel_id": "k1", "name": "moe", "gpu_pct": 10.0}],
        scan={},
        duration_sec=4.2,
    )
    out = assemble_parts(tmp_path)
    assert out["kernel_journey"]["discovery_runs"][0]["duration_sec"] == 4.2


def test_bypass_discovery_decouples_source_from_version_tool(tmp_path: Path) -> None:
    # The bypass route runs the same TraceLens toolchain, so version provenance
    # stays under "tracelens" and mints no versions["bypass"].
    instrument.record_kernel_discovery(
        tmp_path,
        source="bypass",
        tool="tracelens",
        status="success",
        hot_kernels=[{"kernel_id": "k1", "name": "moe", "gpu_pct": 12.0}],
        scan={"analysis_route": "bypass", "candidates_path": str(tmp_path / "c.json")},
        duration_sec=1.5,
    )
    out = assemble_parts(tmp_path)
    runs = out["kernel_journey"]["discovery_runs"]
    assert len(runs) == 1
    assert runs[0]["source"] == "bypass"
    assert runs[0]["hot_kernel_count"] == 1
    assert runs[0]["scan"]["analysis_route"] == "bypass"
    # Version provenance follows the underlying tool, not the route alias.
    versions = out["versions"]
    assert "tracelens" in versions
    assert "bypass" not in versions


def test_discovery_tool_defaults_to_source(tmp_path: Path) -> None:
    # Callers that omit ``tool`` keep version provenance keyed by ``source``.
    instrument.record_kernel_discovery(
        tmp_path,
        source="tracelens",
        status="success",
        hot_kernels=[{"kernel_id": "k1", "name": "moe", "gpu_pct": 9.0}],
        scan={},
    )
    out = assemble_parts(tmp_path)
    assert "tracelens" in out["versions"]


def test_tool_version_probe_git_strategies(tmp_path: Path) -> None:
    sha = _init_git_repo(tmp_path)
    # geak -> git short SHA; commit == version.
    meta = instrument._tool_metadata("geak", root=str(tmp_path))
    assert meta["commit"] == sha
    assert meta["version"] == sha
    # tracelens -> git describe (--always falls back to the short sha here).
    meta_tl = instrument._tool_metadata("tracelens", root=str(tmp_path))
    assert meta_tl["version"]  # non-empty describe output
    # forge -> the distribution version: it is vendored into Hyperloom, not a
    # checkout, so the git strategy geak uses does not apply to it. The commit
    # still comes from the root the caller passed.
    meta_forge = instrument._tool_metadata("forge", root=str(tmp_path))
    assert meta_forge["commit"] == sha
    assert meta_forge["version"] == _distribution_version()
    assert meta_forge["version"] != sha
    # A caller-supplied version always wins over the probe.
    meta_explicit = instrument._tool_metadata(
        "geak",
        root=str(tmp_path),
        version="v9.9",
    )
    assert meta_explicit["version"] == "v9.9"


def test_tool_version_probe_cmd_and_dist() -> None:
    # CLI strategy: python3 --version.
    assert (
        instrument._probe_tool_version(
            ("cmd", ("python3", "--version")),
            "",
        )
        .lower()
        .startswith("python")
    )
    # dist strategy resolves an installed package and rejects a bogus name.
    assert instrument._dist_version(("pytest",))
    assert instrument._dist_version(("definitely-not-a-real-dist-xyz",)) == ""


def test_attach_kernel_roofline_enriches_journey() -> None:
    from hyperloom.inference_optimizer.breakdown.exporter import _attach_kernel_roofline

    kernel_journey = {
        "discovery_runs": [],
        "kernels": [
            {
                "kernel_id": "k001",
                "name": "moe",
                "gpu_pct": 42.0,
                "bound_type": "",
                "discovery": {
                    "kernel_id": "k001",
                    "bound_type": "",
                    "arithmetic_intensity": None,
                    "efficiency_percent": None,
                },
                "backend_attempts": [],
            }
        ],
    }
    kernel_roofline = {
        "kernels": [
            {
                "kernel_id": "k001",
                "name": "moe",
                "bound_type": "memory",
                "arithmetic_intensity": 3.5,
                "flops_per_byte": 2.1,
                "efficiency_percent": 61.0,
                "rocprof_roofline": {"foo": "bar"},
            }
        ],
    }
    _attach_kernel_roofline(kernel_journey, kernel_roofline)
    entry = kernel_journey["kernels"][0]
    assert entry["roofline"]["arithmetic_intensity"] == 3.5
    assert entry["roofline"]["rocprof_roofline"] == {"foo": "bar"}
    # Header + discovery fields backfilled from roofline.
    assert entry["bound_type"] == "memory"
    assert entry["discovery"]["bound_type"] == "memory"
    assert entry["discovery"]["arithmetic_intensity"] == 3.5
    assert entry["discovery"]["efficiency_percent"] == 61.0


def test_merge_phase_timeline_unit_keeps_collector_and_dedups() -> None:
    from hyperloom.inference_optimizer.breakdown.exporter import _merge_phase_timeline

    collector = [
        {"action": "baseline", "ts": "2026-06-12T00:00:01Z", "change": "baseline", "decision": "promoted"},
        {
            "action": "kernel_opt",
            "ts": "2026-06-12T00:00:02Z",
            "change": "kernel_opt:k001",
            "kernel_id": "k001",
            "decision": "KEEP",
        },
    ]
    fragment = [
        # Same attempt as the collector baseline row -> must dedupe.
        {"action": "baseline", "ts": "2026-06-12T00:00:01Z", "task_id": "t1", "decision": "promoted"},
        # An audit row the on-disk state lost -> must be appended.
        {"action": "explore", "ts": "2026-06-12T00:00:03Z", "task_id": "t2", "decision": "promoted"},
    ]
    merged = _merge_phase_timeline(fragment, collector)
    actions = [e["action"] for e in merged]
    assert actions == ["baseline", "kernel_opt", "explore"]
    # Sorted by ts and no duplicate baseline event.
    assert sum(1 for e in merged if e["action"] == "baseline") == 1


def test_build_phase_timeline_merges_journal_and_kernel_lanes(
    tmp_path: Path,
) -> None:
    # A recorder phase_timeline fragment must NOT erase the optimization_journal
    # KEEP/REVERT or the kernel_opt/integrate lanes the collector folds in.
    import json

    from hyperloom.inference_optimizer.breakdown import exporter

    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "optimization_journal.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "ts": "2026-06-12T00:00:01Z",
                        "kind": "baseline",
                        "change": "baseline",
                        "outcome": "KEEP",
                        "phase": "EXPLORE",
                        "throughput_after": 5000.0,
                    },
                    {
                        "ts": "2026-06-12T00:00:05Z",
                        "kind": "explore",
                        "change": "explore",
                        "outcome": "REVERT",
                        "phase": "EXPLORE",
                        "gain_pct": -2.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "kernel_opt_attempts": {
                    "k001": {
                        "last_ts": "2026-06-12T00:00:03Z",
                        "history": [{"ts": "2026-06-12T00:00:03Z", "decision": "KEEP"}],
                    },
                },
                "kernel_integrate_attempts": {
                    "patch-1": {
                        "kernel_id": "k001",
                        "patch_path": "/p.diff",
                        "attempts": [
                            {"ts": "2026-06-12T00:00:04Z", "status": "ok", "decision": "KEEP", "gain_pct": 3.0}
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    # An audit-action fragment triggers the assembled/merge path.
    instrument.record_phase_event(
        tmp_path,
        action="baseline",
        entry={"task_id": "t1", "ts": "2026-06-12T00:00:01Z", "status": "succeeded", "decision": "promoted"},
    )

    out = exporter.build(tmp_path)
    timeline = out["phase_timeline"]
    actions = {e.get("action") for e in timeline}
    assert "baseline" in actions
    assert "explore" in actions
    assert "kernel_opt" in actions
    assert "integrate" in actions
    decisions = {(e.get("action"), e.get("decision")) for e in timeline}
    assert ("explore", "REVERT") in decisions
    assert ("integrate", "KEEP") in decisions
    # action_timeline alias was removed; phase_timeline is the canonical key.
    assert out["phase_timeline"] == timeline
