"""Tests for the ``data_provenance`` breakdown section.

The provenance collector probes (stat / glob only) the on-disk
artifacts each other collector consumes and emits a single
``SectionProvenance`` row per logical section. These tests cover:

1. A fully populated synthetic session — every section ``complete``.
2. Missing ``state.json`` — every state-dependent section becomes
   ``empty`` with the right ``missing_required`` entries.
3. Missing ``runs/baseline`` — baseline section flips to ``empty``.
4. Source artifacts present but session never ran a sweep — sweep is
   ``complete`` (no required missing) and ``populated=False``.
5. Roofline alternative paths (``reports/final.json`` *or*
   ``runs/roofline/<x>/final.json``) both light up the section.
6. Container-image env var unset → env probes report ``found=False``.
7. The data-provenance renderer produces a markdown table containing
   ``Data Provenance`` + the well-known section names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown import build
from inference_optimizer.breakdown.reporters.compose import render_session_report


# ---------------------------------------------------------------------------
# Fixture builder — minimal but realistic shape for every section.
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_full_session(sd: Path) -> None:
    """Create a synthetic session_dir containing one example of every
    artifact the data-provenance collector probes (manifest, state,
    baseline + profile + sweep + params + backends runs, kernel-agent
    artifacts, critic + robustness workdirs, roofline reports)."""

    sd.mkdir(parents=True, exist_ok=True)

    _write_json(sd / "manifest.json", {
        "schema_version": 1, "session_id": "prov-1", "model_name": "TestModel",
        "framework": "sglang", "gpu_type": "mi300x",
    })
    _write_json(sd / "state.json", {
        "session_id": "prov-1", "baseline_tput": 100.0,
        "current_best": {"tput": 120.0, "action": "validate_stack",
                          "extra_sglang_args": "", "extra_envs": {}},
        "optimization_stack": [
            {"action": "backends", "variant_name": "x", "gain_pct": 5.0,
             "extra_sglang_args": "-x", "ts": "2026-05-23T00:00:00+00:00"},
        ],
        "cumulative_gain": 5.0, "cumulative_gain_validated": 5.0,
        "stop_reason": "target_reached", "start_ts": "2026-05-23T00:00:00+00:00",
        "max_minutes": 60, "tick": 1,
        "baseline_attempts": [
            {"ts": "2026-05-23T00:01:00+00:00", "task_id": "b1",
             "status": "succeeded", "decision": "promoted", "key_metric": 100.0,
             "workspace": str(sd / "runs/baseline/b1")},
        ],
        "profile_attempts": [
            {"ts": "2026-05-23T00:02:00+00:00", "task_id": "p1",
             "status": "succeeded", "decision": "promoted", "key_metric": 100.0},
        ],
        "backends_attempts": [
            {"ts": "2026-05-23T00:03:00+00:00", "task_id": "be1",
             "status": "succeeded", "decision": "promoted", "key_metric": 5.0},
        ],
        "params_attempts": [
            {"ts": "2026-05-23T00:04:00+00:00", "task_id": "pa1",
             "status": "succeeded", "decision": "promoted", "key_metric": 3.0,
             "extras": {"round_id": "r1", "best_variant_name": "y"}},
        ],
        "sweep_attempts": [
            {"ts": "2026-05-23T00:05:00+00:00", "task_id": "s1",
             "status": "succeeded", "decision": "promoted", "key_metric": 120.0,
             "workspace": str(sd / "runs/sweep/s1")},
        ],
        "validate_stack_attempts": [
            {"ts": "2026-05-23T00:06:00+00:00", "task_id": "v1",
             "status": "succeeded", "decision": "promoted", "key_metric": 5.0},
        ],
        "last_sweep": {"grid_size": 1, "best_overall": {"variant_name": "v01"},
                        "best_for_each_conc": [], "pareto_front": []},
        "last_select_kernels": {"hot_kernels_top15": [
            {"kernel_id": "k001", "name": "fused_op", "gpu_pct": 30.0,
             "bottleneck": "memory", "recommended_backends": ["claude"],
             "recommended_actions": ["run_optimization"]},
        ], "ts": "2026-05-23T00:02:30+00:00"},
        "last_kernel_opt": {"kernel_id": "k001", "decision": "KEEP",
                              "micro_speedup": 1.1, "compile_passed": True,
                              "correctness_passed": True,
                              "best_artifact_path": "patches/k001/0001.patch",
                              "ts": "2026-05-23T00:07:00+00:00"},
        "kernel_opt_attempts": {"k001": {
            "attempts": 1, "last_decision": "KEEP",
            "last_ts": "2026-05-23T00:07:00+00:00",
            "history": [{"decision": "KEEP", "ts": "2026-05-23T00:07:00+00:00"}],
        }},
        "params_search": {"schema_version": 2, "accepted": [], "rejected": [],
                            "tested": {}, "name_index": {}, "cursor": 0},
        "backends_search": {"schema_version": 1, "accepted": [], "rejected": [],
                              "tested": {}, "name_index": {}, "cursor": 0},
    })

    # baseline
    bdir = sd / "runs/baseline/b1/benchmark_001"
    _write_json(bdir / "benchmark_report.json", {
        "success": True, "output_throughput_tok_s": 100.0,
        "mean_ttft_ms": 100.0, "mean_e2el_ms": 1500.0,
    })
    (sd / "runs/baseline/b1" / "baseline_config.with_envs.yaml").write_text(
        "model: test\n", encoding="utf-8"
    )
    (sd / "runs/baseline/b1" / "server.log").write_text(
        "ServerArgs: ...\n", encoding="utf-8"
    )

    # profile + kernel-agent tracelens
    pdir = sd / "runs/profile/p1/benchmark_001"
    _write_json(pdir / "benchmark_report.json", {
        "success": True,
        "kernel_summary": [
            {"kernel_id": "k001", "name": "fused_op", "gpu_pct": 30.0,
             "time_ms": 0.5, "bottleneck": "memory"},
        ],
    })
    (sd / "runs/profile/p1/torch_trace").mkdir(parents=True, exist_ok=True)
    (sd / "runs/profile/p1/torch_trace/trace.trace.json.gz").write_bytes(b"\x1f\x8b")
    (sd / "runs/profile/p1/kernel_summary.csv").write_text("k,gpu_pct\nk001,30\n",
                                                            encoding="utf-8")
    kag = sd / "kernel-agent/runs/sess1"
    _write_json(kag / "status/tracelens_analysis/a01.json", {"started": 1, "ended": 2})
    _write_json(kag / "tracelens/priority_data.json", {"items": []})
    _write_json(kag / "tracelens/category_data/op_a_metrics.json", {"metrics": {}})
    (kag / "tracelens/analysis.md").write_text("# analysis\n", encoding="utf-8")
    with (kag / "optimization_attempts.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "attempt_id": "a01", "kernel_id": "k001", "backend": "claude",
            "ts": "2026-05-23T00:07:30+00:00", "status": "succeeded",
            "speedup": 1.1, "name": "fused_op",
        }) + "\n")

    # decision-journal variant reports
    _write_json(sd / "runs/params/pa1/variant_01_x/benchmark_001/benchmark_report.json",
                {"success": True, "output_throughput_tok_s": 103.0})
    _write_json(sd / "runs/backends/be1/variant_01_x/benchmark_001/benchmark_report.json",
                {"success": True, "output_throughput_tok_s": 105.0})

    # sweep variants
    _write_json(sd / "runs/sweep/s1/variant_01_conc32/benchmark_001/benchmark_report.json",
                {"success": True, "output_throughput_tok_s": 120.0,
                 "mean_ttft_ms": 90, "mean_tpot_ms": 6, "mean_e2el_ms": 1200})

    # critic + robustness workdirs
    cwd = sd / "critic-workdir/001"
    _write_json(cwd / "review.json", {"verdict": "approve", "topic": "kernel_opt:k001",
                                       "ts": "2026-05-23T00:07:45+00:00"})
    _write_json(cwd / "emit.json", {"topic": "kernel_opt:k001",
                                     "ts": "2026-05-23T00:07:40+00:00"})
    rwd = sd / "robustness-workdir/001"
    _write_json(rwd / "signal.json", {"signal": "stall",
                                       "ts": "2026-05-23T00:07:50+00:00"})

    # roofline (orchestrator-style path)
    _write_json(sd / "reports/final.json", {
        "roofline_comparison": {
            "mode": "baseline-vs-latest",
            "baseline": {"snapshot_id": "b", "compute_pct": 40.0,
                          "top_kernel": {"name": "fused_op", "gpu_pct": 30.0}},
            "latest":   {"snapshot_id": "l", "compute_pct": 45.0},
            "delta":    {"compute_pct": 5.0},
        },
    })


def _entries_by_section(prov: list[dict]) -> dict[str, dict]:
    return {e["section"]: e for e in prov if isinstance(e, dict)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_provenance_full_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete synthetic session: every well-known section must
    report ``status == 'complete'``."""
    monkeypatch.setenv("HYPERLOOM_IMAGE", "registry.example/sgl:latest")
    sd = tmp_path / "sess"
    _build_full_session(sd)
    b = build(sd)
    prov = b["data_provenance"]
    by_sec = _entries_by_section(prov)

    expected_sections = {
        "session", "workload", "baseline", "final", "decision_journal",
        "sweep", "phase_timeline", "kernel_profiling", "kernel_decision_path",
        "kernel_lifecycle", "geak_invocations", "oob_invocations",
        "critic_robustness", "roofline", "attribution", "param_search",
    }
    assert expected_sections.issubset(set(by_sec.keys())), (
        f"missing provenance entries: {expected_sections - set(by_sec.keys())}"
    )
    for sec, entry in by_sec.items():
        assert entry["status"] == "complete", (
            f"{sec} expected complete, got {entry['status']} "
            f"(missing_required={entry['missing_required']})"
        )
        assert entry["missing_required"] == [], (
            f"{sec} had missing_required={entry['missing_required']}"
        )


def test_provenance_missing_state_json(tmp_path: Path) -> None:
    """Removing state.json must make every state-dependent section
    ``empty`` and list ``session state`` (or similar) under
    ``missing_required``."""
    sd = tmp_path / "sess"
    _build_full_session(sd)
    (sd / "state.json").unlink()
    b = build(sd)
    by_sec = _entries_by_section(b["data_provenance"])

    # decision_journal: state is the only required source it depends on.
    dj = by_sec["decision_journal"]
    assert dj["status"] == "empty", dj
    assert any("session state" in m for m in dj["missing_required"]), (
        f"decision_journal missing_required={dj['missing_required']}"
    )
    # final, attribution, param_search, phase_timeline all hinge on state.
    for sec in ("attribution", "param_search"):
        assert by_sec[sec]["status"] == "empty", (sec, by_sec[sec])
        # ``session`` requires both manifest.json + state.json; with
        # state gone but manifest present + ``session_id`` extracted from
        # manifest, the section is populated but partial.
        sess = by_sec["session"]
        assert sess["status"] == "partial", sess
        assert sess["populated"] is True
        assert any("session state" in m for m in sess["missing_required"]), (
            f"session missing_required={sess['missing_required']}"
        )


def test_provenance_missing_runs_baseline(tmp_path: Path) -> None:
    """Deleting runs/baseline/* must collapse the baseline section to
    ``empty`` with ``baseline benchmark_report`` listed missing."""
    sd = tmp_path / "sess"
    _build_full_session(sd)
    # Wipe every baseline benchmark_report.json under runs/baseline.
    for p in (sd / "runs/baseline").rglob("benchmark_report.json"):
        p.unlink()
    # Also remove baseline_attempts so the breakdown can't reconstruct
    # the section from state alone.
    state = json.loads((sd / "state.json").read_text())
    state["baseline_attempts"] = []
    state.pop("last_baseline", None)
    state["baseline_tput"] = None
    (sd / "state.json").write_text(json.dumps(state))

    b = build(sd)
    by_sec = _entries_by_section(b["data_provenance"])
    base = by_sec["baseline"]
    assert base["status"] == "empty", base
    assert any("baseline benchmark_report" in m for m in base["missing_required"]), (
        f"baseline missing_required={base['missing_required']}"
    )


def test_provenance_sweep_complete_but_unpopulated(tmp_path: Path) -> None:
    """Without any sweep on disk, sweep has no required sources missing
    (everything sweep depends on is optional), so status remains
    ``complete`` but ``populated`` is False."""
    sd = tmp_path / "sess"
    _build_full_session(sd)
    # Drop every sweep artifact + the sweep_attempts ledger.
    sweep_dir = sd / "runs/sweep"
    if sweep_dir.exists():
        for p in sorted(sweep_dir.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        sweep_dir.rmdir()
    state = json.loads((sd / "state.json").read_text())
    state["sweep_attempts"] = []
    state.pop("last_sweep", None)
    (sd / "state.json").write_text(json.dumps(state))

    b = build(sd)
    sw = _entries_by_section(b["data_provenance"])["sweep"]
    assert sw["status"] == "complete", sw
    assert sw["populated"] is False, sw
    assert sw["missing_required"] == [], sw


def test_provenance_roofline_alternative_paths(tmp_path: Path) -> None:
    """Both ``reports/final.json`` and ``runs/roofline/<x>/final.json``
    must satisfy the roofline section's required-OR probe."""
    # Variant A: reports/final.json present (the default fixture path).
    sd_a = tmp_path / "sess_a"
    _build_full_session(sd_a)
    ba = build(sd_a)
    roof_a = _entries_by_section(ba["data_provenance"])["roofline"]
    assert roof_a["status"] == "complete", roof_a

    # Variant B: only runs/roofline/<x>/final.json present.
    sd_b = tmp_path / "sess_b"
    _build_full_session(sd_b)
    (sd_b / "reports/final.json").unlink()
    _write_json(sd_b / "runs/roofline/rf1/final.json", {
        "roofline_comparison": {
            "mode": "single", "baseline": {"snapshot_id": "x"},
        },
    })
    bb = build(sd_b)
    roof_b = _entries_by_section(bb["data_provenance"])["roofline"]
    assert roof_b["status"] == "complete", roof_b


def test_provenance_env_image_unset(tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """When no container-image env var is set, the env probes must
    report ``found=False`` (without flipping ``session`` to ``empty``,
    because the env vars are optional)."""
    for name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(name, raising=False)
    sd = tmp_path / "sess"
    _build_full_session(sd)
    b = build(sd)
    sess = _entries_by_section(b["data_provenance"])["session"]
    env_probes = [p for p in sess["sources"] if p["path"].startswith("env:")]
    assert env_probes, "expected env probes in session provenance"
    assert all(p["found"] is False for p in env_probes), env_probes
    assert sess["status"] == "complete"


def test_renderer_data_provenance(tmp_path: Path) -> None:
    """The data-provenance renderer must emit a markdown table containing
    ``Data Provenance`` and at least one section row."""
    sd = tmp_path / "sess"
    _build_full_session(sd)
    b = build(sd)
    result = render_session_report(b)
    md = result.markdown
    assert "Data Provenance" in md
    # Table header
    assert "| section | status | populated | missing_required | sources |" in md
    # Spot-check at least one known section row.
    assert "baseline" in md and "kernel_profiling" in md
