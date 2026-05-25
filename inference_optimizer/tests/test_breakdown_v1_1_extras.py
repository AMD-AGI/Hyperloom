"""Tests covering v1.1 additions in collectors.

Specifically:

* ``server_args=ServerArgs(...)`` regex in ``_extract_framework_args``
  (sglang ≥0.4 startup line) — exercised via the public ``build()`` path
  by writing a synthetic baseline tree.
* TraceLens roofline fallback in ``collect_kernel_profiling`` — when
  ``status/tracelens_analysis/<run>.json`` has empty ``top_kernels``,
  the collector reads ``tracelens/category_data/*_metrics.json`` and
  ``tracelens/priority_data.json``.
* ``decision_note`` propagation: the round-winner ``note`` field on
  ``backend_winners_history[].winners[]`` lands on the journal variant.
* ``duration_seconds`` on phase events and on variant decisions.
* TraceLens timeline event surfaces alongside other phase events.
* Roofline merge into ``kernel_lifecycle.detected``.
* Earliest-round (``round_id is None`` in attempt extras) gets
  emitted as ``"<phase>-000"`` instead of being swallowed by the
  later round's ``params-last`` row.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.breakdown import build
from inference_optimizer.breakdown.collectors import _extract_framework_args


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# A. ServerArgs regex
# ---------------------------------------------------------------------------
def test_extract_framework_args_recognises_serverargs_line(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    # Real-shape line copied (truncated) from a sglang 0.4 baseline:
    log.write_text(
        "[2026-05-21 23:24:59] server_args=ServerArgs(model_path='/m', "
        "attention_backend='aiter', mem_fraction_static=0.68, tp_size=1, "
        "cuda_graph_bs=[1, 2, 4, 8])\n",
        encoding="utf-8",
    )
    args, src = _extract_framework_args(log, None)
    assert src == "log_args_line"
    assert args.startswith("ServerArgs(")
    assert "attention_backend='aiter'" in args
    assert "mem_fraction_static=0.68" in args


# ---------------------------------------------------------------------------
# B. TraceLens roofline fallback (category_data)
# ---------------------------------------------------------------------------
def _kp_session_with_tracelens(tmp_path: Path, with_categories: bool) -> Path:
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "rl"})
    _write_json(sd / "state.json", {"session_id": "rl", "framework": "sglang"})
    kar = sd / "kernel-agent/runs/sess-rl"
    status_dir = kar / "status/tracelens_analysis"
    status_dir.mkdir(parents=True)
    # Status file with EMPTY top_kernels — forces the fallback path.
    _write_json(status_dir / "run-1.json", {
        "status": "ok", "summary": "", "top_kernels": [],
    })
    tl = kar / "tracelens"
    tl.mkdir()
    if with_categories:
        cat = tl / "category_data"
        cat.mkdir()
        _write_json(cat / "gemm_metrics.json", {
            "category": "gemm",
            "operations": [
                {
                    "name": "aten::mm",
                    "count": 60,
                    "time_ms": 800.0,
                    "percent_of_total": 19.4,
                    "percent_of_category": 54.5,
                    "library": "Tensile",
                    "Input Dims": "((1, 2), (2, 3))",
                    "efficiency": {
                        "tflops_achieved": 580.0,
                        "efficiency_percent": 81.86,
                        "bound_type": "compute",
                        "flops_per_byte": 3982.22,
                    },
                },
            ],
        })
        _write_json(tl / "priority_data.json", {
            "priorities": [{"display_name": "GEMM", "category": "gemm", "impact_score": 5.66}],
            "findings": [],
        })
    else:
        # category_data missing — exercise the priority_data fallback.
        _write_json(tl / "priority_data.json", {
            "priorities": [{"display_name": "GEMM", "category": "gemm", "impact_score": 5.66}],
            "findings": [{
                "members": [{
                    "operation": "aten::mm",
                    "category": "gemm",
                    "efficiency_pct": 81.86,
                    "bound_type": "compute",
                    "library": "Tensile",
                    "time_ms": 800.0,
                    "impact_score": 3.08,
                }],
            }],
        })
    (tl / "analysis.md").write_text(
        "# Findings\n\nThe trace shows a clear GEMM-bound regime "
        "with ~81% MFU on the largest contraction.\n",
        encoding="utf-8",
    )
    return sd


def test_kernel_profiling_falls_back_to_category_data(tmp_path: Path) -> None:
    sd = _kp_session_with_tracelens(tmp_path, with_categories=True)
    runs = build(sd)["kernel_profiling"]
    assert len(runs) == 1
    out = runs[0]["outputs"]
    assert out["tool"] == "tracelens_analysis"
    # category_data row landed on top_kernels with roofline fields:
    assert any(
        k.get("efficiency_percent") == 81.86 and k.get("bound_type") == "compute"
        for k in out["top_kernels"]
    )
    # analysis.md surfaced as the summary (preferred over priority synth).
    assert out["analysis_summary"]
    assert "GEMM-bound" in out["analysis_summary"]


def test_kernel_profiling_falls_back_to_priority_data(tmp_path: Path) -> None:
    sd = _kp_session_with_tracelens(tmp_path, with_categories=False)
    runs = build(sd)["kernel_profiling"]
    assert len(runs) == 1
    out = runs[0]["outputs"]
    assert out["top_kernels"]
    k0 = out["top_kernels"][0]
    assert k0["efficiency_percent"] == 81.86
    assert k0["bound_type"] == "compute"


# ---------------------------------------------------------------------------
# C. decision_note + duration on variants, plus first-round visibility
# ---------------------------------------------------------------------------
def _decision_journal_session(tmp_path: Path) -> Path:
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "dj2"})
    # Two attempts: round_id=None (the earliest round whose audit predates
    # round-id assignment) and round_id="params-001". Without v1.1 the
    # first one is invisible because last_round only mirrors the second.
    _write_json(sd / "state.json", {
        "session_id": "dj2",
        "baseline_tput": 1000.0,
        "current_best": {"tput": 1000.0},
        "params_attempts": [
            {
                "ts": "2026-05-15T09:00:00+00:00",
                "task_id": "early-task",
                "status": "succeeded",
                "decision": "discarded",
                "workspace": str(sd / "runs/params/early-task"),
                "extras": {"round_id": None, "best_variant_name": None},
            },
            {
                "ts": "2026-05-15T10:00:00+00:00",
                "task_id": "late-task",
                "status": "succeeded",
                "decision": "discarded",
                "workspace": str(sd / "runs/params/late-task"),
                "extras": {"round_id": "params-001"},
            },
        ],
        "backend_winners_history": [{
            "action": "params",
            "round_id": "params-001",
            "base_tput": 1000.0,
            "ts": "2026-05-15T10:00:00+00:00",
            "winners": [{
                "name": "knob_a",
                "fingerprint": "fp-a",
                "gain_pct": None,
                "tput": 1010.0,
                "extra_sglang_args": "--knob-a 1",
                "extra_envs": {},
                "note": "new_family_scheduler",
            }],
        }],
        "params_search": {
            "schema_version": 2,
            "tested": {
                "fp-a": {
                    "name": "knob_a",
                    "fingerprint": "fp-a",
                    "gain_pct": None,
                    "extra_sglang_args": "--knob-a 1",
                    "result": {
                        "status": "succeeded",
                        "output_throughput": 1010.0,
                        "duration_seconds": 222.5,
                        "note": "new_family_scheduler",
                    },
                },
            },
            "rejected": [],
            "last_round": {
                "round_id": "params-001",
                "base_tput": 1000.0,
                "tested_fp": ["fp-a"],
                "round_winners": ["knob_a"],
                "selected_new": [],
            },
        },
    })
    # Disk benchmark report for early-task to enable disk-walk variant
    # reconstruction and duration backfill.
    early_dir = sd / "runs/params/early-task/variant_00_first_knob/benchmark_001"
    _write_json(early_dir / "benchmark_report.json", {
        "success": True,
        "throughput": {"output_throughput": 980.0},
        "execution_time": 333.3,
    })
    return sd


def test_decision_journal_surfaces_first_round_via_disk_walk(tmp_path: Path) -> None:
    sd = _decision_journal_session(tmp_path)
    journal = build(sd)["decision_journal"]
    rounds = {(e["phase"], e["round_id"]) for e in journal}
    assert ("params", "params-000") in rounds
    assert ("params", "params-001") in rounds
    # No legacy params-last row.
    assert ("params", "params-last") not in rounds
    early = [e for e in journal if e["round_id"] == "params-000"][0]
    assert early["variants"]
    v = early["variants"][0]
    # Disk-walked variant carries duration from execution_time fallback:
    assert v["duration_seconds"] == 333.3
    # gain_pct_vs_base backfilled from baseline_tput when state didn't record it
    assert v["gain_pct_vs_base"] is not None
    assert abs(v["gain_pct_vs_base"] - ((980.0 - 1000.0) / 1000.0 * 100.0)) < 1e-6


def test_decision_journal_decision_note_and_duration_on_variants(tmp_path: Path) -> None:
    sd = _decision_journal_session(tmp_path)
    journal = build(sd)["decision_journal"]
    late = [e for e in journal if e["round_id"] == "params-001"][0]
    knob_a = [v for v in late["variants"] if v["name"] == "knob_a"][0]
    assert knob_a["decision_note"] == "new_family_scheduler"
    assert knob_a["duration_seconds"] == 222.5
    # gain_pct_vs_base backfilled from baseline_tput (state had None):
    assert knob_a["gain_pct_vs_base"] is not None
    assert abs(knob_a["gain_pct_vs_base"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# D. phase_timeline duration + tracelens event
# ---------------------------------------------------------------------------
def test_phase_timeline_synthesizes_closing_event_from_state(tmp_path: Path) -> None:
    """P2-2: when ``final.closing_phase_entered`` is True (via
    ``closing_started_unix``) but no ``closing_attempts`` list exists,
    the timeline still surfaces one synthesized ``closing`` event with
    its duration estimated from the latest audit-attempt ts."""
    import datetime as _dt
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "cls"})
    closing_start_iso = "2026-05-22T01:50:57+00:00"
    closing_start_unix = _dt.datetime.fromisoformat(closing_start_iso).timestamp()
    last_attempt_iso = "2026-05-22T01:52:30+00:00"
    _write_json(sd / "state.json", {
        "session_id": "cls",
        "closing_started_unix": closing_start_unix,
        "closing_report_task_id": "task-closing-1",
        "closing_phase": False,  # mirrors real session — flag is False
                                  # but unix ts is set, so closing was entered.
        "baseline_attempts": [{
            "ts": last_attempt_iso,
            "task_id": "last",
            "status": "succeeded",
            "decision": "promoted",
            "extras": {"duration_seconds": 10.0},
        }],
    })
    timeline = build(sd)["phase_timeline"]
    closing = [e for e in timeline if e.get("action") == "closing"]
    assert len(closing) == 1
    evt = closing[0]
    assert evt["ts"].startswith("2026-05-22T01:50:57")
    assert evt["task_id"] == "task-closing-1"
    # Duration ≈ 1:52:30 - 1:50:57 = 93 seconds.
    assert evt["duration_seconds"] is not None
    assert abs(evt["duration_seconds"] - 93.0) < 1.5
    assert evt["ended_ts_utc"] is not None
    assert (evt.get("extras") or {}).get("synthesized") is True


def test_phase_timeline_closing_attempts_list_surfaces(tmp_path: Path) -> None:
    """P2-2: when ``closing_attempts`` list IS populated (newer
    orchestrator), each entry becomes one timeline event and the
    synthetic fallback does NOT add an extra row."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "cls2"})
    _write_json(sd / "state.json", {
        "session_id": "cls2",
        "closing_started_unix": 1.0,
        "closing_attempts": [{
            "ts": "2026-05-22T02:00:00+00:00",
            "task_id": "cls-task-a",
            "status": "succeeded",
            "decision": "report_written",
            "extras": {"duration_seconds": 42.5},
        }, {
            "ts": "2026-05-22T02:01:00+00:00",
            "task_id": "cls-task-b",
            "status": "succeeded",
            "decision": "completed",
        }],
    })
    timeline = build(sd)["phase_timeline"]
    closing = [e for e in timeline if e.get("action") == "closing"]
    assert len(closing) == 2
    assert {e["task_id"] for e in closing} == {"cls-task-a", "cls-task-b"}
    a = [e for e in closing if e["task_id"] == "cls-task-a"][0]
    assert a["duration_seconds"] == 42.5
    assert a["decision"] == "report_written"


def test_phase_timeline_no_closing_event_when_not_entered(tmp_path: Path) -> None:
    """P2-2: no closing breadcrumbs → no synthesized event."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "nocls"})
    _write_json(sd / "state.json", {
        "session_id": "nocls",
        "baseline_attempts": [{
            "ts": "2026-05-20T10:00:00+00:00",
            "task_id": "b1",
            "status": "succeeded",
            "decision": "promoted",
        }],
    })
    timeline = build(sd)["phase_timeline"]
    closing = [e for e in timeline if e.get("action") == "closing"]
    assert closing == []


def test_phase_timeline_reads_tracelens_started_ended_duration(tmp_path: Path) -> None:
    """P2-3: when the kernel-agent writer records ``started_at`` /
    ``ended_at`` / ``duration_seconds`` on the status JSON, the
    timeline event picks them up (ts = started_at, duration filled,
    ended_ts_utc = ended_at)."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "tl-time"})
    _write_json(sd / "state.json", {"session_id": "tl-time"})
    tl_status = sd / "kernel-agent/runs/sess-tl/status/tracelens_analysis"
    tl_status.mkdir(parents=True)
    _write_json(tl_status / "run-z.json", {
        "tool": "tracelens_analysis",
        "state": "succeeded",
        "status": "ok",
        "summary": "ok",
        "started_at": "2026-05-22T01:00:00+00:00",
        "ended_at":   "2026-05-22T01:02:30+00:00",
        "duration_seconds": 150.0,
    })
    timeline = build(sd)["phase_timeline"]
    tl_evt = [e for e in timeline if e.get("action") == "tracelens_analysis"]
    assert len(tl_evt) == 1
    ev = tl_evt[0]
    assert ev["ts"] == "2026-05-22T01:00:00+00:00"
    assert ev["duration_seconds"] == 150.0
    assert ev["ended_ts_utc"] is not None
    assert ev["ended_ts_utc"].startswith("2026-05-22T01:02:30")


def test_kernel_profiling_reads_tracelens_duration(tmp_path: Path) -> None:
    """P2-3: ``kernel_profiling`` rows for tracelens_analysis runs
    surface ``duration_seconds`` and ``ended_ts_utc`` directly from
    the status JSON's new fields."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "tl-kp"})
    _write_json(sd / "state.json", {"session_id": "tl-kp", "framework": "sglang"})
    tl_root = sd / "kernel-agent/runs/sess-tl/status/tracelens_analysis"
    tl_root.mkdir(parents=True)
    _write_json(tl_root / "run-q.json", {
        "tool": "tracelens_analysis",
        "state": "succeeded",
        "status": "ok",
        "summary": "ok",
        "started_at": "2026-05-22T01:00:00+00:00",
        "ended_at":   "2026-05-22T01:05:00+00:00",
        "duration_seconds": 300.0,
        "top_kernels": [],
    })
    runs = build(sd)["kernel_profiling"]
    tl_runs = [r for r in runs if r.get("outputs", {}).get("tool") == "tracelens_analysis"]
    assert len(tl_runs) == 1
    r = tl_runs[0]
    assert r["ts"] == "2026-05-22T01:00:00+00:00"
    assert r["duration_seconds"] == 300.0
    assert r["ended_ts_utc"] is not None
    assert r["ended_ts_utc"].startswith("2026-05-22T01:05:00")


def test_phase_timeline_duration_and_tracelens_event(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "pt"})
    _write_json(sd / "state.json", {
        "session_id": "pt",
        "baseline_attempts": [{
            "ts": "2026-05-15T09:00:00+00:00",
            "task_id": "b1",
            "status": "succeeded",
            "decision": "promoted",
            "workspace": str(sd / "runs/baseline/b1"),
            "extras": {},
        }],
    })
    bench = sd / "runs/baseline/b1/benchmark_001"
    _write_json(bench / "benchmark_report.json", {
        "success": True,
        "duration_seconds": 120.0,
    })
    # TraceLens analysis run that should surface as a timeline event.
    tl_status = sd / "kernel-agent/runs/sess-pt/status/tracelens_analysis"
    tl_status.mkdir(parents=True)
    _write_json(tl_status / "run-z.json", {"status": "ok", "summary": "ok"})

    timeline = build(sd)["phase_timeline"]
    baseline_evt = [e for e in timeline if e["action"] == "baseline"][0]
    assert baseline_evt["duration_seconds"] == 120.0
    assert baseline_evt["ended_ts_utc"] is not None
    assert baseline_evt["ended_ts_utc"].startswith("2026-05-15T09:02:00")
    assert any(e["action"] == "tracelens_analysis" for e in timeline)


# ---------------------------------------------------------------------------
# E. Roofline merge into kernel_lifecycle.detected
# ---------------------------------------------------------------------------
def test_detected_to_roofline_match_uses_name_and_input_dims(tmp_path: Path) -> None:
    """P2-4: when two detected kernels share a name (e.g. ``aten::mm``)
    but differ in input shapes, the (name, input_dims) keyed merge
    assigns each its own roofline numbers — instead of both rows
    inheriting the highest-impact row's roofline (the pre-P2-4
    behaviour)."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "kl-multi"})
    _write_json(sd / "state.json", {
        "session_id": "kl-multi",
        "framework": "sglang",
        "last_select_kernels": {
            "candidates_path": str(
                sd / "kernel-agent/runs/sess-kl/kernel_candidates.json"
            ),
            "hot_kernels_top15": [],
        },
    })
    # kernel_candidates.json with two aten::mm rows of different shapes,
    # matching the production schema (input_shapes is a list of
    # {call_num, shape} dicts; shapes mirrors them for flat consumers).
    kar = sd / "kernel-agent/runs/sess-kl"
    kar.mkdir(parents=True)
    _write_json(kar / "kernel_candidates.json", {
        "hot_kernels": [
            {
                "kernel_id": "k001",
                "name": "aten::mm",
                "gpu_pct": 19.4,
                "input_shapes": [
                    {"call_num": 60, "shape": "(15360,6144) bf16"},
                    {"call_num": 60, "shape": "(6144,43008) bf16"},
                ],
                "shapes": ["(15360,6144) bf16", "(6144,43008) bf16"],
                "duration_us": 840000.0,
                "bottleneck": "compute",
            },
            {
                "kernel_id": "k002",
                "name": "aten::mm",
                "gpu_pct": 10.08,
                "input_shapes": [
                    {"call_num": 60, "shape": "(15360,21504) bf16"},
                    {"call_num": 60, "shape": "(21504,6144) bf16"},
                ],
                "shapes": ["(15360,21504) bf16", "(21504,6144) bf16"],
                "duration_us": 436000.0,
                "bottleneck": "compute",
            },
        ],
    })
    # Two TraceLens roofline rows, one per shape — note the
    # ``Input Dims`` strings use a different bracket / whitespace
    # convention to confirm the normalizer is shape-equivalent and
    # not string-equal.
    cat = kar / "tracelens/category_data"
    cat.mkdir(parents=True)
    _write_json(cat / "gemm_metrics.json", {
        "category": "gemm",
        "operations": [
            {
                "name": "aten::mm",
                "count": 60,
                "time_ms": 840.0,
                "percent_of_total": 19.4,
                "Input Dims": "((15360, 6144), (6144, 43008))",
                "efficiency": {
                    "efficiency_percent": 81.86,
                    "bound_type": "compute",
                    "flops_per_byte": 3982.0,
                    "tflops_achieved": 580.0,
                },
                "library": "Tensile",
            },
            {
                "name": "aten::mm",
                "count": 60,
                "time_ms": 436.0,
                "percent_of_total": 10.08,
                "Input Dims": "((15360, 21504), (21504, 6144))",
                "efficiency": {
                    "efficiency_percent": 65.4,
                    "bound_type": "compute",
                    "flops_per_byte": 2840.0,
                    "tflops_achieved": 462.0,
                },
                "library": "Tensile",
            },
        ],
    })
    detected = build(sd)["kernel_lifecycle"]["detected"]
    k1 = [d for d in detected if d["kernel_id"] == "k001"][0]
    k2 = [d for d in detected if d["kernel_id"] == "k002"][0]
    # Each detected kernel gets its OWN roofline row, not the highest-
    # impact one for both:
    assert k1["efficiency_percent"] == 81.86
    assert k1["flops_per_byte"] == 3982.0
    assert k1["tflops_achieved"] == 580.0
    assert k2["efficiency_percent"] == 65.4
    assert k2["flops_per_byte"] == 2840.0
    assert k2["tflops_achieved"] == 462.0
    # extras.input_dims surfaced for both
    assert (k1.get("extras") or {}).get("input_dims") == [
        [15360, 6144], [6144, 43008],
    ]
    assert (k2.get("extras") or {}).get("input_dims") == [
        [15360, 21504], [21504, 6144],
    ]


def test_detected_to_roofline_falls_back_to_name_when_no_dims(tmp_path: Path) -> None:
    """P2-4: when the detected row has no shape info we still get the
    pre-P2-4 name-only match (back-compat with old fixtures)."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "kl-noshape"})
    _write_json(sd / "state.json", {
        "session_id": "kl-noshape",
        "framework": "sglang",
        "profile_attempts": [{
            "ts": "2026-05-15T08:00:00+00:00",
            "task_id": "p1",
            "status": "succeeded",
            "decision": "promoted",
            "extras": {},
        }],
    })
    pdir = sd / "runs/profile/p1/benchmark_001"
    _write_json(pdir / "benchmark_report.json", {
        "success": True,
        "kernel_summary": [
            {"kernel_id": "k001", "name": "aten::mm", "gpu_pct": 50.0,
             "time_ms": 5.0, "bottleneck": ""},
        ],
    })
    cat = sd / "kernel-agent/runs/sess-kl/tracelens/category_data"
    cat.mkdir(parents=True)
    _write_json(cat / "gemm_metrics.json", {
        "category": "gemm",
        "operations": [{
            "name": "aten::mm",
            "count": 1,
            "time_ms": 5.0,
            "percent_of_total": 50.0,
            "Input Dims": "((1, 2), (2, 3))",
            "efficiency": {
                "efficiency_percent": 87.5,
                "bound_type": "compute",
                "flops_per_byte": 1234.0,
            },
        }],
    })
    detected = build(sd)["kernel_lifecycle"]["detected"]
    mm = [d for d in detected if d["name"] == "aten::mm"][0]
    # Name-only match still wires the roofline through.
    assert mm["efficiency_percent"] == 87.5
    assert mm["bound_type"] == "compute"


def test_kernel_lifecycle_detected_inherits_roofline(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "kl"})
    _write_json(sd / "state.json", {
        "session_id": "kl",
        "framework": "sglang",
        "profile_attempts": [{
            "ts": "2026-05-15T08:00:00+00:00",
            "task_id": "p1",
            "status": "succeeded",
            "decision": "promoted",
            "extras": {},
        }],
    })
    pdir = sd / "runs/profile/p1/benchmark_001"
    _write_json(pdir / "benchmark_report.json", {
        "success": True,
        "kernel_summary": [
            {"kernel_id": "k001", "name": "aten::mm", "gpu_pct": 50.0,
             "time_ms": 5.0, "bottleneck": ""},
        ],
    })
    # TraceLens roofline data living next to a kernel-agent run dir.
    cat = sd / "kernel-agent/runs/sess-kl/tracelens/category_data"
    cat.mkdir(parents=True)
    _write_json(cat / "gemm_metrics.json", {
        "category": "gemm",
        "operations": [{
            "name": "aten::mm",
            "count": 1,
            "time_ms": 5.0,
            "percent_of_total": 50.0,
            "efficiency": {
                "efficiency_percent": 87.5,
                "bound_type": "compute",
                "flops_per_byte": 1234.0,
            },
        }],
    })
    detected = build(sd)["kernel_lifecycle"]["detected"]
    assert detected
    mm = [k for k in detected if k.get("name") == "aten::mm"][0]
    assert mm.get("efficiency_percent") == 87.5
    assert mm.get("bound_type") == "compute"
    assert mm.get("arithmetic_intensity") == 1234.0
