"""R7 fill-layer tests: ``kernel_decision_path[].steps[].backend`` and
``.speedup`` inference.

These tests exercise the collector helpers that look beyond the explicit
``attempt.backend`` / ``attempt.speedup`` fields when those aren't
present in ``state.json`` — falling back to (a) per-attempt extras,
(b) the kernel-agent ``runs/<sid>/results/<kid>.json`` index, and
(c) path-based inference from artifact / patch / workspace strings.

Each test builds a minimal synthetic state (and, where relevant, a
minimal on-disk ``kernel-agent/runs/.../`` layout) and asserts the
collector resolves ``step.backend`` / ``step.speedup`` from the right
fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.breakdown.collectors import (
    collect_kernel_decision_path,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_kernel_agent_result(
    session_dir: Path,
    kid: str,
    *,
    best_backend: str | None = None,
    selected_backends: list[str] | None = None,
    attempt_backends: list[str] | None = None,
    micro_speedup: float | None = None,
    best_artifact_path: str | None = None,
) -> None:
    """Write a minimal kernel-agent results/<kid>.json under the
    session dir. Mirrors the real on-disk schema closely enough that
    :func:`_load_kernel_agent_kernel_index` indexes it."""
    sid = session_dir.name
    run_dir = session_dir / "kernel-agent" / "runs" / sid
    result = {
        "kernel_id": kid,
        "session_id": sid,
        "attempts": [
            {"backend": b, "attempt_id": f"{b}-x", "status": "completed"}
            for b in (attempt_backends or [])
        ],
        "selected_backends": selected_backends or [],
        "verification": {
            "best_backend": best_backend or "",
            "micro_speedup": micro_speedup,
            "best_artifact_path": best_artifact_path or "",
        },
        "best_artifact_path": best_artifact_path or "",
    }
    _write_json(run_dir / "results" / f"{kid}.json", result)


# ---------------------------------------------------------------------------
# kernel_opt backend inference
# ---------------------------------------------------------------------------
def test_kdp_step_backend_from_attempt_field() -> None:
    """``history[].backend`` (rare) wins over everything else."""
    state = {
        "kernel_opt_attempts": {
            "k001": {
                "attempts": 1,
                "history": [{
                    "decision": "KEEP",
                    "ts": "2026-01-01T00:00:00+00:00",
                    "backend": "geak",
                }],
            },
        },
    }
    out = collect_kernel_decision_path(state, [], session_dir=None)
    step = out[0]["steps"][0]
    assert step["step"] == "kernel_opt"
    assert step["backend"] == "geak"


def test_kdp_step_backend_from_extras() -> None:
    """``history[].extras.backend`` is the second-tier source."""
    state = {
        "kernel_opt_attempts": {
            "k001": {
                "attempts": 1,
                "history": [{
                    "decision": "KEEP",
                    "ts": "2026-01-01T00:00:00+00:00",
                    "extras": {"backend": "claude"},
                }],
            },
        },
    }
    out = collect_kernel_decision_path(state, [], session_dir=None)
    assert out[0]["steps"][0]["backend"] == "claude"


def test_kdp_step_backend_inferred_from_path(tmp_path: Path) -> None:
    """No explicit backend anywhere — recover ``geak`` from
    ``ent.last_artifact_path`` which encodes ``/kernel-agent/geak/...``."""
    state = {
        "kernel_opt_attempts": {
            "k001": {
                "attempts": 1,
                "last_decision": "REVERT",
                "last_ts": "2026-01-01T00:00:00+00:00",
                "last_artifact_path": (
                    "/hl/users/abc/kernel-agent/geak/sid/geak-x/results/r1/"
                    "worktrees/slot_0/aiter/ops/foo.py"
                ),
                "history": [{
                    "decision": "REVERT",
                    "ts": "2026-01-01T00:00:00+00:00",
                }],
            },
        },
    }
    # No on-disk kernel-agent files → fall through to path inference.
    sd = tmp_path / "session"
    sd.mkdir()
    out = collect_kernel_decision_path(state, [], session_dir=sd)
    assert out[0]["steps"][0]["backend"] == "geak"


def test_kdp_step_backend_from_kernel_agent_results(tmp_path: Path) -> None:
    """When ``kernel-agent/runs/<sid>/results/<kid>.json`` exists, its
    ``verification.best_backend`` fills steps with no explicit backend
    — this is the most common case in real sessions because
    ``state.kernel_opt_attempts[].history`` rows don't carry backend."""
    sd = tmp_path / "session"
    sd.mkdir()
    _make_kernel_agent_result(
        sd, "k001", best_backend="codex", selected_backends=["codex"],
    )
    state = {
        "kernel_opt_attempts": {
            "k001": {
                "attempts": 1,
                "history": [{
                    "decision": "KEEP",
                    "ts": "2026-01-01T00:00:00+00:00",
                }],
            },
        },
    }
    out = collect_kernel_decision_path(state, [], session_dir=sd)
    assert out[0]["steps"][0]["backend"] == "codex"


# ---------------------------------------------------------------------------
# kernel_opt speedup inference
# ---------------------------------------------------------------------------
def test_kdp_step_speedup_from_extras() -> None:
    """``history[].extras.kernel_speedup`` flows through directly."""
    state = {
        "kernel_opt_attempts": {
            "k001": {
                "attempts": 1,
                "history": [{
                    "decision": "KEEP",
                    "ts": "2026-01-01T00:00:00+00:00",
                    "extras": {"backend": "geak", "kernel_speedup": 1.45},
                }],
            },
        },
    }
    out = collect_kernel_decision_path(state, [], session_dir=None)
    step = out[0]["steps"][0]
    assert step["speedup"] == 1.45


def test_kdp_step_speedup_from_history_micro_field() -> None:
    """``history[].micro`` (the orchestrator's stored per-attempt
    micro_speedup) maps to ``step.speedup``."""
    state = {
        "kernel_opt_attempts": {
            "k001": {
                "attempts": 1,
                "history": [{
                    "decision": "KEEP",
                    "ts": "2026-01-01T00:00:00+00:00",
                    "micro": 1.33,
                }],
            },
        },
    }
    out = collect_kernel_decision_path(state, [], session_dir=None)
    assert out[0]["steps"][0]["speedup"] == 1.33


def test_kdp_step_speedup_from_kernel_agent_results(tmp_path: Path) -> None:
    """When the history row has no speedup, the kernel-level
    ``verification.micro_speedup`` from results/<kid>.json is patched
    onto the terminal kernel_opt step so the kernel_speedup signal
    isn't lost."""
    sd = tmp_path / "session"
    sd.mkdir()
    _make_kernel_agent_result(
        sd, "k001", best_backend="claude", micro_speedup=4.32,
    )
    state = {
        "kernel_opt_attempts": {
            "k001": {
                "attempts": 1,
                "last_micro_speedup": 4.32,
                "history": [{
                    "decision": "KEEP",
                    "ts": "2026-01-01T00:00:00+00:00",
                }],
            },
        },
    }
    out = collect_kernel_decision_path(state, [], session_dir=sd)
    step = out[0]["steps"][0]
    assert step["backend"] == "claude"
    assert step["speedup"] == 4.32


# ---------------------------------------------------------------------------
# integrate backend / speedup inference
# ---------------------------------------------------------------------------
def test_kdp_integrate_backend_inferred_from_patch_path() -> None:
    """The integrate row's ``patch_path`` lives under
    ``/kernel-agent/<backend>/`` — that's the authoritative source for
    integrate ``step.backend``."""
    state = {
        "kernel_integrate_attempts": {
            "k001|patch|": {
                "kernel_id": "k001",
                "patch_path": (
                    "/hl/users/abc/kernel-agent/oob/sid/tasks/cli/xyz/"
                    "workspace/optimized_versions/v4_foo.py"
                ),
                "target_file": "/sgl/aiter/foo.py",
                "attempts": [{
                    "decision": "REVERT",
                    "status": "ok",
                    "ts": "2026-01-01T00:10:00+00:00",
                    "gain_pct": -12.0,
                }],
            },
        },
    }
    out = collect_kernel_decision_path(state, [], session_dir=None)
    step = out[0]["steps"][0]
    assert step["step"] == "integrate"
    assert step["backend"] == "oob"


def test_kdp_integrate_speedup_carries_kernel_speedup(tmp_path: Path) -> None:
    """When ``results/<kid>.json`` reports a kernel_speedup but the
    integrate attempt only carries an e2e ``gain_pct``, the integrate
    step.speedup is filled from the kernel-level micro_speedup. The
    e2e gain stays in step.gain_pct (different unit)."""
    sd = tmp_path / "session"
    sd.mkdir()
    _make_kernel_agent_result(sd, "k001", best_backend="oob", micro_speedup=1.33)
    state = {
        "kernel_integrate_attempts": {
            "k001|p|": {
                "kernel_id": "k001",
                "patch_path": (
                    "/hl/users/abc/kernel-agent/oob/sid/tasks/cli/x/workspace/"
                    "optimized_versions/v1_foo.py"
                ),
                "attempts": [{
                    "decision": "REVERT",
                    "status": "ok",
                    "ts": "2026-01-01T00:30:00+00:00",
                    "gain_pct": -8.4,  # e2e — not the same as kernel speedup
                }],
            },
        },
    }
    out = collect_kernel_decision_path(state, [], session_dir=sd)
    step = out[0]["steps"][0]
    assert step["step"] == "integrate"
    assert step["speedup"] == 1.33
    assert step["gain_pct"] == -8.4   # e2e gain stays in gain_pct


# ---------------------------------------------------------------------------
# Summary aggregation + select-step invariant
# ---------------------------------------------------------------------------
def test_kdp_summary_backends_attempted_aggregated(tmp_path: Path) -> None:
    """``summary.backends_attempted`` is the de-duped ordered set of
    backends across every step that resolved one — including integrate
    steps inferred from ``patch_path``."""
    sd = tmp_path / "session"
    sd.mkdir()
    _make_kernel_agent_result(sd, "k001", best_backend="claude")
    state = {
        "kernel_opt_attempts": {
            "k001": {
                "attempts": 1,
                "history": [
                    {
                        "decision": "PARTIAL",
                        "ts": "2026-01-01T00:00:00+00:00",
                        "extras": {"backend": "geak"},
                    },
                    {
                        "decision": "KEEP",
                        "ts": "2026-01-01T00:10:00+00:00",
                    },
                ],
            },
        },
        "kernel_integrate_attempts": {
            "k001|p|": {
                "kernel_id": "k001",
                "patch_path": (
                    "/hl/users/abc/kernel-agent/oob/sid/tasks/cli/x/workspace/"
                    "optimized_versions/v1_foo.py"
                ),
                "attempts": [{
                    "decision": "KEEP",
                    "status": "ok",
                    "ts": "2026-01-01T00:30:00+00:00",
                    "gain_pct": 5.0,
                }],
            },
        },
    }
    out = collect_kernel_decision_path(state, [], session_dir=sd)
    summary = out[0]["summary"]
    # geak from extras, claude from results.json, oob from integrate
    # patch_path — all three should appear in insertion order.
    assert summary["backends_attempted"] == ["geak", "claude", "oob"]
    step_backends = {s.get("backend") for s in out[0]["steps"] if s.get("backend")}
    assert step_backends == set(summary["backends_attempted"])


def test_kdp_select_validate_steps_stay_none_when_no_data() -> None:
    """``select`` and ``validate`` steps have no backend / speedup
    semantics — we never invent values for them. The collector should
    leave both fields None even when the kernel does have backend
    information on its kernel_opt / integrate steps."""
    state = {
        "last_select_kernels": {
            "ts": "2026-01-01T00:00:00+00:00",
            "hot_kernels_top15": [{
                "kernel_id": "k001",
                "name": "foo",
                "bottleneck": "memory",
            }],
        },
        "kernel_opt_attempts": {
            "k001": {
                "attempts": 1,
                "history": [{
                    "decision": "KEEP",
                    "ts": "2026-01-01T00:10:00+00:00",
                    "extras": {"backend": "geak"},
                }],
            },
        },
        "validate_stack_attempts": [{
            "ts": "2026-01-01T00:20:00+00:00",
            "decision": "promoted",
            "extras": {"kernel_id": "k001"},
        }],
    }
    out = collect_kernel_decision_path(state, [], session_dir=None)
    by_step = {s["step"]: s for s in out[0]["steps"]}
    assert by_step["select"]["backend"] is None
    assert by_step["select"]["speedup"] is None
    assert by_step["validate"]["backend"] is None
    assert by_step["validate"]["speedup"] is None
    # but kernel_opt picked up the explicit extras.backend
    assert by_step["kernel_opt"]["backend"] == "geak"
