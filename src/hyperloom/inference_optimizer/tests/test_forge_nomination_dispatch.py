# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The auto=true / auto=false fork in run_optimization_handler, end to end.

These lock in the whole Step-12 wiring: with the nomination env set, the handler
projects the candidate list into the manifest forge reads, derives the rewrite
budget, writes the request, runs ``forge-loop --auto`` (mocked at the subprocess
boundary only) and queues every returned sibling for the shared integrate lane.
With the env unset, none of that happens and the legacy selector path runs
unchanged.

Only the subprocess is mocked. The real SharedState, real candidate artifact,
real producers, and the real ``enqueue_nominated_patch`` queue all run.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.state.shared_state import SharedState

_AUTO_ENV = "HYPERLOOM_FORGE_NOMINATION_AUTO"


def _row(kernel_id: str, *, root: Path | None = None, **overrides: Any) -> dict[str, Any]:
    """One candidate row; ``root`` puts its source file inside that workspace."""
    source_file = str(root / f"{kernel_id}.py") if root is not None else f"/repo/{kernel_id}.py"
    row: dict[str, Any] = {
        "kernel_id": kernel_id,
        "name": f"{kernel_id}_kernel",
        "gpu_pct": 12.0,
        "source_file": source_file,
        "reusable_native_kernel": True,
        "skip_reason": "",
    }
    row.update(overrides)
    return row


def _candidates(session_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = session_dir / "kernel_candidates.json"
    path.write_text(json.dumps({"hot_kernels": rows}), encoding="utf-8")
    return path


def _seed_state(session_dir: Path, *, trace: Path, max_minutes: float | None = 600.0) -> None:
    state = SharedState.load_or_init(session_dir)
    if max_minutes is not None:
        state.max_minutes = max_minutes
    state.last_profile_trace = str(trace)
    state.save(session_dir)


def _canned_envelope(patches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "patches": patches,
        "nomination": {"candidates_seen": 3, "resolved": 2, "selected": len(patches)},
        "improved": bool(patches),
    }


def _patch_entry(kernel: str, *, micro: float) -> dict[str, Any]:
    """A forge envelope patch row that parse_outcome accepts and that has a real
    on-disk artifact + target so enqueue_nominated_patch queues it."""
    return {
        "kernel_name": f"{kernel}_kernel",
        "patch_path": f"/repo/{kernel}.patch",
        "target_file": f"/repo/{kernel}.py",
        "micro_speedup": micro,
    }


# --------------------------------------------------------------------------- #
# auto=true
# --------------------------------------------------------------------------- #
def test_auto_true_produces_manifest_request_and_queues_every_sibling(tmp_path, monkeypatch):
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(
        tmp_path,
        [
            _row("k001", root=tmp_path, gpu_pct=30.0),
            # An unroutable row: the manifest must keep it as a superset even
            # though the selector would drop it.
            _row("k002", source_file="", reusable_native_kernel=False, skip_reason="source file not resolved"),
        ],
    )
    _seed_state(tmp_path, trace=trace)

    seen: dict[str, Any] = {}

    def _fake_submit_auto(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return _canned_envelope([_patch_entry("k001", micro=8.0), _patch_entry("k009", micro=3.0)])

    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(forge_submit, "submit_auto", _fake_submit_auto)

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))

    # The handler reports the auto outcome, not a single-kernel result.
    assert result["status"] == "complete"
    assert result["auto"] is True
    assert len(result["nominated_patches"]) == 2
    # No single-kernel stamping leaked onto the auto result.
    assert "kernel_id" not in result
    assert "kernel_id_pinned" not in result
    assert "requested_kernel_id" not in result

    # The manifest was written and keeps the unroutable row (superset).
    manifest = json.loads((tmp_path / "forge_candidate_manifest.json").read_text(encoding="utf-8"))
    by_id = {entry["kernel_id"]: entry for entry in manifest["hot_kernels"]}
    assert set(by_id) == {"k001", "k002"}

    # The request points forge at the MANIFEST, carries a real budget, protocol 1.
    request = json.loads((tmp_path / "forge_nomination_input.json").read_text(encoding="utf-8"))
    assert request["lane"] == "rewrite"
    assert Path(request["candidates_path"]) == (tmp_path / "forge_candidate_manifest.json").resolve()
    assert request["lane_budget_sec"] > 0
    # A 600-min session leaves ~35 700 phase-sec, the rewrite lane takes 50%
    # (~17 850 sec), and the 4500-sec admission floor funds 3 targets -- but
    # forge executes one per call, so the request asks for one.
    assert krh._nomination_lane_budget(SharedState.load_or_init(tmp_path)).max_targets == 3
    assert request["max_kernels"] == 1
    assert request["protocol_version"] == 1

    # submit_auto was handed the request path, not a named kernel.
    assert Path(seen["nomination_input"]) == (tmp_path / "forge_nomination_input.json")

    # Queued on the caller's own state, on the REWRITE lane (no fusion stamping),
    # so the drain lifts them as action="integrate".
    state = SharedState.load_or_init(tmp_path)
    assert krh.queue_nominated_siblings(state, result["nominated_patches"]) == 2
    records = list(state.pending_kernel_integrations.values())
    assert len(records) == 2
    for record in records:
        assert record["status"] == "pending"
        assert record.get("source") != "forge_fusion"
        assert "action_label" not in record
        assert record["task_key"].startswith("forge_rewrite:")


def test_auto_true_empty_nomination_queues_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    _seed_state(tmp_path, trace=trace)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(forge_submit, "submit_auto", lambda **_: _canned_envelope([]))

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))
    assert result["status"] == "complete"
    assert result["nominated_patches"] == []
    state = SharedState.load_or_init(tmp_path)
    assert not state.pending_kernel_integrations


def test_auto_true_forge_timeout_is_surfaced_not_reported_complete(tmp_path, monkeypatch):
    """A crashed/timed-out forge run must not look like a clean empty nomination."""
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    _seed_state(tmp_path, trace=trace)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(
        forge_submit,
        "submit_auto",
        lambda **_: {"status": "timeout", "patches": [], "error": "deadline exceeded"},
    )

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))
    # The forge status and its error ride through; NOT collapsed to complete.
    assert result["status"] == "timeout"
    assert result["auto"] is True
    assert result["nominated_patches"] == []
    assert result["error"] == "deadline exceeded"
    state = SharedState.load_or_init(tmp_path)
    assert not state.pending_kernel_integrations


def test_auto_true_missing_trace_fails_cleanly_without_leaking_request(tmp_path, monkeypatch):
    """A funded run with no decode trace becomes a failed result, not an exception.

    The up-front validation only checks that candidates_path exists; it never
    confirms a trace was captured. build_request then rejects the empty trace, and
    the auto path must degrade to a failed HandlerResult (the phase caller expects
    a result, not a raise) rather than leaking a request artifact.
    """
    monkeypatch.setenv(_AUTO_ENV, "1")
    candidates = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    # Funded budget, but NO last_profile_trace -> _resolve_fusion_decode_trace "".
    state = SharedState.load_or_init(tmp_path)
    state.max_minutes = 600.0
    state.save(tmp_path)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(
        forge_submit,
        "submit_auto",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not reach forge without a trace")),
    )

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))
    assert result["status"] == "failed"
    assert result["auto"] is True
    assert "trace" in result["error"].lower()
    # The request was never written (the producer raised before write_request).
    assert not (tmp_path / "forge_nomination_input.json").exists()


def test_auto_true_unbounded_session_skips_without_calling_forge(tmp_path, monkeypatch):
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    # No max_minutes -> unbounded -> zero budget -> nothing to nominate.
    _seed_state(tmp_path, trace=trace, max_minutes=None)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    called = {"n": 0}

    def _boom(**_: Any) -> dict[str, Any]:
        called["n"] += 1
        raise AssertionError("submit_auto must not run without a budget")

    monkeypatch.setattr(forge_submit, "submit_auto", _boom)

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))
    assert result["status"] == "skipped"
    assert result["reason"] == "no_budget"
    assert called["n"] == 0
    # No artifact leaks: the budget is checked BEFORE anything is written, so
    # neither the request nor the manifest is left behind on the skip.
    assert not (tmp_path / "forge_nomination_input.json").exists()
    assert not (tmp_path / "forge_candidate_manifest.json").exists()


def test_auto_true_without_candidates_path_skips(tmp_path, monkeypatch):
    monkeypatch.setenv(_AUTO_ENV, "1")
    # Trace-analyze validation passes off a cached candidates_path, but the
    # payload itself names none -> the manifest helper has nothing to project and
    # the auto path skips before ever reaching forge.
    cached = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    state = SharedState.load_or_init(tmp_path)
    state.max_minutes = 600.0
    state.last_profile_trace = str(tmp_path / "t.json")
    state.last_trace_analyze = {"candidates_path": str(cached)}
    state.save(tmp_path)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(
        forge_submit,
        "submit_auto",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    result = asyncio.run(krh.run_optimization_handler({}, session_dir=tmp_path))
    assert result["status"] == "skipped"
    assert result["reason"] == "no_candidates"


def test_auto_true_no_nominatable_row_fails_before_writing_anything(tmp_path, monkeypatch):
    """With nothing to stage from, no workspace could make a nomination runnable.

    submit_auto stages a worktree of the tree the candidates live in, so being
    outside the session directory is normal and not a refusal. What is futile is
    a brief whose every row lacks a resolved source or was already rejected:
    there is no tree to branch from, and forge's containment check would refuse
    whatever was picked.
    """
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(
        tmp_path,
        [
            {"kernel_id": "k001", "name": "unlocated", "gpu_pct": 30.0, "reusable_native_kernel": True},
            {"kernel_id": "k002", "name": "also_unlocated", "gpu_pct": 10.0, "reusable_native_kernel": True},
        ],
    )
    _seed_state(tmp_path, trace=trace)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(
        forge_submit,
        "submit_auto",
        lambda **_: (_ for _ in ()).throw(AssertionError("no candidate is stageable; forge must not run")),
    )

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))
    assert result["status"] == "failed"
    assert result["auto"] is True
    assert result["error_class"] == "forge_workspace_staging_unavailable"
    assert "no runnable target" in result["error"]
    # The refusal happens before any write, so no artifact is left behind.
    assert not (tmp_path / "forge_candidate_manifest.json").exists()
    assert not (tmp_path / "forge_nomination_input.json").exists()


def test_auto_true_a_row_outside_the_session_dir_still_proceeds(tmp_path, monkeypatch):
    """The live install tree is never inside the session dir, so it must proceed.

    Refusing here is what kept the lane dormant: every real candidate lives in a
    framework tree elsewhere on disk, and staging is what brings it in.
    """
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    outside = tmp_path.parent / "elsewhere"
    candidates = _candidates(
        tmp_path,
        [_row("k001", root=outside, gpu_pct=30.0), _row("k002", root=outside, gpu_pct=10.0)],
    )
    _seed_state(tmp_path, trace=trace)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(forge_submit, "submit_auto", lambda **_: _canned_envelope([]))

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))
    assert result == {
        "status": "complete",
        "auto": True,
        "nominated_patches": [],
        "dropped": {},
        "nomination": {"candidates_seen": 3, "resolved": 2, "selected": 0},
    }
    assert (tmp_path / "forge_candidate_manifest.json").exists()


def test_auto_true_a_rejected_row_does_not_count_as_nominatable(tmp_path, monkeypatch):
    """A rejected row can never be picked, so it cannot satisfy the guard.

    Its source resolves, which is the only reason it could be mistaken for a
    stageable target; the other row has no source at all.
    """
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(
        tmp_path,
        [
            {"kernel_id": "k001", "name": "unlocated", "gpu_pct": 30.0, "reusable_native_kernel": True},
            _row("k002", root=tmp_path, gpu_pct=10.0),
        ],
    )
    state = SharedState.load_or_init(tmp_path)
    state.max_minutes = 600.0
    state.last_profile_trace = str(trace)
    state.rejected_kernel_ids = ["k002"]
    state.save(tmp_path)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(
        forge_submit,
        "submit_auto",
        lambda **_: (_ for _ in ()).throw(AssertionError("the only resolved row is rejected; forge must not run")),
    )

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))
    assert result["status"] == "failed"
    assert result["error_class"] == "forge_workspace_staging_unavailable"
    assert not (tmp_path / "forge_candidate_manifest.json").exists()


# --------------------------------------------------------------------------- #
# auto=false (env unset)
# --------------------------------------------------------------------------- #
def test_auto_false_never_touches_the_nomination_path(tmp_path, monkeypatch):
    monkeypatch.delenv(_AUTO_ENV, raising=False)
    candidates = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    _seed_state(tmp_path, trace=tmp_path / "t.json", max_minutes=600.0)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(
        forge_submit,
        "submit_auto",
        lambda **_: (_ for _ in ()).throw(AssertionError("auto=false must never call submit_auto")),
    )
    # Prove the legacy selector path is entered instead: stub the selector to
    # return nothing so the handler short-circuits without a real subprocess.
    selector_calls = {"n": 0}

    def _fake_selector(payload, *, session_dir, skipped_out):
        selector_calls["n"] += 1
        return []

    monkeypatch.setattr(krh, "_batch_kernel_candidates", _fake_selector)

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))
    # The selector ran; no manifest / request / auto result was produced.
    assert selector_calls["n"] == 1
    assert result.get("auto") is not True
    assert not (tmp_path / "forge_candidate_manifest.json").exists()
    assert not (tmp_path / "forge_nomination_input.json").exists()


def test_auto_true_a_failed_run_queues_nothing_even_when_it_returns_patches(tmp_path, monkeypatch):
    """A patch from a run whose own tooling broke must never reach the queue.

    The status is decided before anything lands, so a nonzero-exit envelope that
    still lists a sibling leaves the pending queue untouched.
    """
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    _seed_state(tmp_path, trace=trace)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    envelope = {
        "status": "failed",
        "patches": [_patch_entry("k001", micro=1.4)],
        "error": "forge-loop --auto exited rc=2",
    }
    monkeypatch.setattr(forge_submit, "submit_auto", lambda **_: envelope)

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))

    assert result["status"] == "failed"
    assert result["nominated_patches"] == []
    assert result["error"] == "forge-loop --auto exited rc=2"
    state = SharedState.load_or_init(tmp_path)
    assert state.pending_kernel_integrations == {}


def test_auto_true_a_timed_out_run_queues_nothing_even_when_it_returns_patches(tmp_path, monkeypatch):
    """Same rule for a deadline kill: the queue stays empty."""
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    _seed_state(tmp_path, trace=trace)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    envelope = {
        "status": "timeout",
        "patches": [_patch_entry("k001", micro=2.0)],
        "error": "deadline exceeded",
    }
    monkeypatch.setattr(forge_submit, "submit_auto", lambda **_: envelope)

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))

    assert result["status"] == "timeout"
    assert result["nominated_patches"] == []
    state = SharedState.load_or_init(tmp_path)
    assert state.pending_kernel_integrations == {}


def test_auto_true_reports_named_refusals_alongside_the_queued_siblings(tmp_path, monkeypatch):
    """A refused sibling must be visible, not absorbed into a lower queued count."""
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    _seed_state(tmp_path, trace=trace)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    good = _patch_entry("k001", micro=1.5)
    envelope = _canned_envelope(
        [
            good,
            # Same name as the keeper: collapses to the stronger claim.
            {**good, "micro_speedup": 1.1},
            # No patch_path: nothing to apply.
            {"kernel_name": "k002_kernel", "target_file": "/repo/k002.py"},
            "not-an-object",
        ]
    )
    monkeypatch.setattr(forge_submit, "submit_auto", lambda **_: envelope)

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))

    assert result["status"] == "complete"
    assert len(result["nominated_patches"]) == 1
    assert result["dropped"] == {
        "duplicate_kernel_name": 1,
        "missing_patch_path": 1,
        "not_an_object": 1,
    }


def test_auto_true_an_all_malformed_envelope_is_not_a_clean_empty_nomination(tmp_path, monkeypatch):
    """Every entry refused looks identical to "nominated nothing" without this.

    Forge believed it nominated, so nothing usable arriving is a disagreement
    about the contract: it fails, and names each refusal.
    """
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    _seed_state(tmp_path, trace=trace)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    envelope = _canned_envelope([{"patch_path": "/repo/x.patch", "target_file": "/repo/x.py"}])
    monkeypatch.setattr(forge_submit, "submit_auto", lambda **_: envelope)

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))

    assert result["status"] == "failed"
    assert "nominated_patches" not in result
    assert result["dropped"] == {"missing_kernel_name": 1}


def test_auto_true_a_clean_empty_nomination_reports_no_refusals(tmp_path, monkeypatch):
    """The other side of the same signal: nothing offered, nothing refused."""
    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    _seed_state(tmp_path, trace=trace)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(forge_submit, "submit_auto", lambda **_: _canned_envelope([]))

    result = asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))

    assert result["status"] == "complete"
    assert result["nominated_patches"] == []
    assert result["dropped"] == {}


def test_auto_true_says_the_nominator_is_still_a_placeholder(tmp_path, monkeypatch, caplog):
    """An operator flipping the env has to learn the trace is not read yet.

    Without this the path looks like forge-driven kernel selection, when the
    shipped nominator only reranks candidates Hyperloom already resolved.
    """
    import logging

    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(tmp_path, [_row("k001", root=tmp_path, gpu_pct=30.0)])
    _seed_state(tmp_path, trace=trace)

    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(forge_submit, "submit_auto", lambda **_: _canned_envelope([]))

    with caplog.at_level(logging.WARNING, logger=krh.log.name):
        asyncio.run(krh.run_optimization_handler({"candidates_path": str(candidates)}, session_dir=tmp_path))

    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("placeholder" in message and "does not read the trace" in message for message in warnings)
