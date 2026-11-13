# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The export gate: search waits until a KEEP's benchmark really ran an exported module.

Evidence, not source, decides. A module that names the AOTI entry points in a helper
nobody calls, in a dead branch, or in the ``except`` arm of a fallback to eager parses
exactly like one that exports, and benchmarks just as happily. The run's own call
counters are what separate them.

The whole mechanism is inert unless the operator declared ``aoti.enabled``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.measurement.aoti_evidence import evidence_dir
from hyperloom.orchestrator.state.shared_state import SharedState

_SEARCH = ("explore", "sweep", "profile", "roofline")


def _declare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True, aoti: bool = True) -> None:
    module = tmp_path / "hyperloom_optimize.py"
    module.write_text("def hyperloom_optimize(model):\n    return model\n", encoding="utf-8")
    scope: dict[str, object] = {"schema_version": 2, "file": str(module)}
    if aoti:
        scope["aoti"] = {"enabled": enabled}
    (tmp_path / "optimization_scope.json").write_text(json.dumps(scope), encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))


def _counters(workspace: Path, **totals: object) -> None:
    out = evidence_dir(workspace)
    out.mkdir(parents=True, exist_ok=True)
    (out / "1234.json").write_text(json.dumps({"instrumented": True, **totals}), encoding="utf-8")


class _Dispatcher:
    """The denial surface, with only the state the gate reads."""

    def __init__(self, exports: bool = False, export_landed: bool = False) -> None:
        self.shared_state = SharedState()
        self.shared_state.current_best_exports = exports
        self.shared_state.export_landed = export_landed or exports

    def denial(self, action: str):
        from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator

        self._export_gate_denied = DispatcherCollaborator._export_gate_denied
        return DispatcherCollaborator._export_gate_denial_for_action(self, action)


class _Promoter:
    """The promotion surface the gate recorder touches."""

    def __init__(self) -> None:
        self.shared_state = SharedState()

    def record(self, *, kept_flag: bool, workspace: str = "") -> bool:
        from hyperloom.orchestrator.loop._integrate_promotion import IntegratePromotionMixin

        self._keep_really_exported = lambda ws: IntegratePromotionMixin._keep_really_exported(self, ws)
        return IntegratePromotionMixin._record_export_gate(self, kept_flag=kept_flag, workspace=workspace)


@pytest.mark.parametrize("action", _SEARCH)
def test_search_is_denied_until_the_export_lands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    _declare(tmp_path, monkeypatch)
    denial = _Dispatcher().denial(action)
    assert denial is not None
    assert denial.rule == "export_gate"


@pytest.mark.parametrize("action", ("baseline", "integrate", "report", "target_analysis"))
def test_the_work_that_lands_the_export_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """A gate that denied integrate could never be opened."""
    _declare(tmp_path, monkeypatch)
    assert _Dispatcher().denial(action) is None


@pytest.mark.parametrize("action", _SEARCH)
def test_a_landed_export_opens_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    _declare(tmp_path, monkeypatch)
    assert _Dispatcher(exports=True).denial(action) is None


@pytest.mark.parametrize("action", _SEARCH)
def test_a_workload_declaring_no_export_is_never_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    _declare(tmp_path, monkeypatch, enabled=False)
    assert _Dispatcher().denial(action) is None


@pytest.mark.parametrize("action", _SEARCH)
def test_no_declaration_at_all_is_never_gated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    assert _Dispatcher().denial(action) is None


def test_an_unreadable_scope_holds_the_gate_shut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A false denial costs one tick; a false admission spends the budget the gate protects."""
    (tmp_path / "optimization_scope.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    denial = _Dispatcher().denial("explore")
    assert denial is not None and "unreadable" in str(denial)


def test_a_keep_that_invoked_the_exported_path_opens_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch)
    workspace = tmp_path / "run_0"
    _counters(workspace, compiled=1, loaded=1, invoked=12)
    promoter = _Promoter()
    assert promoter.record(kept_flag=True, workspace=str(workspace)) is True
    assert promoter.shared_state.export_landed is True
    assert promoter.shared_state.export_attempts == 0


def test_a_keep_that_compiled_but_never_invoked_is_only_an_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package built and never used benchmarks exactly like one that ran."""
    _declare(tmp_path, monkeypatch)
    workspace = tmp_path / "run_0"
    _counters(workspace, compiled=1, loaded=1, invoked=0)
    promoter = _Promoter()
    assert promoter.record(kept_flag=True, workspace=str(workspace)) is True
    assert promoter.shared_state.export_landed is False
    assert promoter.shared_state.export_attempts == 1


def test_counters_that_never_appeared_hold_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent counters mean the start-up shim did not run -- a broken harness, not a verdict."""
    _declare(tmp_path, monkeypatch)
    promoter = _Promoter()
    promoter.record(kept_flag=True, workspace=str(tmp_path / "run_0"))
    assert promoter.shared_state.export_landed is False
    assert promoter.shared_state.export_attempts == 1


def test_a_revert_is_not_an_export_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A candidate that never reached the tree would read as progress toward an export."""
    _declare(tmp_path, monkeypatch)
    promoter = _Promoter()
    assert promoter.record(kept_flag=False, workspace=str(tmp_path / "run_0")) is False
    assert promoter.shared_state.export_attempts == 0


def test_a_workload_declaring_no_export_records_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch, enabled=False)
    promoter = _Promoter()
    assert promoter.record(kept_flag=True, workspace=str(tmp_path / "run_0")) is False
    assert promoter.shared_state.export_attempts == 0


def test_every_keep_is_re_judged_so_the_gate_can_close_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``current_best`` moves on every KEEP, so "does the deliverable export?" has a fresh
    answer each time rather than being latched at the first landing.

    An eager KEEP arriving here is not a normal flow -- the perf gate refuses one, on
    absent counters too, while an export holds the slot. This is the backstop for the
    lane that bypasses that gate: an enablement round keeps through ``_gate_enablement``,
    which weighs bring-up rather than throughput."""
    _declare(tmp_path, monkeypatch)
    promoter = _Promoter()
    exporting, eager = tmp_path / "run_0", tmp_path / "run_1"
    _counters(exporting, compiled=1, loaded=1, invoked=12)
    _counters(eager, compiled=0, loaded=0, invoked=0)

    assert promoter.record(kept_flag=True, workspace=str(exporting)) is True
    assert promoter.shared_state.current_best_exports is True

    assert promoter.record(kept_flag=True, workspace=str(eager)) is True
    assert promoter.shared_state.current_best_exports is False
    assert promoter.shared_state.export_attempts == 1


def test_an_export_that_ever_landed_stays_landed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``export_landed`` is one-way on purpose: it is read only by the first-export
    throughput waiver, which must never apply twice. If losing the slot re-armed it, a slow
    export would be waived back in, lose to eager again, and the two could trade forever."""
    _declare(tmp_path, monkeypatch)
    promoter = _Promoter()
    exporting, eager = tmp_path / "run_0", tmp_path / "run_1"
    _counters(exporting, compiled=1, loaded=1, invoked=12)
    _counters(eager, compiled=0, loaded=0, invoked=0)

    promoter.record(kept_flag=True, workspace=str(exporting))
    promoter.record(kept_flag=True, workspace=str(eager))
    assert promoter.shared_state.export_landed is True
    assert _landing(tmp_path, export_landed=True) is False


@pytest.mark.parametrize("action", _SEARCH)
def test_search_is_denied_again_once_the_accepted_module_stops_exporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """Search is no more useful on an eager best than it was before the first export. The
    refusal normally stops this arising; the gate closing is the fail-closed backstop."""
    _declare(tmp_path, monkeypatch)
    denial = _Dispatcher(exports=False, export_landed=True).denial(action)
    assert denial is not None
    assert denial.rule == "export_gate"
    assert "no longer exports" in denial.args[0] and "counters are missing" in denial.args[0]


class _Graded:
    def __init__(self, comparable: bool = True, veto_reason: str = "") -> None:
        self.comparable = comparable
        self.veto_reason = veto_reason


class _Throughput:
    """Stand-in for _ThroughputGrade: the graded comparison and its delta travel together."""

    def __init__(self, graded: object = None, delta_pct: float | None = -12.0) -> None:
        self.graded = graded if graded is not None else _Graded()
        self.delta_pct = delta_pct


def _landing(tmp_path: Path, **over: object) -> bool:
    from hyperloom.orchestrator.actions.executors._integrate_perf_gate import _export_landing_keep

    state = SharedState()
    state.export_landed = bool(over.pop("export_landed", False))
    # The waiver reads the run's counters, like the gate does; a workspace with
    # invoked > 0 is what "this benchmark really exported" looks like.
    workspace = tmp_path / "run_0"
    if over.pop("invoked", 1):
        _counters(workspace, compiled=1, loaded=1, invoked=12)
    kwargs: dict[str, object] = {
        "shared_state": state,
        "throughput": _Throughput(),
        "accuracy_clear": True,
        "workspace": str(workspace),
    }
    # The old per-field spellings, so a case that varies one still reads naturally.
    if "graded" in over or "delta_pct" in over:
        kwargs["throughput"] = _Throughput(over.pop("graded", None), over.pop("delta_pct", -12.0))
    if "accuracy_pass" in over or "acc_block" in over:
        kwargs["accuracy_clear"] = over.pop("accuracy_pass", True) is True and not over.pop("acc_block", False)
    kwargs.update(over)
    return _export_landing_keep(**kwargs)


def _exporting_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, exports: bool = True) -> None:
    body = (
        "import torch\n\n\ndef hyperloom_optimize(model):\n    return torch._inductor.aoti_load_package('m.pt2')\n"
        if exports
        else "def hyperloom_optimize(model):\n    return model\n"
    )
    module = tmp_path / "hyperloom_optimize.py"
    module.write_text(body, encoding="utf-8")
    (tmp_path / "optimization_scope.json").write_text(
        json.dumps({"schema_version": 2, "file": str(module), "aoti": {"enabled": True}}), encoding="utf-8"
    )
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))


def test_a_correct_but_slower_first_export_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise the gate deadlocks: the export loses on throughput, reverts, and never opens."""
    _exporting_module(tmp_path, monkeypatch)
    assert _landing(tmp_path) is True


def test_only_the_first_export_gets_the_carve_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _exporting_module(tmp_path, monkeypatch)
    assert _landing(tmp_path, export_landed=True) is False


def test_a_module_that_does_not_export_gets_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _exporting_module(tmp_path, monkeypatch, exports=False)
    assert _landing(tmp_path, invoked=0) is False


def test_a_module_that_names_the_entry_points_but_never_ran_them_gets_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caught fallback to eager parses exactly like a real export.

    Deciding on source here would waive throughput every round: the KEEP lands, the
    counters still read invoked=0 so the gate stays shut, and the next round waives a
    slower eager module again.
    """
    _exporting_module(tmp_path, monkeypatch, exports=True)
    workspace = tmp_path / "run_0"
    _counters(workspace, compiled=1, loaded=1, invoked=0)
    assert _landing(tmp_path, invoked=0) is False


def test_counters_that_never_appeared_waive_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _exporting_module(tmp_path, monkeypatch)
    assert _landing(tmp_path, invoked=0, workspace="") is False


@pytest.mark.parametrize(
    "over",
    [
        {"accuracy_pass": None},
        {"accuracy_pass": False},
        {"acc_block": True},
        {"graded": _Graded(comparable=False)},
        {"graded": _Graded(veto_reason="latency budget")},
        {"delta_pct": None},
    ],
    ids=["no-verdict", "failed", "blocked", "incomparable", "vetoed", "unmeasured"],
)
def test_throughput_is_the_only_thing_waived(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, over: dict) -> None:
    """A declared latency or power budget is a constraint, not a preference."""
    _exporting_module(tmp_path, monkeypatch)
    assert _landing(tmp_path, **over) is False


def test_a_workload_declaring_no_export_gets_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch, enabled=False)
    assert _landing(tmp_path) is False


# ── wiring ───────────────────────────────────────────────────────────────────
# These go through the *callers*. The tests above bind each helper off its class and
# call it on a stub, so they pass even if nothing ever invokes it -- which is how a
# gate that denied only `explore` survived review. Exercise the admission path itself.


def _admission(action: str, *, exports: bool = False, baseline_tput: float = 100.0):
    """The denial the dispatcher's real admission path returns for ``action``."""
    from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator

    stub = _Dispatcher(exports=exports)
    stub.shared_state.baseline_tput = baseline_tput
    stub._export_gate_denial_for_action = lambda a: DispatcherCollaborator._export_gate_denial_for_action(stub, a)
    stub._export_gate_denied = DispatcherCollaborator._export_gate_denied
    return DispatcherCollaborator._sequence_denial_for_action(stub, action)


@pytest.mark.parametrize("action", _SEARCH)
def test_the_admission_path_actually_consults_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """sweep/profile/roofline are not sequence actions, so a gate reached only through
    that set would deny `explore` alone and silently admit the other three."""
    _declare(tmp_path, monkeypatch)
    denial = _admission(action)
    assert denial is not None, f"{action} reached the admission path without being gated"
    assert denial.rule == "export_gate"


def test_the_admission_path_still_enforces_baseline_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding the gate must not displace the ordering rule that was already there."""
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    denial = _admission("explore", baseline_tput=0.0)
    assert denial is not None and denial.rule == "execution_order"


@pytest.mark.parametrize("action", _SEARCH)
def test_the_admission_path_admits_once_the_export_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    _declare(tmp_path, monkeypatch)
    assert _admission(action, exports=True) is None


def test_the_promotion_path_actually_records_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_record_export_gate is only useful if _promote_integrate_patch calls it."""
    import inspect

    from hyperloom.orchestrator.loop._integrate_promotion import IntegratePromotionMixin

    body = inspect.getsource(IntegratePromotionMixin._promote_integrate_patch) + inspect.getsource(
        IntegratePromotionMixin._record_keep_bookkeeping
    )
    assert "_record_keep_bookkeeping(" in body, "the promotion path no longer does its KEEP bookkeeping"
    assert "_record_export_gate(" in body, "the promotion path no longer records the export gate"
    assert "_record_inert_source_layer(" in body, "the promotion path no longer records inert source"


def _attempt_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """The directories one integrate attempt leaves under its task root.

    ``output_root`` is ``<session>/runs/integrate_patch/<task_id>/`` and holds a slot per
    leg. Only the first is the graded candidate.
    """
    root = tmp_path / "integrate_patch" / "task-7"
    candidate = root / "variant_00_integrate-patch-abcd1234" / "suite_run_0"
    parity = root / "variant_00_integrate-patch-abcd1234-parity" / "suite_run_0"
    for path in (candidate, parity):
        path.mkdir(parents=True, exist_ok=True)
    return root, candidate, parity


def test_the_parity_leg_does_not_open_the_gate_for_the_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The switch-off leg runs the same patch with the switches unset, so its counters
    describe the configuration the candidate is being measured against -- not the
    candidate. Summed from the task root they would open the gate anyway."""
    _declare(tmp_path, monkeypatch)
    _root, candidate, parity = _attempt_tree(tmp_path)
    _counters(parity, compiled=1, loaded=1, invoked=9)

    promoter = _Promoter()
    assert promoter.record(kept_flag=True, workspace=str(candidate)) is True
    assert promoter.shared_state.export_landed is False
    assert promoter.shared_state.export_attempts == 1


def test_an_earlier_attempt_under_the_same_task_does_not_open_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``output_root`` is keyed by task id and created with ``exist_ok``, so a retried
    attempt inherits whatever the reverted one left behind."""
    _declare(tmp_path, monkeypatch)
    root, candidate, _parity = _attempt_tree(tmp_path)
    _counters(root / "variant_00_integrate-patch-00000000" / "suite_run_0", compiled=1, loaded=1, invoked=4)

    promoter = _Promoter()
    assert promoter.record(kept_flag=True, workspace=str(candidate)) is True
    assert promoter.shared_state.export_landed is False


def test_the_throughput_waiver_reads_the_candidate_leg_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A slower module that never exported must not have its throughput waived on a
    sibling leg's counters."""
    _declare(tmp_path, monkeypatch)
    _root, candidate, parity = _attempt_tree(tmp_path)
    _counters(parity, compiled=1, loaded=1, invoked=9)

    assert _landing(tmp_path, invoked=0, workspace=str(candidate)) is False
    _counters(candidate, compiled=1, loaded=1, invoked=12)
    assert _landing(tmp_path, invoked=0, workspace=str(candidate)) is True


def test_both_readers_are_handed_the_graded_leg_not_the_task_root() -> None:
    """``output_root`` spans every leg; only ``bench_result["workspace"]`` is the
    candidate's own run directory."""
    import inspect

    from hyperloom.orchestrator.actions.executors import _integrate_perf_gate
    from hyperloom.orchestrator.loop._integrate_promotion import IntegratePromotionMixin

    weigh = inspect.getsource(_integrate_perf_gate._weigh_gate)
    assert 'workspace=str(bench_result.get("workspace") or "")' in weigh
    assert "workspace=str(attempt.output_root" not in weigh

    promote = inspect.getsource(IntegratePromotionMixin._record_keep_bookkeeping)
    assert 'workspace=str(measurement.get("workspace") or "")' in promote
    assert 'workspace=str(result.get("workspace")' not in promote


def test_the_two_flags_are_read_back_independently() -> None:
    """They answer different questions, so a resume must not collapse them into one."""
    state = SharedState.from_dict({"export_landed": True, "current_best_exports": False})
    assert (state.export_landed, state.current_best_exports) == (True, False)


def _regression(tmp_path: Path, *, landed: bool = True, workspace: str | None = None) -> str:
    from hyperloom.orchestrator.actions.executors._integrate_perf_gate import _export_regression_refusal

    state = SharedState()
    state.export_landed = landed
    return _export_regression_refusal(state, workspace if workspace is not None else str(tmp_path / "run_0"))


def test_a_candidate_that_stopped_exporting_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An eager module is not a slower deliverable, it is not a deliverable. Refused on that
    ground alone, so the accepted export keeps the slot however the throughput compares."""
    _declare(tmp_path, monkeypatch)
    _counters(tmp_path / "run_0", compiled=0, loaded=0, invoked=0)
    assert "not a deliverable" in _regression(tmp_path)


def test_a_candidate_that_still_exports_is_not_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch)
    _counters(tmp_path / "run_0", compiled=1, loaded=1, invoked=8)
    assert _regression(tmp_path) == ""


def test_a_candidate_that_never_exported_is_refused_from_the_first_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``current_best`` is what the run reports and ships, so an eager candidate must not
    occupy it even before an export exists -- every figure afterwards would describe a path
    the deployment never takes."""
    _declare(tmp_path, monkeypatch)
    _counters(tmp_path / "run_0", compiled=0, loaded=0, invoked=0)
    assert "did not export" in _regression(tmp_path, landed=False)


def test_absent_counters_refuse_once_an_export_has_landed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Elsewhere absent counters mean "cannot judge" rather than a verdict, because before
    the first landing they cannot be told from a harness that was never instrumented. The
    landing KEEP removes that ambiguity -- it is proof the shim runs -- so counters going
    missing afterwards is a regression, and the conservative reading is to refuse."""
    _declare(tmp_path, monkeypatch)
    assert "harness regression" in _regression(tmp_path)
    assert "no workspace" in _regression(tmp_path, workspace="")


def test_absent_counters_still_do_not_refuse_before_the_first_landing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Here they really are ambiguous, and refusing would revert every KEEP all campaign."""
    _declare(tmp_path, monkeypatch)
    assert _regression(tmp_path, landed=False) == ""
    assert _regression(tmp_path, landed=False, workspace="") == ""


def test_refusing_on_absent_counters_cannot_stall_the_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The accepted export keeps the slot, so a refusal costs a round and nothing else --
    which is what makes the strict reading affordable here but not before the landing."""
    _declare(tmp_path, monkeypatch)
    _counters(tmp_path / "run_0", compiled=1, loaded=1, invoked=5)
    assert _regression(tmp_path) == ""


def test_a_workload_declaring_no_export_is_never_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch, enabled=False)
    _counters(tmp_path / "run_0", compiled=0, loaded=0, invoked=0)
    assert _regression(tmp_path) == ""


def test_the_refusal_is_wired_into_the_gate_and_vetoes_everything() -> None:
    """A correctness fix that gives up the export has not fixed what the run exists to
    produce, so the refusal has to land after the fix and landing carve-outs, not before."""
    import inspect

    from hyperloom.orchestrator.actions.executors import _integrate_perf_gate

    body = inspect.getsource(_integrate_perf_gate._weigh_gate)
    weigh_export = inspect.getsource(_integrate_perf_gate._weigh_export)
    assert "_weigh_export(" in body, "the gate no longer weighs the export"
    assert "_export_regression_refusal(" in weigh_export, "the gate no longer consults the export regression refusal"
    assert "gate_pass = correctness_fix = export_landing = False" in body
    assert weigh_export.index("_export_landing_keep(") < weigh_export.index("_export_regression_refusal(")

    perf = inspect.getsource(_integrate_perf_gate.PerfGateMixin._gate_perf)
    assert "verdict.export_regression" in perf, "the revert no longer says why the patch was refused"


def test_the_refusal_counts_the_round_as_an_export_attempt() -> None:
    """``export_attempts`` describes rounds that went by without an export. The refusal
    turns exactly those into REVERTs, which never reach ``_record_export_gate``, so the
    perf gate has to count them or the figure reads zero for the run it describes."""
    import inspect

    from hyperloom.orchestrator.actions.executors import _integrate_perf_gate

    body = inspect.getsource(_integrate_perf_gate.PerfGateMixin._gate_perf)
    assert "verdict.export_regression" in body
    assert "export_attempts += 1" in body
