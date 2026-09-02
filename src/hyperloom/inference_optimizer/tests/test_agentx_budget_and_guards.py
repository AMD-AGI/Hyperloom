# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX budget profile, search-scope collapse, and the session-level guards.

An AgentX round costs orders of magnitude more wall-clock than a synthetic one,
so budgets sized for the latter reap healthy variants -- and an overtime kill is
terminal (the variant skips the KEEP ladder entirely). These tests pin the
widened budgets, the scope reductions that make the cost affordable, and the two
guards that stop a session from silently measuring something other than what it
reports: the bypass-backend combination, and a resume that changes benchmark
mode or crosses an AgentX measurement epoch.

Every case asserts the synthetic path is untouched.
"""

from __future__ import annotations

import argparse

import pytest

from hyperloom.inference_optimizer.cli import (
    _apply_agentx_budget_profile,
    _preflight_agentx_backend,
)
from hyperloom.inference_optimizer.cli.bootstrap import (
    AGENTX_MEASUREMENT_EPOCH,
    agentx_state_is_stale,
)


def _budget_args(**over) -> argparse.Namespace:
    base = dict(
        max_hours=2.0,
        explore_overtime_kill_ratio=2.0,
        conc_sweep_timeout_sec=1800,
        conc_sweep_total_budget_sec=9000,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _off(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)


def _on(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")


# --- budget profile -----------------------------------------------------------


def test_budget_profile_is_noop_without_agentx(monkeypatch):
    _off(monkeypatch)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert vars(args) == vars(_budget_args())


def test_budget_profile_widens_conc_sweep_defaults_under_agentx(monkeypatch):
    _on(monkeypatch)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert args.conc_sweep_timeout_sec > 1800
    assert args.conc_sweep_total_budget_sec > 9000


def test_budget_profile_leaves_the_kill_ratio_alone(monkeypatch):
    """A duration-based replay compresses runtime spread, it does not widen it.

    The measurement window is fixed, so only warmup scales with how slow a
    config is: a variant with 3x slower warmup still lands at ~1.8x the baseline
    total (measured: 46 min warmup + 60 min window + 5 min setup). The stock
    2.0x guard is not the thing that needed loosening -- the per-variant hard
    cap was, and raising this instead would invert the two.
    """
    _on(monkeypatch)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert args.explore_overtime_kill_ratio == 2.0


def test_hard_cap_stays_above_the_soft_kill_at_agentx_baselines(monkeypatch):
    """The layering `_compute_explore_variant_timeout` documents must hold.

    At the measured ~111 min baseline the stock 4h ceiling clamps the hard cap
    to 240 min while the soft kill sits at 222 min -- barely intact -- and a
    longer baseline inverts it, so the generic timeout fires and the round loses
    its KILLED_OVERTIME diagnosis. The AgentX ceiling keeps the ordering.
    """
    from hyperloom.orchestrator.actions.executors.explore import (
        AGENTX_EXPLORE_TIMEOUT_CEILING_SEC,
        DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC,
        _compute_explore_variant_timeout,
    )

    measured_baseline = 111 * 60  # the E4 round
    for baseline in (measured_baseline, 2 * 60 * 60):
        soft_kill = baseline * 2.0
        stock = _compute_explore_variant_timeout(baseline, 2.0, ceiling_sec=DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC)
        agentx = _compute_explore_variant_timeout(baseline, 2.0, ceiling_sec=AGENTX_EXPLORE_TIMEOUT_CEILING_SEC)
        assert agentx > soft_kill, f"layering inverted at baseline={baseline}"
        assert agentx >= stock


def test_budget_profile_never_touches_max_hours(monkeypatch):
    """``--max-hours`` is the operator's contract with the scheduler.

    Silently extending it gets the job killed from outside the process, which
    is far harder to diagnose than simply running out of budget.
    """
    _on(monkeypatch)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert args.max_hours == 2.0


def test_budget_profile_preserves_operator_values(monkeypatch):
    """A value the operator typed is left exactly as typed."""
    _on(monkeypatch)
    args = _budget_args(
        explore_overtime_kill_ratio=1.5,
        conc_sweep_timeout_sec=600,
        conc_sweep_total_budget_sec=1200,
    )
    _apply_agentx_budget_profile(args)
    assert args.explore_overtime_kill_ratio == 1.5
    assert args.conc_sweep_timeout_sec == 600
    assert args.conc_sweep_total_budget_sec == 1200


# --- explicit-flag detection --------------------------------------------------
def test_bypass_guard_allows_magpie(monkeypatch):
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    _preflight_agentx_backend(argparse.Namespace())  # must not raise


def test_bypass_guard_is_inert_without_agentx(monkeypatch):
    _off(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    _preflight_agentx_backend(argparse.Namespace())  # must not raise


def test_bypass_guard_rejects_the_silent_combination(monkeypatch):
    """AgentX + bypass runs synthetic work and labels it AgentX."""
    _on(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    with pytest.raises(SystemExit) as ei:
        _preflight_agentx_backend(argparse.Namespace())
    assert ei.value.code == 2


def test_guard_rejects_agentx_with_a_scriptable_framework(monkeypatch):
    """The other way for the switch to no-op while every gate still fires.

    ``apply_agentx_switch`` returns early for a scriptable framework, but
    ``agentx_enabled()`` is a bare env read -- so eval is disabled, the conc
    sweep is turned off, the grid is collapsed, budgets are widened and the
    state is stamped ``benchmark_mode="agentx"`` over a workload that never
    touched a trace.
    """
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    with pytest.raises(SystemExit) as ei:
        _preflight_agentx_backend(argparse.Namespace(framework="xdit"))
    assert ei.value.code == 2


def test_guard_allows_agentx_with_a_serving_framework(monkeypatch):
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    for fw in ("vllm", "sglang"):
        _preflight_agentx_backend(argparse.Namespace(framework=fw))  # must not raise


def test_scriptable_guard_is_inert_without_agentx(monkeypatch):
    """A scriptable run on its own is perfectly normal."""
    _off(monkeypatch)
    _preflight_agentx_backend(argparse.Namespace(framework="xdit"))  # must not raise


# --- resume staleness ---------------------------------------------------------


class _St:
    def __init__(self, mode="", epoch=0):
        self.benchmark_mode = mode
        self.agentx_epoch = epoch


def test_resume_accepts_matching_agentx_state(monkeypatch):
    _on(monkeypatch)
    assert agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH)) == ""


def test_resume_accepts_matching_synthetic_state(monkeypatch):
    _off(monkeypatch)
    assert agentx_state_is_stale(_St("synthetic", 0)) == ""


def test_resume_rejects_mode_switch(monkeypatch):
    """The KEEP ledger keys on server args alone, so the rows would collide."""
    _on(monkeypatch)
    assert "benchmark_mode" in agentx_state_is_stale(_St("synthetic", 0))
    _off(monkeypatch)
    assert "benchmark_mode" in agentx_state_is_stale(_St("agentx", 1))


def test_resume_rejects_stale_agentx_epoch(monkeypatch):
    """Same knobs, different workload: the old numbers cannot anchor."""
    _on(monkeypatch)
    reason = agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH - 1))
    assert "epoch" in reason


def test_resume_tolerates_sessions_predating_the_field(monkeypatch):
    """An empty mode means "not asserted", not "mismatch"."""
    _off(monkeypatch)
    assert agentx_state_is_stale(_St("", 0)) == ""


# --- submission verdict gate --------------------------------------------------


def _measurement(**over):
    # Serving shape: positive throughput plus at least one completed request.
    base = {"output_throughput": 100.0, "completed_requests": 42}
    base.update(over)
    return base


def _valid(result):
    from hyperloom.orchestrator.actions.executors.benchmark_result import (
        is_valid_measurement,
    )

    return is_valid_measurement(result)


def test_verdict_gate_rejects_a_failed_submission(monkeypatch):
    """A scenario-rejected run is not comparable and must not reach KEEP."""
    _on(monkeypatch)
    assert _valid(_measurement(submission_valid=False)) is False


def test_verdict_gate_rejects_an_unknown_verdict(monkeypatch):
    """None means no scenario, or an aiperf too old to stamp one.

    ``map_aiperf`` writes the key unconditionally, so an unknown verdict still
    arrives as a present key. Treating unknown as valid is exactly how an
    incomparable run reaches the leaderboard-comparable set.
    """
    _on(monkeypatch)
    assert _valid(_measurement(submission_valid=None)) is False


def test_verdict_gate_accepts_a_valid_submission(monkeypatch):
    _on(monkeypatch)
    assert _valid(_measurement(submission_valid=True)) is True


def test_verdict_gate_is_inert_on_the_synthetic_path(monkeypatch):
    """``is_valid_measurement`` is hot for every synthetic measurement too.

    The synthetic harness runs an InferenceX revision this repo re-pins from
    time to time. Were a future upstream to stamp the key into a synthetic
    ``inferencex_result.json``, a presence-only check would silently invalidate
    every synthetic measurement session-wide while throughput still looked
    healthy. Gating on the mode removes that coupling.
    """
    _off(monkeypatch)
    assert _valid(_measurement(submission_valid=False)) is True
    assert _valid(_measurement(submission_valid=None)) is True


def test_verdict_gate_spares_scriptable_runs_under_agentx(monkeypatch):
    """A scriptable framework skips the aiperf switch entirely.

    ``apply_agentx_switch`` returns early for scriptable frameworks, so their
    results legitimately carry no verdict even with AgentX on. Keying on
    absence rather than presence would reap them.
    """
    _on(monkeypatch)
    assert _valid(_measurement()) is True


# --- inner Magpie timeout follows the AgentX cap ------------------------------


def test_agentx_switch_raises_the_inner_magpie_timeout(monkeypatch):
    """The flat Magpie ``timeout_seconds`` must follow the raised AgentX cap.

    The flat cap covers server boot + warmup + the measurement window + export
    as one deadline. At the model's native context AgentX's boot+warmup alone
    overruns the synthetic 7200s default, so the benchmark is SIGKILLed before
    aiperf writes its result -- a 0-tput baseline that fails the session. The
    switch lifts the inner cap to the same budget the outer subprocess timeout
    uses, so the two layers stay consistent.
    """
    _on(monkeypatch)
    monkeypatch.setenv("AGENTX_BASELINE_TIMEOUT_SEC", "25200")
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )
    from hyperloom.orchestrator.actions.executors.baseline import (
        agentx_baseline_timeout_sec,
    )

    bench = {"framework": "vllm", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    assert bench["timeout_seconds"] == agentx_baseline_timeout_sec()
    assert bench["timeout_seconds"] > 7200
    assert bench["benchmark_script"] == "aiperf_client.sh"


def test_agentx_switch_leaves_the_inner_timeout_alone_without_agentx(monkeypatch):
    """The default (synthetic) cap must be untouched when AgentX is off."""
    _off(monkeypatch)
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )

    bench = {"framework": "vllm", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    assert bench["timeout_seconds"] == 7200
    assert "benchmark_script" not in bench


def test_agentx_switch_skips_scriptable_inner_timeout(monkeypatch):
    """A scriptable framework returns early, so its cap is never rewritten."""
    _on(monkeypatch)
    monkeypatch.setenv("AGENTX_BASELINE_TIMEOUT_SEC", "25200")
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )

    bench = {"framework": "xdit", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    assert bench["timeout_seconds"] == 7200


# --- the client's warmup bound must be the SCALED grace, not the raw one ------


def _switched(monkeypatch, **env):
    """Run the AgentX switch over a vllm bench and hand back its envs."""
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    bench = {"framework": "vllm", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    return bench.get("envs", {})


def test_the_client_is_handed_the_conc_scaled_grace(monkeypatch):
    """One number, two layers.

    ``aiperf_client.sh`` reads AGENTX_WARMUP_GRACE_PERIOD and passes it to
    aiperf as ``--warmup-grace-period`` -- that is what actually stops the
    warmup. This process scales the same knob by CONC to size the subprocess
    cap. Forwarding the operator's RAW value bounds the client below what the
    cap pays for: measured on a Kimi-K3 conc=32 round, a 14400s cap against a
    3600s client bound would have cut warmup at 106 of 354 requests, and the
    round would then report a prefix-reuse figure taken before the cache held
    anything.
    """
    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    envs = _switched(monkeypatch, AGENTX_WARMUP_GRACE_PERIOD="3600", AGENTX_WARMUP_GRACE_CONC="8", CONC="32")
    assert envs["AGENTX_WARMUP_GRACE_PERIOD"] == "14400"


def test_at_or_below_the_anchor_the_operators_grace_round_trips(monkeypatch):
    """Zero drift for every concurrency the old behaviour was validated at."""
    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    envs = _switched(monkeypatch, AGENTX_WARMUP_GRACE_PERIOD="3600", CONC="8")
    assert envs["AGENTX_WARMUP_GRACE_PERIOD"] == "3600"


def test_the_exported_grace_matches_what_the_cap_budgeted(monkeypatch):
    """The invariant itself, asserted directly rather than via two constants."""
    from hyperloom.orchestrator.actions.executors.baseline import (
        agentx_warmup_grace_sec,
    )

    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    envs = _switched(monkeypatch, AGENTX_WARMUP_GRACE_PERIOD="1800", CONC="64")
    assert envs["AGENTX_WARMUP_GRACE_PERIOD"] == str(agentx_warmup_grace_sec())


def test_nothing_is_exported_on_the_default_synthetic_path(monkeypatch):
    """AgentX off: no envs block, no grace, no benchmark_script -- untouched.

    The switch returns early before any of this runs, so the synthetic path
    cannot pick up an AgentX-shaped warmup bound.
    """
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )

    _off(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "32")
    bench = {"framework": "vllm", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    assert "envs" not in bench
    assert "benchmark_script" not in bench
    assert bench["timeout_seconds"] == 7200


def test_a_scriptable_framework_gets_no_grace_either(monkeypatch):
    """The other early return: scriptable frameworks never reach the switch."""
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )

    _on(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "32")
    bench = {"framework": "xdit", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    assert "envs" not in bench


# --- a sweep variant's warmup bound must follow ITS concurrency ----------------


def _variant_envs(monkeypatch, tmp_path, *, session_conc, variant_conc, anchor="3600", grace_conc=None):
    """Materialize one grid variant and hand back the envs it will run with."""
    import yaml

    from hyperloom.orchestrator.actions.executors._grid_runner import (
        GridVariant,
        _build_variant_yaml,
    )

    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", anchor)
    monkeypatch.setenv("CONC", str(session_conc))
    if grace_conc is None:
        monkeypatch.delenv("AGENTX_WARMUP_GRACE_CONC", raising=False)
    else:
        monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", str(grace_conc))

    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "model": "/m/x", "timeout_seconds": 7200, "envs": {}}}),
        encoding="utf-8",
    )
    variant = GridVariant(
        name="v0",
        extra_server_args="",
        extra_envs={"CONC": str(variant_conc)},
    )
    out = tmp_path / "v0"
    out.mkdir(exist_ok=True)
    cfg_path = _build_variant_yaml(base, "", variant, output_subdir=out)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return cfg["benchmark"]["envs"]


def test_a_sweep_variant_is_bounded_by_its_own_concurrency(monkeypatch, tmp_path):
    """A rung's grace follows the rung, not the concurrency the session started at.

    Warmup is ten requests per lane across CONC lanes, so a ladder rung at 128
    has to drain sixteen times the work of a session sitting at 8. Bounding it at
    the session's number cuts the warmup short and the round reports a
    prefix-reuse figure taken before the cache filled.
    """
    envs = _variant_envs(monkeypatch, tmp_path, session_conc=8, variant_conc=128, grace_conc=8)
    assert envs["CONC"] == "128"
    assert envs["AGENTX_WARMUP_GRACE_PERIOD"] == str(3600 * 128 // 8)


def test_a_lower_rung_is_not_given_the_sessions_larger_grace(monkeypatch, tmp_path):
    """The scaling only ever raises; at or below the anchor it is the identity."""
    envs = _variant_envs(monkeypatch, tmp_path, session_conc=32, variant_conc=2, grace_conc=8)
    assert envs["AGENTX_WARMUP_GRACE_PERIOD"] == "3600"


def test_the_inner_cap_moves_with_the_grace(monkeypatch, tmp_path):
    """The two bounds are derived from the same number and must not disagree.

    Raising only the client's grace makes a round wait inside a bound its own
    cap does not cover, and it is SIGKILLed mid-warmup instead.
    """
    import yaml

    from hyperloom.orchestrator.actions.executors._grid_runner import (
        GridVariant,
        _build_variant_yaml,
    )

    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "8")
    monkeypatch.setenv("CONC", "8")

    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "model": "/m/x", "timeout_seconds": 7200, "envs": {}}}),
        encoding="utf-8",
    )

    def _cap_for(conc: int) -> int:
        out = tmp_path / f"v{conc}"
        out.mkdir(exist_ok=True)
        cfg_path = _build_variant_yaml(
            base,
            "",
            GridVariant(name=f"v{conc}", extra_server_args="", extra_envs={"CONC": str(conc)}),
            output_subdir=out,
        )
        return int(yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["benchmark"]["timeout_seconds"])

    assert _cap_for(64) > _cap_for(8)


def test_the_variant_cap_prices_the_rung_it_launches(monkeypatch):
    """The budget gate has to charge what the round will actually be granted."""
    from hyperloom.orchestrator.actions.executors._grid_runner import agentx_variant_timeout_sec

    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "8")
    monkeypatch.setenv("CONC", "8")

    assert agentx_variant_timeout_sec(1800, conc=64) > agentx_variant_timeout_sec(1800, conc=8)
    assert agentx_variant_timeout_sec(1800, conc=None) == agentx_variant_timeout_sec(1800, conc=8)


def test_the_derivation_does_not_narrate_once_per_call_site(monkeypatch, caplog):
    """A ladder resolves this for every rung at several sites; one line each.

    The same arithmetic repeated tens of times per sweep buries the one line
    per rung that carries information.
    """
    from hyperloom.orchestrator.actions.executors import baseline as bl

    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "8")
    monkeypatch.setattr(bl, "_AGENTX_SAID", set())

    with caplog.at_level("INFO"):
        for _ in range(4):
            bl.agentx_baseline_timeout_sec({"CONC": "8", "AGENTX_WARMUP_GRACE_PERIOD": "3600"})
        bl.agentx_baseline_timeout_sec(
            {"CONC": "64", "AGENTX_WARMUP_GRACE_PERIOD": "3600", "AGENTX_WARMUP_GRACE_CONC": "8"}
        )

    lines = [r.getMessage() for r in caplog.records if "agentx_baseline_timeout_sec:" in r.getMessage()]
    assert len(lines) == 2, lines


def test_a_rung_that_names_no_concurrency_reads_the_session(monkeypatch):
    from hyperloom.orchestrator.actions.executors._grid_runner import GridVariant, variant_conc

    assert variant_conc(GridVariant(name="v", extra_envs={"CONC": "16"})) == 16
    assert variant_conc(GridVariant(name="v", extra_envs={})) is None
    assert variant_conc(GridVariant(name="v", extra_envs={"CONC": "nonsense"})) is None
    assert variant_conc(GridVariant(name="v", extra_envs={"CONC": "0"})) is None
    assert variant_conc(None) is None


def test_the_default_grid_never_re_derives_a_grace(monkeypatch, tmp_path):
    """AgentX off: a synthetic variant's env must carry no warmup grace at all."""
    import yaml

    from hyperloom.orchestrator.actions.executors._grid_runner import (
        GridVariant,
        _build_variant_yaml,
    )

    _off(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "8")

    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "model": "/m/x", "timeout_seconds": 7200, "envs": {}}}),
        encoding="utf-8",
    )
    out = tmp_path / "v0"
    out.mkdir(exist_ok=True)
    cfg_path = _build_variant_yaml(
        base, "", GridVariant(name="v0", extra_server_args="", extra_envs={"CONC": "128"}), output_subdir=out
    )
    envs = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["benchmark"]["envs"]
    assert "AGENTX_WARMUP_GRACE_PERIOD" not in envs
    assert envs["CONC"] == "128"
