"""Throwaway self-review: sweep the real pricing functions for disagreements."""

from __future__ import annotations

from types import SimpleNamespace

from hyperloom.orchestrator.phases import machine_state as ms

BOOT, COLD_BENCH, HOT = 350.0, 550.0, 400.0
COLD_ROUND = BOOT + COLD_BENCH


def state(usable, *, phase="PRELUDE", warm=HOT, post_ready=COLD_BENCH, marked=False, double=True):
    return SimpleNamespace(
        phase=phase,
        max_minutes=180,
        baseline_tput=1000.0,
        baseline_runtime_sec=COLD_ROUND,
        baseline_post_ready_runtime_sec=post_ready,
        baseline_warm_runtime_sec=warm,
        baseline_measure_round_dropped=marked,
        baseline_double_run=double,
        session_budget_usable_sec=lambda: usable,
    )


def gate1_need(s):
    """What the pre-ignition gate demands, mirroring _round_affordable."""
    cold = ms.measured_seconds(s, "baseline_runtime_sec")
    rnd = ms.baseline_round_cost_sec(s, double_run=bool(s.baseline_double_run))
    if cold is None or rnd is None:
        return None
    use = (ms.one_more_measurement_sec(s) or cold) if s.phase == "PRELUDE" else 0.0
    return rnd + use


def gate2_need(s, *, warmup_sec, warmup_post_ready):
    """What the post-warmup gate demands, mirroring _measure_round_affordable."""
    bench = ms.measured_seconds(s, "baseline_warm_runtime_sec")
    if bench is None:
        bench = warmup_post_ready
    if bench is None or warmup_sec is None:
        return None
    use = (ms.one_more_measurement_sec(s) or warmup_sec) if s.phase == "PRELUDE" else 0.0
    return bench + use


def main() -> int:
    bad = 0

    # 1. The band: is there a budget gate 1 admits and gate 2 then certainly refuses?
    # Gate 2 is asked after the warmup has spent a cold pass.
    band = []
    for usable in range(0, 6001, 10):
        s = state(float(usable))
        need1 = gate1_need(s)
        admitted = need1 is not None and usable >= need1
        if not admitted:
            continue
        after = state(float(usable) - COLD_ROUND)
        need2 = gate2_need(after, warmup_sec=COLD_ROUND, warmup_post_ready=COLD_BENCH)
        if need2 is not None and (usable - COLD_ROUND) < need2:
            band.append(usable)
    if band:
        bad += 1
        print(f"BAND      gate 1 admits and gate 2 refuses for usable in {band[0]}..{band[-1]}")
    else:
        print("ok        no budget is admitted before ignition only to be refused after the cold pass")

    # 2. Livelock: with the mark set, does every budget either close or admit a retry?
    stuck = []
    for usable in range(0, 8001, 10):
        s = state(float(usable), marked=True)
        closes = ms.exit_cold_anchor_prelude(s) is not None
        need1 = gate1_need(s)
        admits = need1 is not None and usable >= need1
        if not closes and not admits:
            stuck.append(usable)
    if stuck:
        bad += 1
        print(f"LIVELOCK  neither closes nor admits for usable in {stuck[0]}..{stuck[-1]}")
    else:
        print("ok        a marked session always either closes or may retry")

    # 3. A session with no split measured (multi-node / scriptable shape).
    s = state(3000.0, warm=0.0, post_ready=0.0)
    need = gate1_need(s)
    if need is None:
        bad += 1
        print("UNGATED   a round with no boot boundary is waved through")
    else:
        print(f"ok        a round with no split is priced at {need:.0f}s (whole cold rounds)")

    # 4. A first baseline must never be judged.
    first = SimpleNamespace(
        phase="PRELUDE",
        max_minutes=180,
        baseline_tput=0.0,
        baseline_runtime_sec=0.0,
        baseline_post_ready_runtime_sec=0.0,
        baseline_warm_runtime_sec=0.0,
        baseline_measure_round_dropped=False,
        baseline_double_run=True,
        session_budget_usable_sec=lambda: 60.0,
    )
    if gate1_need(first) is not None:
        bad += 1
        print("PREDICTED a first baseline was priced from measurements it cannot have")
    else:
        print("ok        a first baseline is not judged")

    # 5. Later phases ask only whether the round fits.
    later = state(2000.0, phase="EXPLORE")
    if gate1_need(later) != ms.baseline_round_cost_sec(later, double_run=True):
        bad += 1
        print("SCOPED    a re-baseline outside PRELUDE was charged for a successor")
    else:
        print("ok        a re-baseline outside PRELUDE pays only for itself")

    return bad


if __name__ == "__main__":
    raise SystemExit(main())
