# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""EXPLORE's projection router only ever removes variants a projection can judge.

What these pin is the safe direction: a variant whose levers the projection
cannot see is always benchmarked, a projection that fails sends the variant (or
the whole round) to the benchmark, and nothing the router does produces a
number that could reach KEEP/REVERT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _explore_projection as ep
from hyperloom.orchestrator.actions.executors import inferasim_bridge as ib
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant

STACK_ENVS = {"TP": 8, "CONC": 64, "ISL": 1024, "OSL": 1024, "EXTRA_VLLM_ARGS": ""}


def _materialize(variant: GridVariant, subdir: Path) -> Path:
    """Stack plus the variant, the way the round would launch it."""
    envs = dict(STACK_ENVS)
    envs["EXTRA_VLLM_ARGS"] = " ".join(filter(None, [envs["EXTRA_VLLM_ARGS"], variant.extra_server_args]))
    envs.update(variant.extra_envs or {})
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / "config.yaml"
    path.write_text(yaml.safe_dump({"benchmark": {"framework": "vllm", "model": "/models/gpt-oss-120b", "envs": envs}}))
    return path


def _metrics(tput: float, tpot: float = 10.0, calibrated: bool = False, distance: int | None = None):
    extras: dict = {"extrapolation": [], "estimator": "des"}
    if distance is not None:
        extras["anchor_regime_distance"] = distance
    return ib.ProjMetrics(
        output_throughput=tput,
        request_throughput=0.0,
        total_token_throughput=0.0,
        ttft_ms=0.0,
        tpot_ms=tpot,
        itl_ms=tpot,
        e2el_ms=0.0,
        decode_tps_per_gpu=0.0,
        memory_per_gpu_gb=0.0,
        max_concurrency=0,
        calibrated=calibrated,
        extras=extras,
    )


def _by_conc(table: dict[int, float], **kw):
    """A projection keyed on the spec's concurrency (the stack runs at 64)."""

    def project(spec, mode=None):
        return _metrics(table[spec.conc], **kw)

    return project


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setenv(ep.ENV_ENABLED, "1")
    monkeypatch.delenv(ib.ENV_MODE, raising=False)
    for name in (ep.ENV_SIMULATE_MARGIN, ep.ENV_CALIBRATED_MARGIN, ep.ENV_TOP_K):
        monkeypatch.delenv(name, raising=False)


def _route(tmp_path, variants, project, **kw):
    return ep.route_variants(variants, materialize=_materialize, output_root=tmp_path, project=project, **kw)


def test_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(ep.ENV_ENABLED)
    variants = [GridVariant(name="c8", extra_envs={"CONC": "8"})]
    kept, dropped, summary = _route(tmp_path, variants, lambda s, mode=None: pytest.fail("projected while off"))
    assert kept == variants and dropped == [] and summary is None


def test_a_decisively_slower_variant_is_dropped(tmp_path):
    variants = [
        GridVariant(name="c8", extra_envs={"CONC": "8"}),  # -80%
        GridVariant(name="c128", extra_envs={"CONC": "128"}),  # +30%
    ]
    kept, dropped, summary = _route(tmp_path, variants, _by_conc({64: 1000.0, 8: 200.0, 128: 1300.0}))
    assert [v.name for v in kept] == ["c128"]
    assert dropped == [{"name": "c8", "reason": ep.REASON_SLOWER, "detail": dropped[0]["detail"]}]
    assert "simulate margin 15%" in dropped[0]["detail"]
    assert json.loads((tmp_path / "summary.json").read_text())["dropped"] == 1
    assert summary["mode"] == "simulate"


def test_inside_the_simulate_margin_is_benchmarked(tmp_path):
    """-10% is inside the projection's own error in simulate mode."""
    variants = [GridVariant(name="c48", extra_envs={"CONC": "48"})]
    kept, dropped, _ = _route(tmp_path, variants, _by_conc({64: 1000.0, 48: 900.0}))
    assert kept == variants and dropped == []


@pytest.mark.parametrize(
    "variant",
    [
        GridVariant(name="sched", extra_server_args="--enable-chunked-prefill"),
        GridVariant(name="aiter", extra_envs={"VLLM_ROCM_USE_AITER": "0"}),
        GridVariant(name="mixed", extra_server_args="--max-num-seqs 8 --enable-chunked-prefill"),
        GridVariant(name="attn", extra_server_args="--attention-backend TRITON_ATTN"),  # simulate cannot price it
        GridVariant(name="unset", unset_envs=["VLLM_USE_V1"]),
    ],
    ids=lambda v: v.name,
)
def test_levers_the_projection_cannot_see_are_always_benchmarked(tmp_path, variant):
    kept, dropped, summary = _route(tmp_path, [variant], lambda s, mode=None: _metrics(1.0 if s.conc != 64 else 1000.0))
    assert kept == [variant] and dropped == []
    assert summary["variants"][0]["reason"] == "lever_not_projected"


def test_a_variant_that_projects_as_the_stack_is_benchmarked(tmp_path):
    variant = GridVariant(name="same", extra_envs={"CONC": "64"})
    kept, _, summary = _route(tmp_path, [variant], lambda s, mode=None: _metrics(1000.0))
    assert kept == [variant]
    assert summary["variants"][0]["reason"] == "projects_as_stack"


def test_a_failed_variant_projection_is_benchmarked(tmp_path):
    def project(spec, mode=None):
        if spec.conc != 64:
            raise ib.InferasimBridgeError("boom")
        return _metrics(1000.0)

    variant = GridVariant(name="c8", extra_envs={"CONC": "8"})
    kept, dropped, summary = _route(tmp_path, [variant], project)
    assert kept == [variant] and dropped == []
    assert summary["variants"][0]["reason"] == "projection_failed"


def test_a_failed_stack_projection_leaves_the_round_untouched(tmp_path):
    def project(spec, mode=None):
        raise ib.InferasimBridgeError("no preset")

    variants = [GridVariant(name="c8", extra_envs={"CONC": "8"})]
    kept, dropped, summary = _route(tmp_path, variants, project)
    assert kept == variants and dropped == [] and summary is None


def test_an_unknown_mode_leaves_the_round_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv(ib.ENV_MODE, "benchmrak")
    variants = [GridVariant(name="c8", extra_envs={"CONC": "8"})]
    kept, dropped, summary = _route(tmp_path, variants, lambda s, mode=None: _metrics(1.0))
    assert kept == variants and summary is None


def test_top_k_forwards_the_best_visible_variants_and_keeps_order(tmp_path, monkeypatch):
    monkeypatch.setenv(ep.ENV_TOP_K, "2")
    table = {64: 1000.0, 32: 950.0, 96: 1100.0, 128: 1200.0, 48: 980.0}
    variants = [GridVariant(name=f"c{c}", extra_envs={"CONC": str(c)}) for c in (32, 96, 128, 48)]
    sched = GridVariant(name="sched", extra_server_args="--enable-chunked-prefill")
    kept, dropped, _ = _route(tmp_path, [*variants, sched], _by_conc(table))
    # Top two by projected gain, in run order, plus the lever it cannot see.
    assert [v.name for v in kept] == ["c96", "c128", "sched"]
    assert {d["name"] for d in dropped} == {"c32", "c48"}
    assert all(d["reason"] == ep.REASON_OUTRANKED for d in dropped)


def test_top_k_zero_forwards_every_competitive_variant(tmp_path, monkeypatch):
    monkeypatch.setenv(ep.ENV_TOP_K, "0")
    table = {64: 1000.0, 32: 950.0, 96: 1100.0, 128: 1200.0, 48: 980.0}
    variants = [GridVariant(name=f"c{c}", extra_envs={"CONC": str(c)}) for c in (32, 96, 128, 48)]
    kept, dropped, _ = _route(tmp_path, variants, _by_conc(table))
    assert kept == variants and dropped == []


def test_interactivity_sessions_drop_only_what_loses_on_both_axes(tmp_path):
    """Lower concurrency trades throughput for interactivity; that is not decisively worse."""

    def project(spec, mode=None):
        return {
            64: _metrics(1000.0, tpot=10.0),
            8: _metrics(300.0, tpot=4.0),  # -70% tput, +150% interactivity
            16: _metrics(500.0, tpot=20.0),  # -50% tput, -50% interactivity
        }[spec.conc]

    variants = [GridVariant(name="c8", extra_envs={"CONC": "8"}), GridVariant(name="c16", extra_envs={"CONC": "16"})]
    kept, dropped, _ = _route(tmp_path, variants, project, grade_on_intvty=True)
    assert [v.name for v in kept] == ["c8"]
    assert [d["name"] for d in dropped] == ["c16"]


def test_calibrated_comparisons_use_the_tighter_margin(tmp_path, monkeypatch):
    monkeypatch.setenv(ib.ENV_MODE, "benchmark")
    variants = [GridVariant(name="c48", extra_envs={"CONC": "48"})]  # -10%
    project = _by_conc({64: 1000.0, 48: 900.0}, calibrated=True, distance=0)
    kept, dropped, _ = _route(tmp_path, variants, project)
    assert kept == [] and dropped[0]["reason"] == ep.REASON_SLOWER
    assert "calibrated margin 5%" in dropped[0]["detail"]


def test_an_uncalibrated_side_falls_back_to_the_simulate_margin(tmp_path, monkeypatch):
    monkeypatch.setenv(ib.ENV_MODE, "benchmark")

    def project(spec, mode=None):
        return _metrics(1000.0, calibrated=True, distance=0) if spec.conc == 64 else _metrics(900.0, distance=1)

    variants = [GridVariant(name="c48", extra_envs={"CONC": "48"})]
    kept, dropped, _ = _route(tmp_path, variants, project)
    assert kept == variants and dropped == []


def test_benchmark_mode_can_judge_an_attention_backend(tmp_path, monkeypatch):
    """Each regime gets its own anchor in benchmark mode, so the backend is priced."""
    monkeypatch.setenv(ib.ENV_MODE, "benchmark")

    def project(spec, mode=None):
        slow = "TRITON_ATTN" in spec.extra_server_args
        return _metrics(500.0 if slow else 1000.0, calibrated=True, distance=0)

    variant = GridVariant(name="attn", extra_server_args="--attention-backend TRITON_ATTN")
    kept, dropped, _ = _route(tmp_path, [variant], project)
    assert kept == [] and dropped[0]["name"] == "attn"


def test_the_router_reads_the_materialized_config_not_the_process_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CONC", "64")
    variants = [GridVariant(name="c8", extra_envs={"CONC": "8"})]
    kept, dropped, _ = _route(tmp_path, variants, _by_conc({64: 1000.0, 8: 200.0}))
    assert kept == [] and dropped[0]["name"] == "c8"


def test_the_default_shortlist_is_five(tmp_path):
    concs = [16, 24, 32, 40, 48, 56, 72]
    table = {64: 1000.0, **{c: 1000.0 + c for c in concs}}
    variants = [GridVariant(name=f"c{c}", extra_envs={"CONC": str(c)}) for c in concs]
    kept, dropped, _ = _route(tmp_path, variants, _by_conc(table))
    assert [v.name for v in kept] == ["c32", "c40", "c48", "c56", "c72"]
    assert {d["reason"] for d in dropped} == {ep.REASON_OUTRANKED}


def test_a_variant_a_past_measurement_answers_is_not_projected(tmp_path):
    variants = [GridVariant(name="c8", extra_envs={"CONC": "8"})]
    kept, dropped, summary = _route(tmp_path, variants, _by_conc({64: 1000.0, 8: 1.0}), exempt={"c8"})
    assert kept == variants and dropped == []
    assert summary["variants"][0]["reason"] == "measured_before"


def test_two_estimators_are_never_compared(tmp_path):
    def project(spec, mode=None):
        m = _metrics(1000.0 if spec.conc == 64 else 1.0)
        if spec.conc != 64:
            m.extras["estimator"] = "analytical"
        return m

    variants = [GridVariant(name="c8", extra_envs={"CONC": "8"})]
    kept, dropped, summary = _route(tmp_path, variants, project)
    assert kept == variants and dropped == []
    assert summary["variants"][0]["reason"] == "projection_unreadable"


def test_the_round_is_projected_in_the_mode_resolved_on_the_stack(tmp_path, monkeypatch):
    """Auto mode calibrates the whole round when the stack has an anchor, and says so."""
    monkeypatch.setattr(ib, "resolve_mode", lambda spec: ib.MODE_BENCHMARK)
    seen: list[tuple[str, dict]] = []

    def project(spec, mode=None):
        seen.append((mode, dict(spec.runtime)))
        return _metrics(1000.0 if spec.conc == 64 else 500.0, calibrated=True, distance=0)

    variants = [GridVariant(name="c8", extra_envs={"CONC": "8"})]
    kept, dropped, summary = _route(tmp_path, variants, project, runtime={"image_digest": "x"})
    assert {m for m, _ in seen} == {ib.MODE_BENCHMARK}
    assert all(rt == {"image_digest": "x"} for _, rt in seen)
    assert summary["mode"] == ib.MODE_BENCHMARK
    assert dropped and "calibrated margin 5%" in dropped[0]["detail"]
