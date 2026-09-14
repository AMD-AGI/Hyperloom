# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for research-hint artifacts collection + rendering."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.knowledge import research_hints as rh
from hyperloom.inference_optimizer.session import session_paths


# ---- _coerce_hint ----


def test_coerce_hint_valid():
    out = rh._coerce_hint(
        {
            "what": " do x ",
            "source": " paper ",
            "domain_tags": "moe",
            "expected_impact": "+5%",
            "accuracy_risk": "low",
        }
    )
    assert out["what"] == "do x"
    assert out["source"] == "paper"
    assert out["domain_tags"] == ["moe"]
    assert out["status"] == "proposed"


def test_coerce_hint_rejects():
    assert rh._coerce_hint("x") is None
    assert rh._coerce_hint({"what": "x"}) is None
    assert rh._coerce_hint({"source": "s"}) is None


# ---- load / append ----


def test_load_hints_missing(tmp_path):
    assert rh.load_hints(tmp_path) == []


def test_append_and_load_hints(tmp_path):
    added, dropped = rh.append_hints(
        tmp_path,
        [
            {"what": "enable cudagraph", "source": "blog"},
            {"what": "no source here"},
            {"what": "enable cudagraph", "source": "blog"},
        ],
    )
    assert added == 1
    assert dropped == 1
    hints = rh.load_hints(tmp_path)
    assert len(hints) == 1
    assert session_paths.research_hints_json(tmp_path).exists()
    assert session_paths.research_hints_md(tmp_path).exists()


def test_load_hints_bad_json(tmp_path):
    p = session_paths.research_hints_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert rh.load_hints(tmp_path) == []


def test_write_hints_skeleton(tmp_path):
    rh.write_hints_skeleton(tmp_path)
    md = session_paths.research_hints_md(tmp_path)
    assert md.exists()
    assert "No proven priors" in md.read_text(encoding="utf-8")
    rh.write_hints_skeleton(tmp_path)


def test_render_md_with_hints():
    md = rh._render_md(
        [
            {
                "what": "x",
                "expected_impact": "",
                "accuracy_risk": "",
                "domain_tags": [],
                "status": "proposed",
                "source": "s",
            }
        ]
    )
    assert "## 1. x" in md
    assert "domain_tags: -" in md


# ---- competitor target ----


def test_write_competitor_target_no_source(tmp_path):
    assert rh.write_competitor_target(tmp_path, {"per_conc": [{"conc": 1}]}) is False
    assert rh.write_competitor_target(tmp_path, "x") is False


def test_write_and_load_competitor_target(tmp_path):
    ok = rh.write_competitor_target(
        tmp_path,
        {
            "gpu": "MI300",
            "model": "m",
            "framework": "sglang",
            "precision": "fp8",
            "per_conc": [{"conc": 8, "tput_per_gpu": 100.0, "source": "vendor"}],
            "notes": "n",
        },
    )
    assert ok is True
    loaded = rh.load_competitor_target(tmp_path)
    assert loaded["gpu"] == "MI300"
    assert loaded["per_conc"][0]["conc"] == 8


def test_load_competitor_target_missing(tmp_path):
    assert rh.load_competitor_target(tmp_path) is None


def test_load_competitor_target_bad(tmp_path):
    p = session_paths.competitor_target_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"per_conc": []}), encoding="utf-8")
    assert rh.load_competitor_target(tmp_path) is None


# ---- gap analysis ----


def _target():
    return {
        "per_conc": [
            {"conc": 8, "tput_per_gpu": 100.0, "tpot_ms": 10.0, "interactivity": 100.0, "source": "v"},
            {"conc": 16, "tput_per_gpu": 200.0, "tpot_ms": 20.0, "source": "v"},
        ],
    }


def test_gap_analysis_none():
    assert rh.gap_analysis(None, our_tput_per_gpu=1, our_tpot_ms=1) is None


def test_gap_analysis_throughput():
    gap = rh.gap_analysis(_target(), our_tput_per_gpu=50.0, our_tpot_ms=10.0, conc=8)
    assert gap["throughput_gap_pct"] == 50.0
    assert gap["tpot_ratio"] == 1.0
    assert gap["primary_gap"] == "throughput"
    assert gap["target_conc"] == 8.0


def test_gap_analysis_latency_primary():
    gap = rh.gap_analysis(_target(), our_tput_per_gpu=95.0, our_tpot_ms=40.0, conc=8)
    assert gap["primary_gap"] == "latency"


def test_match_target_row_nearest():
    gap = rh.gap_analysis(_target(), our_tput_per_gpu=100.0, our_tpot_ms=10.0, conc=10)
    assert gap["target_conc"] == 8.0


def test_match_target_row_no_conc():
    gap = rh.gap_analysis(_target(), our_tput_per_gpu=100.0, our_tpot_ms=10.0)
    assert gap["target_conc"] == 16.0


# ---- summaries ----


def test_full_gap_summary_empty():
    assert rh.full_gap_summary(None) == ""


def test_full_gap_summary_with_priority():
    gap = {
        "throughput_gap_pct": 10.0,
        "tpot_ratio": 1.5,
        "interactivity_gap_pct": 5.0,
        "source": "v",
    }
    out = rh.full_gap_summary(gap)
    assert "TPOT ratio" in out
    assert "Priority" in out


def test_agentx_summary_keeps_missing_axes_independent():
    gap = {"benchmark_mode": "agentx", "throughput_gap_pct": 50.0, "interactivity_gap_pct": None}
    text = rh.full_gap_summary(gap)
    assert "total throughput/GPU gap vs target: +50.0%" in text
    assert "E2E normalized interactivity P90: unavailable" in text


# ---- variant matching ----


def test_match_variants_to_priors():
    hints = [{"what": "enable cudagraph decode", "domain_tags": ["decode"]}]
    variants = [
        {"name": "v1", "description": "use cudagraph for decode"},
        {"name": "v2", "description": "unrelated thing zzz"},
    ]
    out = rh.match_variants_to_priors(variants, hints, primary_gap="latency")
    assert "v1" in out
    assert out["v1"]["latency_aligned"] is True
    assert "v2" not in out


@pytest.mark.parametrize("primary_gap", ["latency", "throughput", None])
def test_only_latency_gap_automatically_aligns_decode_keywords(primary_gap):
    variants = [{"name": "decode_variant", "description": "cudagraph decode path"}]
    matches = rh.match_variants_to_priors(variants, [], primary_gap=primary_gap)
    summary = rh.priors_match_summary(variants, [], primary_gap=primary_gap)
    if primary_gap == "latency":
        assert matches == {"decode_variant": {"hints": [], "latency_aligned": True}}
        assert "aligns-with-latency-gap" in summary
    else:
        assert matches == {}
        assert summary == ""


@pytest.mark.parametrize("primary_gap", ["latency", "throughput", None])
def test_primary_gap_does_not_replace_source_backed_hint_matching(primary_gap):
    hints = [{"what": "cudagraph decode", "domain_tags": ["decode"]}]
    variants = [{"name": "decode_variant", "description": "cudagraph decode path"}]
    matches = rh.match_variants_to_priors(variants, hints, primary_gap=primary_gap)
    assert matches["decode_variant"]["hints"] == ["cudagraph decode"]
    assert matches["decode_variant"]["latency_aligned"] is (primary_gap == "latency")


def test_priors_match_summary_empty():
    assert rh.priors_match_summary([], []) == ""


def test_priors_match_summary_rows():
    hints = [{"what": "enable cudagraph decode", "domain_tags": ["decode"]}]
    variants = [{"name": "v1", "description": "cudagraph decode path"}]
    out = rh.priors_match_summary(variants, hints, primary_gap="latency")
    assert "v1" in out


def test_summarise_for_prompt(tmp_path):
    rh.append_hints(
        tmp_path,
        [
            {"what": "do x", "source": "s", "expected_impact": "+5%", "accuracy_risk": "low"},
        ],
    )
    out = rh.summarise_for_prompt(tmp_path)
    assert "do x" in out
    assert "source=s" in out


def test_summarise_for_prompt_empty(tmp_path):
    assert rh.summarise_for_prompt(tmp_path) == ""


def test_to_num():
    assert rh._to_num("1.5") == 1.5
    assert rh._to_num(None) is None
    assert rh._to_num("x") is None


def test_tokens():
    toks = rh._tokens("Enable CUDAGraph for-decode the")
    assert "cudagraph" in toks
    assert "the" not in toks


def test_load_hints_items_not_list(tmp_path):
    p = session_paths.research_hints_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"hints": {"not": "a list"}}), encoding="utf-8")
    assert rh.load_hints(tmp_path) == []


def test_load_hints_bare_scalar(tmp_path):
    p = session_paths.research_hints_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(5), encoding="utf-8")
    assert rh.load_hints(tmp_path) == []


def test_load_hints_top_level_list(tmp_path):
    p = session_paths.research_hints_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"what": "x", "source": "s"}]), encoding="utf-8")
    out = rh.load_hints(tmp_path)
    assert len(out) == 1
    assert out[0]["what"] == "x"


def test_persist_oserror_is_soft(tmp_path, monkeypatch, caplog):
    def _boom(_path, _text):
        raise OSError("disk full")

    monkeypatch.setattr(rh._common_io, "atomic_write_text", _boom)
    with caplog.at_level("WARNING"):
        rh._persist(tmp_path, [{"what": "x", "source": "s"}])
    assert any("persist failed" in r.getMessage() for r in caplog.records)


def test_coerce_per_conc_not_dict():
    assert rh._coerce_per_conc("nope") is None
    assert rh._coerce_per_conc(None) is None


def test_coerce_per_conc_picks_fields():
    row = rh._coerce_per_conc({"source": " v ", "conc": 8, "tput_per_gpu": 100.0, "tpot_ms": None})
    assert row["source"] == "v"
    assert row["conc"] == 8
    assert row["tput_per_gpu"] == 100.0
    assert "tpot_ms" not in row


def test_write_competitor_target_per_conc_not_list(tmp_path):
    assert rh.write_competitor_target(tmp_path, {"per_conc": "oops"}) is False


def test_write_competitor_target_oserror(tmp_path, monkeypatch, caplog):
    def _boom(_path, _text):
        raise OSError("nope")

    monkeypatch.setattr(rh._common_io, "atomic_write_text", _boom)
    with caplog.at_level("WARNING"):
        ok = rh.write_competitor_target(
            tmp_path,
            {"per_conc": [{"conc": 8, "tput_per_gpu": 100.0, "source": "v"}]},
        )
    assert ok is False
    assert any("write failed" in r.getMessage() for r in caplog.records)


def test_load_competitor_target_bad_json(tmp_path, caplog):
    p = session_paths.competitor_target_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert rh.load_competitor_target(tmp_path) is None
    assert any("failed to read" in r.getMessage() for r in caplog.records)


def test_load_competitor_target_not_dict(tmp_path):
    p = session_paths.competitor_target_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert rh.load_competitor_target(tmp_path) is None


def test_load_competitor_target_per_conc_not_list(tmp_path):
    p = session_paths.competitor_target_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"per_conc": "oops"}), encoding="utf-8")
    assert rh.load_competitor_target(tmp_path) is None


def test_gap_analysis_empty_per_conc():
    assert (
        rh.gap_analysis(
            {"per_conc": []},
            our_tput_per_gpu=1.0,
            our_tpot_ms=1.0,
        )
        is None
    )


def test_match_target_row_no_rows_direct():
    assert rh._match_target_row({"per_conc": []}, conc=8) is None
    assert rh._match_target_row({}, conc=None) is None


def test_gap_analysis_latency_via_elif():
    target = {
        "per_conc": [
            {"conc": 8, "tpot_ms": 10.0, "source": "v"},
        ],
    }
    gap = rh.gap_analysis(target, our_tput_per_gpu=None, our_tpot_ms=20.0, conc=8)
    assert gap["throughput_gap_pct"] is None
    assert gap["tpot_ratio"] == 2.0
    assert gap["primary_gap"] == "latency"


def test_gap_analysis_latency_elif_ratio_not_over_one():
    target = {
        "per_conc": [
            {"conc": 8, "tpot_ms": 20.0, "source": "v"},
        ],
    }
    gap = rh.gap_analysis(target, our_tput_per_gpu=None, our_tpot_ms=10.0, conc=8)
    assert gap["tpot_ratio"] == 0.5
    assert gap["primary_gap"] == "throughput"


def test_match_variants_skips_bad_hints_and_variants():
    hints = [
        "not a dict",
        {"what": "  ", "source": "s"},
        {"what": "enable cudagraph decode", "domain_tags": ["decode"]},
    ]
    variants = [
        "not a dict",
        {"description": "cudagraph decode path"},
        {"name": "v1", "description": "cudagraph decode path"},
    ]
    out = rh.match_variants_to_priors(variants, hints)
    assert list(out.keys()) == ["v1"]
    assert "enable cudagraph decode" in out["v1"]["hints"]


def test_summarise_for_prompt_extra_more(tmp_path):
    incoming = [{"what": f"hint {i}", "source": f"s{i}"} for i in range(10)]
    rh.append_hints(tmp_path, incoming)
    out = rh.summarise_for_prompt(tmp_path, max_entries=3)
    assert "... and 7 more in research_hints.md." in out


def _agentx_target():
    return {
        "benchmark_mode": "agentx",
        "throughput_basis": "total_token_throughput_per_gpu",
        "model": "GLM-5.2",
        "framework": "sglang",
        "gpu": "b300",
        "precision": "fp4",
        "notes": "cross-system reference",
        "per_conc": [
            {
                "conc": 4,
                "decode_tp": 4,
                "benchmark_id": "2",
                "tput_per_gpu": 800.0,
                "e2e_norm_intvty_p90": 20.0,
                "source": "api",
            },
            {
                "conc": 8,
                "decode_tp": 8,
                "benchmark_id": "3",
                "tput_per_gpu": 2000.0,
                "e2e_norm_intvty_p90": 50.0,
                "source": "api",
            },
        ],
    }


def _agentx_state(**overrides):
    values = {
        "benchmark_mode": "agentx",
        "tp": 2,
        "conc": 4,
        "current_best": {"total_throughput": 800.0, "e2e_norm_intvty_p90": 5.0},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_agentx_target_roundtrip_preserves_agentx_fields(tmp_path):
    target = _agentx_target()
    assert rh.write_competitor_target(tmp_path, target)
    assert rh.load_competitor_target(tmp_path) == target


@pytest.mark.parametrize(
    "total,p10,throughput_gap,interactivity_gap,primary_gap",
    [
        (800.0, 5.0, 50.0, 75.0, "latency"),
        (400.0, 16.0, 75.0, 20.0, "throughput"),
        (800.0, 10.0, 50.0, 50.0, "latency"),
        (1600.0, 20.0, 0.0, 0.0, None),
        (2000.0, 25.0, -25.0, -25.0, None),
    ],
)
def test_state_gap_uses_accepted_state_without_raw_files(total, p10, throughput_gap, interactivity_gap, primary_gap):
    state = _agentx_state(current_best={"total_throughput": total, "e2e_norm_intvty_p90": p10})
    gap = rh.gap_for_state(_agentx_target(), state)
    assert gap["throughput_gap_pct"] == throughput_gap
    assert gap["interactivity_gap_pct"] == interactivity_gap
    assert gap["primary_gap"] == primary_gap
    assert gap["target_conc"] == 4
    assert gap["tpot_ratio"] is None


@pytest.mark.parametrize(
    "state,throughput_gap,interactivity_gap",
    [
        (_agentx_state(current_best={"e2e_norm_intvty_p90": 5.0}), None, 75.0),
        (_agentx_state(current_best={"total_throughput": 800.0}), 50.0, None),
        (_agentx_state(tp=0), None, 75.0),
        (_agentx_state(current_best={"tput": 800.0, "tpot_mean_ms": 0.001}), None, None),
    ],
)
def test_state_gap_keeps_available_axis(state, throughput_gap, interactivity_gap):
    gap = rh.gap_for_state(_agentx_target(), state)
    assert gap["throughput_gap_pct"] == throughput_gap
    assert gap["interactivity_gap_pct"] == interactivity_gap


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), 0])
@pytest.mark.parametrize("axis", ["throughput", "interactivity"])
def test_agentx_missing_axis_preserves_the_other(value, axis):
    gap = rh.gap_analysis(
        _agentx_target(),
        benchmark_mode="agentx",
        our_tput_per_gpu=value if axis == "throughput" else 400.0,
        our_tpot_ms=999.0,
        our_e2e_norm_intvty_p90=value if axis == "interactivity" else 5.0,
        conc=4,
    )
    assert gap["throughput_gap_pct"] == (None if axis == "throughput" else 50.0)
    assert gap["interactivity_gap_pct"] == (None if axis == "interactivity" else 75.0)
    assert gap["tpot_ratio"] is None


@pytest.mark.parametrize("field", ["tput_per_gpu", "e2e_norm_intvty_p90"])
def test_agentx_missing_reference_axis_preserves_the_other(field):
    target = _agentx_target()
    del target["per_conc"][0][field]
    gap = rh.gap_for_state(target, _agentx_state())
    assert gap["throughput_gap_pct"] == (None if field == "tput_per_gpu" else 50.0)
    assert gap["interactivity_gap_pct"] == (None if field == "e2e_norm_intvty_p90" else 75.0)


def test_agentx_uses_one_reference_row_for_both_axes():
    target = _agentx_target()
    target["per_conc"].append({"conc": 4, "tput_per_gpu": 400.0, "e2e_norm_intvty_p90": 40.0, "source": "api"})
    gap = rh.gap_for_state(target, _agentx_state())
    assert gap["throughput_gap_pct"] == 50.0
    assert gap["interactivity_gap_pct"] == 75.0


def test_state_gap_uses_exact_state_concurrency(caplog):
    gap = rh.gap_for_state(_agentx_target(), _agentx_state(conc=5))
    assert gap["status"] == "unavailable"
    assert gap["reason"] == "concurrency_mismatch"
    assert gap["throughput_gap_pct"] is None
    assert gap["primary_gap"] is None
    assert "requested_conc=5" in caplog.text
    assert "target_concs=[4, 8]" in caplog.text


def test_comparison_reason_contract_is_finite():
    from typing import get_args

    assert set(get_args(rh.ComparisonReason)) == {
        "target_unavailable",
        "concurrency_mismatch",
        "measurement_unavailable",
    }


def test_partial_comparison_logs_missing_axis_and_preserves_valid_axis(caplog):
    gap = rh.gap_for_state(_agentx_target(), _agentx_state(current_best={"e2e_norm_intvty_p90": 5.0}))
    assert gap["throughput_gap_pct"] is None
    assert gap["interactivity_gap_pct"] == 75.0
    assert gap["reason"] is None
    assert "throughput_gap=None" in caplog.text
    assert "interactivity_gap=75.0" in caplog.text


def test_complete_comparison_does_not_log_failure(caplog):
    assert rh.gap_for_state(_agentx_target(), _agentx_state())["status"] == "ok"
    assert not caplog.records


def test_state_gap_is_independent_of_advisory_toggle():
    enabled = rh.gap_for_state(_agentx_target(), _agentx_state(target_advisory_enabled=True))
    disabled = rh.gap_for_state(_agentx_target(), _agentx_state(target_advisory_enabled=False))
    assert disabled == enabled


def test_synthetic_state_gap_keeps_output_and_mean_tpot_contract(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    state = SimpleNamespace(
        benchmark_mode="synthetic", current_best={"tput": 200.0, "tpot_mean_ms": 40.0}, tp=2, conc=8
    )
    expected = rh.gap_analysis(_target(), our_tput_per_gpu=100.0, our_tpot_ms=40.0, conc=8)
    assert rh.gap_for_state(_target(), state) == expected
