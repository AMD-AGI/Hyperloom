# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for research-hint artifacts collection + rendering."""

from __future__ import annotations

import json

from hyperloom.orchestrator.knowledge import research_hints as rh
from inference_optimizer import session_paths


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
    assert rh._coerce_hint({"what": "x"}) is None  # no source
    assert rh._coerce_hint({"source": "s"}) is None  # no what


# ---- load / append ----


def test_load_hints_missing(tmp_path):
    assert rh.load_hints(tmp_path) == []


def test_append_and_load_hints(tmp_path):
    added, dropped = rh.append_hints(
        tmp_path,
        [
            {"what": "enable cudagraph", "source": "blog"},
            {"what": "no source here"},  # dropped
            {"what": "enable cudagraph", "source": "blog"},  # dup
        ],
    )
    assert added == 1
    assert dropped == 1
    hints = rh.load_hints(tmp_path)
    assert len(hints) == 1
    # artifacts written
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
    # idempotent second call
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
    # conc=10 not exact -> nearest is conc 8
    gap = rh.gap_analysis(_target(), our_tput_per_gpu=100.0, our_tpot_ms=10.0, conc=10)
    assert gap["target_conc"] == 8.0


def test_match_target_row_no_conc():
    # conc unknown -> picks highest tput row (200)
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
    assert "the" not in toks  # stopword


# ---- targeted coverage: load_hints non-list items (line 85) ----


def test_load_hints_items_not_list(tmp_path):
    # top-level dict whose "hints" key is a non-list -> items not a list -> []
    p = session_paths.research_hints_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"hints": {"not": "a list"}}), encoding="utf-8")
    assert rh.load_hints(tmp_path) == []


def test_load_hints_bare_scalar(tmp_path):
    # top-level is neither dict nor list -> items = data (an int) -> not a list -> []
    p = session_paths.research_hints_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(5), encoding="utf-8")
    assert rh.load_hints(tmp_path) == []


def test_load_hints_top_level_list(tmp_path):
    # top-level list (not wrapped in {"hints": ...}) is accepted directly
    p = session_paths.research_hints_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"what": "x", "source": "s"}]), encoding="utf-8")
    out = rh.load_hints(tmp_path)
    assert len(out) == 1
    assert out[0]["what"] == "x"


# ---- targeted coverage: _persist OSError branch (lines 164-165) ----


def test_persist_oserror_is_soft(tmp_path, monkeypatch, caplog):
    def _boom(_path, _text):
        raise OSError("disk full")

    monkeypatch.setattr(rh, "_atomic_write", _boom)
    with caplog.at_level("WARNING"):
        # should not raise; failure is logged and swallowed
        rh._persist(tmp_path, [{"what": "x", "source": "s"}])
    assert any("persist failed" in r.getMessage() for r in caplog.records)


# ---- targeted coverage: _coerce_per_conc non-dict (line 210) ----


def test_coerce_per_conc_not_dict():
    assert rh._coerce_per_conc("nope") is None
    assert rh._coerce_per_conc(None) is None


def test_coerce_per_conc_picks_fields():
    row = rh._coerce_per_conc(
        {"source": " v ", "conc": 8, "tput_per_gpu": 100.0, "tpot_ms": None}
    )
    assert row["source"] == "v"
    assert row["conc"] == 8
    assert row["tput_per_gpu"] == 100.0
    # tpot_ms is None -> not copied
    assert "tpot_ms" not in row


# ---- targeted coverage: write_competitor_target per_conc not list (line 237) ----


def test_write_competitor_target_per_conc_not_list(tmp_path):
    # per_conc is a non-list -> coerced to [] -> no sourced rows -> False
    assert rh.write_competitor_target(tmp_path, {"per_conc": "oops"}) is False


# ---- targeted coverage: write_competitor_target OSError (lines 254-256) ----


def test_write_competitor_target_oserror(tmp_path, monkeypatch, caplog):
    def _boom(_path, _text):
        raise OSError("nope")

    monkeypatch.setattr(rh, "_atomic_write", _boom)
    with caplog.at_level("WARNING"):
        ok = rh.write_competitor_target(
            tmp_path,
            {"per_conc": [{"conc": 8, "tput_per_gpu": 100.0, "source": "v"}]},
        )
    assert ok is False
    assert any("write failed" in r.getMessage() for r in caplog.records)


# ---- targeted coverage: load_competitor_target branches (274-276, 278, 281) ----


def test_load_competitor_target_bad_json(tmp_path, caplog):
    p = session_paths.competitor_target_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert rh.load_competitor_target(tmp_path) is None
    assert any("failed to read" in r.getMessage() for r in caplog.records)


def test_load_competitor_target_not_dict(tmp_path):
    # top-level JSON list -> not a dict -> None
    p = session_paths.competitor_target_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert rh.load_competitor_target(tmp_path) is None


def test_load_competitor_target_per_conc_not_list(tmp_path):
    # per_conc present but not a list -> None
    p = session_paths.competitor_target_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"per_conc": "oops"}), encoding="utf-8")
    assert rh.load_competitor_target(tmp_path) is None


# ---- targeted coverage: _match_target_row / gap_analysis no rows (310, 346) ----


def test_gap_analysis_empty_per_conc():
    # target dict with empty per_conc -> _match_target_row returns None -> gap None
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


# ---- targeted coverage: gap_analysis latency-only elif (368-369) ----


def test_gap_analysis_latency_via_elif():
    # tpot_ratio computable and > 1.0, but throughput_gap_pct is None
    # (no our_tput_per_gpu) -> hits the elif branch setting primary_gap.
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
    # tpot_ratio <= 1.0 with no throughput gap -> stays "throughput"
    target = {
        "per_conc": [
            {"conc": 8, "tpot_ms": 20.0, "source": "v"},
        ],
    }
    gap = rh.gap_analysis(target, our_tput_per_gpu=None, our_tpot_ms=10.0, conc=8)
    assert gap["tpot_ratio"] == 0.5
    assert gap["primary_gap"] == "throughput"


# ---- targeted coverage: match_variants_to_priors skip branches (526, 529, 537, 540) ----


def test_match_variants_skips_bad_hints_and_variants():
    hints = [
        "not a dict",  # line 526: skip non-dict hint
        {"what": "  ", "source": "s"},  # line 529: empty 'what' -> skip
        {"what": "enable cudagraph decode", "domain_tags": ["decode"]},
    ]
    variants = [
        "not a dict",  # line 537: skip non-dict variant
        {"description": "cudagraph decode path"},  # line 540: no name -> skip
        {"name": "v1", "description": "cudagraph decode path"},
    ]
    out = rh.match_variants_to_priors(variants, hints)
    assert list(out.keys()) == ["v1"]
    assert "enable cudagraph decode" in out["v1"]["hints"]


# ---- targeted coverage: summarise_for_prompt extra-more line (629) ----


def test_summarise_for_prompt_extra_more(tmp_path):
    incoming = [
        {"what": f"hint {i}", "source": f"s{i}"} for i in range(10)
    ]
    rh.append_hints(tmp_path, incoming)
    out = rh.summarise_for_prompt(tmp_path, max_entries=3)
    assert "... and 7 more in research_hints.md." in out
