# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the ``baseline_comparison`` package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# name_mapping
def test_name_mapping_known_display_name_passthrough():
    from hyperloom.inference_optimizer.baseline_comparison.target_analyzer import to_inferencex_name

    assert to_inferencex_name("MiniMax-M2.5") == "MiniMax-M2.5"


def test_name_mapping_strips_vendor_prefix_from_path():
    from hyperloom.inference_optimizer.baseline_comparison.target_analyzer import to_inferencex_name

    assert to_inferencex_name("/path/models/MiniMaxAI-MiniMax-M2.5") == "MiniMax-M2.5"


def test_name_mapping_case_insensitive():
    from hyperloom.inference_optimizer.baseline_comparison.target_analyzer import to_inferencex_name

    assert to_inferencex_name("/path/x/minimaxai-minimax-m2.5") == "MiniMax-M2.5"


def test_name_mapping_canonical_names_starting_with_vendor_token():
    """Canonical names beginning with a vendor-like token must not be mangled
    by the prefix strip (regression: DeepSeek / Qwen mapped to None)."""
    from hyperloom.inference_optimizer.baseline_comparison.target_analyzer import to_inferencex_name

    assert to_inferencex_name("DeepSeek-R1-0528") == "DeepSeek-R1-0528"
    assert to_inferencex_name("Qwen-3.5-397B-A17B") == "Qwen-3.5-397B-A17B"
    # HF-style paths for the same models still resolve via basename + strip.
    assert to_inferencex_name("/path/models/deepseek-ai/DeepSeek-R1-0528") == "DeepSeek-R1-0528"


def test_name_mapping_unknown_returns_none():
    from hyperloom.inference_optimizer.baseline_comparison.target_analyzer import to_inferencex_name

    assert to_inferencex_name("/path/models/MyCorp-Custom-FT-7B") is None
    assert to_inferencex_name("") is None
    assert to_inferencex_name(None) is None  # type: ignore[arg-type]


def test_known_models_list_is_nonempty():
    from hyperloom.inference_optimizer.baseline_comparison.target_analyzer import (
        KNOWN_INFERENCEX_MODELS,
    )

    assert "MiniMax-M2.5" in KNOWN_INFERENCEX_MODELS
    assert len(KNOWN_INFERENCEX_MODELS) >= 5


# inferencex_client — happy path against a local HTTP server
_SAMPLE_ROW = {
    "hardware": "b300",
    "framework": "vllm",
    "model": "minimaxm2.5",
    "precision": "fp8",
    "spec_method": "none",
    "disagg": False,
    "is_multinode": False,
    "prefill_tp": 2,
    "prefill_ep": 1,
    "decode_tp": 2,
    "decode_ep": 1,
    "num_prefill_gpu": 2,
    "num_decode_gpu": 2,
    "isl": 1024,
    "osl": 1024,
    "conc": 64,
    "image": "vllm-rocm:test",
    "metrics": {
        "tput_per_gpu": 2781.5,
        "output_tput_per_gpu": 1390.7,
        "input_tput_per_gpu": 1390.8,
        "mean_ttft": 0.094,  # seconds
        "mean_tpot": 0.022,
        "mean_e2el": 20.6,
    },
    "date": "2026-04-17",
    "run_url": "https://example/runs/x",
}


# target_analyzer — end-to-end with mocked upstream
def _make_rows() -> list[dict[str, Any]]:
    """Construct a small, realistic-shaped row set covering multiple combos."""
    base = _SAMPLE_ROW
    out: list[dict[str, Any]] = []
    for conc, tput in [(4, 412.5), (16, 1167.9), (64, 2781.5), (256, 6624.1)]:
        row = json.loads(json.dumps(base))
        row["conc"] = conc
        row["metrics"]["tput_per_gpu"] = tput
        out.append(row)
    mi = json.loads(json.dumps(base))
    mi["hardware"] = "mi300x"
    mi["conc"] = 64
    mi["metrics"]["tput_per_gpu"] = 1596.05
    out.append(mi)
    fp4 = json.loads(json.dumps(base))
    fp4["precision"] = "fp4"
    fp4["metrics"]["tput_per_gpu"] = 4066.79
    out.append(fp4)
    return out


def _patch_fetch_rows(monkeypatch, rows: list[dict[str, Any]] | None) -> None:
    """Patch the module-level ``fetch_rows`` used by ``analyze`` with a stub."""
    import hyperloom.inference_optimizer.baseline_comparison.target_analyzer as ta

    monkeypatch.setattr(ta, "fetch_rows", lambda _name: rows)


def test_analyze_happy_path_writes_files(tmp_path: Path, monkeypatch):
    _patch_fetch_rows(monkeypatch, _make_rows())

    from hyperloom.inference_optimizer.baseline_comparison import analyze

    summary = analyze(
        session_dir=tmp_path,
        model_path="/path/models/MiniMaxAI-MiniMax-M2.5",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )

    # Only fp8 / b300 / 1024-1024 rows match (mi300x + fp4 rows filtered out).
    assert summary.status == "ok"
    assert summary.reason == "ok"
    assert summary.row_count == 4
    assert summary.best is not None
    assert summary.best.tput_per_gpu == 6624.1
    assert summary.best.conc == 256
    assert summary.best.decode_tp == 2
    # Latencies converted from seconds to ms.
    assert round(summary.best.mean_tpot_ms, 1) == 22.0
    # Provenance is the live API URL, never the old ``llm_authored`` marker.
    assert "llm_authored" not in summary.source
    assert summary.source.startswith("http")

    json_path = tmp_path / "target_analysis" / "target_baseline.json"
    md_path = tmp_path / "target_analysis" / "target_analysis_report.md"
    assert json_path.exists()
    assert md_path.exists()

    on_disk = json.loads(json_path.read_text())
    assert on_disk["status"] == "ok"
    assert on_disk["query"]["model"] == "MiniMax-M2.5"
    assert on_disk["query"]["gpu"] == "b300"
    assert on_disk["best"]["tput_per_gpu"] == 6624.1
    assert on_disk["source"].startswith("http")

    md_text = md_path.read_text()
    assert "## Reference best" in md_text
    assert "6624.1" in md_text


def test_analyze_excludes_disagg_and_multinode_from_best(tmp_path: Path, monkeypatch):
    """A disaggregated / multinode row with inflated per-GPU throughput must not
    be promoted to ``best`` — only single-node colocated rows are comparable."""
    rows = _make_rows()
    disagg = json.loads(json.dumps(_SAMPLE_ROW))
    disagg["disagg"] = True
    disagg["conc"] = 512
    disagg["metrics"]["tput_per_gpu"] = 999999.0
    rows.append(disagg)
    _patch_fetch_rows(monkeypatch, rows)

    from hyperloom.inference_optimizer.baseline_comparison import analyze

    summary = analyze(
        session_dir=tmp_path,
        model_path="MiniMax-M2.5",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )
    assert summary.status == "ok"
    assert summary.best is not None
    assert summary.best.tput_per_gpu == 6624.1  # not the 999999 disagg row


def test_analyze_writes_measured_advisory_target(tmp_path: Path, monkeypatch):
    """On success, a measured ``competitor_target.json`` (source = API URL) is
    written so the EXPLORE advisory gap is driven by real InferenceX data."""
    _patch_fetch_rows(monkeypatch, _make_rows())

    from hyperloom.inference_optimizer.baseline_comparison import analyze
    from hyperloom.orchestrator.knowledge import research_hints

    summary = analyze(
        session_dir=tmp_path,
        model_path="MiniMax-M2.5",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )
    assert summary.status == "ok"

    ct = research_hints.load_competitor_target(tmp_path)
    assert ct is not None
    assert ct["per_conc"]
    assert all(row["source"] == summary.source for row in ct["per_conc"])

    gap = research_hints.gap_analysis(
        ct, our_tput_per_gpu=100.0, our_tpot_ms=50.0, conc=64,
    )
    assert gap is not None
    assert gap["source"] == summary.source


def test_analyze_mapping_miss_writes_skipped_summary(tmp_path, monkeypatch):
    _patch_fetch_rows(monkeypatch, _make_rows())

    from hyperloom.inference_optimizer.baseline_comparison import analyze

    summary = analyze(
        session_dir=tmp_path,
        model_path="/path/models/MyCorp-Custom-FT-7B",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )
    assert summary.status == "skipped"
    assert summary.reason == "model_mapping_miss"
    assert summary.best is None
    assert "mapping miss" in summary.warning.lower()

    on_disk = json.loads((tmp_path / "target_analysis" / "target_baseline.json").read_text())
    assert on_disk["status"] == "skipped"
    assert on_disk["reason"] == "model_mapping_miss"


def test_analyze_no_target_gpu_writes_marker(tmp_path):
    """``compare_against_gpu=""`` short-circuits and persists a marker JSON."""
    from hyperloom.inference_optimizer.baseline_comparison import analyze

    summary = analyze(
        session_dir=tmp_path,
        model_path="/path/models/MiniMaxAI-MiniMax-M2.5",
        compare_against_gpu="",
    )
    assert summary.status == "skipped"
    assert summary.reason == "no_target_gpu_configured"
    assert summary.best is None
    assert summary.query.gpu == ""

    on_disk = json.loads((tmp_path / "target_analysis" / "target_baseline.json").read_text())
    assert on_disk["status"] == "skipped"
    assert on_disk["reason"] == "no_target_gpu_configured"


def test_analyze_unsupported_target_gpu(tmp_path, monkeypatch):
    """A GPU InferenceX has no data for → ``no_match`` / ``unsupported_target_gpu``.
    Crucially, unknown GPUs are NOT back-filled by any LLM estimate."""
    _patch_fetch_rows(monkeypatch, _make_rows())

    from hyperloom.inference_optimizer.baseline_comparison import analyze

    summary = analyze(
        session_dir=tmp_path,
        model_path="MiniMax-M2.5",
        compare_against_gpu="mi999x",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )
    assert summary.status == "no_match"
    assert summary.reason == "unsupported_target_gpu"
    assert summary.best is None


def test_analyze_dimension_mismatch(tmp_path, monkeypatch):
    """GPU present but no row matches the run's isl/osl → ``dimension_mismatch``."""
    _patch_fetch_rows(monkeypatch, _make_rows())

    from hyperloom.inference_optimizer.baseline_comparison import analyze

    summary = analyze(
        session_dir=tmp_path,
        model_path="MiniMax-M2.5",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=2048,
        osl=2048,
    )
    assert summary.status == "no_match"
    assert summary.reason == "dimension_mismatch"
    assert summary.best is None


def test_analyze_precision_mismatch(tmp_path, monkeypatch):
    """GPU + shape exist but only at a different precision → ``precision_mismatch``.
    An fp4 run must never be compared against fp8 reference numbers."""
    # _make_rows() has b300/1024/1024 rows at fp8 (and one fp4 row); ask for a
    # precision InferenceX does not carry for this shape.
    _patch_fetch_rows(monkeypatch, _make_rows())

    from hyperloom.inference_optimizer.baseline_comparison import analyze

    summary = analyze(
        session_dir=tmp_path,
        model_path="MiniMax-M2.5",
        compare_against_gpu="b300",
        framework="vllm",
        precision="bf16",
        isl=1024,
        osl=1024,
    )
    assert summary.status == "no_match"
    assert summary.reason == "precision_mismatch"
    assert summary.best is None


def test_analyze_fetch_error(tmp_path, monkeypatch):
    """API fetch failure (``fetch_rows`` returns ``None``) → ``fetch_error``."""
    _patch_fetch_rows(monkeypatch, None)

    from hyperloom.inference_optimizer.baseline_comparison import analyze

    summary = analyze(
        session_dir=tmp_path,
        model_path="MiniMax-M2.5",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )
    assert summary.status == "no_match"
    assert summary.reason == "fetch_error"
    assert summary.best is None


def test_analyze_no_match_clears_stale_competitor_target(tmp_path, monkeypatch):
    """A pre-existing (e.g. scout-authored) competitor_target.json must be
    dropped when analyze() ends in no_match, so the advisory feed never reads a
    non-API source. Guards the 'API-measured only' invariant."""
    from hyperloom.inference_optimizer.session import session_paths
    from hyperloom.orchestrator.knowledge import research_hints

    # Seed a stale scout-authored target on disk.
    research_hints.write_competitor_target(
        tmp_path,
        {
            "gpu": "b300",
            "model": "MiniMax-M2.5",
            "per_conc": [{"conc": 64, "tput_per_gpu": 12345.0, "source": "some blog"}],
        },
    )
    assert session_paths.competitor_target_json(tmp_path).exists()

    _patch_fetch_rows(monkeypatch, None)  # force fetch_error / no_match
    from hyperloom.inference_optimizer.baseline_comparison import analyze

    summary = analyze(
        session_dir=tmp_path,
        model_path="MiniMax-M2.5",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )
    assert summary.status == "no_match"
    assert not session_paths.competitor_target_json(tmp_path).exists()
    assert research_hints.load_competitor_target(tmp_path) is None


def test_analyze_ok_write_failure_clears_stale_competitor_target(tmp_path, monkeypatch):
    """When measured advisory write fails, any pre-existing competitor_target.json
    must be removed so the EXPLORE gap block never reads a non-API source."""
    from hyperloom.inference_optimizer.session import session_paths
    from hyperloom.orchestrator.knowledge import research_hints

    research_hints.write_competitor_target(
        tmp_path,
        {
            "gpu": "b300",
            "model": "MiniMax-M2.5",
            "per_conc": [{"conc": 64, "tput_per_gpu": 12345.0, "source": "some blog"}],
        },
    )
    assert session_paths.competitor_target_json(tmp_path).exists()

    _patch_fetch_rows(monkeypatch, _make_rows())
    monkeypatch.setattr(
        "hyperloom.orchestrator.knowledge.research_hints.write_competitor_target",
        lambda *_args, **_kwargs: False,
    )

    from hyperloom.inference_optimizer.baseline_comparison import analyze

    summary = analyze(
        session_dir=tmp_path,
        model_path="MiniMax-M2.5",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )
    assert summary.status == "ok"
    assert not session_paths.competitor_target_json(tmp_path).exists()
    assert research_hints.load_competitor_target(tmp_path) is None


def test_analyze_no_inferencex_data(tmp_path, monkeypatch):
    """Empty API result (``fetch_rows`` returns ``[]``) → ``no_inferencex_data``."""
    _patch_fetch_rows(monkeypatch, [])

    from hyperloom.inference_optimizer.baseline_comparison import analyze

    summary = analyze(
        session_dir=tmp_path,
        model_path="MiniMax-M2.5",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )
    assert summary.status == "no_match"
    assert summary.reason == "no_inferencex_data"
    assert summary.best is None


# report.py renderer — _format_external_baseline_section branches on reason
def _ext_payload(**overrides) -> dict[str, Any]:
    base: dict[str, Any] = {
        "query": {
            "model": "MiniMax-M2.5",
            "gpu": "b300",
            "framework": "vllm",
            "precision": "fp8",
            "isl": 1024,
            "osl": 1024,
        },
        "fetched_at": "2026-05-12T07:00:34Z",
        "row_count": 1,
        "best": {
            "tput_per_gpu": 2781.5,
            "output_tput_per_gpu": 1390.7,
            "conc": 64,
            "decode_tp": 2,
            "mean_ttft_ms": 94.0,
            "mean_tpot_ms": 22.0,
            "mean_e2el_ms": 20600.0,
            "date": "2026-04-17",
        },
        "all_concurrencies": [],
        "status": "ok",
        "reason": "ok",
        "warning": "",
        "source": "https://inferencex.semianalysis.com/api/v1",
    }
    base.update(overrides)
    return base


def test_report_section_renders_no_target_gpu_marker():
    from hyperloom.orchestrator.actions.executors.report import (
        _format_external_baseline_section,
    )

    ext = _ext_payload(
        status="skipped",
        reason="no_target_gpu_configured",
        warning="compare_against_gpu is empty",
        best=None,
        row_count=0,
    )
    ext["query"]["gpu"] = ""
    md = "\n".join(_format_external_baseline_section(ext))
    assert "## External baseline (not requested)" in md
    assert "no_target_gpu_configured" in md
    assert "No `--compare-against-gpu`" in md
    # Must NOT render a reference-best line.
    assert "Reference best per-GPU throughput" not in md


def test_report_section_renders_ok_with_reference_best():
    from hyperloom.orchestrator.actions.executors.report import (
        _format_external_baseline_section,
    )

    md = "\n".join(_format_external_baseline_section(_ext_payload()))
    assert "## External baseline (competitor target, advisory)" in md
    assert "Reference best per-GPU throughput" in md
    assert "2781.5" in md


def test_report_section_renders_fetch_error_with_warning():
    from hyperloom.orchestrator.actions.executors.report import (
        _format_external_baseline_section,
    )

    ext = _ext_payload(
        status="fetch_error",
        reason="fetch_error",
        warning="upstream timed out",
        best=None,
        row_count=0,
    )
    md = "\n".join(_format_external_baseline_section(ext))
    assert "## External baseline (competitor target, advisory)" in md
    assert "fetch_error" in md
    assert "upstream timed out" in md
    assert "No reference best available" in md


def test_session_paths_helpers_under_target_analysis(tmp_path):
    from hyperloom.inference_optimizer.session.session_paths import (
        target_analysis_dir,
        target_analysis_report_md,
        target_baseline_json,
    )

    sd = tmp_path / "sess"
    assert target_analysis_dir(sd) == sd / "target_analysis"
    assert target_baseline_json(sd) == sd / "target_analysis" / "target_baseline.json"
    assert target_analysis_report_md(sd) == (sd / "target_analysis" / "target_analysis_report.md")
