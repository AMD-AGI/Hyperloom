# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.tools import estimate_no_run
from hyperloom.inference_optimizer.tools.estimate_no_run import main as estimate_main
from hyperloom.orchestrator.knowledge.remote_recipe.no_run import (
    DEFAULT_PARALLELISM_LABEL,
    estimate_from_sessions,
    extract_parallelism,
    identity_dimensions,
    parallelism_whatif,
    percentile,
    project_session,
    shape_key,
    sharding_whatif,
)

_CURRENT = {
    "knowledge_schema_version": 1,
    "record_kind": "hyperloom_recipe",
    "optimized_throughput": 130.0,
    "validated_e2e_gain": 30.0,
    "value": {
        "explore": {},
        "framework": {},
        "kernel": {},
        "patch_timeline": [],
    },
}


def test_percentile_interpolates() -> None:
    assert percentile([10.0, 20.0, 30.0], 50) == 20.0
    assert percentile([], 50) is None


def _shaped(
    *,
    cid: str,
    gain: float,
    optimized: float,
    tp: int,
    conc: int = 64,
    isl: int = 1024,
    osl: int = 256,
) -> dict:
    return {
        "canonical_id": cid,
        "session_id": f"{cid}-tp{tp}-{gain}",
        "knowledge": {
            "knowledge_schema_version": 1,
            "record_kind": "hyperloom_recipe",
            "optimized_throughput": optimized,
            "validated_e2e_gain": gain,
            "workload_shape": {"tp": tp, "conc": conc, "isl": isl, "osl": osl},
            "value": {"explore": {}, "framework": {}, "kernel": {}, "patch_timeline": []},
        },
    }


_MI355_SGLANG = "inference:llama:mi355x:sglang:llama:llamaforcausallm:0.5.17:bf16"
_MI300_VLLM = "inference:llama:mi300x:vllm:llama:llamaforcausallm:0.9.1:bf16"


def _sharded(
    *,
    cid: str = _MI355_SGLANG,
    gain: float,
    optimized: float = 1000.0,
    tp: int = 2,
    args: str = "",
) -> dict:
    doc = _shaped(cid=cid, gain=gain, optimized=optimized, tp=tp)
    doc["session_id"] = f"{cid}-{gain}-{args}"
    doc["knowledge"]["value"]["config"] = {"extra_server_args": args, "extra_envs": {}}
    return doc


def test_extract_parallelism_normalizes_across_frameworks() -> None:
    sglang = extract_parallelism("--kv-cache-dtype fp8_e4m3 --tp-size 1 --dp-size 2 --schedule-policy fcfs")
    vllm = extract_parallelism("--tensor-parallel-size=1 --data-parallel-size=2 --block-size 128")
    assert sglang == {"tp": "1", "dp": "2"}
    assert vllm == sglang


def test_extract_parallelism_reads_boolean_switches_and_moe_knobs() -> None:
    knobs = extract_parallelism(
        "--ep-size 4 --enable-dp-attention --dp-size 8 --enable-dp-lm-head --moe-dense-tp-size 1"
    )
    assert knobs == {
        "ep": "4",
        "dp": "8",
        "dp_attention": "on",
        "dp_lm_head": "on",
        "moe_dense_tp": "1",
    }
    assert extract_parallelism("--kv-cache-dtype fp8_e4m3") == {}
    assert extract_parallelism("") == {}


def test_project_session_labels_the_accepted_layout() -> None:
    row = project_session(_sharded(gain=23.0, args="--tp-size 1 --dp-size 2"))
    assert row["parallelism"] == {"tp": "1", "dp": "2"}
    assert row["parallelism_label"] == "dp=2 tp=1"
    assert project_session(_sharded(gain=5.0))["parallelism_label"] == DEFAULT_PARALLELISM_LABEL


def test_sharding_whatif_ranks_layouts_at_fixed_world_size() -> None:
    rows = [
        project_session(_sharded(gain=8.0, tp=8)),
        project_session(_sharded(gain=6.0, tp=8)),
        project_session(_sharded(gain=23.0, tp=8, args="--ep-size 4 --enable-dp-attention")),
    ]
    report = sharding_whatif(rows)
    assert report["sessions_with_parallelism_config"] == 1
    assert report["knobs_seen"] == {"dp_attention": 1, "ep": 1}
    scope = "llama/bf16/sglang/tp8/conc64/isl1024/osl256"
    entry = report["scopes_with_alternatives"][scope]
    assert entry["ranked_by_p50_gain"][0] == "dp_attention=on ep=4"
    assert entry["best"]["p50_validated_e2e_gain_pct"] == 23.0
    assert entry["strategies"][DEFAULT_PARALLELISM_LABEL]["sessions"] == 2
    # 23.0 against the 7.0 median of the two default runs
    assert entry["gain_pct_points_over_default"] == pytest.approx(16.0)


def test_sharding_whatif_does_not_compare_layouts_across_world_sizes() -> None:
    rows = [
        project_session(_sharded(gain=23.0, tp=2, args="--tp-size 1 --dp-size 2")),
        project_session(_sharded(gain=8.0, tp=8, args="--tp-size 1 --dp-size 2")),
    ]
    report = sharding_whatif(rows)
    assert report["scopes_with_alternatives"] == {}
    assert report["scopes_without_alternatives"] == 2


def test_estimate_flags_winners_only_visibility() -> None:
    # A record keeps the layout its session settled on and nothing it rejected,
    # so silence about a layout has to read as untried rather than as beaten.
    report = estimate_from_sessions([_sharded(gain=23.0, args="--tp-size 1 --dp-size 2")])
    assert report["sharding_whatif"]["winners_only"] is True
    assert any("not worse" in item for item in report["limitations"])


def test_identity_dimensions_split_the_canonical_id() -> None:
    dims = identity_dimensions(_MI355_SGLANG)
    assert dims["hardware"] == "mi355x"
    assert dims["framework_name"] == "sglang"
    assert dims["precision"] == "bf16"
    assert identity_dimensions("inference:too:short") == {}


def test_project_session_carries_shape_and_per_gpu_throughput() -> None:
    row = project_session(_shaped(cid=_MI355_SGLANG, gain=20.0, optimized=800.0, tp=8))
    assert (row["tp"], row["conc"], row["isl"], row["osl"]) == (8, 64, 1024, 256)
    assert row["tput_per_gpu"] == 100.0
    assert shape_key(row) == "tp8/conc64/isl1024/osl256"


def test_mixed_hardware_and_framework_pool_is_flagged() -> None:
    report = estimate_from_sessions(
        [
            _shaped(cid=_MI355_SGLANG, gain=20.0, optimized=800.0, tp=8),
            _shaped(cid=_MI300_VLLM, gain=40.0, optimized=400.0, tp=4),
        ]
    )
    joined = " ".join(report["pool_warnings"])
    assert "mixes hardware" in joined
    assert "mixes frameworks" in joined
    assert report["identity_mix"]["model"] == ["llama"]
    assert report["identity_mix"]["framework_name"] == ["sglang", "vllm"]


def test_shape_filter_scopes_the_pool_and_counts_drops() -> None:
    docs = [
        _shaped(cid=_MI355_SGLANG, gain=10.0, optimized=800.0, tp=8),
        _shaped(cid=_MI355_SGLANG, gain=50.0, optimized=200.0, tp=1),
    ]
    scoped = estimate_from_sessions(docs, shape={"tp": 8})
    assert scoped["sessions_scored"] == 1
    assert scoped["sessions_dropped_by_shape_filter"] == 1
    assert scoped["historical"]["p50_validated_e2e_gain_pct"] == 10.0
    assert not any("workload shapes" in w for w in scoped["pool_warnings"])

    pooled = estimate_from_sessions(docs)
    assert pooled["historical"]["p50_validated_e2e_gain_pct"] == 30.0
    assert any("workload shapes" in w for w in pooled["pool_warnings"])
    assert set(pooled["by_shape"]) == {
        "tp8/conc64/isl1024/osl256",
        "tp1/conc64/isl1024/osl256",
    }


def test_parallelism_whatif_measures_retention_and_projects_target() -> None:
    rows = [
        project_session(_shaped(cid=_MI355_SGLANG, gain=10.0, optimized=200.0, tp=2)),
        project_session(_shaped(cid=_MI355_SGLANG, gain=12.0, optimized=600.0, tp=8)),
    ]
    report = parallelism_whatif(rows, target_tp=4)
    assert report["observed_tp"] == [2, 8]
    assert report["replayable_across_tp"] is False
    family = report["families"]["llama/bf16/conc64/isl1024/osl256"]
    # per-GPU falls 100 -> 75 tok/s across the 4x TP step
    assert family["measured_scaling"]["per_gpu_retention"] == pytest.approx(0.75)
    projection = family["projection"]
    assert projection["source_tp"] == 2
    assert projection["ideal_flat_per_gpu"] == pytest.approx(400.0)
    assert projection["efficiency_adjusted"] == pytest.approx(300.0)

    observed = parallelism_whatif(rows, target_tp=8)["families"]["llama/bf16/conc64/isl1024/osl256"]["projection"]
    assert observed["source"] == "observed"
    assert observed["p50_optimized_throughput"] == 600.0

    assert parallelism_whatif([], target_tp=4)["projection"]["reason"]


def test_whatif_does_not_compare_tp_across_different_models() -> None:
    """A small model at TP1 must not set the scaling baseline for a large model at TP8."""
    small = "inference:qwen3-0.6b:mi355x:sglang:qwen3:qwen3forcausallm:0.5.17:bf16"
    large = "inference:llama-70b:mi355x:sglang:llama:llamaforcausallm:0.5.17:bf16"
    rows = [
        project_session(_shaped(cid=small, gain=30.0, optimized=20000.0, tp=1)),
        project_session(_shaped(cid=large, gain=14.0, optimized=1200.0, tp=8)),
    ]
    report = parallelism_whatif(rows, target_tp=4)
    assert set(report["families"]) == {
        "qwen3-0.6b/bf16/conc64/isl1024/osl256",
        "llama-70b/bf16/conc64/isl1024/osl256",
    }
    for family in report["families"].values():
        assert "measured_scaling" not in family


def test_whatif_does_not_compare_tp_across_different_isl() -> None:
    """A short-ISL TP1 run must not set the scaling baseline for a long-ISL TP8 run."""
    rows = [
        project_session(_shaped(cid=_MI355_SGLANG, gain=30.0, optimized=20000.0, tp=1, isl=1024)),
        project_session(_shaped(cid=_MI355_SGLANG, gain=14.0, optimized=1200.0, tp=8, isl=8192)),
    ]
    report = parallelism_whatif(rows, target_tp=4)
    assert set(report["families"]) == {"llama/bf16/conc64/isl1024/osl256", "llama/bf16/conc64/isl8192/osl256"}
    for family in report["families"].values():
        assert "measured_scaling" not in family
        assert family["projection"]["source"] == "scaled"
        assert family["projection"]["ideal_flat_per_gpu"] == family["projection"]["efficiency_adjusted"]


def test_whatif_reaches_the_cli(tmp_path: Path) -> None:
    docs = [
        _shaped(cid=_MI355_SGLANG, gain=10.0, optimized=200.0, tp=2),
        _shaped(cid=_MI355_SGLANG, gain=12.0, optimized=600.0, tp=8),
    ]
    src = tmp_path / "sessions.json"
    src.write_text(json.dumps(docs), encoding="utf-8")
    out = tmp_path / "report.json"
    assert estimate_main(["--input", str(src), "--target-tp", "4", "--output", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    family = report["parallelism_whatif"]["families"]["llama/bf16/conc64/isl1024/osl256"]
    assert family["projection"]["target_tp"] == 4


def _args(**overrides):
    from argparse import Namespace

    base = {"kb_store_url": "", "kb_store_token": "", "kb_store_token_file": None}
    base.update(overrides)
    return Namespace(**base)


def test_credentials_prefer_flags_then_file_then_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_STORE_URL", "https://env-host/kb")
    monkeypatch.setenv("KB_STORE_TOKEN", "env-token")

    assert estimate_no_run.resolve_credentials(_args()) == ("https://env-host/kb", "env-token")

    token_file = tmp_path / "kb_store_token"
    token_file.write_text("file-token\n", encoding="utf-8")
    url, token = estimate_no_run.resolve_credentials(
        _args(kb_store_url="https://flag-host/kb/", kb_store_token_file=token_file)
    )
    assert (url, token) == ("https://flag-host/kb", "file-token")

    _, token = estimate_no_run.resolve_credentials(_args(kb_store_token="flag-token", kb_store_token_file=token_file))
    assert token == "flag-token"


def test_missing_store_url_exits_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KB_STORE_URL", raising=False)
    monkeypatch.delenv("KB_STORE_TOKEN", raising=False)
    assert estimate_main([]) == 2


def test_report_never_echoes_the_token(tmp_path: Path) -> None:
    src = tmp_path / "sessions.json"
    src.write_text(json.dumps([_sharded(gain=10.0)]), encoding="utf-8")
    out = tmp_path / "report.json"
    assert estimate_main(["--input", str(src), "--kb-store-token", "super-secret", "--output", str(out)]) == 0
    assert "super-secret" not in out.read_text(encoding="utf-8")


def test_estimate_cli_offline(tmp_path: Path) -> None:
    payload = [
        _sharded(gain=30.0, args="--tp-size 2 --enable-dp-attention"),
        _sharded(gain=10.0, args=""),
    ]
    src = tmp_path / "sessions.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "report.json"
    rc = estimate_main(["--input", str(src), "--output", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["sessions_scored"] == 2
    assert report["historical"]["p50_validated_e2e_gain_pct"] == pytest.approx(20.0)


def test_the_estimator_reads_only_replay_fields(tmp_path: Path) -> None:
    """Execution evidence is the session pipeline's to report, not the KB's.

    Pinned as a test rather than left to review, because the cheap way to add
    a headroom forecast later is to start reading roofline arms back out of a
    replay record, which is the boundary this tool was split to respect.
    """
    src = tmp_path / "sessions.json"
    src.write_text(json.dumps([_sharded(gain=30.0, args="--tp-size 2")]), encoding="utf-8")
    out = tmp_path / "report.json"
    assert estimate_main(["--input", str(src), "--output", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    for section in ("forecast", "savings_prior"):
        assert section not in report
    assert set(report["coverage"]) == {"with_gain"}
