# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kbmine import cli as estimate_no_run
from kbmine.cli import main as estimate_main
from kbmine.mine import (
    DEFAULT_PARALLELISM_LABEL,
    estimate_from_sessions,
    extract_parallelism,
    identity_dimensions,
    parallelism_whatif,
    percentile,
    project_session,
    recipe_knobs,
    shape_key,
    sharding_whatif,
)
from kbmine.recipe import knob_family


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


def _kimi_like(*, gain: float, args: str, envs: dict | None = None, conc: int = 64, learnings: dict | None = None):
    """A session shaped like the live Kimi-K3 records: rich config, no sharding."""
    doc = _shaped(cid=_MI355_SGLANG, gain=gain, optimized=800.0, tp=8, conc=conc, isl=8192, osl=1024)
    doc["knowledge"]["value"]["config"] = {"extra_server_args": args, "extra_envs": envs or {}}
    doc["knowledge"].update(learnings or {})
    return doc


def test_knobs_are_read_even_when_nothing_is_re_sharded() -> None:
    """The live failure: five Kimi-K3 sessions, rich recipes, zero sharding.

    ``sharding_whatif`` reported no config at all, which was true of the
    parallelism vocabulary and false of the recipe.
    """
    row = project_session(
        _kimi_like(
            gain=21.4,
            args="--max-running-requests 64 --kv-cache-dtype fp8_e4m3 --disable-radix-cache",
            envs={"SGLANG_AITER_K3_OPT": "1"},
        )
    )
    assert row["parallelism_label"] == DEFAULT_PARALLELISM_LABEL
    assert row["knobs"] == {
        "max_running_requests": "64",
        "kv_cache_dtype": "fp8_e4m3",
        "disable_radix_cache": "on",
    }
    assert row["envs"] == {"SGLANG_AITER_K3_OPT": "1"}

    report = recipe_knobs([row], requested_shape={"conc": 64})
    assert report["sessions_with_config"] == 1
    assert report["knobs_by_family"] == {"cache": 1, "kv_cache": 1, "scheduling": 1}
    assert report["envs"]["SGLANG_AITER_K3_OPT"]["values"] == {"1": 1}


def test_knob_values_sized_to_a_scope_are_flagged_not_copied() -> None:
    row = project_session(_kimi_like(gain=14.2, args="--max-running-requests 64 --watchdog-timeout 1800"))
    report = recipe_knobs([row], requested_shape={"conc": 1})

    pool = report["knobs"]["max_running_requests"]
    assert pool["in_requested_scope"] is False
    assert any("equals conc=64" in note for note in pool["scope_coupled"])

    # A timeout is not a pool size, so it must not inherit the note.
    assert report["knobs"]["watchdog_timeout"]["scope_coupled"] == []
    assert knob_family("watchdog-timeout") == "timeout"


def test_learnings_survive_a_scope_miss_and_say_so() -> None:
    """A cold target has no in-scope evidence by definition.

    Returning nothing was correct for a gain prior and useless for the actual
    question, so the prose crosses the shape filter carrying its own scope.
    """
    docs = [
        _kimi_like(
            gain=27.8,
            args="--max-mamba-cache-size 160",
            learnings={
                "what_worked": [{"name": "mamba-slots-match-concurrency-64", "gain_pct": 27.8}],
                "pitfalls": [{"description": "--ep-size 8 regressed", "severity": "regress"}],
                "remaining_gaps": [{"symptom": "vendor patch wins ONLY at concurrency 1", "severity": "medium"}],
            },
        )
    ]
    report = estimate_from_sessions(docs, shape={"conc": 1})

    assert report["sessions_scored"] == 0
    assert report["historical"]["p50_validated_e2e_gain_pct"] is None
    assert report["coverage"]["with_learnings"] == 1
    assert any("no session at the requested scope" in item for item in report["limitations"])

    digest = report["learnings"]
    assert digest["counts"]["total"] == 3
    assert digest["counts"]["in_requested_scope"] == 0
    assert digest["counts"]["outside_requested_scope"] == 3
    # The record naming the requested concurrency leads, even though every
    # session was measured at 64.
    first = digest["items"][0]
    assert first["mentions_requested_scope"] is True
    assert "concurrency 1" in first["statement"]
    assert any("the concurrency asked for" in note for note in first["scope_notes"])
    assert all(item["observed_scopes"] == ["tp8/conc64/isl8192/osl1024"] for item in digest["items"])


def test_learnings_are_not_claimed_to_transfer() -> None:
    report = estimate_from_sessions([_kimi_like(gain=5.0, args="", learnings={"lessons": ["a statement"]})])
    assert "attribution" in report["recipe_knobs"]["note"]
    assert "scoped to the run that produced it" in report["learnings"]["note"]


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
    # A replay record carries no roofline ceiling, so capture is reported as
    # unmeasured rather than as zero. Reading it from the KB is the boundary
    # violation this pins; reading it from session evidence is not.
    assert report["coverage"]["with_capture"] == 0
    assert report["capture"] == {"sessions": 0, "p50_capture_pct": None, "p90_capture_pct": None}


def test_pulse_url_without_a_value_reads_the_environment(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def fake_documents(args):
        seen["url"] = args.pulse_url
        return [], []

    monkeypatch.setattr(estimate_no_run, "_pulse_documents", fake_documents)
    monkeypatch.setenv("PULSE_URL", " https://env-pulse/api ")
    assert estimate_main(["--pulse-url", "--output", str(tmp_path / "r.json")]) == 0
    assert seen["url"] == "https://env-pulse/api"


def test_pulse_url_without_a_value_or_environment_refuses(monkeypatch, capsys) -> None:
    monkeypatch.delenv("PULSE_URL", raising=False)
    assert estimate_main(["--pulse-url"]) == 2
    assert "PULSE_URL" in capsys.readouterr().err


def _write_pool(tmp_path: Path, docs: list[dict]) -> Path:
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(docs), encoding="utf-8")
    return path


def _run(argv: list[str], tmp_path: Path) -> dict:
    out = tmp_path / "report.json"
    assert estimate_main([*argv, "--output", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_cli_warns_when_the_pool_mixes_shapes(tmp_path: Path) -> None:
    pool = _write_pool(
        tmp_path,
        [
            _shaped(cid=_MI355_SGLANG, gain=10.0, optimized=800.0, tp=8, conc=1),
            _shaped(cid=_MI355_SGLANG, gain=242.0, optimized=900.0, tp=8, conc=64),
            _shaped(cid=_MI355_SGLANG, gain=5.0, optimized=200.0, tp=1, conc=64),
        ],
    )
    unscoped = _run(["--input", str(pool)], tmp_path)
    assert any("pool mixes 3 workload shapes; pass tp/conc/isl/osl" in w for w in unscoped["pool_warnings"])

    partly = _run(["--input", str(pool), "--tp", "8"], tmp_path)
    assert partly["sessions_scored"] == 2
    assert any("pool mixes 2 workload shapes; pass conc/isl/osl" in w for w in partly["pool_warnings"])

    scoped = _run(["--input", str(pool), "--tp", "8", "--conc", "64"], tmp_path)
    assert not any("workload shapes" in w for w in scoped["pool_warnings"])


class _FakeStore:
    """A store holding *total* identities with one session each."""

    def __init__(self, total: int, *, report_total: bool = True) -> None:
        self.total, self.report_total = total, report_total

    def search_identities(self, *, scheme, match, hardware_in, offset, limit):
        items = [{"canonical_id": f"{_MI355_SGLANG}-{i}"} for i in range(offset, min(offset + limit, self.total))]
        return {"items": items, **({"total": self.total} if self.report_total else {})}

    def get_rollup(self, cid):
        return {"sessions": [f"{cid}-s"]}

    def get_session(self, cid, sid):
        return _shaped(cid=_MI355_SGLANG, gain=10.0, optimized=800.0, tp=8) | {"session_id": sid}


@pytest.mark.parametrize(
    ("total", "report_total", "expected"),
    [
        (82, True, "read 50 of the 82 identities matching the search (--max-identities 50)"),
        (82, False, "stopped at --max-identities 50 and the store reported no total"),
        (30, True, None),
    ],
    ids=["truncated", "truncated-without-total", "complete"],
)
def test_a_truncated_identity_search_is_reported(monkeypatch, tmp_path: Path, total, report_total, expected) -> None:
    monkeypatch.setenv("KB_STORE_URL", "https://kb.invalid")
    monkeypatch.setattr(estimate_no_run, "KBStoreClient", lambda *a, **k: _FakeStore(total, report_total=report_total))
    report = _run(["--hardware", "mi355x"], tmp_path)
    fetched = min(total, 50)
    assert report["coverage"]["identities_fetched"] == fetched
    assert report["coverage"]["identities_matched"] == (total if report_total else None)
    assert report["sessions_scored"] == fetched
    matching = [line for line in report["limitations"] if "identities" in line]
    if expected is None:
        assert matching == []
    else:
        assert len(matching) == 1 and expected in matching[0]


def _pulse_row(**overrides) -> dict:
    row = {
        "session_id": "s1",
        "model_name": "m",
        "gpu_type": "mi355x",
        "framework": "sglang",
        "tp": 8,
        "conc": 64,
        "isl": 1024,
        "osl": 1024,
        "baseline_tok_per_s_per_gpu": 100.0,
        "opt_tok_per_s_per_gpu": 150.0,
        # Server totals, like the arms above: Pulse's *_per_gpu arms equal the snapshot's achieved throughput.
        "roofline": {
            "roofline_bound_kind": "memory",
            "roofline_mem_ceiling_tok_per_sec": 300.0,
            "roofline_cmp_ceiling_tok_per_sec": 500.0,
            "achieved_tok_per_sec": 150.0,
        },
    }
    row.update(overrides)
    return row


def test_capture_is_the_share_of_the_roofline_gap_closed() -> None:
    from kbmine.pulse import capture_pct, roofline_ceiling

    assert roofline_ceiling(_pulse_row()) == (300.0, "memory")
    assert capture_pct(_pulse_row()) == pytest.approx(25.0)
    compute = _pulse_row(roofline={**_pulse_row()["roofline"], "roofline_bound_kind": "compute"})
    assert roofline_ceiling(compute) == (500.0, "compute")
    assert capture_pct(compute) == pytest.approx(12.5)


def test_an_unlabelled_bound_takes_the_lower_ceiling() -> None:
    from kbmine.pulse import roofline_ceiling

    row = _pulse_row(roofline={"roofline_mem_ceiling_tok_per_sec": 300.0, "roofline_cmp_ceiling_tok_per_sec": 200.0})
    assert roofline_ceiling(row) == (200.0, "inferred min")


@pytest.mark.parametrize(
    "overrides",
    [
        {"roofline": None},
        {"roofline": {"roofline_bound_kind": "memory"}},
        {"baseline_tok_per_s_per_gpu": 300.0},
        {"baseline_tok_per_s_per_gpu": 400.0},
        {"opt_tok_per_s_per_gpu": None},
    ],
    ids=["no-snapshot", "no-ceiling", "zero-gap", "negative-gap", "no-optimized-arm"],
)
def test_an_unmeasurable_capture_is_none_not_zero(overrides) -> None:
    from kbmine.pulse import capture_pct

    assert capture_pct(_pulse_row(**overrides)) is None


def test_a_pulse_row_scales_per_gpu_throughput_back_to_a_total() -> None:
    from kbmine.pulse import LAYOUT_UNKNOWN, project_pulse_row

    projected = project_pulse_row(_pulse_row())
    assert projected["optimized_throughput"] == 150.0, "the arm is already the server total"
    assert projected["tput_per_gpu"] == pytest.approx(150.0 / 8)
    assert projected["baseline_tput_per_gpu"] == pytest.approx(100.0 / 8)
    assert (projected["ceiling_tput_per_gpu"], projected["ceiling_kind"]) == (pytest.approx(300.0 / 8), "memory")
    assert projected["capture_pct"] == pytest.approx(25.0)
    assert projected["parallelism_label"] == LAYOUT_UNKNOWN
    assert project_pulse_row(_pulse_row(tp=None))["tput_per_gpu"] == 150.0


def test_an_agentx_scope_without_isl_or_osl_encodes() -> None:
    from kbmine.kb_store_client import KBStoreClient

    query = KBStoreClient._scope_query({"kernel_optimizer": "forge", "tp": 8, "conc": 1})
    assert query.startswith("?") and "conc=1" in query and "isl" not in query


@pytest.mark.parametrize("n_ids", [49, 50, 60])
def test_explicit_canonical_ids_are_never_reported_as_truncated(monkeypatch, tmp_path: Path, n_ids) -> None:
    monkeypatch.setenv("KB_STORE_URL", "https://kb.invalid")
    monkeypatch.setattr(estimate_no_run, "KBStoreClient", lambda *a, **k: _FakeStore(0, report_total=False))
    ids = [arg for i in range(n_ids) for arg in ("--canonical-id", f"{_MI355_SGLANG}-{i}")]
    report = _run(ids, tmp_path)
    assert report["coverage"]["identities_fetched"] == n_ids
    assert report["sessions_scored"] == n_ids
    assert not [line for line in report["limitations"] if "identities" in line]


def _pulse_client(pages):
    """A PulseClient whose GETs answer from *pages*, a function of (limit, offset), and count the requests."""
    from kbmine.pulse import PulseClient

    client = PulseClient("https://pulse.invalid", "t")
    client.requests = 0

    def fake_get(path, params=None):
        client.requests += 1
        assert client.requests < 50, "the walk must terminate"
        return {"results": pages(params["limit"], params["offset"])}

    client.get = fake_get
    return client


def _rows(start: int, stop: int) -> list[dict]:
    return [{"session_id": f"s{i}"} for i in range(start, stop)]


def test_the_pulse_walk_pages_to_a_short_page() -> None:
    client = _pulse_client(lambda limit, offset: _rows(offset, min(offset + limit, 450)))
    rows = list(client.session_breakdowns(max_rows=1000))
    assert len(rows) == 450 and len({r["session_id"] for r in rows}) == 450
    assert client.walk_notes == []


def test_a_server_ignoring_offset_is_read_once_and_stops() -> None:
    client = _pulse_client(lambda limit, offset: _rows(0, limit))
    rows = list(client.session_breakdowns(max_rows=1000))
    page = len(rows)
    assert page > 0 and len({r["session_id"] for r in rows}) == page, "one page read once, no session twice"
    assert client.requests == 2
    assert any("may be ignoring offset" in note for note in client.walk_notes)
    assert any(f"skipped {page} row(s) repeating" in note for note in client.walk_notes)


def test_a_page_of_non_objects_ends_the_walk() -> None:
    client = _pulse_client(lambda limit, offset: ["x"] * limit)
    assert list(client.session_breakdowns(max_rows=1000)) == []
    assert client.requests == 1
    assert any("not objects" in note for note in client.walk_notes)


def test_the_pulse_walk_notes_reach_the_report(monkeypatch, tmp_path: Path) -> None:
    from kbmine import pulse

    monkeypatch.setattr(
        pulse.PulseClient, "get", lambda self, path, params=None: {"results": _rows(0, params["limit"])}
    )
    report = _run(["--pulse-url", "https://pulse.invalid"], tmp_path)
    assert any("may be ignoring offset" in note for note in report["fetch_errors"])


@pytest.mark.parametrize(
    ("report_total", "expected"),
    [
        (True, "read 1000 of the 5000 identities matching the search (the search's 1000-identity page limit)"),
        (False, "stopped at the search's 1000-identity page limit and the store reported no total"),
    ],
    ids=["with-total", "without-total"],
)
def test_the_search_page_limit_is_reported_as_the_cause(monkeypatch, tmp_path: Path, report_total, expected) -> None:
    monkeypatch.setenv("KB_STORE_URL", "https://kb.invalid")
    monkeypatch.setattr(estimate_no_run, "KBStoreClient", lambda *a, **k: _FakeStore(5000, report_total=report_total))
    report = _run(["--hardware", "mi355x", "--max-identities", "2000"], tmp_path)
    assert report["coverage"]["identities_fetched"] == 1000
    matching = [line for line in report["limitations"] if "identities" in line or "identity" in line]
    assert len(matching) == 1 and expected in matching[0]
    assert "--max-identities" not in matching[0]


def test_capture_compares_server_totals_as_pulse_carries_them() -> None:
    """A tp8 row as Pulse carries it: arms equal the snapshot's achieved total, ceilings are server-wide."""
    from kbmine.pulse import capture_pct

    assert capture_pct(_pulse_row()) == pytest.approx(25.0)
    assert capture_pct(_pulse_row(tp=1)) == pytest.approx(25.0), "tp1 reads the same"


def test_a_row_whose_snapshot_shows_per_gpu_arms_is_scaled_up() -> None:
    """If achieved is tp times an arm, the arms are per GPU: 12.5 and 18.75 per GPU are 100 and 150 in total."""
    from kbmine.pulse import capture_pct, project_pulse_row

    row = _pulse_row(baseline_tok_per_s_per_gpu=12.5, opt_tok_per_s_per_gpu=18.75)
    assert capture_pct(row) == pytest.approx(25.0)
    assert project_pulse_row(row)["optimized_throughput"] == pytest.approx(150.0)


def test_kb_and_input_reports_say_the_gains_are_winners_only(tmp_path: Path, monkeypatch) -> None:
    pool = _write_pool(tmp_path, [_shaped(cid=_MI355_SGLANG, gain=10.0, optimized=800.0, tp=8)])
    offline = _run(["--input", str(pool)], tmp_path)
    assert any("conditional on a session having won" in line for line in offline["limitations"])

    from kbmine import pulse

    monkeypatch.setattr(pulse.PulseClient, "get", lambda self, path, params=None: {"results": []})
    from_pulse = _run(["--pulse-url", "https://pulse.invalid"], tmp_path)
    assert not any("conditional on a session having won" in line for line in from_pulse["limitations"])


def test_a_non_json_pulse_response_is_a_pulse_error(monkeypatch, capsys) -> None:
    """A 200 HTML page (an SSO login, say) fails like every other Pulse error, not with a traceback."""
    import io
    import urllib.request

    from kbmine.pulse import PulseClient, PulseError

    class _Page(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Page(b"<html>Sign in</html>"))
    with pytest.raises(PulseError, match="response was not JSON"):
        PulseClient("https://pulse.invalid", "t").get("/v1/session-breakdowns")
    assert estimate_main(["--pulse-url", "https://pulse.invalid"]) == 1
    assert "Pulse fetch failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error", [TimeoutError("read timed out"), ConnectionResetError("reset")], ids=["timeout", "reset"]
)
def test_a_pulse_read_failure_is_a_pulse_error(monkeypatch, capsys, error) -> None:
    import urllib.request

    from kbmine.pulse import PulseClient, PulseError

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(PulseError, match="transport error"):
        PulseClient("https://pulse.invalid", "t").get("/v1/session-breakdowns")
    assert estimate_main(["--pulse-url", "https://pulse.invalid"]) == 1
    assert "Pulse fetch failed" in capsys.readouterr().err


@pytest.mark.parametrize("content", [None, "{not json"], ids=["missing", "invalid-json"])
def test_an_unreadable_input_file_exits_with_a_message(tmp_path: Path, content) -> None:
    path = tmp_path / "pool.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        estimate_main(["--input", str(path)])
    assert "cannot read a session JSON file" in str(exc.value)


@pytest.mark.parametrize("route", ["kb", "pulse"])
@pytest.mark.parametrize("bundle", ["missing", "not-pem"])
def test_an_unusable_ca_bundle_fails_with_one_line(tmp_path: Path, capsys, route, bundle) -> None:
    path = tmp_path / "bundle.pem"
    if bundle == "not-pem":
        path.write_text("not a certificate\n", encoding="utf-8")
    url_flag = ["--kb-store-url", "https://kb.invalid"] if route == "kb" else ["--pulse-url", "https://pulse.invalid"]
    assert estimate_main([*url_flag, "--ca-bundle", str(path)]) in (1, 2)
    err = capsys.readouterr().err
    assert "cannot load the CA bundle" in err and "Traceback" not in err


def test_a_truncated_pulse_body_is_a_pulse_error(monkeypatch, capsys) -> None:
    import http.client
    import urllib.request

    from kbmine.pulse import PulseClient, PulseError

    class _Truncated:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            raise http.client.IncompleteRead(b"{", 100)

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Truncated())
    with pytest.raises(PulseError, match="transport error"):
        PulseClient("https://pulse.invalid", "t").get("/v1/session-breakdowns")
    assert estimate_main(["--pulse-url", "https://pulse.invalid"]) == 1
    assert "Pulse fetch failed" in capsys.readouterr().err


def test_an_undecodable_token_file_exits_with_a_message(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(SystemExit) as exc:
        estimate_main(["--kb-store-url", "https://kb.invalid", "--kb-store-token-file", str(token_file)])
    assert "cannot read --kb-store-token-file" in str(exc.value)


def test_an_unwritable_output_fails_with_one_line(tmp_path: Path, capsys) -> None:
    pool = _write_pool(tmp_path, [_shaped(cid=_MI355_SGLANG, gain=10.0, optimized=800.0, tp=8)])
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert estimate_main(["--input", str(pool), "--output", str(blocker / "report.json")]) == 1
    assert "cannot write --output" in capsys.readouterr().err


def test_an_http_error_with_an_unreadable_body_is_still_a_pulse_error(monkeypatch) -> None:
    import http.client
    import urllib.error
    import urllib.request

    from kbmine.pulse import PulseClient, PulseError

    class _BadBody(urllib.error.HTTPError):
        def read(self, *args):
            raise http.client.IncompleteRead(b"", 10)

    def fail(*args, **kwargs):
        raise _BadBody("https://pulse.invalid", 502, "Bad Gateway", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(PulseError, match="HTTP 502"):
        PulseClient("https://pulse.invalid", "t").get("/v1/session-breakdowns")


class _Flaky:
    """urlopen stand-in: raises each queued failure in turn, then answers with *rows*."""

    def __init__(self, failures, rows):
        self.failures, self.rows, self.calls = list(failures), rows, 0

    def __call__(self, *args, **kwargs):
        import io

        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        body = json.dumps({"results": self.rows}).encode()

        class _Body(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Body(body)


def test_a_truncated_page_is_retried_and_noted(monkeypatch) -> None:
    import http.client
    import urllib.request

    from kbmine.pulse import PulseClient

    flaky = _Flaky([http.client.IncompleteRead(b"{", 10), ConnectionResetError("reset")], _rows(0, 3))
    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    client = PulseClient("https://pulse.invalid", "t")
    rows = list(client.session_breakdowns(max_rows=1000))
    assert len(rows) == 3 and flaky.calls == 3
    assert any("2 request(s) repeated" in note for note in client.walk_notes)


def test_a_persistent_transport_failure_gives_up_after_three_attempts(monkeypatch) -> None:
    import http.client
    import urllib.request

    from kbmine.pulse import PulseClient, PulseError

    flaky = _Flaky([http.client.IncompleteRead(b"", 10)] * 5, [])
    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    with pytest.raises(PulseError, match="after 3 attempts"):
        PulseClient("https://pulse.invalid", "t").get("/v1/session-breakdowns")
    assert flaky.calls == 3


def test_an_http_status_is_not_retried(monkeypatch) -> None:
    import urllib.error
    import urllib.request

    from kbmine.pulse import PulseClient, PulseError

    error = urllib.error.HTTPError("https://pulse.invalid", 500, "Server Error", {}, None)
    flaky = _Flaky([error], [])
    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    with pytest.raises(PulseError, match="HTTP 500"):
        PulseClient("https://pulse.invalid", "t").get("/v1/session-breakdowns")
    assert flaky.calls == 1


def test_retries_reach_the_report(monkeypatch, tmp_path: Path) -> None:
    import http.client
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _Flaky([http.client.IncompleteRead(b"", 10)], _rows(0, 2)))
    report = _run(["--pulse-url", "https://pulse.invalid"], tmp_path)
    assert any("1 request(s) repeated" in note for note in report["fetch_errors"])


def test_the_report_names_the_service_it_read(monkeypatch, tmp_path: Path) -> None:
    from kbmine import pulse

    monkeypatch.setattr(pulse.PulseClient, "get", lambda self, path, params=None: {"results": []})
    from_pulse = _run(["--pulse-url", "https://pulse.invalid/api"], tmp_path)
    assert from_pulse["pulse_url"] == "https://pulse.invalid/api" and "kb_store_url" not in from_pulse

    pool = _write_pool(tmp_path, [_shaped(cid=_MI355_SGLANG, gain=10.0, optimized=800.0, tp=8)])
    offline = _run(["--input", str(pool)], tmp_path)
    assert offline["kb_store_url"] == "" and "pulse_url" not in offline
