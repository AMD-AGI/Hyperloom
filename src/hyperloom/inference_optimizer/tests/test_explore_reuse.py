# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exact-match reuse of past EXPLORE decision rounds."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors import _explore_reuse as reuse


def _identity(**overrides):
    kw = dict(
        model="/models/Qwen3-8B",
        framework="sglang",
        gpu="mi355x",
        workload_signature="abc123",
        runtime={"image_digest": "sha256:aaa"},
        benchmark={"framework": "sglang", "envs": {"CONC": "8"}},
        inherited_args="",
        stack={"extra_server_args": ""},
        variant={"fingerprint": "f" * 16},
        pythonpath_entries=[],
        decision_protocol="cold",
    )
    kw.update(overrides)
    return reuse.measurement_identity(**kw)


def _result(tput=840.0, **kw):
    fields = dict(
        output_throughput=tput,
        input_throughput=tput,
        total_token_throughput=2 * tput,
        request_throughput=tput / 256,
        tpot_mean_ms=12.0,
        ttft_mean_ms=140.0,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "measurements"
    monkeypatch.setenv(reuse.ENV_STORE, str(root))
    monkeypatch.delenv(reuse.ENV_MAX_AGE_DAYS, raising=False)
    return root


def test_off_without_a_store(monkeypatch):
    monkeypatch.delenv(reuse.ENV_STORE, raising=False)
    assert not reuse.reuse_enabled()
    assert reuse.lookup("k" * 64, accuracy_required=False, keep_threshold_pct=1.0) is None


def test_a_measurement_that_cannot_be_pinned_is_not_keyed():
    assert _identity(runtime={}) is None
    assert _identity(runtime_override={"pythonpath_prefix": "/x"}) is None
    assert _identity(runtime={"framework_version": "0.5.1"}) is not None


def test_runtime_identity_reads_the_boot_fingerprint():
    meta = {"sglang": "0.5.1", "vllm": "0.9", "image_digest": "sha256:aaa", "rocm": "unknown"}
    assert reuse.runtime_identity(meta, "sglang") == {"framework_version": "0.5.1", "image_digest": "sha256:aaa"}
    assert reuse.runtime_identity(None, "sglang") == {}


def test_each_part_of_the_identity_changes_the_key():
    base = reuse.measurement_key(_identity())
    for change in (
        {"runtime": {"image_digest": "sha256:bbb"}},
        {"gpu": "mi300x"},
        {"workload_signature": "zzz"},
        {"stack": {"extra_server_args": "--kept 1"}},
        {"variant": {"fingerprint": "e" * 16}},
        {"decision_protocol": "warm"},
        {"inherited_args": "--tp 8"},
    ):
        assert reuse.measurement_key(_identity(**change)) != base, change


def test_overlays_are_identified_by_content_not_path(tmp_path):
    a, b, c = (tmp_path / n for n in ("a", "b", "c"))
    for d, body in ((a, "def k(): return 1\n"), (b, "def k(): return 1\n"), (c, "def k(): return 2\n")):
        d.mkdir()
        (d / "kernel.py").write_text(body)
    key_a = reuse.measurement_key(_identity(pythonpath_entries=[str(a)]))
    assert reuse.measurement_key(_identity(pythonpath_entries=[str(b)])) == key_a
    assert reuse.measurement_key(_identity(pythonpath_entries=[str(c)])) != key_a
    (a / "kernel.py").write_text("def k(): return 3\n")
    assert reuse.measurement_key(_identity(pythonpath_entries=[str(a)])) != key_a


def test_an_overlay_too_large_to_hash_refuses_reuse(tmp_path, monkeypatch):
    big = tmp_path / "big"
    big.mkdir()
    for i in range(3):
        (big / f"f{i}.py").write_text("x")
    monkeypatch.setattr(reuse, "_OVERLAY_MAX_FILES", 2)
    assert _identity(pythonpath_entries=[str(big)]) is None


def test_benchmark_identity_drops_where_things_are_written(tmp_path):
    bench = {
        "framework": "SGLang",
        "model": "/m",
        "envs": {"CONC": 8, "RESULT_DIR": "/x", "MODEL_PATH": "/m", "NOTE": f"{tmp_path}/run", "ISL": 256},
    }
    ident = reuse.benchmark_identity(bench, volatile_roots=[str(tmp_path)])
    assert ident["framework"] == "sglang"
    assert ident["envs"] == {"CONC": "8", "ISL": "256"}


def test_record_then_lookup_replays_the_paired_gain(store):
    key = reuse.measurement_key(_identity())
    reuse.record(key, _identity(), _result(840.0), anchor_tput=800.0, accuracy=0.8, variant_name="v")
    hit = reuse.lookup(key, accuracy_required=True, keep_threshold_pct=1.0)
    assert hit is not None
    assert hit.gain_pct == pytest.approx(5.0)
    assert hit.accuracy == pytest.approx(0.8)
    # Replayed against today's anchor, not yesterday's absolute number.
    replay = hit.as_result(name="v", extra_server_args="", extra_envs={}, anchor_tput=1000.0)
    assert replay.status == "succeeded"
    assert replay.output_throughput == pytest.approx(1050.0)
    assert replay.total_token_throughput == pytest.approx(2100.0)
    assert replay.tpot_mean_ms == pytest.approx(12.0)
    assert hit.provenance()["base_tput"] == pytest.approx(800.0)


def test_a_stale_record_is_not_reused(store, monkeypatch):
    key = reuse.measurement_key(_identity())
    reuse.record(key, _identity(), _result(), anchor_tput=800.0)
    later = time.time() + 15 * 86400
    assert reuse.lookup(key, accuracy_required=False, keep_threshold_pct=1.0, now=later) is None
    monkeypatch.setenv(reuse.ENV_MAX_AGE_DAYS, "30")
    assert reuse.lookup(key, accuracy_required=False, keep_threshold_pct=1.0, now=later) is not None


def test_an_accuracy_gated_session_needs_the_score_only_for_a_would_be_keep(store):
    keep_key, revert_key = "a" * 64, "b" * 64
    reuse.record(keep_key, {}, _result(840.0), anchor_tput=800.0, accuracy=None)
    reuse.record(revert_key, {}, _result(800.4), anchor_tput=800.0, accuracy=None)
    assert reuse.lookup(keep_key, accuracy_required=True, keep_threshold_pct=1.0) is None
    assert reuse.lookup(keep_key, accuracy_required=False, keep_threshold_pct=1.0) is not None
    assert reuse.lookup(revert_key, accuracy_required=True, keep_threshold_pct=1.0) is not None


def test_nothing_is_recorded_without_an_anchor_or_a_measurement(store):
    reuse.record("c" * 64, {}, _result(840.0), anchor_tput=0.0)
    reuse.record("d" * 64, {}, _result(0.0), anchor_tput=800.0)
    assert reuse.lookup("c" * 64, accuracy_required=False, keep_threshold_pct=1.0) is None
    assert reuse.lookup("d" * 64, accuracy_required=False, keep_threshold_pct=1.0) is None
