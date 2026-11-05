# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Skipping an EXPLORE variant whose exact launch already reverted."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors.explore_prefilter import _prefilter as prefilter
from hyperloom.orchestrator.actions.executors.explore_prefilter import _reuse as reuse
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant

_SESSION = {
    "model": "/models/Qwen3-8B",
    "framework": "sglang",
    "gpu": "mi355x",
    "workload_signature": "abc123",
    "runtime": {"image_digest": "sha256:aaa", "framework_version": "0.5.1"},
    "benchmark": {"framework": "sglang", "envs": {"CONC": "8"}},
    "inherited_args": "",
    "decision_protocol": "cold",
    "grade_on_intvty": False,
    "keep_threshold_pct": 1.0,
}


def _identity(pythonpath=(), **session):
    return reuse.measurement_identity(
        {**_SESSION, **session}, {"extra_server_args": ""}, {"fingerprint": "f" * 16}, list(pythonpath)
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "measurements"
    monkeypatch.setenv(reuse.ENV_STORE, str(root))
    monkeypatch.delenv(reuse.ENV_MAX_AGE_DAYS, raising=False)
    return root


def test_off_without_a_store(monkeypatch):
    monkeypatch.delenv(reuse.ENV_STORE, raising=False)
    assert not reuse.reuse_enabled()
    assert reuse.lookup_revert("k" * 64) is None


def test_a_measurement_that_cannot_be_pinned_is_not_keyed():
    assert _identity(runtime={}) is None
    assert _identity(runtime={"framework_version": "0.5.1"}) is None
    assert _identity(runtime={"image_digest": "sha256:aaa"}) is None
    assert _identity(runtime={"image_digest": "sha256:aaa", "framework_version": "0.5.1"}) is not None


def test_runtime_identity_reads_the_boot_fingerprint():
    meta = {"sglang": "0.5.1", "vllm": "0.9", "image_digest": "sha256:aaa", "rocm": "unknown"}
    assert reuse.runtime_identity(meta, "sglang") == {"framework_version": "0.5.1", "image_digest": "sha256:aaa"}
    assert reuse.runtime_identity(None, "sglang") == {}


@pytest.mark.parametrize(
    "change",
    [
        {"runtime": {"image_digest": "sha256:bbb", "framework_version": "0.5.1"}},
        {"gpu": "mi300x"},
        {"workload_signature": "zzz"},
        {"decision_protocol": "warm"},
        {"grade_on_intvty": True},
        {"keep_threshold_pct": 0.5},
    ],
)
def test_each_part_of_the_identity_changes_the_key(change):
    assert reuse.measurement_key(_identity(**change)) != reuse.measurement_key(_identity())


def test_overlay_content_not_its_path_is_the_identity(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "kernel.py").write_text("x = 1\n")
    assert reuse.measurement_key(_identity([str(a)])) == reuse.measurement_key(_identity([str(b)]))
    (b / "kernel.py").write_text("x = 2\n")
    assert reuse.measurement_key(_identity([str(a)])) != reuse.measurement_key(_identity([str(b)]))


def test_a_stored_revert_is_found_and_ages_out(store):
    key = reuse.measurement_key(_identity())
    reuse.record_revert(key, _identity(), {"reason": "gain_below_threshold", "round_id": "explore-001"})
    record = reuse.lookup_revert(key)
    assert record["outcome"] == "REVERT" and record["round_id"] == "explore-001"
    assert reuse.lookup_revert(key, now=time.time() + 15 * 86400) is None


def test_records_of_an_older_schema_are_ignored(store):
    key = reuse.measurement_key(_identity())
    reuse.record_revert(key, _identity(), {})
    path = store / key[:2] / f"{key}.json"
    path.write_text(path.read_text().replace(f'"schema": {reuse.SCHEMA}', '"schema": 1'))
    assert reuse.lookup_revert(key) is None


def test_the_launch_digest_ignores_priced_envs_and_locations_but_not_levers():
    priced = frozenset({"CONC", "EXTRA_SGLANG_ARGS"})
    base = {"CONC": "8", "EXTRA_SGLANG_ARGS": "--x", "RESULT_DIR": "/out/a", "SGLANG_USE_AITER": "1"}
    digest = reuse.launch_digest(base, priced)
    assert reuse.launch_digest({**base, "CONC": "64", "RESULT_DIR": "/out/b"}, priced) == digest
    assert reuse.launch_digest({**base, "SGLANG_USE_AITER": "0"}, priced) != digest


def test_the_launch_digest_follows_overlay_content(tmp_path):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "kernel.py").write_text("x = 1\n")
    digest = reuse.launch_digest({"PYTHONPATH": str(overlay)}, frozenset())
    assert digest != reuse.launch_digest({}, frozenset())
    (overlay / "kernel.py").write_text("x = 2\n")
    assert reuse.launch_digest({"PYTHONPATH": str(overlay)}, frozenset()) != digest


@pytest.mark.parametrize(
    ("outcome", "reason", "anchor_tput", "recorded"),
    [
        ("REVERT", "below_threshold", 100.0, True),
        ("KEEP", "above_threshold", 100.0, False),
        ("REVERT", "accuracy_drop", 100.0, False),
        ("REVERT", "accuracy_unavailable", 100.0, False),
        ("REVERT", "below_threshold", 0.0, False),
    ],
)
def test_only_a_performance_revert_is_recorded(store, outcome, reason, anchor_tput, recorded):
    identity = _identity()
    ctx = SimpleNamespace(identity=lambda gv, stack: identity, session_dir=None)
    verdict = prefilter.Verdict(outcome, reason, anchor_tput, "r1", prefilter.Stack())
    prefilter._record_revert(ctx, GridVariant(name="v"), SimpleNamespace(output_throughput=90.0), verdict)
    assert (reuse.lookup_revert(reuse.measurement_key(identity)) is not None) is recorded
