# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""integrate_patch accuracy-gate routing: image-diff (offline) vs lm-eval (online).

Exercises ``IntegratePatchExecutor._bench_patch`` with ``run_grid`` and the
gate helpers mocked, so we validate ONLY the gate-selection logic (which gate
runs, and that its verdict is threaded into ``accuracy_pass``).
"""

from __future__ import annotations

import asyncio

import pytest

from inference_optimizer.orchestrator.action_executors import integrate_patch as ip
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    VariantResult,
)


def _ok_variant(result_dir="/ws/patch", status="succeeded"):
    return VariantResult(
        name="integrate-patch-abcd1234",
        extra_server_args="",
        extra_envs={},
        status=status,
        output_throughput=1.5,
        workspace=result_dir,
    )


def _patch_common(monkeypatch, variant, tmp_path):
    """Stub run_grid + config materialization so _bench_patch reaches the gate."""

    async def fake_run_grid(*args, **kwargs):
        return [variant]

    monkeypatch.setattr(ip, "run_grid", fake_run_grid)
    monkeypatch.setattr(
        ip,
        "materialize_config_with_envs",
        lambda *a, **k: tmp_path / "integrate_patch.with_envs.yaml",
    )
    # result_dir on the variant drives the gate; ensure the config file "exists".
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("benchmark: {framework: xdit}\n", encoding="utf-8")
    return cfg


def _run_bench(executor, params, tmp_path):
    return asyncio.run(
        executor._bench_patch(
            params=params,
            output_root=tmp_path / "out",
            config_changes_applied={},
            specialist_task_id="abcd1234deadbeef",
        )
    )


@pytest.fixture
def executor(tmp_path):
    return ip.IntegratePatchExecutor(session_dir=tmp_path)


class TestOfflineImageDiffGate:
    def test_offline_runs_image_diff_and_passes(self, executor, tmp_path, monkeypatch):
        variant = _ok_variant(result_dir=str(tmp_path / "patch"))
        cfg = _patch_common(monkeypatch, variant, tmp_path)

        called = {"image_diff": 0, "lm_eval": 0}

        def fake_image_diff(baseline, candidate, threshold_db=None):
            called["image_diff"] += 1
            return True

        # image_diff_passed is imported lazily inside _bench_patch from _image_diff
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors._image_diff."
            "image_diff_passed",
            fake_image_diff,
        )
        monkeypatch.setattr(
            ip, "parse_eval_results",
            lambda *a, **k: called.__setitem__("lm_eval", called["lm_eval"] + 1) or {},
        )

        _bench, evidence = _run_bench(
            executor,
            {
                "config_path": str(cfg),
                "framework": "xdit",
                "baseline_image_dir": str(tmp_path / "baseline"),
            },
            tmp_path,
        )

        assert called["image_diff"] == 1
        assert called["lm_eval"] == 0
        assert evidence["accuracy_pass"] is True

    def test_offline_image_diff_failure_rejects(self, executor, tmp_path, monkeypatch):
        variant = _ok_variant(result_dir=str(tmp_path / "patch"))
        cfg = _patch_common(monkeypatch, variant, tmp_path)
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors._image_diff."
            "image_diff_passed",
            lambda *a, **k: False,
        )
        _bench, evidence = _run_bench(
            executor,
            {
                "config_path": str(cfg),
                "framework": "xdit",
                "baseline_image_dir": str(tmp_path / "baseline"),
            },
            tmp_path,
        )
        assert evidence["accuracy_pass"] is False

    def test_offline_without_baseline_dir_skips_gate(
        self, executor, tmp_path, monkeypatch
    ):
        variant = _ok_variant(result_dir=str(tmp_path / "patch"))
        cfg = _patch_common(monkeypatch, variant, tmp_path)
        _bench, evidence = _run_bench(
            executor,
            {"config_path": str(cfg), "framework": "xdit"},
            tmp_path,
        )
        assert evidence["accuracy_pass"] is None

    def test_offline_failed_bench_skips_gate(self, executor, tmp_path, monkeypatch):
        variant = _ok_variant(
            result_dir=str(tmp_path / "patch"), status="failed"
        )
        cfg = _patch_common(monkeypatch, variant, tmp_path)
        called = {"image_diff": 0}
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors._image_diff."
            "image_diff_passed",
            lambda *a, **k: called.__setitem__("image_diff", 1) or True,
        )
        _bench, evidence = _run_bench(
            executor,
            {
                "config_path": str(cfg),
                "framework": "xdit",
                "baseline_image_dir": str(tmp_path / "baseline"),
            },
            tmp_path,
        )
        assert called["image_diff"] == 0
        assert evidence["accuracy_pass"] is None


class TestOnlineLmEvalGate:
    def test_online_runs_lm_eval_not_image_diff(self, executor, tmp_path, monkeypatch):
        variant = _ok_variant(result_dir=str(tmp_path / "patch"))
        cfg = _patch_common(monkeypatch, variant, tmp_path)

        called = {"image_diff": 0, "lm_eval": 0}
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors._image_diff."
            "image_diff_passed",
            lambda *a, **k: called.__setitem__("image_diff", 1) or True,
        )

        def fake_eval(_dir):
            called["lm_eval"] += 1
            return {"score": 0.80}

        monkeypatch.setattr(ip, "parse_eval_results", fake_eval)
        monkeypatch.setattr(ip, "accuracy_passed", lambda new, base: True)

        _bench, evidence = _run_bench(
            executor,
            {
                "config_path": str(cfg),
                "framework": "sglang",
                "accuracy_baseline": 0.80,
            },
            tmp_path,
        )
        assert called["lm_eval"] == 1
        assert called["image_diff"] == 0
        assert evidence["accuracy_pass"] is True

    def test_online_without_baseline_accuracy_skips(
        self, executor, tmp_path, monkeypatch
    ):
        variant = _ok_variant(result_dir=str(tmp_path / "patch"))
        cfg = _patch_common(monkeypatch, variant, tmp_path)
        _bench, evidence = _run_bench(
            executor,
            {"config_path": str(cfg), "framework": "sglang"},
            tmp_path,
        )
        assert evidence["accuracy_pass"] is None
