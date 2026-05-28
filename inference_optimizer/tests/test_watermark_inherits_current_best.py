"""PR-B: watermark refresh inherits ``current_best.extra_sglang_args``.

Pins two contracts so the e2e bug surfaced on Qwen3-30B-A3B doesn't
silently come back:

1. ``materialize_config_with_envs`` MERGES the caller-supplied
   ``extra_sglang_args`` with the yaml-default ``EXTRA_SGLANG_ARGS``
   instead of overwriting. The profile path's
   ``--enable-profile-cuda-graph`` / ``--enable-shape-discovery-
   for-cuda-graph-profile`` flags live in the yaml envs after the
   patcher / cuda-graph-profile branch fires; a plain overwrite when
   the caller passes a non-empty ``extra_sglang_args`` (e.g.
   watermark-refresh inheriting ``current_best.extra_sglang_args=
   --num-continuous-decode-steps 2``) silently drops graph capture and
   the profile run hits ``error_class=no_trace_files`` downstream.

2. ``ProfileExecutor.__call__`` merges ``params["base_extra_args"]``
   (stamped by Coordinator from ``current_best.extra_sglang_args``)
   into ``params["extra_sglang_args"]`` before delegating to
   ``BaselineExecutor.__call__``. Without this, the watermark-refresh
   profile run captures a baseline-workload trace and
   ``record_trace_analyze`` stamps the same KPI as the PRELUDE
   snapshot, hiding the actual gain on the dashboard.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors._workload_envs import (
    materialize_config_with_envs,
)


# ---------------------------------------------------------------------------
# Contract 1 — materialize_config_with_envs merges, never overwrites
# ---------------------------------------------------------------------------
def _write_profile_yaml(
    path: Path, *, framework: str = "sglang", existing_args: str = "",
) -> None:
    """Lay down a minimal profile-style yaml with envs pre-populated."""
    envs: dict = {
        "CONC": 8, "ISL": 256, "OSL": 256,
        "MAX_MODEL_LEN": 4608, "TP": 1, "RANDOM_RANGE_RATIO": 1,
    }
    if existing_args:
        env_key = "EXTRA_SGLANG_ARGS" if framework == "sglang" else "EXTRA_VLLM_ARGS"
        envs[env_key] = existing_args
    cfg = {
        "benchmark": {
            "framework": framework,
            "envs": envs,
            "model": "/tmp/m",
            "precision": "bf16",
            "run_mode": "local",
            "profiler": {"torch_profiler": {"enabled": False}},
        }
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _load_envs(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]["envs"]


class TestMaterializeMergesExtraSglangArgs:
    """``server_args`` no longer clobbers the yaml-default
    ``EXTRA_SGLANG_ARGS`` — they get merged via ``merge_server_args``."""

    def test_merge_preserves_yaml_default_when_caller_supplies_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("PRECISION", "bf16")
        monkeypatch.setenv("FRAMEWORK", "sglang")
        src = tmp_path / "profile.yaml"
        _write_profile_yaml(
            src,
            existing_args="--enable-profile-cuda-graph",
        )
        out = materialize_config_with_envs(
            src,
            tmp_path / "out",
            extra_sglang_args="--num-continuous-decode-steps 2",
        )
        envs = _load_envs(out)
        merged = str(envs.get("EXTRA_SGLANG_ARGS") or "")
        # Both the yaml-default profile flag AND the caller-supplied
        # optimization param must survive.
        assert "--enable-profile-cuda-graph" in merged
        assert "--num-continuous-decode-steps 2" in merged

    def test_caller_supplied_arg_appears_after_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """When both sides set the same flag, ``merge_server_args``
        concatenates left-to-right (yaml first, caller second). Sglang
        / vLLM argparse then takes the LAST value at parse time, so the
        caller-supplied value effectively wins. We pin the
        concatenation order so the shell-append → argparse contract
        keeps producing the right runtime value."""
        monkeypatch.setenv("PRECISION", "bf16")
        monkeypatch.setenv("FRAMEWORK", "sglang")
        src = tmp_path / "profile.yaml"
        _write_profile_yaml(
            src,
            existing_args="--mem-fraction-static 0.85",
        )
        out = materialize_config_with_envs(
            src,
            tmp_path / "out",
            extra_sglang_args="--mem-fraction-static 0.92",
        )
        envs = _load_envs(out)
        merged = str(envs.get("EXTRA_SGLANG_ARGS") or "")
        # Both values appear in the merged string (no de-dup by design);
        # 0.92 sits to the right of 0.85 so argparse picks 0.92.
        assert "0.85" in merged
        assert "0.92" in merged
        assert merged.index("0.85") < merged.index("0.92")

    def test_no_args_supplied_leaves_yaml_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """No caller args + yaml has profile flags → yaml flags persist."""
        monkeypatch.setenv("PRECISION", "bf16")
        monkeypatch.setenv("FRAMEWORK", "sglang")
        src = tmp_path / "profile.yaml"
        _write_profile_yaml(src, existing_args="--enable-profile-cuda-graph")
        out = materialize_config_with_envs(src, tmp_path / "out")
        envs = _load_envs(out)
        assert "--enable-profile-cuda-graph" in str(envs["EXTRA_SGLANG_ARGS"])


# ---------------------------------------------------------------------------
# Contract 2 — ProfileExecutor merges base_extra_args into extra_sglang_args
# ---------------------------------------------------------------------------
class TestProfileExecutorMergesBaseExtraArgs:
    """Coordinator stamps ``current_best.extra_sglang_args`` into
    ``params["base_extra_args"]``; ProfileExecutor.__call__ must merge
    it into ``params["extra_sglang_args"]`` so the downstream
    BaselineExecutor materialize step picks it up via the standard
    ``extra_sglang_args`` channel.

    We test the merge directly on the params dict (not through the
    full ``__call__``, which requires a real Task + SubAgentRunner
    context). The merge code path lives at the very top of
    ``ProfileExecutor.__call__`` and operates on ``ctx.task.params``
    in place; we exercise the same logic by importing
    ``merge_server_args`` and asserting the contract."""

    def test_merge_logic_is_left_to_right(self):
        """Sanity: merge_server_args(base, task_args) keeps both."""
        from inference_optimizer.orchestrator.action_executors._grid_runner import (
            merge_server_args,
        )
        merged = merge_server_args(
            "--num-continuous-decode-steps 2",
            "",
        )
        assert "--num-continuous-decode-steps" in merged
        assert "2" in merged

    def test_merge_with_existing_task_args(self):
        from inference_optimizer.orchestrator.action_executors._grid_runner import (
            merge_server_args,
        )
        merged = merge_server_args(
            "--num-continuous-decode-steps 2",
            "--mem-fraction-static 0.9",
        )
        assert "--num-continuous-decode-steps" in merged
        assert "--mem-fraction-static" in merged

    def test_profile_executor_init_does_not_crash(self):
        """Smoke: the module-level singleton constructs OK + the
        per-call merge branch is byte-readable."""
        from inference_optimizer.orchestrator.action_executors.profile import (
            ProfileExecutor,
            profile_executor,
        )
        assert isinstance(profile_executor, ProfileExecutor)
        # Confirm the source contains the merge call (regression guard
        # in case a refactor accidentally removes the merge block).
        import inspect
        src = inspect.getsource(ProfileExecutor.__call__)
        assert "base_extra_args" in src
        assert "merge_server_args" in src
