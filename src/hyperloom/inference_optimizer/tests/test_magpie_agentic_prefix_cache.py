# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An agentic replay resends each turn's prompt as the next turn's prefix, so it must keep its prefix cache."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp


#: The generic SGLang scripts' default-server-args block, verbatim.
_RUNNER = """#!/bin/bash
DEFAULT_ARGS=""
for flag_val in "--mem-fraction-static=0.8" "--disable-radix-cache"; do
  flag="${flag_val%%=*}"
  if [[ -z "$EXTRA_SGLANG_ARGS" ]] || ! echo "$EXTRA_SGLANG_ARGS" | grep -q -- "$flag"; then
    DEFAULT_ARGS="$DEFAULT_ARGS $flag_val"
  fi
done
echo "$DEFAULT_ARGS"
"""


def _runners(root: Path, body: str = _RUNNER) -> Path:
    """Write both generic scripts where ``$INFERENCEX_PATH/benchmarks`` puts the copy that runs."""
    benchmarks = root / "benchmarks"
    benchmarks.mkdir(parents=True, exist_ok=True)
    for name in mp._AGENTIC_RADIX_SCRIPTS:
        (benchmarks / name).write_text(body, encoding="utf-8")
    return root


def _mlperf_env(root: Path) -> dict[str, str]:
    """The environment an MLPerf AgentX run prepares under.

    Only that backend reaches ``prepare_agentx_runtime``: the native Magpie one
    pops ``benchmark_script`` and drives Magpie's managed server, so no generic
    script of ours launches its server. ``MAGPIE_PATH`` points at a tree with no
    scripts so the patch under test is the one applied to the ``inferencex_path``
    copy, which is the copy a session actually runs.
    """
    return {
        "HYPERLOOM_AGENTIC_BACKEND": "mlperf",
        "MAGPIE_PATH": str(root / "absent-magpie"),
    }


def _default_args(script: Path, *, is_agentic: str, extra: str = "") -> str:
    """What the patched script resolves DEFAULT_ARGS to, run as the real harness runs it."""
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env={"IS_AGENTIC": is_agentic, "EXTRA_SGLANG_ARGS": extra, "PATH": "/usr/bin:/bin"},
        check=True,
    )
    return proc.stdout.strip()


def test_an_agentic_run_keeps_the_prefix_cache_and_a_sweep_does_not(tmp_path):
    """The whole point: the flag cannot be lifted from the request side, because naming it passes it."""
    _runners(tmp_path)

    assert mp.ensure_agentic_prefix_cache(None, tmp_path) is True

    script = tmp_path / "benchmarks" / "sglang_mi355x.sh"
    assert _default_args(script, is_agentic="1") == "--mem-fraction-static=0.8"
    assert _default_args(script, is_agentic="") == "--mem-fraction-static=0.8 --disable-radix-cache"


def test_the_spelling_upstream_accepts_is_accepted_here(tmp_path):
    """InferenceX's own multi-node script takes either, and a caller may carry either."""
    _runners(tmp_path)
    mp.ensure_agentic_prefix_cache(None, tmp_path)

    assert _default_args(tmp_path / "benchmarks" / "sglang_mi355x.sh", is_agentic="true") == "--mem-fraction-static=0.8"


def test_the_patch_reaches_every_generic_script(tmp_path):
    _runners(tmp_path)

    mp.ensure_agentic_prefix_cache(None, tmp_path)

    for name in mp._AGENTIC_RADIX_SCRIPTS:
        assert mp._AGENTIC_RADIX_SENTINEL in (tmp_path / "benchmarks" / name).read_text(encoding="utf-8")


def test_patching_twice_changes_nothing(tmp_path):
    """Every Magpie invocation re-asserts this, so a second pass must be a no-op rather than a second guard."""
    _runners(tmp_path)

    assert mp.ensure_agentic_prefix_cache(None, tmp_path) is True
    once = (tmp_path / "benchmarks" / "sglang_mi355x.sh").read_text(encoding="utf-8")
    assert mp.ensure_agentic_prefix_cache(None, tmp_path) is True

    assert (tmp_path / "benchmarks" / "sglang_mi355x.sh").read_text(encoding="utf-8") == once


def test_a_request_that_overrides_a_default_still_overrides_it(tmp_path):
    """The script skips any default the request names; the patch must not disturb that."""
    _runners(tmp_path)
    mp.ensure_agentic_prefix_cache(None, tmp_path)

    resolved = _default_args(
        tmp_path / "benchmarks" / "sglang_mi355x.sh",
        is_agentic="1",
        extra="--mem-fraction-static=0.9",
    )

    assert resolved == ""


def test_a_script_whose_block_moved_is_reported_not_silently_accepted(tmp_path):
    """Reporting success here would serve every AgentX round with the cache off and say nothing."""
    _runners(tmp_path, body="#!/bin/bash\nDEFAULT_ARGS='--mem-fraction-static=0.8'\n")

    assert mp.ensure_agentic_prefix_cache(None, tmp_path) is False


def test_a_tree_with_no_generic_script_has_nothing_to_patch(tmp_path):
    (tmp_path / "benchmarks").mkdir()

    assert mp.ensure_agentic_prefix_cache(None, tmp_path) is True


def test_preparing_an_agentx_run_keeps_its_prefix_cache(tmp_path, monkeypatch):
    """Baseline and profile shell out without the grid runner, so the patch has to live where both arrive.

    Called from the grid runner alone it never reached a baseline --- the very
    measurement every later round is compared against --- and a GLM-5.3 run
    served its replay with the cache off while reporting the patch applied.
    """
    from hyperloom.orchestrator.actions.executors import _workload_envs as we

    _runners(tmp_path)
    # Stop the AgentX asset deployment this function also drives; the scripts are what this pins.
    monkeypatch.setattr(we, "agentx_enabled", lambda *_a, **_k: True)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.agentx.runtime.maybe_prepare_agentx",
        lambda **_kwargs: None,
    )

    we.prepare_agentx_runtime(
        env=_mlperf_env(tmp_path), inferencex_path=str(tmp_path), active=True
    )

    for name in mp._AGENTIC_RADIX_SCRIPTS:
        assert mp._AGENTIC_RADIX_SENTINEL in (tmp_path / "benchmarks" / name).read_text(encoding="utf-8")


def test_a_non_agentx_run_keeps_the_harness_defaults(tmp_path, monkeypatch):
    """A fixed-sequence throughput sweep wants the cache off; only an agentic replay does not."""
    from hyperloom.orchestrator.actions.executors import _workload_envs as we

    _runners(tmp_path)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    we.prepare_agentx_runtime(
        env=_mlperf_env(tmp_path), inferencex_path=str(tmp_path), active=False
    )

    body = (tmp_path / "benchmarks" / "sglang_mi355x.sh").read_text(encoding="utf-8")
    assert mp._AGENTIC_RADIX_SENTINEL not in body
