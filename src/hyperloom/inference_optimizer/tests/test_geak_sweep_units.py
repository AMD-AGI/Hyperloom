# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the GEAK post-optimization sweep and the sweep/kernel helpers
it shares with the native concurrency sweep.

These cover the ``_geak_sweep.sweep_via_geak`` branches not exercised by
``test_geak_breakdown_unit`` -- the ``validated_regimes`` protocol fallback (used
when ``result`` carries no explicit ``bench_protocol``), the ``pin_num_prompts``
single-point replay, and the per-variant subprocess-error path -- plus the pure
row/flatten helpers the GEAK sweep result feeds into downstream.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import _geak_sweep
from hyperloom.orchestrator.actions.executors._geak_sweep import sweep_via_geak
from hyperloom.orchestrator.actions.executors._grid_base import coerce_extra_envs
from hyperloom.orchestrator.actions.executors._grid_runner import VariantResult
from hyperloom.orchestrator.kernel.attempt_summary import _backend_results_dir
from hyperloom.orchestrator.kernel.conc_sweep import (
    _budget_limited_without_valid_pair,
    _point_from_variant,
)
from hyperloom.orchestrator.loop.coordinator_helpers import _parse_server_arg_value


def _bench_script(tmp_path: Path) -> Path:
    """A ``bench_e2e.sh`` stub that records NUM_PROMPTS and reports throughput."""
    bench = tmp_path / "bench_e2e.sh"
    bench.parent.mkdir(parents=True, exist_ok=True)
    bench.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_DIR"])
summary = {
    "output_throughput_tok_s_median": 200.0,
    "ttft_ms_median": 10.0,
    "tpot_ms_median": 3.0,
    "e2el_ms_median": 50.0,
}
(out / "bench_summary.json").write_text(json.dumps(summary), encoding="utf-8")
(out / "env.json").write_text(
    json.dumps({"NUM_PROMPTS": os.environ.get("NUM_PROMPTS")}), encoding="utf-8"
)
PY
""",
        encoding="utf-8",
    )
    return bench


@pytest.mark.asyncio
async def test_sweep_via_geak_uses_validated_regimes_and_pins_num_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``bench_protocol`` -> fall back to the first ``validated_regimes`` entry;
    ``pin_num_prompts`` forwards that regime's NUM_PROMPTS onto the point."""
    bench = _bench_script(tmp_path)
    monkeypatch.setenv("MODEL_PATH", "/models/x")
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("TP", "1")

    def _fake_run(_cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        env = kwargs["env"]
        out = Path(env["OUT_DIR"])
        summary = {
            "output_throughput_tok_s_median": 200.0,
            "ttft_ms_median": 10.0,
            "tpot_ms_median": 3.0,
            "e2el_ms_median": 50.0,
        }
        (out / "bench_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (out / "env.json").write_text(json.dumps({"NUM_PROMPTS": env.get("NUM_PROMPTS")}), encoding="utf-8")
        (out / "server.log").write_text(
            "server_args=ServerArgs(model_path='/models/x', tp_size=1, context_length=4096)\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(_cmd, 0, "", "")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _fake_run)

    result = await sweep_via_geak(
        result={
            "bench_script": str(bench),
            "output_dir": str(tmp_path),
            # No ``bench_protocol`` -> validated_regimes[0] fallback.
            "validated_regimes": [
                {"num_warmups": 3, "seed": 7, "num_prompts": 64},
                {"num_warmups": 9},
            ],
            "accepted_config": {"flags": "", "env": ""},
        },
        conc_values=[1],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=30,
        repeats=1,
        pin_num_prompts=True,
    )

    assert result["status"] == "succeeded"
    out_dir = tmp_path / "sweep" / "variant_0_conc1_isl16_osl16"
    env = json.loads((out_dir / "env.json").read_text(encoding="utf-8"))
    assert env["NUM_PROMPTS"] == "64"
    evidence = result["promotion_measurement"]["launch_evidence"]
    assert result["promotion_measurement"]["server_log_path"] == str(out_dir / "server.log")
    assert evidence["actual_server_log_path"] == str(out_dir / "server.log")
    assert evidence["observed_server_identity"] == {
        "context_length": 4096,
        "model_path": "/models/x",
        "tp_size": 1,
    }


@pytest.mark.asyncio
async def test_sweep_via_geak_prefers_executable_final_launch_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2a runs the GEAK final script without changing its single-point protocol."""
    bench = _bench_script(tmp_path)
    final = tmp_path / "final_launch.sh"
    final.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    final.chmod(0o755)
    monkeypatch.setenv("MODEL_PATH", "/models/x")
    captured: dict[str, Any] = {}

    def _fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        captured["command"] = command
        captured["env"] = kwargs["env"]
        out = Path(command[2])
        (out / "bench_summary.json").write_text(
            json.dumps({"output_throughput_tok_s_median": 200.0}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _fake_run)
    result = await sweep_via_geak(
        result={
            "bench_script": str(bench),
            "final_launch_script": str(final),
            "accepted_config": {},
        },
        conc_values=[1],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=30,
        repeats=3,
    )

    out_dir = tmp_path / "sweep" / "variant_0_conc1_isl16_osl16"
    assert result["replay_mode"] == "final_launch_script"
    assert captured["command"] == ["bash", str(final), str(out_dir)]
    assert captured["env"]["REPLICAS"] == "3"


@pytest.mark.asyncio
async def test_sweep_via_geak_marks_variant_failed_on_subprocess_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising ``subprocess.run`` is caught and recorded as a failed variant."""
    bench = _bench_script(tmp_path)
    monkeypatch.setenv("MODEL_PATH", "/models/x")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("cannot spawn bench process")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _boom)

    result = await sweep_via_geak(
        result={"bench_script": str(bench), "output_dir": str(tmp_path), "accepted_config": {}},
        conc_values=[1],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=30,
    )

    assert result["status"] == "failed"
    entry = result["sweep_grid"][0]
    assert entry["status"] == "failed"
    assert "cannot spawn bench process" in entry["error"]


def test_point_from_variant_defaults_conc_zero_on_bad_env() -> None:
    """A non-numeric CONC env coerces the row's ``conc`` to 0 rather than raising."""
    v = VariantResult(
        name="optimized_concX",
        extra_server_args="",
        extra_envs={"CONC": "not-a-number"},
        status="failed",
    )
    point = _point_from_variant(v, arm="optimized")
    assert point["conc"] == 0
    assert point["arm"] == "optimized"
    assert point["status"] == "failed"


def test_budget_limited_without_valid_pair_paths() -> None:
    """Empty-points guard, a genuine failure short-circuit, and the budget-skip case."""
    summary_no_pairs = {"successful_pairs": 0}
    # No points at all -> not attributable to budget gating.
    assert (
        _budget_limited_without_valid_pair(
            budget_exhausted=True,
            summary=summary_no_pairs,
            baseline_points=[],
            optimized_points=[],
        )
        is False
    )
    # A genuinely-failed (non budget) point -> not budget gating.
    assert (
        _budget_limited_without_valid_pair(
            budget_exhausted=True,
            summary=summary_no_pairs,
            baseline_points=[{"status": "failed", "error_class": "crash"}],
            optimized_points=[],
        )
        is False
    )
    # Every remaining point was a budget skip -> budget-limited.
    assert (
        _budget_limited_without_valid_pair(
            budget_exhausted=True,
            summary=summary_no_pairs,
            baseline_points=[{"status": "skipped", "error_class": "budget_exhausted"}],
            optimized_points=[],
        )
        is True
    )


def test_backend_results_dir_keyed_and_single_subdir(tmp_path: Path) -> None:
    """Resolve the results dir by session key, and via the lone-subdir fallback."""
    from hyperloom.inference_optimizer.session.session_paths import (
        kernel_agent_runs_root,
    )

    # Keyed by ``session_dir.name``.
    sd = tmp_path / "sess-A"
    runs = kernel_agent_runs_root(sd)
    (runs / sd.name / "results").mkdir(parents=True)
    assert _backend_results_dir(sd, "") == runs / sd.name / "results"

    # Migrated-key recovery: a single subdir under the runs root.
    sd2 = tmp_path / "sess-B"
    runs2 = kernel_agent_runs_root(sd2)
    (runs2 / "migrated-key" / "results").mkdir(parents=True)
    assert _backend_results_dir(sd2, "") == runs2 / "migrated-key" / "results"

    # Ambiguous (no keyed match, not exactly one subdir) -> None.
    sd3 = tmp_path / "sess-C"
    runs3 = kernel_agent_runs_root(sd3)
    (runs3 / "one").mkdir(parents=True)
    (runs3 / "two").mkdir(parents=True)
    assert _backend_results_dir(sd3, "") is None


def test_coerce_extra_envs_skips_malformed_tokens() -> None:
    """The GEAK/sweep env coercion drops empty tokens and empty keys in both the
    shell-string and token-list shapes rather than emitting junk keys."""
    # Shell-string shape: leading separator -> empty token; ``=v`` -> empty key.
    assert coerce_extra_envs("; =v FOO=1") == {"FOO": "1"}
    # Token-list shape: dict item with a None key, a token without ``=``, a
    # non-string item, and an empty-key ``=v`` are all skipped.
    assert coerce_extra_envs([{None: "x", "A": "1"}, "noeq", 123, "=v", "B=2"]) == {"A": "1", "B": "2"}


def test_parse_server_arg_value_falls_back_on_unbalanced_quotes() -> None:
    """The GEAK handoff recovers a flag value even when the server-args string is
    not shlex-parseable (unbalanced quote -> plain ``str.split`` fallback)."""
    got = _parse_server_arg_value('--max-model-len 4096 "unbalanced', "--max-model-len")
    assert got == "4096"
    # ``--flag=value`` form is also handled.
    assert _parse_server_arg_value("--gpu-memory-utilization=0.9", "--gpu-memory-utilization") == "0.9"
    # Absent flag -> None.
    assert _parse_server_arg_value("--tp 8", "--max-model-len") is None
