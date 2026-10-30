# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel wins the AgentX P50 threshold alone refused are deferred into one bundled E2E.

On GLM-5.3 a fmoe_ck tune (+2.97%), an a8w8 tune (+2.77%) and a Controller rewrite (+2.7%) were each discarded, and
the same pod measured them together at +13.1%.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller.paths import operator_directory_name
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.kernel import KernelPhase
from hyperloom.orchestrator.roles import MockBackend, MockCriticBackend, ScriptedPlan
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.shared_state import SharedState

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "bundle-test",
    "GIT_AUTHOR_EMAIL": "bundle-test@local",
    "GIT_COMMITTER_NAME": "bundle-test",
    "GIT_COMMITTER_EMAIL": "bundle-test@local",
}
_P50 = "p50_gain_below_threshold"


@pytest.fixture(autouse=True)
def _isolated_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "user_data"))


def _revert(gain_pct: float, refused_by: str, **fields) -> dict:
    return {"status": "ok", "decision": "REVERT", "gain_pct": gain_pct, "refused_by": refused_by, **fields}


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, **_GIT_IDENTITY}
    return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path: Path, files: tuple[str, ...]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    for name in files:
        (repo / name).write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


def _patch(repo: Path, relative: str, content: str) -> str:
    path = repo / relative
    original = path.read_text(encoding="utf-8")
    path.write_text(content, encoding="utf-8")
    patch = _git(repo, "diff", "--binary", "--", relative)
    path.write_text(original, encoding="utf-8")
    return patch + "\n"


def _publish(patches_root: Path, repo: Path, base: str, name: str) -> Path:
    operator_id = f"kernel:forge-loop:{name}:standalone:unknown:triton:mi355x"
    patch_dir = patches_root / operator_directory_name(operator_id)
    patch_dir.mkdir(parents=True)
    (patch_dir / "change.patch").write_text(_patch(repo, f"{name}.py", "VALUE = 2\n"), encoding="utf-8")
    (patch_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    publication = {
        "schema_version": 2,
        "operator_id": operator_id,
        "identity": {
            "producer": "forge-loop",
            "kernel_name": name,
            "framework": "standalone",
            "framework_version": "unknown",
            "backend": "triton",
            "gpu": "mi355x",
        },
        "base_commit": base,
        "best_commit": "b" * 40,
        "repo_root": str(repo),
        "kernel_path": f"{name}.py",
        "operator_name": name,
        "micro_validated": True,
        "manifest": {"changed_files": [f"{name}.py"]},
    }
    (patch_dir / "publication.json").write_text(json.dumps(publication), encoding="utf-8")
    return patch_dir


def _coordinator(session_dir: Path, repo: Path) -> Coordinator:
    silent = ScriptedPlan(
        turns=[], default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})
    )
    coordinator = Coordinator(
        session_dir,
        backends={"orchestration": MockBackend(silent, name="orch"), "critic": MockCriticBackend()},
    )
    state = coordinator.shared_state
    state.baseline_tput = 100.0
    state.current_best = {"action": "baseline", "tput": 100.0}
    state.framework_repo_path = str(repo)
    state.macro_cycle = 2
    state.save(session_dir)
    return coordinator


def _stub_integrate(monkeypatch: pytest.MonkeyPatch, verdicts: dict[str, dict]) -> list[dict]:
    calls: list[dict] = []

    async def _fake(payload, *, session_dir, **_kwargs):
        calls.append(payload)
        return verdicts[payload["kernel_id"]]

    monkeypatch.setattr("hyperloom.orchestrator.kernel.request_handlers.integrate_handler", _fake)
    return calls


def _tuner_csv(tmp_path: Path, tuner: str) -> str:
    csv = tmp_path / f"candidate_{tuner}.csv"
    csv.write_text("gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n", encoding="utf-8")
    return str(csv)


@pytest.mark.asyncio
async def test_gemm_e2e_defers_only_candidates_the_p50_threshold_alone_refused(tmp_path, monkeypatch) -> None:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        model_path=str(tmp_path / "absent"), tp=4, framework="sglang", baseline_tput=100.0, macro_cycle=3
    )
    monkeypatch.setattr(KernelPhase, "_serving_log_text", lambda self: "INFO server started\n")
    monkeypatch.setattr(KernelPhase, "_merge_gemm_candidate_with_runtime", lambda self, env_var, env_value: env_value)
    envs = {
        "fmoe_ck": {"AITER_CONFIG_FMOE": _tuner_csv(tmp_path, "fmoe_ck")},
        "a8w8_blockscale_bpreshuffle": {
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": _tuner_csv(tmp_path, "a8w8_blockscale_bpreshuffle")
        },
        "dense_bf16": {"AITER_CONFIG_GEMM_BF16": _tuner_csv(tmp_path, "dense_bf16")},
    }
    _stub_integrate(
        monkeypatch,
        {
            "gemm_tune_fmoe_ck": _revert(2.97, _P50),
            "gemm_tune_a8w8_blockscale_bpreshuffle": _revert(2.77, "p90_regressed"),
            "gemm_tune_dense_bf16": _revert(2.5, _P50, veto_reason="ttft p90 regressed"),
        },
    )
    result = {"candidates": [{"tuner": tuner, "env": env} for tuner, env in envs.items()]}

    await coord.phase_kernel._validate_gemm_tuning_e2e(result)

    assert coord.shared_state.kernel_bundle_pending == [
        {
            "kind": "env",
            "id": "gemm_tune_fmoe_ck",
            "envs": envs["fmoe_ck"],
            "server_args": "--moe-runner-backend aiter",
            "gain_pct": 2.97,
            "macro_cycle": 3,
        }
    ]
    assert [row["tuner"] for row in result["e2e_results"]["reverted"]] == list(envs)


@pytest.mark.asyncio
async def test_controller_integration_defers_only_patches_the_p50_threshold_alone_refused(
    tmp_path, monkeypatch
) -> None:
    repo, base = _repo(tmp_path, ("mqa.py", "rope.py", "norm.py"))
    patches = tmp_path / "attempt" / "result" / "patches"
    dirs = {name: _publish(patches, repo, base, name) for name in ("mqa", "rope", "norm")}
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    coordinator = _coordinator(session_dir, repo)
    operator = "kernel:forge-loop:{}:standalone:unknown:triton:mi355x".format
    _stub_integrate(
        monkeypatch,
        {
            operator("mqa"): _revert(2.7, _P50),
            operator("rope"): _revert(2.9, "p90_regressed"),
            operator("norm"): _revert(2.4, _P50, veto_reason="output tput regressed"),
        },
    )

    await coordinator.phase_kernel._integrate_controller_output(
        {"patch_count": 3, "patches_root": str(patches)}, tmp_path / "attempt"
    )

    expected = [
        {
            "kind": "patch",
            "id": operator("mqa"),
            "repo_root": str(repo),
            "patch_path": str(dirs["mqa"] / "change.patch"),
            "kernel_path": "mqa.py",
            "operator_name": "mqa",
            "gain_pct": 2.7,
            "macro_cycle": 2,
        }
    ]
    assert coordinator.shared_state.kernel_bundle_pending == expected
    assert SharedState.load_or_init(session_dir).kernel_bundle_pending == expected
    assert [(repo / f"{name}.py").read_text(encoding="utf-8") for name in dirs] == ["VALUE = 1\n"] * 3
