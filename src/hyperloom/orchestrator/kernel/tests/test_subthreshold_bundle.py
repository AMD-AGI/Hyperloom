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


_FMOE_ENVS = {"AITER_CONFIG_FMOE": "/session/merged_fmoe.csv"}
_A8W8_ENVS = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "/session/merged_a8w8.csv"}


def _env_entry(tuner: str, envs: dict, server_args: str, cycle: int = 2) -> dict:
    return {
        "kind": "env",
        "id": f"gemm_tune_{tuner}",
        "envs": envs,
        "server_args": server_args,
        "gain_pct": 2.9,
        "macro_cycle": cycle,
    }


def _patch_entry(patch_dir: Path, repo: Path, name: str, cycle: int = 2) -> dict:
    return {
        "kind": "patch",
        "id": f"kernel:forge-loop:{name}:standalone:unknown:triton:mi355x",
        "repo_root": str(repo),
        "patch_path": str(patch_dir / "change.patch"),
        "kernel_path": f"{name}.py",
        "operator_name": name,
        "gain_pct": 2.7,
        "macro_cycle": cycle,
    }


def _jit_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    from hyperloom.orchestrator.actions.executors import _aiter_jit

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(_aiter_jit, "prepare_serving_so_for_csvs", lambda envs, **_kw: calls.append(("prepare", envs)))
    monkeypatch.setattr(_aiter_jit, "drop_serving_so_for_envs", lambda envs, **_kw: calls.append(("drop", envs)))
    return calls


@pytest.mark.asyncio
async def test_kernel_entry_measures_the_bundle_once_and_reverts_every_member(tmp_path, monkeypatch) -> None:
    from hyperloom.orchestrator.kernel import campaign_baseline, forge_handoff
    from hyperloom.orchestrator.phases import machine_state

    monkeypatch.setattr(machine_state, "phase_budget_remaining_seconds", lambda _state, **_kw: 3600.0)
    repo, base = _repo(tmp_path, ("mqa.py",))
    patch_dir = _publish(tmp_path / "attempt" / "result" / "patches", repo, base, "mqa")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    coordinator = _coordinator(session_dir, repo)
    state = coordinator.shared_state
    state.current_best = {"action": "gemm_tuning", "tput": 104.0}
    state.baseline_runtime_sec = 600.0
    state.kernel_bundle_pending = [
        _env_entry("fmoe_ck", _FMOE_ENVS, "--moe-runner-backend aiter"),
        _env_entry("a8w8_blockscale_bpreshuffle", _A8W8_ENVS, ""),
        _patch_entry(patch_dir, repo, "mqa"),
    ]
    seen_on_disk: list[str] = []

    async def _integrate(payload, *, session_dir, **_kwargs):
        calls.append(payload)
        seen_on_disk.append((repo / "mqa.py").read_text(encoding="utf-8"))
        return _revert(2.2, "p90_regressed")

    calls: list[dict] = []
    monkeypatch.setattr("hyperloom.orchestrator.kernel.request_handlers.integrate_handler", _integrate)
    jit = _jit_calls(monkeypatch)
    phase = coordinator.phase_kernel

    async def _nothing(*_args, **_kwargs):
        return None

    monkeypatch.setattr(KernelPhase, "_maybe_reprofile_for_kernel", _nothing)
    monkeypatch.setattr(KernelPhase, "_maybe_run_forge_fusion_before_kernel_opt", _nothing)
    monkeypatch.setattr(KernelPhase, "_run_kernel_rewrite_controller", _nothing)
    monkeypatch.setattr(campaign_baseline, "seal_campaign_baseline", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(forge_handoff, "write_forge_handoff", lambda *_args, handoff_dir, **_kwargs: handoff_dir)

    await phase._finish_kernel_entry()

    union = {**_FMOE_ENVS, **_A8W8_ENVS}
    assert [
        {
            key: payload[key]
            for key in ("kernel_id", "mode", "extra_envs", "extra_server_args", "base_tput", "preapplied_patches")
        }
        for payload in calls
    ] == [
        {
            "kernel_id": "kernel_bundle_c2",
            "mode": "env_only",
            "extra_envs": union,
            "extra_server_args": "--moe-runner-backend aiter",
            "base_tput": 104.0,
            "preapplied_patches": [str(patch_dir / "change.patch")],
        }
    ]
    assert seen_on_disk == ["VALUE = 2\n"]
    assert jit == [("prepare", union), ("drop", union)]
    assert (repo / "mqa.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert _git(repo, "rev-parse", "HEAD") == base
    assert state.optimization_stack == []
    assert SharedState.load_or_init(session_dir).kernel_bundle_pending == []


@pytest.mark.asyncio
async def test_a_kept_bundle_lands_as_one_stack_entry_with_its_envs_args_and_source_layer(
    tmp_path, monkeypatch
) -> None:
    from hyperloom.orchestrator.kernel.controller_patch_integration import KeepSession
    from hyperloom.orchestrator.kernel.subthreshold_bundle import validate_subthreshold_bundle

    repo, base = _repo(tmp_path, ("mqa.py",))
    patch_dir = _publish(tmp_path / "attempt" / "result" / "patches", repo, base, "mqa")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    coordinator = _coordinator(session_dir, repo)
    state = coordinator.shared_state
    patch_member = _patch_entry(patch_dir, repo, "mqa")
    state.kernel_bundle_pending = [_env_entry("fmoe_ck", _FMOE_ENVS, "--moe-runner-backend aiter"), patch_member]
    jit = _jit_calls(monkeypatch)

    async def _measure(_payload):
        return {
            "status": "ok",
            "decision": "KEEP",
            "base_tput": 100.0,
            "new_tput": 113.0,
            "gain_pct": 13.1,
            "extra_server_args": "--moe-runner-backend aiter",
            "extra_envs": _FMOE_ENVS,
            "bench_result": {"output_throughput": 113.0},
        }

    await validate_subthreshold_bundle(
        KeepSession(session_dir, state, coordinator.writeback.record_integrate_keep),
        measure=_measure,
        round_stop=lambda: None,
    )

    head = _git(repo, "rev-parse", "HEAD")
    keys = (
        "action",
        "variant_name",
        "kernel_id",
        "candidate_extra_server_args",
        "extra_envs",
        "stack_kernel_ids",
        "scope",
        "base_sha",
        "keep_commit",
        "framework_root",
        "target_files",
        "gain_pct",
    )
    assert [{key: entry.get(key) for key in keys} for entry in state.optimization_stack] == [
        {
            "action": "integrate",
            "variant_name": "kernel_bundle_c2",
            "kernel_id": "kernel_bundle_c2",
            "candidate_extra_server_args": "--moe-runner-backend aiter",
            "extra_envs": _FMOE_ENVS,
            "stack_kernel_ids": ["gemm_tune_fmoe_ck", patch_member["id"]],
            "scope": "source_patch",
            "base_sha": base,
            "keep_commit": head,
            "framework_root": str(repo),
            "target_files": ["mqa.py"],
            "gain_pct": 13.1,
        }
    ]
    assert Path(state.optimization_stack[0]["source_snapshot"]).is_dir()
    assert state.current_best["tput"] == 113.0
    assert _git(repo, "log", "-1", "--format=%P") == base
    assert (repo / "mqa.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert jit == [("prepare", _FMOE_ENVS)]
    assert SharedState.load_or_init(session_dir).kernel_bundle_pending == []


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["one_member_this_cycle", "no_round_fits"])
async def test_a_bundle_that_cannot_be_measured_is_cleared_without_a_round(tmp_path, monkeypatch, case) -> None:
    from hyperloom.orchestrator.actions.stop_attribution import PHASE_TIME_EXHAUSTED_CLASS, STOPPED_BY_THE_RUN
    from hyperloom.orchestrator.kernel.controller_patch_integration import KeepSession
    from hyperloom.orchestrator.kernel.subthreshold_bundle import validate_subthreshold_bundle

    repo, base = _repo(tmp_path, ("mqa.py",))
    patch_dir = _publish(tmp_path / "attempt" / "result" / "patches", repo, base, "mqa")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    state = _coordinator(session_dir, repo).shared_state
    stale_cycle = 1 if case == "one_member_this_cycle" else 2
    state.kernel_bundle_pending = [
        _env_entry("fmoe_ck", _FMOE_ENVS, "--moe-runner-backend aiter"),
        _patch_entry(patch_dir, repo, "mqa", cycle=stale_cycle),
    ]
    jit = _jit_calls(monkeypatch)
    calls: list[dict] = []

    async def _measure(payload):
        calls.append(payload)
        return {}

    stop = STOPPED_BY_THE_RUN[PHASE_TIME_EXHAUSTED_CLASS] if case == "no_round_fits" else None
    await validate_subthreshold_bundle(
        KeepSession(session_dir, state, lambda _result: None), measure=_measure, round_stop=lambda: stop
    )

    assert calls == []
    assert jit == []
    assert state.kernel_bundle_pending == []
    assert SharedState.load_or_init(session_dir).kernel_bundle_pending == []
    assert (repo / "mqa.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_an_env_only_round_over_preapplied_patches_is_not_refused_for_carrying_no_envs() -> None:
    from hyperloom.orchestrator.kernel._integrate_outcomes import _integrate_request_refusal

    bare = {"base_tput": 100.0, "mode": "env_only", "extra_envs": {}, "extra_server_args": ""}

    assert [
        _integrate_request_refusal(bare, requested_controls=False),
        _integrate_request_refusal(
            {**bare, "preapplied_patches": ["/attempt/mqa/change.patch"]}, requested_controls=False
        ),
    ] == [
        {
            "status": "failed",
            "error_class": "env_only_missing_envs",
            "error": "env_only integrate requires runtime configuration or an explicit paired reference",
        },
        None,
    ]
