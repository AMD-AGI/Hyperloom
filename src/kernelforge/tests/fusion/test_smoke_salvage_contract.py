# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a killed or env-blocked serving smoke owes the caller.

Two promises the gate makes, both of which need the SMOKE to say what happened
rather than the gate guessing from a message:

* A micro KEEP survives a smoke that never judged the kernel, and it survives it
  WITH an applicable patch -- including on a plain pip install, where ``git diff``
  yields nothing and only a pristine-snapshot diff can produce one.
* Only a GPU fault discards a KEEP. A boot-time HIP OOM and an HTTP probe error
  are the environment failing, and clearing the patch for them throws away a
  kernel that parity and the microbench both passed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from kernelforge.fusion import command as cli_module
from kernelforge.fusion.models import FusionArtifacts, ValidationResult
from kernelforge.fusion.validate import (
    SMOKE_STAGE_DECODE_CRASH,
    SMOKE_STAGE_DECODE_PROBE,
    SMOKE_STAGE_STARTUP_CRASH,
    SmokeVerdict,
)

# A boot that dies on memory, reported the way ROCm reports it.
OOM_BOOT_LOG = """
(EngineCore pid=1) INFO 08-15 15:25:16 [core.py:114] Initializing a V1 LLM engine
(EngineCore pid=1) ERROR 08-15 15:32:33 [core.py:1231] RuntimeError: HIP error: out of memory
"""

FAULT_DECODE_LOG = """
Application startup complete.
Memory access fault by GPU node-1 on address 0x7f0000000000
"""


def _pip_install(tmp_path):
    """A framework tree that is NOT a git checkout, as pip leaves it."""
    repo = tmp_path / "site-packages"
    (repo / "vllm" / "model_executor" / "models").mkdir(parents=True)
    source = repo / "vllm" / "model_executor" / "models" / "minimax.py"
    source.write_text("def forward(x):\n    return norm(x)\n", encoding="utf-8")
    return repo, source


def _kept_result(source_file: str):
    vr = ValidationResult(
        correctness_passed=True,
        max_abs_err=0.0,
        rtol=0.02,
        kernel_speedup=2.5,
        eager_us=100.0,
        fused_us=40.0,
        kept=True,
        note="KERNEL OK",
    )
    result = SimpleNamespace(
        kept=True,
        best=vr,
        best_recipe=SimpleNamespace(
            env_flag="MINIMAX_FUSED",
            pattern_id="llm:rmsnorm",
            source_file=str(source_file),
        ),
        termination_reason="",
    )
    return result, vr


def _run_gate(monkeypatch, tmp_path, verdict, *, repo_root="", pristine_dir="", source=None):
    """Drive the gate against one smoke verdict; report what the smoke saw on disk."""
    monkeypatch.setattr(cli_module, "_serving_check_enabled", lambda: True)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    observed: dict[str, object] = {}

    def fake_smoke(*_a, **_k):
        patch = out / "fusion.patch"
        observed["patch"] = patch.read_text(encoding="utf-8") if patch.is_file() else ""
        observed["checkpoint"] = (out / cli_module.KERNEL_KEEP_CHECKPOINT).is_file()
        return verdict

    monkeypatch.setattr(cli_module, "serving_smoke_verdict", fake_smoke)
    result, vr = _kept_result(source or "/m.py")
    cli_module.apply_serving_gate(
        result,
        framework="vllm",
        out=out,
        gpu="0",
        model_path="/models/minimax",
        isl=8,
        osl=8,
        repo_root=repo_root,
        pristine_dir=pristine_dir,
        tp=8,
        block_size=128,
        max_model_len=13312,
    )
    return result, vr, out, observed


def test_a_non_git_install_hands_over_a_patch_when_the_smoke_is_killed(monkeypatch, tmp_path):
    """The salvage target: a pip framework must still produce fusion.patch.

    ``export_artifacts`` has no git to diff here, so without the pristine
    snapshot it returns empty and a process killed during the smoke leaves a
    checkpoint pointing at nothing.
    """
    repo, source = _pip_install(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pristine = cli_module._snapshot_fusion_source(str(repo), str(source), out)
    assert pristine, "snapshot precondition"
    source.write_text("def forward(x):\n    return fused_norm(x)  # fused\n", encoding="utf-8")

    _result, _vr, out, observed = _run_gate(
        monkeypatch,
        tmp_path,
        SmokeVerdict(ok=True, reason="serving smoke ok", stage="ok"),
        repo_root=str(repo),
        pristine_dir=pristine,
        source=source,
    )

    assert "fused_norm" in observed["patch"], "a killed smoke would have no patch to hand over"
    assert observed["checkpoint"] is True


def test_the_checkpoint_is_only_written_once_a_patch_exists(monkeypatch, tmp_path):
    """The checkpoint is the completion marker, so it must not precede the patch."""
    repo, source = _pip_install(tmp_path)

    # No pristine snapshot and no git: nothing can be exported.
    _result, _vr, out, observed = _run_gate(
        monkeypatch,
        tmp_path,
        SmokeVerdict(ok=True, reason="serving smoke ok", stage="ok"),
        repo_root=str(repo),
        pristine_dir="",
        source=source,
    )

    assert observed["patch"] == ""
    assert observed["checkpoint"] is False
    assert not (out / cli_module.KERNEL_KEEP_CHECKPOINT).exists()


def test_a_stale_patch_cannot_complete_the_current_checkpoint(monkeypatch, tmp_path):
    """Only the patch returned by THIS export can authorize a checkpoint."""
    out = tmp_path / "out"
    out.mkdir()
    stale_patch = out / "fusion.patch"
    stale_checkpoint = out / cli_module.KERNEL_KEEP_CHECKPOINT
    stale_patch.write_text("OLD KERNEL\n", encoding="utf-8")
    stale_checkpoint.write_text('{"kept": true, "pattern_id": "old"}', encoding="utf-8")
    state_seen_by_export: dict[str, bool] = {}

    def empty_export(*_args, **_kwargs):
        state_seen_by_export["patch"] = stale_patch.exists()
        state_seen_by_export["checkpoint"] = stale_checkpoint.exists()
        return FusionArtifacts()

    monkeypatch.setattr(cli_module, "export_artifacts", empty_export)

    exported = cli_module._export_salvage_patch(
        out,
        "/site-packages/vllm/model.py",
        repo_root="/site-packages",
        pristine_dir=str(out / ".pristine"),
    )

    assert state_seen_by_export == {"patch": False, "checkpoint": False}
    assert exported is False
    assert not stale_patch.exists()
    assert not stale_checkpoint.exists()


def test_a_boot_time_oom_keeps_the_kernel_and_its_patch(monkeypatch, tmp_path):
    """A server that never served cannot be evidence against the kernel."""
    repo, source = _pip_install(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pristine = cli_module._snapshot_fusion_source(str(repo), str(source), out)
    source.write_text("def forward(x):\n    return fused_norm(x)\n", encoding="utf-8")

    result, vr, out, _observed = _run_gate(
        monkeypatch,
        tmp_path,
        SmokeVerdict(
            ok=False,
            reason="server exited rc=1 before ready: RuntimeError: HIP error: out of memory",
            stage=SMOKE_STAGE_STARTUP_CRASH,
            blames_kernel=False,
        ),
        repo_root=str(repo),
        pristine_dir=pristine,
        source=source,
    )

    assert result.kept is True
    assert vr.kept is True
    assert vr.kernel_speedup == 2.5
    assert result.termination_reason == "serving_unconfirmed"
    assert (out / "fusion.patch").is_file()
    ckpt = out / cli_module.KERNEL_KEEP_CHECKPOINT
    assert json.loads(ckpt.read_text(encoding="utf-8"))["kept"] is True


def test_a_probe_error_keeps_the_kernel_and_its_patch(monkeypatch, tmp_path):
    """An HTTP probe that could not ask the question did not get an answer."""
    repo, source = _pip_install(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pristine = cli_module._snapshot_fusion_source(str(repo), str(source), out)
    source.write_text("def forward(x):\n    return fused_norm(x)\n", encoding="utf-8")

    result, vr, out, _observed = _run_gate(
        monkeypatch,
        tmp_path,
        SmokeVerdict(
            ok=False,
            reason="decode probe failed: /v1/models probe error: OSError: boom",
            stage=SMOKE_STAGE_DECODE_PROBE,
            blames_kernel=False,
        ),
        repo_root=str(repo),
        pristine_dir=pristine,
        source=source,
    )

    assert result.kept is True
    assert vr.kernel_speedup == 2.5
    assert (out / "fusion.patch").is_file()
    assert (out / cli_module.KERNEL_KEEP_CHECKPOINT).is_file()


def test_a_gpu_fault_in_decode_discards_the_keep_and_the_patch(monkeypatch, tmp_path):
    """The one failure that IS about the kernel still reverts, and cleans up."""
    repo, source = _pip_install(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pristine = cli_module._snapshot_fusion_source(str(repo), str(source), out)
    source.write_text("def forward(x):\n    return fused_norm(x)\n", encoding="utf-8")

    result, vr, out, _observed = _run_gate(
        monkeypatch,
        tmp_path,
        SmokeVerdict(
            ok=False,
            reason="scheduler crashed during CUDA-graph decode: Memory access fault by GPU node-1",
            stage=SMOKE_STAGE_DECODE_CRASH,
            blames_kernel=True,
        ),
        repo_root=str(repo),
        pristine_dir=pristine,
        source=source,
    )

    assert result.kept is False
    assert vr.kept is False
    assert vr.kernel_speedup is None
    assert result.termination_reason == "serving_crash"
    assert not (out / "fusion.patch").exists()
    assert not (out / cli_module.KERNEL_KEEP_CHECKPOINT).exists()
