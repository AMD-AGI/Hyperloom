# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preparing and running the SGLang dense BF16 tuner against fixture rows, shared by its tests and the err-ratio filter tests."""

from __future__ import annotations

from pathlib import Path

from kernelforge.workflows.gemm_tune.model_analyzer import ModelProfile
from kernelforge.workflows.gemm_tune.script_probe import ScriptSurface
from kernelforge.workflows.gemm_tune.tuners import sglang_dense_bf16 as sd
from kernelforge.workflows.gemm_tune.tuners.base import TuneContext


# Real header written by csrc/gemm_a16w16/gemm_a16w16_tune.py for both the tuned (-o) and the full-candidate profile
# (-o2) CSV.
_HDR = (
    "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw"
)

_NK = [(4096, 4096)]
_M = [1, 512]


def _csv(rows):
    return "\n".join([_HDR, *rows]) + "\n"


def _aiter_root(tmp_path, *, direct=True, shim=False, gradlib=False) -> Path:
    root = tmp_path / "aiter"
    targets = []
    if direct:
        targets.append(root / "csrc" / "gemm_a16w16" / "gemm_a16w16_tune.py")
    if shim:
        targets.append(root / "csrc" / "gemm_a16w16" / "gemm_tuner.py")
    if gradlib:
        targets.append(root / "gradlib" / "gradlib" / "gemm_tuner.py")
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text("# stub", encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ctx(tmp_path, **overrides) -> TuneContext:
    base = dict(
        profile=ModelProfile(
            model_path="/fake",
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
        ),
        framework="sglang",
        precision="bf16",
        quant_type="none",
        gpu_type="mi355x",
        tp=1,
        conc=64,
        tokens=[1, 512],
        mp=1,
        output_dir=tmp_path,
        iters=20,
        warmup=5,
        min_improvement_pct=1.0,
        timeout_s=3600,
    )
    base.update(overrides)
    return TuneContext(**base)


def _permissive_surface(script):
    """What probe_script returns when --help could not be read: veto nothing."""
    return ScriptSurface(str(script), frozenset(), False)


def _prep(tmp_path, monkeypatch, *, tuned_rows, profile_rows=(), rc=1, root=None, stderr="", surface=None):
    """Wire the tuner so run() executes without aiter, and capture its argv."""
    monkeypatch.setattr(sd, "probe_script", surface or _permissive_surface)
    monkeypatch.setattr(sd, "resolve_aiter_root", lambda: root or _aiter_root(tmp_path))
    monkeypatch.setattr(sd, "_compute_nk_shapes", lambda geometry, tp: list(_NK))
    monkeypatch.setattr(sd, "_compute_m_values", lambda conc, thorough=False: list(_M))
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        if tuned_rows is not None:
            Path(cmd[cmd.index("-o") + 1]).write_text(_csv(tuned_rows), encoding="utf-8")
            Path(cmd[cmd.index("-o2") + 1]).write_text(_csv(profile_rows), encoding="utf-8")
        return rc, "", stderr

    monkeypatch.setattr(sd, "run_subprocess", _fake_run)
    return captured


def _run(tmp_path, **ctx_kwargs):
    return sd.SglangDenseBf16Tuner(_ctx(tmp_path, **ctx_kwargs)).run()
