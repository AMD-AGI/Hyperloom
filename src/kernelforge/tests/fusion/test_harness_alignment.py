# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The gate that stops a campaign benching an implementation nobody runs.

Every number a campaign produces -- parity, speedup, both launch counts -- is
relative to whatever the harness's eager arm calls, and nothing downstream
re-examines that choice. These tests cover the one place it can still be caught:
between authoring the harness and starting the loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import kernelforge.fusion.command as cli
from kernelforge.fusion.models import Recipe

_ANCHOR = "void bf16_to_fp32_copy(int)"
_AFTER = "void sglang::flash_c4_prefill<512l>(Params)"
_DEAD = "triton_hip_compress_forward"


def _recipe() -> Recipe:
    return Recipe(
        pattern_id="compress_prologue",
        description="Drop the fp32 cast into the compress prologue.",
        env_flag="SGLANG_FUSED_COMPRESS",
        source_file="/sgl/models/dsv4.py",
        source_hints=["linear_bf16_fp32("],
        fusion_math="compress(norm(gemm(x)))",
        eager_reference_hint="Import the real compress op.",
        shapes={"T": 16},
        matched_categories=["elementwise"],
        trigger_share=0.3,
        trace_kernels={"anchor": _ANCHOR, "before": [], "after": [_AFTER], "span": [_ANCHOR, _AFTER]},
    )


def _harness_src(kernels: list[str] | None) -> str:
    """A harness reporting the kernels its eager arm launched, or reporting none.

    ``None`` is the harness that omits the fields altogether -- it says nothing
    about which path it ran, which is different from saying it ran the wrong one.
    """
    payload = {
        "compiled": True,
        "is_triton": True,
        "error": "",
        "parity": [{"snr_db": 45.0, "max_abs_err": 1e-3, "label": "T16"}],
        "eager_us": 100.0,
        "fused_us": 100.0,
        "eager_launches": 4,
        "fused_launches": 4,
        "skipped": False,
        "skip_reason": "",
    }
    if kernels is not None:
        payload["eager_kernels"] = kernels
        payload["eager_matches_trace"] = _ANCHOR in kernels
    return "print(%r)\n" % json.dumps(payload)


class _Author:
    """Stands in for the agent: writes one scripted harness per attempt."""

    def __init__(self, *scripts: str):
        self.scripts = list(scripts)
        self.prompts: list[str] = []

    def __call__(self, prompt, *, workdir, log_path, gpu, model, max_turns, backend, timeout_s, target_files, **kw):
        self.prompts.append(prompt)
        script = self.scripts[min(len(self.prompts), len(self.scripts)) - 1]
        Path(target_files[0]).write_text(script, encoding="utf-8")
        return 0


def _run(tmp_path: Path, monkeypatch, author: _Author) -> tuple[bool, str]:
    monkeypatch.setattr(cli, "run_author", author)
    monkeypatch.setattr(cli, "_agent_timeout_sec", lambda: 60)
    out = tmp_path / "out"
    out.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    return cli._author_baseline_harness(
        _recipe(),
        harness_path=str(out / "kernel_harness.py"),
        repo_root=str(repo),
        out=out,
        gpu="0",
        llm_model=None,
        max_turns=10,
        backend=None,
    )


def test_a_harness_on_the_traced_path_is_accepted_first_time(tmp_path, monkeypatch):
    author = _Author(_harness_src([_ANCHOR, _AFTER]))

    published, reason = _run(tmp_path, monkeypatch, author)

    assert published is True and reason == ""
    assert len(author.prompts) == 1


def test_a_harness_on_a_dead_path_is_sent_back_with_what_it_actually_ran(tmp_path, monkeypatch):
    author = _Author(_harness_src([_DEAD]), _harness_src([_ANCHOR, _AFTER]))

    published, reason = _run(tmp_path, monkeypatch, author)

    assert published is True and reason == ""
    assert len(author.prompts) == 2
    retry = author.prompts[1]
    # The retry has to name both sides, or the next attempt is the same guess again.
    assert "WRONG code path" in retry
    assert _DEAD in retry and _ANCHOR in retry


def test_a_harness_that_never_finds_the_path_fails_the_recipe(tmp_path, monkeypatch):
    """Better no campaign than hours of numbers describing the wrong software."""
    monkeypatch.setenv("FORGE_HARNESS_ALIGN_ATTEMPTS", "2")
    author = _Author(_harness_src([_DEAD]))

    published, reason = _run(tmp_path, monkeypatch, author)

    assert published is False
    assert "never matched the traced kernels after 2 attempts" in reason
    assert len(author.prompts) == 2


def test_a_silent_harness_leaves_the_gate_unchecked_rather_than_failed(tmp_path, monkeypatch):
    """Saying nothing about the path is not the same as admitting the wrong one."""
    author = _Author(_harness_src(None))

    published, reason = _run(tmp_path, monkeypatch, author)

    assert published is True and reason == ""
    assert len(author.prompts) == 1


def test_a_harness_that_admits_the_mismatch_is_believed(tmp_path, monkeypatch):
    """Its own verdict counts even when it cannot produce the names behind it."""
    monkeypatch.setenv("FORGE_HARNESS_ALIGN_ATTEMPTS", "1")
    author = _Author("print(%r)\n" % json.dumps({"compiled": True, "eager_matches_trace": False}))

    published, reason = _run(tmp_path, monkeypatch, author)

    assert published is False
    assert "never matched the traced kernels" in reason


def test_the_first_prompt_carries_the_traced_kernel_names(tmp_path, monkeypatch):
    author = _Author(_harness_src([_ANCHOR, _AFTER]))

    _run(tmp_path, monkeypatch, author)

    assert _ANCHOR in author.prompts[0]
    assert _AFTER in author.prompts[0]
