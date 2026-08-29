"""Coverage completion tests for kernel_backends/base.py.

Covers the single prompt builder, AITER-operator detection, and the
gbrain-combination branches of the combined-KB builder (gbrain mocked).
"""

from __future__ import annotations

from kernelforge.config import Config
from kernelforge.kernel_backends.base import (
    _is_aiter_operator,
    build_single_kernel_backend_prompt,
)
from kernelforge.kernel_backends.constants import KERNEL_BACKENDS, resolve_language_dir


# ─── _is_aiter_operator ───


def test_is_aiter_operator_true():
    assert _is_aiter_operator("repository", ["/work/aiter/ops/triton/x.py"])
    assert _is_aiter_operator("image_kernel", ["/repo/aiter/csrc/pa/k.cpp"])


def test_is_aiter_operator_false_cases():
    # Wrong task type.
    assert not _is_aiter_operator("snippet", ["/work/aiter/ops/x.py"])
    # Right task type but no aiter path component (substring 'aiter' in name
    # must not count).
    assert not _is_aiter_operator("repository", ["/work/aiter_pa_decode/x.py"])
    assert not _is_aiter_operator("repository", None)
    assert not _is_aiter_operator("", ["/work/aiter/ops/x.py"])


# ─── build_single_kernel_backend_prompt ───


def test_build_single_kernel_backend_prompt_unknown():
    config = Config(gpu_target="gfx950")
    assert build_single_kernel_backend_prompt(config, "nope") == ""


def test_build_single_kernel_backend_prompt_ck():
    config = Config(gpu_target="gfx950")
    prompt = build_single_kernel_backend_prompt(config, "ck")
    assert isinstance(prompt, str)
    assert len(prompt) > 200
    assert "gfx950" in prompt


def test_build_single_kernel_backend_prompt_supports_every_registered_backend():
    config = Config(gpu_target="gfx950")

    for backend in KERNEL_BACKENDS:
        prompt = build_single_kernel_backend_prompt(config, backend)
        assert prompt, f"{backend} is registered but has no forge-loop prompt"


# ─── resolve_language_dir ───


def test_resolve_language_dir_matches_backend_named_folders(tmp_path):
    (tmp_path / "languages" / "triton").mkdir(parents=True)
    assert resolve_language_dir("triton", tmp_path) == "triton"


def test_resolve_language_dir_none_without_folder(tmp_path):
    (tmp_path / "languages").mkdir()
    assert resolve_language_dir("hipblaslt", tmp_path) is None
    assert resolve_language_dir("", tmp_path) is None


def test_build_single_kernel_backend_prompt_aiter_operator():
    config = Config(gpu_target="gfx950")
    # AITER operator path exercises include_aiter=True branch.
    prompt = build_single_kernel_backend_prompt(
        config,
        "ck",
        task_type="repository",
        source_paths=["/work/aiter/ops/ck/gemm.py"],
    )
    assert len(prompt) > 200


def test_prompt_build_does_not_probe_provider_sdks(monkeypatch):
    """Prompt assembly must not resolve an agent provider."""
    from kernelforge.agent_backends import registry

    def reject(*_args, **_kwargs):
        raise AssertionError("provider selection must not run")

    monkeypatch.setattr(registry, "select_default_agent_provider", reject)
    monkeypatch.setattr(registry, "resolve_agent_runtime", reject)

    assert build_single_kernel_backend_prompt(Config(gpu_target="gfx950"), "ck")
