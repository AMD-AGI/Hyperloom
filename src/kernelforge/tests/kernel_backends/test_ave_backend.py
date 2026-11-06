# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ave backend discovery, default knowledge and CLI selection."""

from kernelforge.cli.forge_loop_command import forge_loop
from kernelforge.config import Config
from kernelforge.kernel_backends.base import build_single_kernel_backend_prompt
from kernelforge.workflows.forge_loop.campaign.campaign_config import infer_kernel_backend


def test_ave_backend_discovery_knowledge_and_cli(tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("import avelang as ave\n")
    assert infer_kernel_backend([kernel]) == "ave"
    prompt = build_single_kernel_backend_prompt(Config(gpu_target="gfx950"), "ave")
    assert "AveLang" in prompt
    assert "avelang.md" in prompt and "lds_layout.md" in prompt
    with forge_loop.make_context("forge-loop", ["--workspace", ".", "--kernel-backend", "ave"]) as context:
        assert context.params["kernel_backend"] == "ave"
