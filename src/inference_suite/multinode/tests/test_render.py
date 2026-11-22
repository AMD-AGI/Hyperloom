# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from inference_suite.frameworks import Framework
from inference_suite.multinode.render import render_role


def test_sglang_runner_owned_flags_are_stripped():
    args = ["--model-path", "/m", "--tp-size", "8", "--mem-fraction-static", "0.8", "--port=30000"]
    args += ["--disaggregation-transfer-backend", "mori", "--context-length", "8192", "--disable-radix-cache"]
    rendered = render_role(Framework.SGLANG, args, {})
    assert rendered.args == ("--mem-fraction-static", "0.8", "--disable-radix-cache")
    assert rendered.stripped == (
        "--model-path",
        "--tp-size",
        "--port",
        "--disaggregation-transfer-backend",
        "--context-length",
    )
    assert rendered.transfer_backend == "mori"


def test_atom_runner_owned_flags_are_stripped():
    rendered = render_role(Framework.ATOM, ["-tp", "8", "--max-model-len", "4096", "--kv_cache_dtype", "fp8"], {})
    assert rendered.args == ("--kv_cache_dtype", "fp8")
    assert rendered.stripped == ("-tp", "--max-model-len")


def test_host_env_is_stripped():
    env = {"HF_HOME": "/hf", "NCCL_IB_HCA": "@fabric.ib_devices", "MORI_SHMEM_MODE": "ISOLATION", "N": 3}
    rendered = render_role(Framework.SGLANG, [], env)
    assert rendered.env == {"MORI_SHMEM_MODE": "ISOLATION", "N": "3"}
    assert rendered.stripped == ("env HF_HOME", "env NCCL_IB_HCA")
