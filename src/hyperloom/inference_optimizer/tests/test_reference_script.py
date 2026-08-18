# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for reference-script parsing and rendering."""

from __future__ import annotations

import subprocess

import pytest
from pathlib import Path
from types import SimpleNamespace

from hyperloom.common.env_safety import GPU_MASK_ENV_NAMES
from hyperloom.inference_optimizer.reference_script import (
    parse_reference_script,
    render_reference_script,
)
from hyperloom.inference_optimizer.cli.bootstrap import _resolve_reference_recipe

_M3_RECIPE = """#!/usr/bin/env bash
source "$(dirname "$0")/../benchmark_lib.sh"
check_env_vars MODEL TP CONC ISL OSL

export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ROCM_USE_AITER=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_USE_RUST_FRONTEND=1
export NCCL_DMABUF_ENABLE=0
export VLLM_CACHE_ROOT=/workspace/cache
export SOME_SECRET=hunter2

PORT=${PORT:-8888}
set -x
vllm serve $MODEL --port $PORT \\
--tensor-parallel-size=$TP \\
--block-size 128 \\
--attention-backend TRITON_ATTN \\
--no-enable-prefix-caching \\
--language-model-only \\
--served-model-name $NAME \\
--max-model-len $MAX_MODEL_LEN \\
--trust-remote-code > $SERVER_LOG 2>&1 &
"""


def _write(tmp_path: Path, text: str, name: str = "recipe.sh") -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_parse_lifts_static_flags(tmp_path):
    src = _write(tmp_path, _M3_RECIPE)
    r = parse_reference_script(src, framework="vllm")
    assert "--block-size 128" in r.server_args
    assert "--attention-backend TRITON_ATTN" in r.server_args
    assert "--no-enable-prefix-caching" in r.server_args
    assert "--language-model-only" in r.server_args
    assert "--trust-remote-code" in r.server_args


def test_parse_drops_var_tokens(tmp_path):
    src = _write(tmp_path, _M3_RECIPE)
    r = parse_reference_script(src, framework="vllm")
    assert "$" not in r.server_args
    assert "--port" not in r.server_args
    assert "--tensor-parallel-size" not in r.server_args
    assert "--max-model-len" not in r.server_args


def test_parse_orphan_flag_dropped_with_value(tmp_path):
    """R1: --served-model-name $NAME drops BOTH tokens, no orphan flag."""
    src = _write(tmp_path, _M3_RECIPE)
    r = parse_reference_script(src, framework="vllm")
    assert "--served-model-name" not in r.server_args


def test_parse_flag_equals_var_dropped_whole(tmp_path):
    text = "vllm serve $MODEL --tensor-parallel-size=$TP --block-size=32\n"
    src = _write(tmp_path, text)
    r = parse_reference_script(src, framework="vllm")
    assert "--tensor-parallel-size" not in r.server_args
    assert "--block-size=32" in r.server_args


def test_parse_env_denylist(tmp_path):
    """R6: tuning envs are carried; credential-shaped names are not."""
    src = _write(tmp_path, _M3_RECIPE)
    r = parse_reference_script(src, framework="vllm")
    assert r.envs.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "0"
    assert r.envs.get("VLLM_ROCM_USE_AITER") == "1"
    assert r.envs.get("VLLM_ALLREDUCE_USE_FLASHINFER") == "1"
    assert r.envs.get("VLLM_USE_RUST_FRONTEND") == "1"
    assert r.envs.get("NCCL_DMABUF_ENABLE") == "0"
    # Path-typed vars pass: the denylist gates names, not value shapes.
    assert r.envs.get("VLLM_CACHE_ROOT") == "/workspace/cache"
    assert "SOME_SECRET" not in r.envs


def test_parse_env_execution_and_redirect_vectors_blocked(tmp_path):
    """A recipe may tune the server, not hijack the interpreter or reroute traffic."""
    text = """\
export PYTHONUSERBASE=/tmp/evil
export NODE_OPTIONS=--require=/tmp/evil.js
export PERL5OPT=-M/tmp/evil
export GIT_SSH_COMMAND=/tmp/evil.sh
export HTTPS_PROXY=http://attacker
export HF_ENDPOINT=http://attacker
export SSL_CERT_FILE=/tmp/evil.pem
export REQUESTS_CA_BUNDLE=/tmp/evil.pem
export TMPDIR=/tmp/evil
export VLLM_ROCM_USE_AITER=1
vllm serve $MODEL --trust-remote-code
"""
    src = _write(tmp_path, text)
    r = parse_reference_script(src, framework="vllm")
    for blocked in (
        "PYTHONUSERBASE",
        "NODE_OPTIONS",
        "PERL5OPT",
        "GIT_SSH_COMMAND",
        "HTTPS_PROXY",
        "HF_ENDPOINT",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "TMPDIR",
    ):
        assert blocked not in r.envs, f"{blocked} should be blocked"
    assert r.envs.get("VLLM_ROCM_USE_AITER") == "1"


def test_parse_env_gpu_masks_blocked(tmp_path):
    """A recipe selects how the run is tuned, never which devices it sees."""
    text = """\
export CUDA_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=1
export HSA_VISIBLE_DEVICES=2
export HIP_VISIBLE_DEVICES=3
export ROCR_VISIBLE_DEVICES=4
export VLLM_ROCM_USE_AITER=1
vllm serve $MODEL --trust-remote-code
"""
    src = _write(tmp_path, text)
    r = parse_reference_script(src, framework="vllm")
    for blocked in sorted(GPU_MASK_ENV_NAMES):
        assert blocked not in r.envs, f"{blocked} should be blocked"
    assert r.envs.get("VLLM_ROCM_USE_AITER") == "1"


def test_parse_env_tokenizer_knob_is_not_a_credential(tmp_path):
    """The TOKEN fragment must not swallow TOKENIZERS_PARALLELISM, and the
    exemption must not turn TOKENIZER into a way past the fragment rule."""
    text = """\
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=secret
export TOKENIZER_API_KEY=secret
export MY_TOKENIZER_SECRET=secret
vllm serve $MODEL --trust-remote-code
"""
    src = _write(tmp_path, text)
    r = parse_reference_script(src, framework="vllm")
    assert r.envs.get("TOKENIZERS_PARALLELISM") == "false"
    for blocked in ("HF_TOKEN", "TOKENIZER_API_KEY", "MY_TOKENIZER_SECRET"):
        assert blocked not in r.envs, f"{blocked} should be blocked"


def test_parse_env_workload_owned_blocked(tmp_path):
    """Workload keys the optimizer owns must never come from a reference script."""
    text = """\
export TP=16
export MODEL=/bad/path
export CONC=128
export RUN_EVAL=false
export RESULT_DIR=/bad
export VLLM_ROCM_USE_AITER=1
vllm serve $MODEL --trust-remote-code
"""
    src = _write(tmp_path, text)
    r = parse_reference_script(src, framework="vllm")
    for blocked in ("TP", "MODEL", "CONC", "RUN_EVAL", "RESULT_DIR"):
        assert blocked not in r.envs, f"{blocked} should be blocked"
    assert r.envs.get("VLLM_ROCM_USE_AITER") == "1"


def test_parse_env_shell_safety_blocked(tmp_path):
    """Shell-unsafe vars (BLOCKED_UNTRUSTED_ENV_NAMES) must be denied."""
    text = """\
export LD_LIBRARY_PATH=/bad
export PYTHONPATH=/bad
export PATH=/bad/bin
export VLLM_ROCM_USE_AITER=1
vllm serve $MODEL
"""
    src = _write(tmp_path, text)
    r = parse_reference_script(src, framework="vllm")
    for blocked in ("LD_LIBRARY_PATH", "PYTHONPATH", "PATH"):
        assert blocked not in r.envs, f"{blocked} should be blocked"
    assert r.envs.get("VLLM_ROCM_USE_AITER") == "1"


def test_parse_env_self_defaults(tmp_path):
    text = """\
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE-1}"
export VLLM_ROCM_USE_AITER=${OTHER:-1}
export VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-$DEFAULT}
"""
    r = parse_reference_script(_write(tmp_path, text), framework="vllm")
    assert r.envs["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
    assert r.envs["NCCL_DMABUF_ENABLE"] == "1"
    assert "VLLM_ROCM_USE_AITER" not in r.envs
    assert "VLLM_USE_RUST_FRONTEND" not in r.envs


def test_parse_continuation_parity(tmp_path):
    """A \\-continuation recipe parses identically to its single-line form."""
    multi = _write(tmp_path, _M3_RECIPE, "multi.sh")
    single_line = (
        "vllm serve $MODEL --block-size 128 "
        "--attention-backend TRITON_ATTN --no-enable-prefix-caching "
        "--language-model-only --trust-remote-code\n"
    )
    single = _write(tmp_path, single_line, "single.sh")
    rm = parse_reference_script(multi, framework="vllm")
    rs = parse_reference_script(single, framework="vllm")
    assert rm.server_args == rs.server_args


def test_parse_missing_file_raises():
    with pytest.raises(OSError):
        parse_reference_script("/nonexistent/recipe.sh", framework="vllm")


def test_parse_sglang_entrypoint(tmp_path):
    text = (
        "export SGLANG_USE_AITER=1\n"
        "python3 -m sglang.launch_server --model-path=$MODEL "
        "--port $PORT --tp $TP --chunked-prefill-size 8192\n"
    )
    src = _write(tmp_path, text)
    r = parse_reference_script(src, framework="sglang")
    assert "--chunked-prefill-size 8192" in r.server_args
    assert "$" not in r.server_args
    assert r.envs.get("SGLANG_USE_AITER") == "1"


def test_render_round_trip(tmp_path):
    src = _write(tmp_path, _M3_RECIPE)
    r = parse_reference_script(src, framework="vllm")
    text = render_reference_script(
        framework="vllm",
        server_args=r.server_args,
        envs=r.envs,
        model=r.model,
    )
    rt_src = _write(tmp_path, text, "rt.sh")
    r2 = parse_reference_script(rt_src, framework="vllm")
    assert r2.server_args == r.server_args
    assert r2.envs == r.envs


def test_render_carries_validated_rocm_envs():
    # Regression: validated KEEP envs must survive into current_setting.sh.
    text = render_reference_script(
        framework="vllm",
        server_args="--kv-cache-dtype fp8",
        envs={
            "VLLM_ROCM_USE_AITER_MHA": "1",
            "HIP_FORCE_DEV_KERNARG": "1",
            "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT6",
            "NCCL_MIN_NCHANNELS": "112",
        },
    )
    assert "export VLLM_ROCM_USE_AITER_MHA=1" in text
    assert "export HIP_FORCE_DEV_KERNARG=1" in text
    assert "export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT6" in text
    assert "export NCCL_MIN_NCHANNELS=112" in text


def test_render_model_path_is_exported():
    text = render_reference_script(framework="sglang", server_args="", model="/models/M")
    assert "export MODEL=/models/M" in text


def test_render_bare_model_name_stays_a_comment():
    """parse_reference_script records a basename; exporting it would break --model-path."""
    text = render_reference_script(framework="sglang", server_args="", model="minimaxm3")
    assert "export MODEL=" not in text
    assert "# model: minimaxm3" in text


def test_render_quotes_env_values():
    text = render_reference_script(
        framework="sglang",
        server_args="",
        envs={"HL_OPTS": "a b; rm -rf /"},
        gpu_type="mi300x",
    )
    assert "export HL_OPTS='a b; rm -rf /'" in text


def test_render_redacts_secret_shaped_envs():
    """The script is archived and uploaded, so credentials are named but not written."""
    text = render_reference_script(
        framework="sglang",
        server_args="",
        envs={"HF_TOKEN": "hf_realsecret", "VLLM_ROCM_USE_AITER": "1"},
    )
    assert "hf_realsecret" not in text
    assert "# export HF_TOKEN=<redacted; supply manually>" in text
    assert "export VLLM_ROCM_USE_AITER=1" in text


def test_render_no_model_no_export():
    text = render_reference_script(framework="sglang", server_args="")
    assert "export MODEL=" not in text


def test_render_enablement_has_strict_mode():
    text = render_reference_script(
        framework="sglang",
        server_args="",
        setup_commands=["pip install vllm==0.24"],
    )
    assert "set -euo pipefail" in text
    assert "pip install vllm==0.24" in text


def test_render_patches_emit_apply_function():
    text = render_reference_script(
        framework="sglang",
        server_args="",
        patches=["patches/001_fix.patch"],
        framework_root="/sgl-workspace/sglang",
    )
    assert "export FRAMEWORK_ROOT=/sgl-workspace/sglang" in text
    assert "apply_patch patches/001_fix.patch" in text
    assert "for lvl in 1 0 2" in text


def test_render_enablement_script_is_valid_bash(tmp_path):
    text = render_reference_script(
        framework="sglang",
        server_args="--tp 8",
        envs={"VLLM_ROCM_USE_AITER": "1"},
        model="/models/M",
        setup_commands=["pip install aiter==0.1.4"],
        patches=["patches/001.patch"],
        framework_root="/sgl-workspace/sglang",
        runtime="/session/enablement/stacks/sglang/s1/venv",
    )
    sh = tmp_path / "enablement_setting.sh"
    sh.write_text(text, encoding="utf-8")
    proc = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_render_runtime_note_emitted():
    text = render_reference_script(
        framework="vllm",
        server_args="",
        runtime="/session/enablement/stacks/vllm/spec-1/venv",
    )
    assert "isolated attempt venv" in text
    assert "/session/enablement/stacks/vllm/spec-1/venv" in text


def test_render_base_params_unchanged_without_enablement():
    """No setup_commands/patches/framework_root → output identical to original."""
    text = render_reference_script(
        framework="vllm",
        server_args="--block-size 128",
        envs={"VLLM_ROCM_USE_AITER": "1"},
        tp=8,
        gpu_type="mi300x",
    )
    assert "set -euo pipefail" not in text
    assert "apply_patch" not in text
    assert "export TP=8" in text
    assert "vllm serve $MODEL" in text
    assert "VLLM_ROCM_USE_AITER" in text


def test_render_round_trip_with_enablement_params(tmp_path):
    """Inserting setup/patch lines does not confuse parse_reference_script."""
    text = render_reference_script(
        framework="sglang",
        server_args="--tp 8",
        envs={"VLLM_ROCM_USE_AITER": "1"},
        model="/models/M",
        setup_commands=["pip install vllm==0.24"],
        patches=["patches/fix.patch"],
        framework_root="/sgl-workspace/sglang",
    )
    sh = tmp_path / "enablement_setting.sh"
    sh.write_text(text, encoding="utf-8")
    r = parse_reference_script(str(sh), framework="sglang")
    assert "--tp 8" in r.server_args
    assert r.envs.get("VLLM_ROCM_USE_AITER") == "1"


# --- bootstrap._resolve_reference_recipe tests ---

def test_resolve_no_flag_returns_empty(monkeypatch):
    """No --reference-script → empty tuple, nothing attempted."""
    monkeypatch.setenv("FRAMEWORK", "vllm")
    args = SimpleNamespace(reference_script=None)
    assert _resolve_reference_recipe(args) == ("", {}, "", "")


def test_resolve_valid_flag_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    src = _write(tmp_path, _M3_RECIPE, "explicit.sh")
    args = SimpleNamespace(reference_script=src)
    server_args, envs, model, source = _resolve_reference_recipe(args)
    assert "--block-size 128" in server_args
    assert source == src


def test_resolve_unreadable_flag_raises_system_exit(monkeypatch):
    """Given but unreadable path → SystemExit(2)."""
    monkeypatch.setenv("FRAMEWORK", "vllm")
    args = SimpleNamespace(reference_script="/no/such/recipe.sh")
    with pytest.raises(SystemExit) as exc_info:
        _resolve_reference_recipe(args)
    assert exc_info.value.code == 2


def test_resolve_script_with_no_flags_raises_system_exit(tmp_path, monkeypatch):
    """A readable script that yields no server flags and no envs → SystemExit(2)."""
    monkeypatch.setenv("FRAMEWORK", "vllm")
    empty = _write(tmp_path, "#!/usr/bin/env bash\necho hello\n")
    args = SimpleNamespace(reference_script=empty)
    with pytest.raises(SystemExit) as exc_info:
        _resolve_reference_recipe(args)
    assert exc_info.value.code == 2
