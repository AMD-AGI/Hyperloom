# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for reference-script parsing, rendering, and discovery."""

from __future__ import annotations

from pathlib import Path

from inference_optimizer.reference_script import (
    discover_reference_script,
    models_compatible,
    parse_reference_script,
    render_reference_script,
)

_M3_RECIPE = """#!/usr/bin/env bash
source "$(dirname "$0")/../benchmark_lib.sh"
check_env_vars MODEL TP CONC ISL OSL

export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ROCM_USE_AITER=1
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


# ── parse ──────────────────────────────────────────────────────────────────
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
    # $-valued flags and the positional model are dropped entirely.
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


def test_parse_env_whitelist(tmp_path):
    """R6: only enumerated exports are carried; path-valued / secret vars are not."""
    src = _write(tmp_path, _M3_RECIPE)
    r = parse_reference_script(src, framework="vllm")
    assert r.envs.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "0"
    assert r.envs.get("VLLM_ROCM_USE_AITER") == "1"
    assert "VLLM_CACHE_ROOT" not in r.envs
    assert "SOME_SECRET" not in r.envs


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


def test_parse_missing_file_failsoft():
    r = parse_reference_script("/nonexistent/recipe.sh", framework="vllm")
    assert r.server_args == ""
    assert r.envs == {}
    assert r.model is None


def test_parse_unreachable_url_failsoft():
    r = parse_reference_script("https://127.0.0.1:1/none.sh", framework="vllm")
    assert r.server_args == ""
    assert r.envs == {}


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


# ── render / round-trip ──────────────────────────────────────────────────────
def test_render_round_trip(tmp_path):
    src = _write(tmp_path, _M3_RECIPE)
    r = parse_reference_script(src, framework="vllm")
    text = render_reference_script(
        framework="vllm", server_args=r.server_args, envs=r.envs, model=r.model,
    )
    rt_src = _write(tmp_path, text, "rt.sh")
    r2 = parse_reference_script(rt_src, framework="vllm")
    assert r2.server_args == r.server_args
    assert r2.envs == r.envs


# ── discovery ────────────────────────────────────────────────────────────────
def _mk_tree(tmp_path: Path, names: list[str]) -> Path:
    d = tmp_path / "InferenceX" / "benchmarks" / "single_node"
    d.mkdir(parents=True)
    for n in names:
        (d / n).write_text("vllm serve $MODEL\n", encoding="utf-8")
    return tmp_path / "InferenceX"


def test_discovery_exact(tmp_path):
    root = _mk_tree(tmp_path, [
        "dsr1_fp8_mi300x.sh", "gptoss_fp4_mi300x.sh",
    ])
    path, tier = discover_reference_script(
        str(root), model_path="dsr1", precision="fp8",
        gpu_type="mi300x", framework="vllm",
    )
    assert tier == "exact"
    assert path.endswith("dsr1_fp8_mi300x.sh")


def test_discovery_version_mismatch_is_none(tmp_path):
    root = _mk_tree(tmp_path, ["minimaxm2.5_fp8_mi300x.sh"])
    path, tier = discover_reference_script(
        str(root), model_path="minimaxm3", precision="fp8",
        gpu_type="mi300x", framework="vllm",
    )
    assert tier == "none"
    assert path is None


def test_discovery_fuzzy(tmp_path):
    root = _mk_tree(tmp_path, ["kimik2.5_int4_mi300x.sh"])
    path, tier = discover_reference_script(
        str(root), model_path="/wekafs/models/moonshotai-Kimi-K2.5-Instruct",
        precision="int4", gpu_type="mi300x", framework="vllm",
    )
    assert tier == "fuzzy"
    assert path.endswith("kimik2.5_int4_mi300x.sh")


def test_discovery_framework_gate(tmp_path):
    """An atom-tagged script must not match a vllm run."""
    root = _mk_tree(tmp_path, ["dsr1_fp8_mi300x_atom.sh"])
    _, tier = discover_reference_script(
        str(root), model_path="dsr1", precision="fp8",
        gpu_type="mi300x", framework="vllm",
    )
    assert tier == "none"


def test_discovery_precision_gpu_filter(tmp_path):
    root = _mk_tree(tmp_path, ["dsr1_fp8_b200.sh", "dsr1_fp4_mi300x.sh"])
    _, tier = discover_reference_script(
        str(root), model_path="dsr1", precision="fp8",
        gpu_type="mi300x", framework="vllm",
    )
    assert tier == "none"  # fp8 only exists for b200, fp4 only for mi300x


def test_discovery_missing_path_failsoft():
    path, tier = discover_reference_script(
        "/nonexistent", model_path="x", precision="fp8",
        gpu_type="mi300x", framework="vllm",
    )
    assert path is None
    assert tier == "none"


# ── model-gate (R2): shared by discovery + baseline executor ─────────────────
def test_models_compatible_exact_and_fuzzy():
    assert models_compatible("dsr1", "/wekafs/models/dsr1") is True
    # filename-style alias vs a real HF path (normalized substring → fuzzy)
    assert models_compatible(
        "minimaxm2.5", "/wekafs/models/MiniMaxAI-MiniMax-M2.5",
    ) is True


def test_models_compatible_version_mismatch_blocked():
    """R2: a near-name version mismatch must NOT apply the recipe."""
    assert models_compatible("minimaxm2.5", "/wekafs/models/minimax-m3") is False
    assert models_compatible("kimik2", "/wekafs/models/kimi-k2.5") is False


def test_models_compatible_empty_is_ungated():
    assert models_compatible("", "/wekafs/models/anything") is True


# ── _resolve_reference_recipe gating (discovery only when flag is given) ──────
from types import SimpleNamespace

from inference_optimizer.cli import _resolve_reference_recipe


def test_resolve_no_flag_does_not_discover(tmp_path, monkeypatch):
    """No --reference-script → empty, and discovery never runs (0 degrade)."""
    root = _mk_tree(tmp_path, ["dsr1_fp8_mi300x.sh"])
    monkeypatch.setenv("INFERENCEX_PATH", str(root))
    monkeypatch.setenv("FRAMEWORK", "vllm")
    args = SimpleNamespace(
        reference_script=None, model="/wekafs/models/dsr1",
        precision="fp8", gpu_type="mi300x",
    )
    server_args, envs, model, source = _resolve_reference_recipe(args)
    # Even though an EXACT recipe exists on disk, omitting the flag must NOT use it.
    assert (server_args, envs, model, source) == ("", {}, "", "")


def test_resolve_valid_flag_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    src = _write(tmp_path, _M3_RECIPE, "explicit.sh")
    args = SimpleNamespace(
        reference_script=src, model="/wekafs/models/whatever",
        precision="fp8", gpu_type="mi300x",
    )
    server_args, envs, model, source = _resolve_reference_recipe(args)
    assert "--block-size 128" in server_args
    assert source == src


def test_resolve_invalid_flag_falls_back_to_discovery(tmp_path, monkeypatch):
    """Given but unreadable path → auto-discover an exact match instead."""
    root = _mk_tree(tmp_path, ["dsr1_fp8_mi300x.sh"])
    monkeypatch.setenv("INFERENCEX_PATH", str(root))
    monkeypatch.setenv("FRAMEWORK", "vllm")
    args = SimpleNamespace(
        reference_script="/no/such/recipe.sh", model="/wekafs/models/dsr1",
        precision="fp8", gpu_type="mi300x",
    )
    server_args, envs, model, source = _resolve_reference_recipe(args)
    assert source.endswith("dsr1_fp8_mi300x.sh")  # discovered exact match


def test_resolve_invalid_flag_no_inferencex_returns_empty(tmp_path, monkeypatch):
    """Given but unreadable + no INFERENCEX_PATH → empty (still 0 degrade)."""
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.setenv("FRAMEWORK", "vllm")
    args = SimpleNamespace(
        reference_script="/no/such/recipe.sh", model="/wekafs/models/dsr1",
        precision="fp8", gpu_type="mi300x",
    )
    assert _resolve_reference_recipe(args) == ("", {}, "", "")
