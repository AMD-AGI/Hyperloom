"""sglang ``--context-length`` cap injection tests.

sglang sizes its context window from the model's ``max_position_embeddings``,
so a huge native window OOMs the aiter backend. Hyperloom injects a
workload-sized ``--context-length`` (capped to the native window) into
``EXTRA_SGLANG_ARGS`` unless the operator already pinned one. Exercised at both
the pure-helper and the ``materialize_config_with_envs`` layers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors._grid_runner import (
    DEFAULT_SGLANG_CONTEXT_FLOOR_TOKENS,
    DEFAULT_SGLANG_CONTEXT_HEADROOM_TOKENS,
    inject_sglang_attention_backend,
    inject_sglang_context_length,
    resolve_sglang_context_cap,
)
from inference_optimizer.orchestrator.action_executors._workload_envs import (
    materialize_config_with_envs,
)

# Mistral-Nemo-12B's native window — the value that triggered the production aiter OOM.
_HUGE_MAX_POS = 1_024_000


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Clear the workload env knobs and disable the TP clamp so the rendered YAML is hermetic."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    for key in (
        "SGLANG_CONTEXT_HEADROOM_TOKENS", "SGLANG_CONTEXT_FLOOR_TOKENS",
        "SGLANG_WATCHDOG_TIMEOUT",
        "CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "RANDOM_RANGE_RATIO",
        "ROCR_VISIBLE_DEVICES", "PRECISION", "RUN_EVAL", "FRAMEWORK",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_model(tmp_path: Path, max_pos: int | None, *, nested: bool = False) -> str:
    """Create a model dir with a config.json (``max_pos=None`` omits the key; ``nested=True`` uses a ``text_config`` block)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg: dict = {"model_type": "mistral"}
    if max_pos is not None:
        if nested:
            cfg["text_config"] = {"max_position_embeddings": max_pos}
        else:
            cfg["max_position_embeddings"] = max_pos
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(model_dir)


def _write_yaml(path: Path, *, framework: str, model: str) -> None:
    cfg: dict = {
        "benchmark": {
            "framework": framework,
            "model": model,
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _materialize_envs(
    tmp_path: Path,
    *,
    framework: str,
    model: str,
    extra_server_args: str = "",
    extra_envs: dict | None = None,
) -> dict:
    """Render a YAML via the production choke point and return its envs map."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework=framework, model=model)
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base, out,
        extra_server_args=extra_server_args,
        extra_envs=extra_envs,
    )
    cfg = yaml.safe_load(materialized.read_text())
    return cfg["benchmark"]["envs"]


def _context_length_value(server_args: str) -> int:
    """Extract the single ``--context-length`` value (space- or =-separated)."""
    tokens = server_args.replace("=", " ").split()
    idx = tokens.index("--context-length")
    return int(tokens[idx + 1])


# resolve_sglang_context_cap (pure helper)
def test_cap_defaults():
    assert DEFAULT_SGLANG_CONTEXT_HEADROOM_TOKENS == 2048
    assert DEFAULT_SGLANG_CONTEXT_FLOOR_TOKENS == 8192


def test_cap_uses_floor_for_small_workload():
    # 256 + 256 + 2048 = 2560 < 8192 -> floor wins.
    assert resolve_sglang_context_cap(256, 256) == 8192


def test_cap_uses_workload_plus_headroom_above_floor():
    # 4096 + 4096 + 2048 = 10240 > 8192 -> workload+headroom wins.
    assert resolve_sglang_context_cap(4096, 4096) == 10240


def test_cap_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("SGLANG_CONTEXT_HEADROOM_TOKENS", "1024")
    monkeypatch.setenv("SGLANG_CONTEXT_FLOOR_TOKENS", "4096")
    assert resolve_sglang_context_cap(2048, 2048) == 2048 + 2048 + 1024
    assert resolve_sglang_context_cap(256, 256) == 4096


@pytest.mark.parametrize("bad", ["", "  ", "abc", "1.5", "-5"])
def test_cap_bad_env_falls_back_to_default(monkeypatch, bad):
    monkeypatch.setenv("SGLANG_CONTEXT_HEADROOM_TOKENS", bad)
    monkeypatch.setenv("SGLANG_CONTEXT_FLOOR_TOKENS", bad)
    assert resolve_sglang_context_cap(256, 256) == 8192


# inject_sglang_context_length (pure helper)
def test_inject_caps_huge_window_for_sglang(tmp_path):
    """Case 1: a huge max_position_embeddings → the injected ``--context-length`` never exceeds the workload cap."""
    model = _write_model(tmp_path, _HUGE_MAX_POS)
    out = inject_sglang_context_length("--foo bar", "sglang", model, 256, 256)
    assert "--context-length" in out
    cap = resolve_sglang_context_cap(256, 256)
    value = _context_length_value(out)
    assert value <= cap
    assert value <= _HUGE_MAX_POS
    # Pre-existing flags must survive the merge.
    assert "--foo bar" in out


def test_inject_caps_huge_window_via_nested_text_config(tmp_path):
    """Multimodal configs hide the window under ``text_config``; the cap must still apply."""
    model = _write_model(tmp_path, _HUGE_MAX_POS, nested=True)
    out = inject_sglang_context_length("", "sglang", model, 256, 256)
    assert _context_length_value(out) == 8192


def test_inject_clamps_to_small_native_window(tmp_path):
    """A native window below the cap is never advertised above itself."""
    model = _write_model(tmp_path, 4096)
    out = inject_sglang_context_length("", "sglang", model, 256, 256)
    assert _context_length_value(out) == 4096


def test_inject_uses_cap_below_native_window(tmp_path):
    """When the native window is large but finite, the workload cap wins."""
    model = _write_model(tmp_path, 131072)
    out = inject_sglang_context_length("", "sglang", model, 4096, 4096)
    assert _context_length_value(out) == 10240


@pytest.mark.parametrize("existing", [
    "--context-length 6144",
    "--context-length=6144",
    "--foo 1 --context-length 6144 --bar 2",
])
def test_inject_does_not_override_existing(tmp_path, existing):
    """Required case 2: an operator-supplied ``--context-length`` wins."""
    model = _write_model(tmp_path, _HUGE_MAX_POS)
    out = inject_sglang_context_length(existing, "sglang", model, 256, 256)
    assert out == existing
    assert out.count("--context-length") == 1
    assert "8192" not in out


@pytest.mark.parametrize("framework", ["vllm", "atom"])
def test_inject_noop_for_non_sglang(tmp_path, framework):
    """Required case 3: vllm / atom get no ``--context-length`` injected."""
    model = _write_model(tmp_path, _HUGE_MAX_POS)
    assert (
        inject_sglang_context_length("--foo", framework, model, 256, 256)
        == "--foo"
    )
    assert "--context-length" not in inject_sglang_context_length(
        "--gpu-memory-utilization 0.9", framework, model, 256, 256,
    )


def test_inject_noop_when_maxpos_unreadable(tmp_path):
    """Required case 4: an unreadable window injects nothing (fail-safe)."""
    # config.json present but carries no max-length key.
    no_key = _write_model(tmp_path / "a", None)
    assert inject_sglang_context_length("--foo", "sglang", no_key, 256, 256) == "--foo"
    # config.json absent entirely.
    missing = str(tmp_path / "does-not-exist")
    assert inject_sglang_context_length("--foo", "sglang", missing, 256, 256) == "--foo"
    # empty / None model path.
    assert inject_sglang_context_length("--foo", "sglang", "", 256, 256) == "--foo"
    assert inject_sglang_context_length("--foo", "sglang", None, 256, 256) == "--foo"


def test_inject_empty_or_unknown_framework_treated_as_sglang(tmp_path):
    """An empty/unknown framework routes to EXTRA_SGLANG_ARGS, so the cap applies there too."""
    model = _write_model(tmp_path, _HUGE_MAX_POS)
    assert _context_length_value(
        inject_sglang_context_length("", "", model, 256, 256)
    ) == 8192
    assert _context_length_value(
        inject_sglang_context_length("", None, model, 256, 256)
    ) == 8192


# materialize_config_with_envs (the production choke point)
def test_materialize_sglang_injects_context_length_cap(tmp_path):
    model = _write_model(tmp_path, _HUGE_MAX_POS)
    envs = _materialize_envs(tmp_path, framework="sglang", model=model)
    sglang_args = envs["EXTRA_SGLANG_ARGS"]
    # YAML envs default ISL=OSL=256 -> floor cap 8192.
    assert _context_length_value(sglang_args) == 8192
    # The watchdog injection at the same choke point must still fire.
    assert "--watchdog-timeout 1800" in sglang_args


def test_materialize_sglang_respects_user_context_length(tmp_path):
    model = _write_model(tmp_path, _HUGE_MAX_POS)
    envs = _materialize_envs(
        tmp_path, framework="sglang", model=model,
        extra_server_args="--context-length 6144",
    )
    sglang_args = envs["EXTRA_SGLANG_ARGS"]
    assert sglang_args.count("--context-length") == 1
    assert _context_length_value(sglang_args) == 6144
    assert "8192" not in sglang_args


def test_materialize_sglang_noop_when_maxpos_unreadable(tmp_path):
    # Model dir without a config.json -> no cap, but watchdog still injected.
    model = str(tmp_path / "model")
    Path(model).mkdir()
    envs = _materialize_envs(tmp_path, framework="sglang", model=model)
    sglang_args = envs["EXTRA_SGLANG_ARGS"]
    assert "--context-length" not in sglang_args
    assert "--watchdog-timeout 1800" in sglang_args


def test_materialize_vllm_no_context_length(tmp_path):
    model = _write_model(tmp_path, _HUGE_MAX_POS)
    envs = _materialize_envs(tmp_path, framework="vllm", model=model)
    assert "EXTRA_SGLANG_ARGS" not in envs
    assert "--context-length" not in envs.get("EXTRA_VLLM_ARGS", "")


# ---------------------------------------------------------------------------
# inject_sglang_attention_backend (dual chunk attention)
# ---------------------------------------------------------------------------
def _write_dual_chunk_model(
    tmp_path: Path, *, dual_chunk: bool, nested: bool = False
) -> str:
    model_dir = tmp_path / "dcmodel"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg: dict = {"model_type": "qwen2", "max_position_embeddings": 1_010_000}
    block = {"chunk_size": 262144} if dual_chunk else None
    if dual_chunk:
        if nested:
            cfg["text_config"] = {"dual_chunk_attention_config": block}
        else:
            cfg["dual_chunk_attention_config"] = block
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(model_dir)


def test_dual_chunk_injects_flash_attn_backend(tmp_path):
    """A model with dual_chunk_attention_config gets the compatible backend."""
    model = _write_dual_chunk_model(tmp_path, dual_chunk=True)
    out = inject_sglang_attention_backend("--foo bar", "sglang", model)
    assert "--attention-backend dual_chunk_flash_attn" in out
    assert "--foo bar" in out


def test_dual_chunk_injects_via_nested_text_config(tmp_path):
    model = _write_dual_chunk_model(tmp_path, dual_chunk=True, nested=True)
    out = inject_sglang_attention_backend("", "sglang", model)
    assert "--attention-backend dual_chunk_flash_attn" in out


def test_dual_chunk_noop_without_config(tmp_path):
    model = _write_dual_chunk_model(tmp_path, dual_chunk=False)
    assert inject_sglang_attention_backend("--foo", "sglang", model) == "--foo"


def test_dual_chunk_does_not_override_operator_backend(tmp_path):
    model = _write_dual_chunk_model(tmp_path, dual_chunk=True)
    existing = "--attention-backend triton"
    out = inject_sglang_attention_backend(existing, "sglang", model)
    assert out == existing
    assert out.count("--attention-backend") == 1


@pytest.mark.parametrize("framework", ["vllm", "atom"])
def test_dual_chunk_noop_for_non_sglang(tmp_path, framework):
    model = _write_dual_chunk_model(tmp_path, dual_chunk=True)
    assert (
        inject_sglang_attention_backend("--foo", framework, model) == "--foo"
    )


def test_dual_chunk_noop_when_config_unreadable(tmp_path):
    missing = str(tmp_path / "nope")
    assert inject_sglang_attention_backend("--foo", "sglang", missing) == "--foo"
    assert inject_sglang_attention_backend("--foo", "sglang", "") == "--foo"


def test_materialize_sglang_injects_dual_chunk_backend(tmp_path):
    model = _write_dual_chunk_model(tmp_path, dual_chunk=True)
    envs = _materialize_envs(tmp_path, framework="sglang", model=model)
    sglang_args = envs["EXTRA_SGLANG_ARGS"]
    assert "--attention-backend dual_chunk_flash_attn" in sglang_args
