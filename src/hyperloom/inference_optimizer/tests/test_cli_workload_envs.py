# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI workload-env export regressions."""

from __future__ import annotations

import argparse
import json
import os
import re
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.inference_optimizer.cli import (
    _export_workload_envs_for_optimize,
    _redact_unknown_args,
    _resolve_run_max_model_len,
    _resolve_workload_knobs,
)
from hyperloom.orchestrator.actions.executors._workload_envs import (
    FrameworkScriptMismatchError,
    materialize_config_with_envs,
)


# ``_export_workload_envs_for_optimize`` writes TP/CONC/EP straight into
# ``os.environ``, which ``monkeypatch`` cannot undo, so restore them here.
_EXPORTED_WORKLOAD_ENVS = ("TP", "CONC", "EP")


@pytest.fixture(autouse=True)
def _restore_exported_workload_envs():
    saved = {key: os.environ.get(key) for key in _EXPORTED_WORKLOAD_ENVS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _ns(**kwargs) -> argparse.Namespace:
    defaults = {"conc": 64}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _write_yaml(path, framework, benchmark_script=None):
    bench = {"framework": framework, "model": "/m", "envs": {}}
    if benchmark_script:
        bench["benchmark_script"] = benchmark_script
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")


def _write_yaml_with_envs(path, framework, envs):
    bench = {"framework": framework, "model": "/m", "envs": dict(envs)}
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")


def _knob_ns(**kwargs) -> argparse.Namespace:
    base = {"isl": None, "osl": None, "conc": None, "tp": None, "ep": None, "precision": None}
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_resolve_workload_knobs_fresh_defaults():
    """Fresh launch, no flags: unset knobs fall back to the shared defaults."""
    a = _knob_ns()
    _resolve_workload_knobs(a)
    assert (a.isl, a.osl, a.conc, a.tp, a.ep, a.precision) == (1024, 1024, 64, 1, 1, "bf16")


def test_resolve_workload_knobs_explicit_flags_win():
    """Explicit flags are preserved verbatim over defaults."""
    a = _knob_ns(isl=512, osl=512, conc=32, tp=2, ep=2, precision="fp8")
    _resolve_workload_knobs(a)
    assert (a.isl, a.osl, a.conc, a.tp, a.ep, a.precision) == (512, 512, 32, 2, 2, "fp8")


def test_resolve_workload_knobs_resume_restores_state():
    """Resume without workload flags: persisted SharedState values win over defaults."""
    a = _knob_ns()
    state = SimpleNamespace(isl=4096, osl=2048, conc=128, tp=8, ep=8, precision="fp8")
    _resolve_workload_knobs(a, state)
    assert (a.isl, a.osl, a.conc, a.tp, a.ep, a.precision) == (4096, 2048, 128, 8, 8, "fp8")


def test_resolve_workload_knobs_resume_explicit_flag_overrides_state():
    """Resume WITH an explicit flag: the flag wins over the persisted state."""
    a = _knob_ns(tp=1)
    state = SimpleNamespace(isl=4096, osl=2048, conc=128, tp=8, ep=8, precision="fp8")
    _resolve_workload_knobs(a, state)
    assert a.tp == 1  # explicit --tp 1 wins
    assert a.isl == 4096  # unset -> restored from state


def test_framework_script_mismatch_fails_fast(tmp_path):
    """vllm framework + sglang script must raise before server boot."""
    src = tmp_path / "cfg.yaml"
    _write_yaml(src, "vllm", benchmark_script="sglang_mi300x.sh")
    with pytest.raises(FrameworkScriptMismatchError, match="framework/script mismatch"):
        materialize_config_with_envs(
            src,
            tmp_path / "out",
            model_path="/m",
            gpu_type="mi300x",
            benchmark_script="sglang_mi300x.sh",
        )


def test_framework_script_match_ok(tmp_path):
    """vllm framework derives vllm_mi300x.sh; no mismatch raised."""
    src = tmp_path / "cfg.yaml"
    _write_yaml(src, "vllm")
    out = materialize_config_with_envs(
        src,
        tmp_path / "out",
        model_path="/m",
        gpu_type="mi300x",
    )
    bench = yaml.safe_load(out.read_text())["benchmark"]
    assert bench["benchmark_script"] == "vllm_mi300x.sh"


def test_materializer_emits_explicit_hip_visibility_epoch(tmp_path, monkeypatch):
    src = tmp_path / "cfg.yaml"
    _write_yaml_with_envs(src, "vllm", {"ROCR_VISIBLE_DEVICES": "9"})
    monkeypatch.setenv("HYPERLOOM_PREFER_HIP_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TP", "2")

    out = materialize_config_with_envs(src, tmp_path / "out")
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]

    assert envs["HIP_VISIBLE_DEVICES"] == "4,5"
    assert "ROCR_VISIBLE_DEVICES" not in envs


def test_single_node_explicit_tp_overrides_stale_env(monkeypatch):
    """`optimize --tp N` must reach YAML materialization on single-node."""
    monkeypatch.setenv("TP", "8")

    _export_workload_envs_for_optimize(
        _ns(conc=64),
        nodes_resolved=1,
        tp_resolved=4,
        ep_resolved=1,
        argv=["optimize", "--tp", "4"],
    )

    assert os.environ["TP"] == "4"


def test_single_node_exports_resolved_workload_envs(monkeypatch):
    """Resolved workload knobs project into env unconditionally so SharedState,
    manifest, and the materialized YAML agree (issue #903). The resolver has
    already folded flag > resume-state > default into ``args``/``*_resolved``."""
    for key in ("TP", "CONC", "EP"):
        monkeypatch.delenv(key, raising=False)

    _export_workload_envs_for_optimize(
        _ns(conc=64),
        nodes_resolved=1,
        tp_resolved=1,
        ep_resolved=1,
        argv=["optimize", "--model", "/m"],
    )

    assert os.environ["TP"] == "1"
    assert os.environ["CONC"] == "64"
    assert os.environ["EP"] == "1"


def test_multi_node_always_exports_workload_envs(monkeypatch):
    """Multi-node child workers still receive resolved workload values."""
    for key in ("TP", "CONC", "EP"):
        monkeypatch.delenv(key, raising=False)

    _export_workload_envs_for_optimize(
        _ns(conc=32),
        nodes_resolved=2,
        tp_resolved=8,
        ep_resolved=2,
        argv=["optimize", "--nodes", "2"],
    )

    assert os.environ["TP"] == "8"
    assert os.environ["CONC"] == "32"
    assert os.environ["EP"] == "2"


def test_operator_server_args_env_routes_to_vllm_args(tmp_path, monkeypatch):
    """One server-args injection point must reach vLLM YAML materialization."""
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_SERVER_ARGS",
        "--gpu-memory-utilization 0.85 --kv-cache-dtype fp8_e4m3",
    )
    src = tmp_path / "cfg.yaml"
    _write_yaml(src, "vllm")

    out = materialize_config_with_envs(src, tmp_path / "out")
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]

    assert "EXTRA_VLLM_ARGS" in envs
    assert "--gpu-memory-utilization 0.85" in envs["EXTRA_VLLM_ARGS"]
    assert "--kv-cache-dtype fp8_e4m3" in envs["EXTRA_VLLM_ARGS"]


def test_operator_server_args_dedup_vllm_single_value_flags(tmp_path, monkeypatch):
    """Operator flags should override YAML defaults without duplicate vLLM keys."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SERVER_ARGS", "--gpu-memory-utilization 0.85")
    src = tmp_path / "cfg.yaml"
    _write_yaml_with_envs(
        src,
        "vllm",
        {"EXTRA_VLLM_ARGS": "--gpu-memory-utilization 0.95 --trust-remote-code"},
    )

    out = materialize_config_with_envs(src, tmp_path / "out")
    args = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]

    assert args.count("--gpu-memory-utilization") == 1
    assert "--gpu-memory-utilization 0.85" in args
    assert "--trust-remote-code" in args


def test_json_serve_arg_survives_append_merge_as_valid_json(tmp_path):
    """JSON-valued flags appended via extra_server_args stay valid JSON.

    The GEMM shape-capture path appends ``current_best.extra_server_args`` (which
    can carry ``--compilation-config`` / ``--speculative-config`` JSON blobs) onto
    the reused baseline config. A prior shlex round-trip in that merge stripped
    the double quotes -> ``{cudagraph_mode:FULL}`` -> vLLM ``json.loads`` rejects
    it at boot -> server never starts -> ``shape_capture_failed`` (no CSV). The
    materialized config must therefore carry VALID JSON. Regression guard.
    """
    src = tmp_path / "cfg.yaml"
    _write_yaml_with_envs(src, "vllm", {"EXTRA_VLLM_ARGS": "--kv-cache-dtype fp8_e4m3"})
    # The extra_server_args reaching shape-capture already had their JSON quotes
    # stripped by an upstream shlex round-trip (observed in the session configs),
    # so materialize receives the unquoted-bareword form. It must not pass this
    # broken JSON through to the launch config; the repair restores valid JSON.
    out = materialize_config_with_envs(
        src,
        tmp_path / "out",
        extra_server_args=(
            "--compilation-config {cudagraph_mode:FULL} --speculative-config {method:ngram,num_speculative_tokens:7}"
        ),
        args_mode="append",
    )
    args = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    blobs = re.findall(r"\{[^{}]*\}", args)
    assert blobs, f"no JSON blob found in materialized args: {args!r}"
    for blob in blobs:
        json.loads(blob)  # must parse: broken JSON was repaired, not passed through


def test_json_serve_arg_survives_shape_capture_port_removal(tmp_path):
    """The shape-capture materialization path must remove its inherited port
    without corrupting a sibling JSON-valued server flag."""
    src = tmp_path / "cfg.yaml"
    _write_yaml_with_envs(
        src,
        "vllm",
        {"EXTRA_VLLM_ARGS": ('--compilation-config {"cudagraph_mode":"FULL"} --port 8888')},
    )

    out = materialize_config_with_envs(
        src,
        tmp_path / "out",
        remove_args=["--port"],
    )
    args = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]

    assert "--port" not in args
    blob = args.split("--compilation-config ", 1)[1].strip()
    assert json.loads(blob) == {"cudagraph_mode": "FULL"}


def test_conc_env_ladder_materializes_as_single_baseline_conc(
    tmp_path,
    monkeypatch,
):
    """CONC=4,16,128 is recognized as a ladder, not crashed by int()."""
    monkeypatch.setenv("CONC", "4,16,128")
    src = tmp_path / "cfg.yaml"
    _write_yaml(src, "vllm")

    out = materialize_config_with_envs(src, tmp_path / "out")
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]

    assert envs["CONC"] == 4


def test_explicit_max_model_len_wins_over_auto(tmp_path, monkeypatch):
    """Explicit --max-model-len / $MAX_MODEL_LEN must not be recomputed."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"max_position_embeddings": 4096}', encoding="utf-8")
    monkeypatch.delenv("MAX_MODEL_LEN", raising=False)

    value, source = _resolve_run_max_model_len(
        _ns(model=str(model), isl=1024, osl=1024, max_model_len=200000),
    )

    assert value == 200000
    assert source == "--max-model-len"


def test_env_max_model_len_wins_over_auto(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"max_position_embeddings": 4096}', encoding="utf-8")
    monkeypatch.setenv("MAX_MODEL_LEN", "200000")

    value, source = _resolve_run_max_model_len(
        _ns(model=str(model), isl=1024, osl=1024, max_model_len=None),
    )

    assert value == 200000
    assert source == "$MAX_MODEL_LEN"


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        # The platform hands pod env through --extra-env, so the secret sits in
        # the value rather than in the flag name.
        (["--extra-env", "HF_TOKEN=hf_live"], "--extra-env HF_TOKEN=***"),
        (["--extra-env=HF_TOKEN=hf_live"], "--extra-env=HF_TOKEN=***"),
        (["--api-key", "sk-live"], "--api-key ***"),
        (["--api-key=sk-live"], "--api-key=***"),
        (["--extra-env", "OPENAI_API_KEY=sk-live"], "--extra-env OPENAI_API_KEY=***"),
        (["--extra-env", "AWS_SECRET_ACCESS_KEY=abc"], "--extra-env AWS_SECRET_ACCESS_KEY=***"),
        # Non-secret flags stay fully readable: a misspelled real flag lands here
        # too, and redacting its value would hide the mistake.
        (["--pod-cpu", "8"], "--pod-cpu 8"),
        (["--gpus-per-nod", "4"], "--gpus-per-nod 4"),
        (["--extra-env", "LOG_LEVEL=debug"], "--extra-env LOG_LEVEL=debug"),
        # A secret flag must not leak its sensitivity onto the next flag's value.
        (["--api-key", "sk-live", "--pod-cpu", "8"], "--api-key *** --pod-cpu 8"),
        ([], ""),
    ],
)
def test_unknown_args_are_logged_without_credential_values(tokens, expected):
    """Unrecognised argv is warned about verbatim, which used to print secrets.

    The platform forwards one FLAGS block to both itself and ``optimize``, so its
    own flags -- including pod credentials -- always land in the unknown list.
    """
    assert _redact_unknown_args(tokens) == expected
