# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rendering launch scripts, and for re-feeding one as a ``--reference-script`` launch file."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys

import pytest

from hyperloom.inference_optimizer.reference_script import (
    LaunchSettings,
    ServingTarget,
    render_reference_script,
)
from inference_suite import UserFileError, parse_user_file


def _refeed(text: str, *, framework: str):
    return parse_user_file("rendered.sh", text.encode("utf-8"), framework=framework).launch


def test_render_round_trip():
    args, envs = "--block-size 128 --attention-backend TRITON_ATTN --trust-remote-code", {"VLLM_ROCM_USE_AITER": "1"}
    text = render_reference_script(
        framework="vllm", launch=LaunchSettings(server_args=args, envs=envs), target=ServingTarget(model="/models/M")
    )
    launch = _refeed(text, framework="vllm")
    assert " ".join(launch.args) == args
    assert launch.env == {"MODEL": "/models/M", **envs}


@pytest.mark.parametrize(
    ("framework", "command"),
    [
        ("sglang", "python3 -m sglang.launch_server --model-path=$MODEL --tp 8"),
        ("vllm", "vllm serve $MODEL --tp 8"),
        ("atom", "python3 -m atom.entrypoints.openai_server --tp 8"),
    ],
)
def test_render_launches_the_frameworks_entrypoint(framework, command):
    text = render_reference_script(framework=framework, launch=LaunchSettings(server_args="--tp 8"))
    assert text.endswith(command + "\n")
    assert _refeed(text, framework=framework).args == ("--tp", "8")


def test_render_carries_validated_rocm_envs():
    # Regression: validated KEEP envs must survive into current_setting.sh.
    text = render_reference_script(
        framework="vllm",
        launch=LaunchSettings(
            server_args="--kv-cache-dtype fp8",
            envs={
                "VLLM_ROCM_USE_AITER_MHA": "1",
                "HIP_FORCE_DEV_KERNARG": "1",
                "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT6",
                "NCCL_MIN_NCHANNELS": "112",
            },
        ),
    )
    assert "export VLLM_ROCM_USE_AITER_MHA=1" in text
    assert "export HIP_FORCE_DEV_KERNARG=1" in text
    assert "export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT6" in text
    assert "export NCCL_MIN_NCHANNELS=112" in text


def test_render_model_path_is_exported():
    text = render_reference_script(
        framework="sglang", launch=LaunchSettings(server_args=""), target=ServingTarget(model="/models/M")
    )
    assert "export MODEL=/models/M" in text


def test_render_bare_model_name_stays_a_comment():
    """A model named without a path is recorded; exporting it would break --model-path."""
    text = render_reference_script(
        framework="sglang", launch=LaunchSettings(server_args=""), target=ServingTarget(model="minimaxm3")
    )
    assert "export MODEL=" not in text
    assert "# model: minimaxm3" in text


def test_render_quotes_env_values():
    text = render_reference_script(
        framework="sglang",
        launch=LaunchSettings(server_args="", envs={"HL_OPTS": "a b; rm -rf /"}),
        target=ServingTarget(gpu_type="mi300x"),
    )
    assert "export HL_OPTS='a b; rm -rf /'" in text


@pytest.mark.parametrize(
    "value",
    [
        "/runs/a table 'quoted' $literal/candidate.csv",
        '{"path":"$literal", "name":"a b"}',
        "line one\nline two",
        "",
        "$(printf unexpected) `printf unexpected`",
    ],
)
def test_literal_environment_survives_export_child_and_reference_import(tmp_path, value):
    name = "AITER_CONFIG_GEMM_BF16"
    text = render_reference_script(framework="sglang", launch=LaunchSettings(server_args="", envs={name: value}))
    assert _refeed(text, framework="sglang").env[name] == value
    observer = tmp_path / "observe.py"
    observer.write_text("import json,os; print(json.dumps(os.environ['AITER_CONFIG_GEMM_BF16']))")
    wrapper = tmp_path / "run.sh"
    wrapper.write_text("python3() { " + shlex.join([sys.executable, "-S", str(observer)]) + "; }\n" + text)
    child = subprocess.run(["bash", str(wrapper)], text=True, capture_output=True, check=True)
    assert json.loads(child.stdout) == value


def test_render_redacts_secret_shaped_envs():
    """The script is archived and uploaded, so credentials are named but not written."""
    text = render_reference_script(
        framework="sglang",
        launch=LaunchSettings(server_args="", envs={"HF_TOKEN": "hf_realsecret", "VLLM_ROCM_USE_AITER": "1"}),
    )
    assert "hf_realsecret" not in text
    assert "# export HF_TOKEN=<redacted; supply manually>" in text
    assert "export VLLM_ROCM_USE_AITER=1" in text


def test_render_no_model_no_export():
    text = render_reference_script(framework="sglang", launch=LaunchSettings(server_args=""))
    assert "export MODEL=" not in text


def test_render_enablement_runs_its_setup_script_under_strict_mode():
    text = render_reference_script(
        framework="sglang", launch=LaunchSettings(server_args=""), setup_script="enablement_setup.sh"
    )
    assert "set -euo pipefail" in text
    assert 'bash "$SCRIPT_DIR"/enablement_setup.sh' in text


def test_render_enablement_script_is_valid_bash(tmp_path):
    text = render_reference_script(
        framework="sglang",
        launch=LaunchSettings(server_args="--tp 8", envs={"VLLM_ROCM_USE_AITER": "1"}),
        target=ServingTarget(model="/models/M"),
        setup_script="enablement_setup.sh",
    )
    sh = tmp_path / "enablement_setting.sh"
    sh.write_text(text, encoding="utf-8")
    proc = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_render_base_params_unchanged_without_enablement():
    """No setup_commands/patches/framework_root → output identical to original."""
    text = render_reference_script(
        framework="vllm",
        launch=LaunchSettings(server_args="--block-size 128", envs={"VLLM_ROCM_USE_AITER": "1"}),
        target=ServingTarget(tp=8, gpu_type="mi300x"),
    )
    assert "set -euo pipefail" not in text
    assert "enablement_setup.sh" not in text
    assert "export TP=8" in text
    assert "vllm serve $MODEL" in text
    assert "VLLM_ROCM_USE_AITER" in text


def test_render_round_trip_with_enablement_params():
    """The setup call does not confuse the launch file parser, so the script still re-feeds."""
    text = render_reference_script(
        framework="sglang",
        launch=LaunchSettings(server_args="--tp 8", envs={"VLLM_ROCM_USE_AITER": "1"}),
        target=ServingTarget(model="/models/M"),
        setup_script="enablement_setup.sh",
    )
    launch = _refeed(text, framework="sglang")
    assert launch.args == ("--tp", "8")
    assert launch.env.get("VLLM_ROCM_USE_AITER") == "1"


@pytest.mark.parametrize("invalid", ["missing", "empty_manifest", "bad_manifest", "path_list"])
def test_export_rejects_unloadable_overlay(tmp_path, invalid):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    if invalid != "missing":
        (overlay / "sitecustomize.py").write_text("")
    if invalid == "empty_manifest":
        (overlay / "_overlay_manifest.json").write_text('{"modules": []}')
    if invalid == "bad_manifest":
        (overlay / "_overlay_manifest.json").write_text("invalid")
    value = str(overlay) + (":/another" if invalid == "path_list" else "")
    with pytest.raises(ValueError, match="overlay_pythonpath"):
        render_reference_script(framework="sglang", launch=LaunchSettings(server_args="", overlay_pythonpath=value))


def test_a_rendered_overlay_cannot_be_refed(tmp_path):
    """A launch file cannot authorize Python code: the overlay's controls line refuses the re-feed."""
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("raise RuntimeError('must not be imported')")
    text = render_reference_script(
        framework="sglang", launch=LaunchSettings(server_args="--tp 8", overlay_pythonpath=str(overlay))
    )
    with pytest.raises(UserFileError, match="hyperloom-launch-controls"):
        _refeed(text, framework="sglang")


def test_launch_controls_are_recorded_and_refuse_a_refeed():
    controls = {"unset_envs": ["SGLANG_USE_AITER"], "remove_args": ["--disable-cuda-graph"], "args_mode": "replace"}
    text = render_reference_script(framework="sglang", launch=LaunchSettings(server_args="--tp 8", **controls))
    assert "# hyperloom-launch-controls: " + json.dumps(controls, sort_keys=True) in text.splitlines()
    assert "unset SGLANG_USE_AITER" in text.splitlines()
    with pytest.raises(UserFileError, match="hyperloom-launch-controls"):
        _refeed(text, framework="sglang")
