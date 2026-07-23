# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``multi_node/scripts/launch_infera_node.py``."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module():
    path = _repo_root() / "multi_node" / "scripts" / "launch_infera_node.py"
    spec = importlib.util.spec_from_file_location("launch_infera_node", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_build_sglang_cmd_uses_infera_engine():
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {"model": "/models/x", "tp": 8, "nnodes": 2, "dist_init_port": 5000, "ep": 8, "extra_args": ""},
    )()
    cmd = mod._build_sglang_cmd(ns, node_rank=1, leader="10.0.0.1", advertise_host="10.0.0.2")
    assert cmd[0:3] == ["python3", "-m", "infera.engine.sglang"]
    assert "--discovery-backend" in cmd and "kubernetes" in cmd
    assert "--advertise-host" in cmd and "10.0.0.2" in cmd


def test_build_vllm_cmd_uses_infera_engine():
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {"model": "/models/x", "tp": 8, "ep": 1, "extra_args": ""},
    )()
    cmd = mod._build_vllm_cmd(ns, advertise_host="10.0.0.3")
    assert cmd[0:3] == ["python3", "-m", "infera.engine.vllm"]
    assert "--model-path" in cmd
    assert "--advertise-host" in cmd and "10.0.0.3" in cmd


def test_build_sglang_cmd_autofills_dp_size_for_dp_attention():
    # --enable-dp-attention without --dp-size => sglang would disable it at
    # dp_size==1; the launcher must inject --dp-size=tp so it takes effect.
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {
            "model": "/models/x",
            "tp": 8,
            "nnodes": 1,
            "dist_init_port": 5000,
            "ep": 8,
            "extra_args": "--enable-dp-attention --enable-dp-lm-head",
        },
    )()
    cmd = mod._build_sglang_cmd(ns, node_rank=0, leader="10.0.0.1", advertise_host="10.0.0.2")
    assert "--enable-dp-attention" in cmd
    assert cmd[cmd.index("--dp-size") + 1] == "8"


def test_build_sglang_cmd_respects_explicit_dp_size():
    # An explicit --dp-size must not be overridden by the auto-fill.
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {
            "model": "/models/x",
            "tp": 8,
            "nnodes": 1,
            "dist_init_port": 5000,
            "ep": 8,
            "extra_args": "--enable-dp-attention --dp-size 4",
        },
    )()
    cmd = mod._build_sglang_cmd(ns, node_rank=0, leader="10.0.0.1", advertise_host="10.0.0.2")
    assert cmd.count("--dp-size") == 1
    assert cmd[cmd.index("--dp-size") + 1] == "4"


def test_build_sglang_cmd_injects_skip_server_warmup_for_pd_leg():
    # PD warmup can hang until SGLANG_WARMUP_TIMEOUT; the launcher skips it for
    # PD-disaggregated legs. Aggregated keeps warmup ON (single-node parity).
    mod = _load_module()
    pd_ns = type(
        "NS",
        (),
        {
            "model": "/models/x",
            "tp": 8,
            "nnodes": 1,
            "dist_init_port": 5000,
            "ep": 8,
            "extra_args": "--disaggregation-mode decode",
        },
    )()
    pd_cmd = mod._build_sglang_cmd(pd_ns, node_rank=0, leader="l", advertise_host="h")
    assert pd_cmd.count("--skip-server-warmup") == 1

    agg_ns = type(
        "NS",
        (),
        {"model": "/models/x", "tp": 8, "nnodes": 1, "dist_init_port": 5000, "ep": 8, "extra_args": ""},
    )()
    agg_cmd = mod._build_sglang_cmd(agg_ns, node_rank=0, leader="l", advertise_host="h")
    assert "--skip-server-warmup" not in agg_cmd


def test_build_sglang_cmd_skip_warmup_not_duplicated():
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {
            "model": "/models/x",
            "tp": 8,
            "nnodes": 1,
            "dist_init_port": 5000,
            "ep": 8,
            "extra_args": "--skip-server-warmup --mem-fraction-static 0.8",
        },
    )()
    cmd = mod._build_sglang_cmd(ns, node_rank=0, leader="l", advertise_host="h")
    assert cmd.count("--skip-server-warmup") == 1


def test_build_sglang_cmd_no_dp_size_without_dp_attention():
    # No DP-attention flag => no --dp-size injection.
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {
            "model": "/models/x",
            "tp": 8,
            "nnodes": 1,
            "dist_init_port": 5000,
            "ep": 8,
            "extra_args": "--mem-fraction-static 0.8",
        },
    )()
    cmd = mod._build_sglang_cmd(ns, node_rank=0, leader="10.0.0.1", advertise_host="10.0.0.2")
    assert "--dp-size" not in cmd


def test_forward_controls_apply_after_recovered_pid1_env(monkeypatch):
    """Infera must apply unset then explicit overrides after PID1 recovery."""
    mod = _load_module()
    from hyperloom.inference_optimizer.multi_node.commands import infera

    monkeypatch.setenv(
        "HYPERLOOM_MN_UNSET_FWD_ENV",
        json.dumps(["NCCL_DEBUG", "NCCL_PROTO"]),
    )
    monkeypatch.setenv(
        "HYPERLOOM_MN_EXTRA_FWD_ENV",
        json.dumps({"NCCL_PROTO": "LL"}),
    )
    recovered = {
        "NCCL_DEBUG": "INFO",
        "NCCL_PROTO": "Simple",
        "RCCL_MSCCL_ENABLE": "1",
    }

    forwarded = infera._collect_forward_env()
    assert json.loads(forwarded["HYPERLOOM_MN_UNSET_FWD_ENV"]) == ["NCCL_DEBUG"]
    assert json.loads(forwarded["HYPERLOOM_MN_EXTRA_FWD_ENV"]) == {"NCCL_PROTO": "LL"}
    recovered["HYPERLOOM_MN_UNSET_FWD_ENV"] = forwarded["HYPERLOOM_MN_UNSET_FWD_ENV"]
    recovered["HYPERLOOM_MN_EXTRA_FWD_ENV"] = forwarded["HYPERLOOM_MN_EXTRA_FWD_ENV"]

    env = mod._apply_forward_env_controls(recovered)

    assert "NCCL_DEBUG" not in env
    assert env["NCCL_PROTO"] == "LL"
    assert env["RCCL_MSCCL_ENABLE"] == "1"
    assert "HYPERLOOM_MN_UNSET_FWD_ENV" not in env
    assert "HYPERLOOM_MN_EXTRA_FWD_ENV" not in env


def test_infera_resume_identity_includes_variant_env(monkeypatch):
    """An env-only Infera variant must not resume a server with stale env."""
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli
    from hyperloom.inference_optimizer.multi_node.commands import infera

    monkeypatch.delenv("HYPERLOOM_MN_EXTRA_FWD_ENV", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_UNSET_FWD_ENV", raising=False)
    state = {
        "last_restart_framework": "sglang",
        "last_restart_model": "/m",
        "last_restart_tp": 8,
        "last_restart_ep": 8,
        "last_restart_pd_mode": "aggregated",
        "last_restart_extra_args": "--foo 1",
        "last_restart_env_fingerprint": mn_cli._variant_env_fingerprint(),
    }
    args = argparse.Namespace(model="/m", tp=8, ep=8, extra_args="--foo 1")
    assert infera._infera_restart_config_matches(state, args, "sglang", "aggregated") is True

    monkeypatch.setenv("HYPERLOOM_MN_EXTRA_FWD_ENV", json.dumps({"NCCL_PROTO": "LL"}))
    assert infera._infera_restart_config_matches(state, args, "sglang", "aggregated") is False

    monkeypatch.delenv("HYPERLOOM_MN_EXTRA_FWD_ENV")
    monkeypatch.setenv("HYPERLOOM_MN_UNSET_FWD_ENV", json.dumps(["NCCL_PROTO"]))
    assert infera._infera_restart_config_matches(state, args, "sglang", "aggregated") is False
