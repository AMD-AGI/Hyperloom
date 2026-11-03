# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from .test_common_utils import _restart_args


def test_credentials_endpoint_resolution_and_geak_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import credentials

    # An inherited key on either side resolves that side's official URL, so both must be absent for "unconfigured".
    for name in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    # Each side resolves on its own; an unconfigured side stays empty.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://open.example/v1")
    assert credentials._resolve_llm_endpoints() == ("", "https://open.example/v1")

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example")
    assert credentials._resolve_llm_endpoints() == ("https://anthropic.example", "https://open.example/v1")

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert credentials._resolve_llm_endpoints() == ("https://anthropic.example", "")

    cfg = tmp_path / "geak.yaml"
    cfg.write_text("model: x\n  base_url: https://old/v1\n", encoding="utf-8")
    assert credentials._sync_geak_config_base_url(str(cfg), r"https://new.example/\g/v1") is True
    assert r"https://new.example/\g/v1" in cfg.read_text(encoding="utf-8")
    assert credentials._sync_geak_config_base_url(str(cfg), r"https://new.example/\g/v1") is False
    assert credentials._sync_geak_config_base_url(str(tmp_path / "missing.yaml"), "https://x") is False


def test_pd_restart_against_an_aggregated_state_fails_instead_of_reporting_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart that touches nothing must not report that it launched servers."""
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    state = {
        "backend": "infera",
        "framework": "sglang",
        "nodes": 2,
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
        "ssh_key_path": "/tmp/k",
        "ssh_port": 2222,
    }
    fanout_calls: list[str] = []
    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(state))
    monkeypatch.setattr(inf._mn_cli, "_save_state", lambda _payload: None)
    monkeypatch.setattr(
        inf,
        "_infera_fanout_launch",
        lambda st, args, targets, **kw: fanout_calls.append(kw["label"]) or (0, [{"ok": True}]),
    )

    with pytest.raises(inf._mn_cli.ConfigurationError, match="no prefill or decode pods"):
        inf._infera_restart_server(_restart_args(pd_mode="disaggregated"))

    assert fanout_calls == []


@pytest.mark.parametrize(
    ("state", "match"),
    [
        ({"backend": "rayjob"}, "backend is not 'infera'"),
        ({"backend": "infera", "pd_mode": "aggregated"}, "no GPU pod IPs"),
        ({"backend": "infera", "worker_pod_ips": ["10.0.1.0"]}, "no ssh_key_path"),
    ],
)
def test_infera_state_errors_are_config_errors_not_transient(
    monkeypatch: pytest.MonkeyPatch,
    state: dict,
    match: str,
) -> None:
    """Rerunning cannot supply an SSH key, so these must not read as retryable."""
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    monkeypatch.setattr(inf._mn_cli, "_load_state", lambda: dict(state))

    with pytest.raises(inf._mn_cli.ConfigurationError, match=match):
        inf._infera_require_state()


def test_missing_gpu_pods_names_the_mode_that_chose_the_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message has to explain why IPs that ARE set look absent."""
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    monkeypatch.setattr(
        inf._mn_cli,
        "_load_state",
        lambda: {"backend": "infera", "pd_mode": "aggregated", "prefill_pod_ips": ["10.0.2.0"]},
    )

    with pytest.raises(inf._mn_cli.ConfigurationError) as excinfo:
        inf._infera_require_state()

    message = str(excinfo.value)
    assert "pd_mode='aggregated'" in message
    assert "HYPERLOOM_MN_EXT_WORKER_IPS" in message
    assert "PD_MODE=disaggregated" in message


def test_resolve_pd_node_counts_infers_from_pod_lists() -> None:
    """PD node counts fall back to discovered pod-list lengths when CLI is unset."""
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    args = argparse.Namespace(pd_prefill_nodes=0, pd_decode_nodes=0)
    state = {"prefill_pod_ips": ["10.0.1.1", "10.0.1.2"], "decode_pod_ips": ["10.0.2.1"]}
    assert inf._resolve_pd_node_counts(args, state) == (2, 1)
    assert inf._resolve_pd_node_counts(
        argparse.Namespace(pd_prefill_nodes=3, pd_decode_nodes=0),
        state,
    ) == (3, 1)


def test_server_lifecycle_remaining_resolution_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml

    from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl
    from hyperloom.orchestrator.actions.executors import benchmark_backend as bb

    monkeypatch.delenv(bb.BENCHMARK_BACKEND_ENV, raising=False)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("benchmark: [", encoding="utf-8")
    info = sl.resolve_lifecycle_params(bad_yaml)
    assert info["eligible"] is False
    assert "could not read" in info["reason"]

    invalid_port = tmp_path / "invalid-port.yaml"
    invalid_port.write_text(
        yaml.safe_dump({"benchmark": {"framework": "xdit", "envs": {"PORT": "bad"}}}),
        encoding="utf-8",
    )
    info = sl.resolve_lifecycle_params(invalid_port)
    assert info["port"] == sl.REUSE_PORT_DEFAULT
    assert "scriptable framework" in info["reason"]

    profiler = tmp_path / "profiler.yaml"
    profiler.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "benchmark_script": "vllm_mi300x.sh",
                    "envs": {"PORT": 8888},
                    "profiler": {"torch_profiler": {"enabled": True}},
                }
            }
        ),
        encoding="utf-8",
    )
    info = sl.resolve_lifecycle_params(profiler)
    assert info["eligible"] is False
    assert "torch_profiler" in info["reason"]


def test_canonical_fingerprint_remaining_normalization_branches() -> None:
    from hyperloom.inference_optimizer import canonical_fingerprint as fp

    with_controls = fp.canonical_fingerprint(
        '--flag "unterminated',
        {"B": 2},
        remove_args="--old",
        unset_envs=123,
        args_mode="bad-mode",
    )
    without_controls = fp.canonical_fingerprint("--flag unterminated", {"B": 2})
    assert len(with_controls) == 16
    assert len(without_controls) == 16
    assert with_controls != without_controls


def test_conc_sweep_plot_arm_series_sorts_by_conc() -> None:
    from hyperloom.orchestrator.kernel import conc_sweep_plot as plot

    points = [
        {"conc": 4, "output_throughput": 800},
        {"conc": 2, "output_throughput": 600},
        {"conc": 0, "output_throughput": 100},
        {"conc": None, "output_throughput": 100},
        {"conc": 1, "output_throughput": None},
    ]

    axes = plot._axes_for_metric("output_throughput", 2.0)
    xs, ys, concs = plot._arm_series(points, 2.0, axes)
    assert xs == [300.0, 200.0]
    assert ys == [300.0, 400.0]
    assert concs == [2.0, 4.0]
    assert plot._arm_series([{"conc": None, "output_throughput": -1}], 1.0, axes) == ([], [], [])
