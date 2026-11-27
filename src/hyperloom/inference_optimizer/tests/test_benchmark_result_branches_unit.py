# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measurement validity, and the multi-node fold of the pods' GPU sampler CSVs into a round's directory."""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors import benchmark_result as br


# ---- is_valid_measurement -------------------------------------------------
def test_is_valid_measurement():
    assert br.is_valid_measurement(None) is False
    assert br.is_valid_measurement("nope") is False
    assert br.is_valid_measurement({"output_throughput": 10.0, "completed_requests": 5}) is True
    assert br.is_valid_measurement({"output_throughput": 0.0, "completed_requests": 5}) is False
    assert br.is_valid_measurement({"output_throughput": 10.0, "completed_requests": 0}) is False


# ---- harvest_mn_gpu_metrics -----------------------------------------------
def test_harvest_mn_gpu_metrics_single_node_noop(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: False)
    assert br.harvest_mn_gpu_metrics(tmp_path) == {}


def test_harvest_mn_gpu_metrics_folds_pod_csvs(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setattr(
        mn,
        "pd_topology_from_state",
        lambda: {"prefill_pod_ips": ["10.0.0.1"], "decode_pod_ips": ["10.0.0.2"]},
    )

    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))

    header = "timestamp,Temperature (Junction) (C),Average Socket Power (W)"
    (shared / "gpu_metrics_10.0.0.1.csv").write_text(f"{header}\n1000,55,300\n1001,56,310\n", encoding="utf-8")
    (shared / "gpu_metrics_10.0.0.2.csv").write_text(f"{header}\n1000,50,280\n", encoding="utf-8")

    dest = tmp_path / "ws"
    dest.mkdir()

    out = br.harvest_mn_gpu_metrics(dest, subprocess_started_unix=900.0)
    assert out["rows"] == 3
    assert (dest / "gpu_metrics.csv").read_text(encoding="utf-8").splitlines() == [
        f"host,role,{header}",
        "10.0.0.1,prefill,1000,55,300",
        "10.0.0.1,prefill,1001,56,310",
        "10.0.0.2,decode,1000,50,280",
    ]


def test_harvest_mn_gpu_metrics_no_csvs(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))
    assert br.harvest_mn_gpu_metrics(tmp_path / "ws") == {}


# ---- is_valid_measurement, scriptable runs --------------------------------
def test_is_valid_measurement_scriptable_gate(monkeypatch):
    # Scriptable run: valid on positive throughput when the quality gate passes.
    assert br.is_valid_measurement({"quality_gate": {"passed": True}, "output_throughput": 5.0}) is True
    # Scriptable run whose quality gate fails is not selectable.
    monkeypatch.setattr(br, "quality_gate_passed", lambda qg, require=False: False, raising=False)
    import hyperloom.orchestrator.actions.executors._accuracy_gate as ag

    monkeypatch.setattr(ag, "quality_gate_passed", lambda qg, require=False: False)
    assert br.is_valid_measurement({"quality_gate": {"passed": False}, "output_throughput": 5.0}) is False


def test_harvest_mn_gpu_metrics_non_absolute_dir(monkeypatch, tmp_path):
    # Unresolved / relative shared dir is treated as absent.
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", "relative/path")
    assert br.harvest_mn_gpu_metrics(tmp_path / "ws") == {}


def test_harvest_mn_gpu_metrics_window_and_malformed_rows(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setattr(mn, "pd_topology_from_state", dict)

    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))

    header = "timestamp,Temperature (Junction) (C),Average Socket Power (W)"
    # Rows: one in-window, one out-of-window (dropped), one blank, one bad ts.
    (shared / "gpu_metrics_hostA.csv").write_text(f"{header}\n1000,55,300\n5,50,280\n\n,51,290\n", encoding="utf-8")
    # A too-short file (header only) is skipped.
    (shared / "gpu_metrics_hostB.csv").write_text(f"{header}\n", encoding="utf-8")

    dest = tmp_path / "ws"
    dest.mkdir()
    out = br.harvest_mn_gpu_metrics(dest, subprocess_started_unix=900.0)
    assert out["rows"] == 1
    assert (dest / "gpu_metrics.csv").read_text(encoding="utf-8").splitlines() == [
        f"host,{header}",
        "hostA,1000,55,300",
    ]


@pytest.mark.parametrize(
    "pod_rows",
    [
        pytest.param("5,50,280\n", id="every row outside the round's window"),
        pytest.param("1000," + "x" * 200_000 + "\n", id="a field the csv module refuses"),
    ],
)
def test_harvest_mn_gpu_metrics_writes_nothing_without_usable_rows(monkeypatch, tmp_path, pod_rows):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setattr(mn, "pd_topology_from_state", dict)
    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))
    (shared / "gpu_metrics_hostA.csv").write_text(f"timestamp,temp,power\n{pod_rows}", encoding="utf-8")

    dest = tmp_path / "ws"
    assert br.harvest_mn_gpu_metrics(dest, subprocess_started_unix=900.0) == {}
    assert not (dest / "gpu_metrics.csv").exists()
