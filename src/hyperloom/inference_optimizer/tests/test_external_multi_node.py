# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cluster hand-off: env-synthesized multi-node state and its adoption guards."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
import types
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli.multi_node import _prepare_multi_node_state
from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.inference_optimizer.multi_node.state_paths import resolve_state_file


@pytest.fixture()
def _external_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Minimal infera PD external env without SaFE credentials."""
    monkeypatch.delenv("SAFE_API_URL", raising=False)
    monkeypatch.delenv("SAFE_API_KEY", raising=False)
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://frontend:8000")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_PREFILL_IPS", "10.0.1.1")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_DECODE_IPS", "10.0.1.2")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_KEY", str(tmp_path / "id_ed25519"))
    (tmp_path / "id_ed25519").write_text("fake-key", encoding="utf-8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("PD_MODE", "disaggregated")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "infera")
    session = tmp_path / "session"
    session.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    return session


def test_build_external_state_from_env_pd_topology(_external_env: Path) -> None:
    state = ext.build_external_state_from_env()
    assert state["external"] is True
    assert state["service_url"] == "http://frontend:8000"
    assert state["backend"] == "infera"
    assert state["prefill_pod_ips"] == ["10.0.1.1"]
    assert state["decode_pod_ips"] == ["10.0.1.2"]
    assert state["last_restart_pd_prefill_nodes"] == 1
    assert state["last_restart_pd_decode_nodes"] == 1


@pytest.mark.parametrize(
    ("explicit", "ssh", "head_ip", "expected"),
    [
        # An explicit backend wins even against the opposite hand-off shape.
        ("rayjob", True, "", "rayjob"),
        ("infera", False, "10.0.0.9", "infera"),
        # Without one, the shape the platform handed over decides.
        ("", True, "", "infera"),
        ("", False, "10.0.0.9", "rayjob"),
        # SSH outranks a head IP, matching external_has_server_control.
        ("", True, "10.0.0.9", "infera"),
        # Benchmark-only (neither shape): restarts are skipped, CLI default applies.
        ("", False, "", "rayjob"),
        # An unusable explicit value defers to the shape instead of guessing.
        ("bogus", False, "10.0.0.9", "rayjob"),
    ],
)
def test_backend_follows_handoff_shape_without_an_explicit_env(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    explicit: str,
    ssh: bool,
    head_ip: str,
    expected: str,
) -> None:
    """``state["backend"]`` routes every hyperloom-mn subcommand to SSH or Ray.

    The platform can export the ``HYPERLOOM_MN_EXT_*`` block without the
    companion backend var, so a hardcoded default sent those subcommands at the
    wrong control plane; the hand-off itself has to decide.
    """
    if explicit:
        monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", explicit)
    else:
        monkeypatch.delenv("INFERENCE_OPTIMIZER_MN_BACKEND", raising=False)
    if ssh:
        monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_KEY", str(tmp_path / "id_ed25519"))
        monkeypatch.setenv("HYPERLOOM_MN_EXT_WORKER_IPS", "10.0.2.1")
    else:
        monkeypatch.delenv("HYPERLOOM_MN_EXT_SSH_KEY", raising=False)
        for role in ("PREFILL", "DECODE", "WORKER"):
            monkeypatch.delenv(f"HYPERLOOM_MN_EXT_{role}_IPS", raising=False)
    if head_ip:
        monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", head_ip)
    else:
        monkeypatch.delenv("HYPERLOOM_MN_EXT_HEAD_IP", raising=False)

    assert ext.build_external_state_from_env()["backend"] == expected


def test_external_state_default_ssh_port_matches_image(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the platform omits ``HYPERLOOM_MN_EXT_SSH_PORT``, use the image default."""
    from hyperloom.inference_optimizer.multi_node._internal.ssh_client import DEFAULT_SSH_PORT

    monkeypatch.delenv("HYPERLOOM_MN_EXT_SSH_PORT", raising=False)
    monkeypatch.setenv("HYPERLOOM_MN_EXT_PREFILL_IPS", "10.0.1.1")
    monkeypatch.delenv("HYPERLOOM_MN_EXT_DECODE_IPS", raising=False)

    state = ext.build_external_state_from_env()
    assert state["ssh_port"] == DEFAULT_SSH_PORT
    assert state["prefill_pods"][0]["sshPort"] == DEFAULT_SSH_PORT


def test_external_pod_ssh_ports_follow_the_lws_ordinal(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each pod of a multi-node role listens on ``base + role_offset + ordinal``.

    Regression guard. ``mn-idle.sh`` is launched with
    ``MN_SSH_PORT=$(( role_base + ${LWS_WORKER_INDEX:-0} ))``, so the pods of one
    LeaderWorkerSet group listen on consecutive ports. External mode used to hand
    the same ``role_base`` to every IP in a role, which is invisible while a role
    has a single pod but makes every non-leader unreachable as soon as the role
    spans nodes. The platform supplies the IP lists in LWS ordinal order (leader
    first), so the list index IS the ordinal.
    """
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_PORT", "2222")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_PREFILL_IPS", "10.0.1.1,10.0.1.2")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_DECODE_IPS", "10.0.2.1,10.0.2.2")

    state = ext.build_external_state_from_env()

    # prefill keeps the base; decode is strided so co-located roles never collide.
    assert [p["sshPort"] for p in state["prefill_pods"]] == [2222, 2223]
    assert [p["sshPort"] for p in state["decode_pods"]] == [2232, 2233]
    # Ports must pair with the right IP, in the order the platform listed them.
    assert [(p["podIP"], p["sshPort"]) for p in state["prefill_pods"]] == [
        ("10.0.1.1", 2222),
        ("10.0.1.2", 2223),
    ]


def test_external_aggregated_worker_ports_follow_the_lws_ordinal(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The aggregated worker role is the common multi-node shape: one LWS group."""
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_PORT", "2222")
    monkeypatch.setenv("PD_MODE", "aggregated")
    monkeypatch.delenv("HYPERLOOM_MN_EXT_PREFILL_IPS", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_DECODE_IPS", raising=False)
    monkeypatch.setenv("HYPERLOOM_MN_EXT_WORKER_IPS", "10.0.3.1,10.0.3.2,10.0.3.3")

    state = ext.build_external_state_from_env()

    assert [p["sshPort"] for p in state["worker_pods"]] == [2222, 2223, 2224]


def test_load_multi_node_state_falls_back_to_env(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ext.load_multi_node_state()["service_url"] == "http://frontend:8000"


def test_load_multi_node_state_prefers_env_over_stale_disk(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External env must override a leftover SaFE state file in the same session."""
    state_path = resolve_state_file()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "backend": "infera",
                "service_url": "http://old-safe:8000",
                "prefill_pod_ips": ["10.9.9.9"],
                "ssh_key_path": "/old/key",
            }
        ),
        encoding="utf-8",
    )
    loaded = ext.load_multi_node_state()
    assert loaded["service_url"] == "http://frontend:8000"
    assert loaded["prefill_pod_ips"] == ["10.0.1.1"]
    assert loaded.get("external") is True


def test_load_multi_node_state_honours_handoff_alongside_safe_creds(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The platform sandbox case: LLM creds and a cluster hand-off coexist.

    SAFE_API_* authenticate the LLM gateway and are set in essentially every
    platform sandbox, so gating the hand-off on them made multi-node unusable
    exactly where the integration runs. The hand-off must still win here.
    """
    monkeypatch.setenv("SAFE_API_URL", "http://safe")
    monkeypatch.setenv("SAFE_API_KEY", "key")
    state_path = resolve_state_file()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"backend": "infera", "service_url": "http://stale-cluster:8000"}),
        encoding="utf-8",
    )
    loaded = ext.load_multi_node_state()
    assert loaded["service_url"] == "http://frontend:8000"
    assert loaded["prefill_pod_ips"] == ["10.0.1.1"]
    assert loaded.get("external") is True


def test_external_mode_signals_ignore_safe_creds(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every external-mode signal keys off the hand-off, never off SaFE creds."""
    monkeypatch.setenv("SAFE_API_URL", "http://safe")
    monkeypatch.setenv("SAFE_API_KEY", "key")
    assert ext.external_service_url() == "http://frontend:8000"
    assert ext.external_has_ssh_control() is True
    assert ext.external_has_server_control() is True


def test_provision_external_writes_state_and_skips_safe(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        nodes=2,
        mn_backend="infera",
        mn_image=None,
        model="/models/test",
        no_kernel=True,
    )
    _prepare_multi_node_state(args)
    state_path = resolve_state_file()
    assert state_path.is_file()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["external"] is True
    assert saved["prefill_pod_ips"] == ["10.0.1.1"]
    assert os.environ["BENCHMARK_BASE_URL"] == "http://frontend:8000"
    assert os.environ["MAGPIE_RUN_PHASE"] == "client"


def test_provision_external_infera_requires_ssh(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HYPERLOOM_MN_EXT_SSH_KEY", raising=False)
    args = argparse.Namespace(
        nodes=2,
        mn_backend="infera",
        mn_image=None,
        model="/models/test",
        no_kernel=True,
    )
    with pytest.raises(SystemExit) as ei:
        _prepare_multi_node_state(args)
    assert ei.value.code == 2


def _adopt_rayjob(monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    """Neutralize the session-side setup so adoption reduces to its reporting."""
    from hyperloom.inference_optimizer.cli import multi_node as mn
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "rayjob")
    monkeypatch.setattr(mncli, "cmd_bootstrap", lambda _ns: 0)
    monkeypatch.setattr(mncli, "install_geak_on_pods_best_effort", lambda: 0)
    monkeypatch.setattr(mn, "_replay_kernel_patches_for_multi_node", lambda _a: None)
    return argparse.Namespace(nodes=2, mn_backend="rayjob", mn_image=None, model="/models/test", no_kernel=True)


def test_rayjob_with_head_ip_is_not_reported_benchmark_only(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """rayjob restarts through the Ray dashboard, with no SSH anywhere.

    Keying the adoption line on SSH labelled this fully controllable cluster
    "benchmark-only" -- the same words the genuinely uncontrollable one gets --
    so the one line an operator reads to learn whether the run can tune anything
    said the opposite of the truth.
    """
    # The distinguishing shape: a real rayjob hand-off carries HEAD_IP and no SSH
    # material at all, so keying on SSH flips the verdict while control is intact.
    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.2.1")
    monkeypatch.delenv("HYPERLOOM_MN_EXT_SSH_KEY", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_PREFILL_IPS", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_DECODE_IPS", raising=False)
    _prepare_multi_node_state(_adopt_rayjob(monkeypatch))

    out = capsys.readouterr()
    assert "server_control=yes" in out.out
    assert "ssh_control=no" in out.out
    assert "benchmark-only" not in out.out
    assert "WARNING: no server control" not in out.err


def test_rayjob_without_head_ip_warns_results_are_meaningless(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Benchmark-only is legal but must not pass for a real optimization run.

    Without server control every candidate re-measures the one unchanged server,
    so the run still reports gains no config produced. That was a single info
    line from the per-round restart helper; adoption has to say it where the
    operator is actually looking.
    """
    # A rayjob hand-off carries neither the SSH key nor the pod IPs; only
    # SERVICE_URL is guaranteed, and HEAD_IP is the documented opt-in.
    monkeypatch.delenv("HYPERLOOM_MN_EXT_HEAD_IP", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_SSH_KEY", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_PREFILL_IPS", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_DECODE_IPS", raising=False)
    _prepare_multi_node_state(_adopt_rayjob(monkeypatch))

    out = capsys.readouterr()
    assert "server_control=no (benchmark-only)" in out.out
    assert "WARNING: no server control" in out.err
    assert "measures the SAME unchanged server" in out.err


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        # A ClusterIP name resolves only inside the cluster; the head pod is the
        # way in from outside it, on the URL's own port.
        (
            {"service_url": "http://wid.ns.svc.cluster.local:8000", "head_pod_ip": "10.0.2.1"},
            "http://10.0.2.1:8000",
        ),
        # No port on the URL: fall back to the frontend port the launch scripts publish.
        (
            {"service_url": "http://wid.ns.svc.cluster.local", "head_pod_ip": "10.0.2.1"},
            "http://10.0.2.1:8888",
        ),
        # Nothing to rewrite through.
        ({"service_url": "http://wid.ns.svc.cluster.local:8000"}, "http://wid.ns.svc.cluster.local:8000"),
        # Already an address this sandbox can dial.
        ({"service_url": "http://10.0.9.9:8000", "head_pod_ip": "10.0.2.1"}, "http://10.0.9.9:8000"),
        ({}, ""),
    ],
)
def test_reachable_service_url_prefers_the_head_pod_over_a_clusterip_name(
    state: dict[str, object],
    expected: str,
) -> None:
    """One definition of "where the frontend answers" for every caller.

    The rewrite was copy-pasted into the benchmark env builder and the
    post-restart health wait, so the two could drift into disagreeing about
    which endpoint a run is actually talking to.
    """
    assert ext.reachable_service_url(state) == expected


def _write_disk_state(**overrides: object) -> Path:
    """Persist an external state file carrying resume bookkeeping."""
    state_file = resolve_state_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "external": True,
        "backend": "infera",
        "service_url": "http://frontend:8000",
        "prefill_pod_ips": ["10.0.1.1"],
        "decode_pod_ips": ["10.0.1.2"],
        "worker_pod_ips": [],
        "last_restart_submission_id": "launch-1",
        "last_restart_framework": "sglang",
    }
    state.update(overrides)
    state_file.write_text(json.dumps(state), encoding="utf-8")
    return state_file


def test_bookkeeping_carries_over_within_the_same_handoff(_external_env: Path) -> None:
    """The whole point of the merge: one cluster's session survives a reload."""
    _write_disk_state()

    state = ext.load_multi_node_state()

    assert state["last_restart_submission_id"] == "launch-1"
    assert state["last_restart_framework"] == "sglang"


def test_pd_leg_urls_carry_over_within_the_same_handoff(_external_env: Path) -> None:
    """PD leg URLs lack the ``last_`` prefix but must survive a reload too.

    They are read from the launcher summary and describe this cluster's legs.
    Dropping them made the CLI's mid-restart checkpoint persist a state without
    them, so a later launch failure wiped them on disk, and the PD serving/resume
    probe then saw no legs and could never resume.
    """
    _write_disk_state(
        pd_prefill_url="http://10.32.17.187:30000",
        pd_decode_url="http://10.32.17.185:30001",
    )

    state = ext.load_multi_node_state()

    assert state["pd_prefill_url"] == "http://10.32.17.187:30000"
    assert state["pd_decode_url"] == "http://10.32.17.185:30001"


def test_pd_leg_urls_are_dropped_when_the_cluster_was_replaced(_external_env: Path) -> None:
    """Same identity guard as the bookkeeping: a replacement cluster's legs differ."""
    _write_disk_state(
        service_url="http://other-frontend:8000",
        pd_prefill_url="http://10.9.9.1:30000",
        pd_decode_url="http://10.9.9.2:30001",
    )

    state = ext.load_multi_node_state()

    assert "pd_prefill_url" not in state
    assert "pd_decode_url" not in state


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("service_url", "http://other-frontend:8000"),
        ("prefill_pod_ips", ["10.9.9.1"]),
        ("decode_pod_ips", ["10.9.9.2"]),
        ("head_pod_ip", "10.9.9.3"),
    ],
)
def test_bookkeeping_is_dropped_when_the_cluster_was_replaced(
    _external_env: Path,
    field: str,
    value: object,
) -> None:
    """A state file outlives the cluster it describes, so identity must be checked.

    ``last_restart_submission_id`` names a job on one Ray cluster and
    ``last_server_pid_dir`` names PIDs on one set of pods. Carrying either onto a
    replacement cluster hands the resume fast path a launch identity that was
    never valid here, letting it skip a relaunch for a server this cluster never
    started. The merge only tested the ``external`` flag, which a stale file from
    a previous hand-off also carries.
    """
    _write_disk_state(**{field: value})

    state = ext.load_multi_node_state()

    assert "last_restart_submission_id" not in state
    assert "last_restart_framework" not in state
    # The hand-off itself still wins: only the bookkeeping was dropped.
    assert state["service_url"] == "http://frontend:8000"
    assert state["prefill_pod_ips"] == ["10.0.1.1"]


def test_unset_nodes_stays_single_pod_and_says_so(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unstated count is reported, never inferred.

    ``nodes`` is passed verbatim to the framework as ``--nnodes``, which then
    waits for exactly that many ranks, so guessing high hangs the launch instead
    of degrading it. Guessing low only takes the single-pod path, which is
    recoverable once it is visible -- so the run stays at 1 and names the pods it
    saw, rather than inventing a size the caller never stated.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)

    with caplog.at_level(logging.WARNING):
        assert ext.build_external_state_from_env()["nodes"] == 1

    assert "carries 2 GPU pod IPs" in caplog.text
    assert "--nodes 2" in caplog.text


def test_a_single_pod_handoff_without_nodes_warns_about_nothing(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One pod and one node agree, so there is nothing to report."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_DECODE_IPS", raising=False)
    monkeypatch.setenv("PD_MODE", "aggregated")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_WORKER_IPS", "10.0.1.1")

    with caplog.at_level(logging.WARNING):
        assert ext.build_external_state_from_env()["nodes"] == 1

    assert "GPU pod IPs" not in caplog.text


@pytest.mark.parametrize(
    ("pd_mode", "env", "expected"),
    [
        # On contract: the unused list is empty either way.
        ("disaggregated", {"HYPERLOOM_MN_EXT_PREFILL_IPS": "a,b", "HYPERLOOM_MN_EXT_DECODE_IPS": "c,d"}, 4),
        ("aggregated", {"HYPERLOOM_MN_EXT_WORKER_IPS": "a,b,c,d"}, 4),
        # Off contract: the worker list repeats the PD pods. Summing all three
        # reported eight pods for a cluster of four.
        (
            "disaggregated",
            {
                "HYPERLOOM_MN_EXT_PREFILL_IPS": "a,b",
                "HYPERLOOM_MN_EXT_DECODE_IPS": "c,d",
                "HYPERLOOM_MN_EXT_WORKER_IPS": "a,b,c,d",
            },
            4,
        ),
        # A role listing the same pod twice is still one pod.
        ("aggregated", {"HYPERLOOM_MN_EXT_WORKER_IPS": "a,a,b"}, 2),
    ],
)
def test_pod_count_follows_the_mode_that_will_use_the_pods(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pd_mode: str,
    env: dict[str, str],
    expected: int,
) -> None:
    """The three IP lists are alternatives, and the mode picks between them.

    ``gpu_ssh_targets_from_state`` selects prefill+decode or worker by mode, and
    SKILL.md documents the same split, so a count that adds all three describes
    a cluster nobody has.
    """
    for role in ("PREFILL", "DECODE", "WORKER"):
        monkeypatch.delenv(f"HYPERLOOM_MN_EXT_{role}_IPS", raising=False)
    monkeypatch.setenv("PD_MODE", pd_mode)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # A deliberately wrong stated count, so the reported one lands in the log.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "99")

    with caplog.at_level(logging.WARNING):
        ext.build_external_state_from_env()

    assert f"disagrees with the {expected} GPU pod IPs" in caplog.text


def test_stated_single_node_survives_a_multi_pod_handoff(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single-node guarantee outranks the hand-off's shape.

    ``is_multi_node()`` reads ``state["nodes"]`` before it reads the environment,
    so deriving a larger count from the handed-over pod IPs would drag a
    single-node run onto the multi-node path in any sandbox that exports them for
    an unrelated cluster. A stated value is therefore honoured verbatim.
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")

    assert ext.build_external_state_from_env()["nodes"] == 1


def test_a_rayjob_handoff_takes_its_node_count_only_from_the_flag(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rayjob hand-off names its cluster by head IP and never by pod, so its
    size can only come from the caller.

    Nothing in ``HYPERLOOM_MN_EXT_HEAD_IP`` says how large the cluster is -- it
    is a Service address -- which is why inferring a count from the hand-off
    could never have helped this backend.
    """
    monkeypatch.delenv("HYPERLOOM_MN_EXT_PREFILL_IPS", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_DECODE_IPS", raising=False)
    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.2.1")

    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "4")
    assert ext.build_external_state_from_env()["nodes"] == 4

    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES")
    assert ext.build_external_state_from_env()["nodes"] == 1


@pytest.mark.parametrize(
    ("builder", "build"),
    [
        ("_build_kill_single_entrypoint", lambda fn, path: fn(path)),
        ("_build_multinode_kill_entrypoint", lambda fn, path: fn(path)),
    ],
)
def test_kill_entrypoints_quote_state_derived_paths(builder: str, build: object) -> None:
    """A PID path reaches the pod as one argument, whatever it contains.

    These paths come from ``--pid-file`` or the state file's
    ``last_server_pid_dir``, and were interpolated raw while the neighbouring
    restart entrypoint quoted the same values. A directory with a space split
    into two arguments and the kill silently addressed the wrong path.
    """
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    entrypoint = build(getattr(mncli, builder), "/tmp/my pids/rank.pid")  # type: ignore[operator]

    assert "'/tmp/my pids/rank.pid'" in entrypoint


def test_launch_entrypoint_quotes_state_derived_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same for the launch entrypoint's pid/log directories."""
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_TRACE_DIR", "")
    args = argparse.Namespace(
        framework="sglang",
        model="/models/test",
        tp=8,
        ep=1,
        pd_mode="aggregated",
        extra_args="",
        no_wait_health=False,
    )

    entrypoint = mncli._build_multinode_launch_entrypoint(args, 2, "/tmp/my pids", "/tmp/my logs")

    assert "--pid-dir '/tmp/my pids'" in entrypoint
    assert "--log-dir '/tmp/my logs'" in entrypoint


_AGGREGATED_STATE = {"service_url": "http://frontend:8000"}
_PD_STATE = {
    "service_url": "http://frontend:8000",
    "pd_prefill_url": "http://10.0.1.1:30000",
    "pd_decode_url": "http://10.0.1.2:30001",
}


def _fake_serving_httpx(
    *,
    health: int = 200,
    down_legs: tuple[str, ...] = (),
    models: tuple[str, ...] = ("model-a",),
    completion_status: int = 200,
    completion_tokens: int = 8,
    seen: list[str] | None = None,
    budgets: list[float] | None = None,
    delay_s: float = 0.0,
) -> types.SimpleNamespace:
    """Stand-in httpx routing by URL across the three serving checks."""
    if budgets is None:
        budgets = []

    class _Resp:
        def __init__(self, status: int, payload: dict | None = None) -> None:
            self.status_code = status
            self._payload = payload or {}

        def json(self) -> dict:
            return self._payload

    class _Client:
        def __init__(self, timeout: object = None) -> None:
            self.timeout = timeout

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

        def get(self, url: str, timeout: float | None = None) -> _Resp:
            if seen is not None:
                seen.append(url)
            if timeout is not None:
                budgets.append(timeout)
            if delay_s:
                time.sleep(delay_s)
            if url.endswith("/health"):
                return _Resp(503 if url[: -len("/health")] in down_legs else health)
            if url.endswith("/v1/models"):
                return _Resp(200, {"data": [{"id": m} for m in models]})
            raise AssertionError(f"unexpected GET {url}")

        def post(self, url: str, json: dict | None = None, timeout: float | None = None) -> _Resp:
            if seen is not None:
                seen.append(url)
            if timeout is not None:
                budgets.append(timeout)
            return _Resp(completion_status, {"usage": {"completion_tokens": completion_tokens}})

    return types.SimpleNamespace(Client=_Client)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"usage": {"completion_tokens": 5}}, 5),
        # No usage block: a non-empty text still proves something was generated.
        ({"choices": [{"text": "hello"}]}, 1),
        # The broken PD KV handoff: 200, and nothing came back.
        ({"choices": [{"text": "   "}]}, 0),
        ({"usage": {"completion_tokens": 0}, "choices": [{"text": ""}]}, 0),
        # Shapes a server should never send, which must not raise here.
        ({"usage": {"completion_tokens": "many"}}, 0),
        ({"usage": "nonsense", "choices": []}, 0),
        ({}, 0),
        ("not a body", 0),
    ],
)
def test_generated_tokens_counts_only_what_was_actually_produced(body: object, expected: int) -> None:
    """The reading that separates a serving cluster from one that answers 200."""
    from hyperloom.inference_optimizer.multi_node._internal import serving_probe

    assert serving_probe.generated_tokens(body) == expected


def test_probe_thresholds_survive_a_junk_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed tunable must not decide whether a cluster looks alive."""
    from hyperloom.inference_optimizer.multi_node._internal import serving_probe

    monkeypatch.setenv("HYPERLOOM_MN_COMPLETION_PROBE_TOKENS", "lots")

    assert serving_probe._int_env("HYPERLOOM_MN_COMPLETION_PROBE_TOKENS", 8) == 8


def test_cluster_is_serving_survives_a_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable endpoint is a "not serving" answer, never an exception.

    The caller treats any raise as a failed probe, but letting one escape here
    would bypass the liveness check that decides whether a relaunch is safe.
    """
    from hyperloom.inference_optimizer.multi_node._internal import serving_probe

    class _Exploding:
        def __init__(self, timeout: object = None) -> None:
            self.timeout = timeout

        def __enter__(self) -> "_Exploding":
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

        def get(self, _url: str, timeout: float | None = None) -> object:
            raise ConnectionError("connection refused")

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=_Exploding))

    assert serving_probe.cluster_is_serving(_AGGREGATED_STATE, pd_mode="aggregated", timeout_s=5) is False


def test_cluster_is_serving_without_httpx_is_not_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight backfills httpx; until it does, nothing can be confirmed."""
    from hyperloom.inference_optimizer.multi_node._internal import serving_probe

    monkeypatch.setitem(sys.modules, "httpx", None)

    assert serving_probe.cluster_is_serving(_AGGREGATED_STATE, pd_mode="aggregated", timeout_s=5) is False


def test_cluster_is_serving_accepts_a_cluster_that_generates_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one answer that lets a resume skip a relaunch outright."""
    from hyperloom.inference_optimizer.multi_node._internal import serving_probe

    monkeypatch.setitem(sys.modules, "httpx", _fake_serving_httpx())

    assert serving_probe.cluster_is_serving(_AGGREGATED_STATE, pd_mode="aggregated", timeout_s=5) is True


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ({"health": 503}, "the group is not up"),
        ({"models": ()}, "the workers died during the weight load, so nothing registered"),
        ({"completion_status": 503}, "registered but refusing traffic"),
        # The case a status-only check calls healthy: a broken PD KV handoff
        # answers 200 with an empty completion.
        ({"completion_tokens": 0}, "200 with no tokens generated"),
    ],
)
def test_cluster_is_serving_rejects_each_way_a_cluster_can_look_up_but_not_serve(
    monkeypatch: pytest.MonkeyPatch,
    failure: dict,
    reason: str,
) -> None:
    """Each check exists because it catches what the previous one misses."""
    from hyperloom.inference_optimizer.multi_node._internal import serving_probe

    monkeypatch.setitem(sys.modules, "httpx", _fake_serving_httpx(**failure))

    assert serving_probe.cluster_is_serving(_AGGREGATED_STATE, pd_mode="aggregated", timeout_s=5) is False, reason


def test_pd_disaggregated_health_checks_both_legs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frontend answers while a leg is still loading, so ask the legs."""
    from hyperloom.inference_optimizer.multi_node._internal import serving_probe

    seen: list[str] = []
    monkeypatch.setitem(sys.modules, "httpx", _fake_serving_httpx(seen=seen))

    assert serving_probe.cluster_is_serving(_PD_STATE, pd_mode="disaggregated", timeout_s=5) is True
    assert "http://10.0.1.1:30000/health" in seen
    assert "http://10.0.1.2:30001/health" in seen


def test_pd_disaggregated_rejects_a_single_dead_leg(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prefill that serves cannot stand in for a decode leg that does not."""
    from hyperloom.inference_optimizer.multi_node._internal import serving_probe

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        _fake_serving_httpx(down_legs=("http://10.0.1.2:30001",)),
    )

    assert serving_probe.cluster_is_serving(_PD_STATE, pd_mode="disaggregated", timeout_s=5) is False


def test_serving_probe_budget_is_shared_by_every_request_it_makes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The budget is for the probe, not for each call inside it.

    PD asks up to four times over: both legs, /v1/models, then a completion. A
    per-request timeout would let a slow cluster spend several multiples of the
    stated budget while the caller believes it bounded the wait.
    """
    from hyperloom.inference_optimizer.multi_node._internal import serving_probe

    budgets: list[float] = []
    monkeypatch.setitem(sys.modules, "httpx", _fake_serving_httpx(budgets=budgets, delay_s=0.05))

    assert serving_probe.cluster_is_serving(_PD_STATE, pd_mode="disaggregated", timeout_s=2) is True
    assert len(budgets) == 4
    assert budgets == sorted(budgets, reverse=True), "each call must inherit what the last one left"
    assert budgets[0] <= 2


def test_serving_probe_stops_once_its_budget_is_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spent budget ends the probe instead of starting another request."""
    from hyperloom.inference_optimizer.multi_node._internal import serving_probe

    seen: list[str] = []
    monkeypatch.setitem(sys.modules, "httpx", _fake_serving_httpx(seen=seen, delay_s=0.05))

    # Enough for the first leg, nothing left for the second.
    assert serving_probe.cluster_is_serving(_PD_STATE, pd_mode="disaggregated", timeout_s=0.04) is False
    assert seen == ["http://10.0.1.1:30000/health"]


def test_unresolvable_endpoints_short_circuit_without_asking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not knowing where to look is not evidence, and must cost no request.

    A PD state with no recorded leg URLs cannot be verified. Returning False is
    only half of it: issuing a request to a nonsense URL would spend the whole
    probe budget on a connection that was never going to resolve.
    """
    from hyperloom.inference_optimizer.multi_node._internal import serving_probe

    seen: list[str] = []
    monkeypatch.setitem(sys.modules, "httpx", _fake_serving_httpx(seen=seen))

    assert serving_probe.cluster_is_serving(_AGGREGATED_STATE, pd_mode="disaggregated", timeout_s=5) is False
    assert serving_probe.cluster_is_serving({}, pd_mode="aggregated", timeout_s=5) is False
    assert seen == []


@pytest.mark.parametrize(
    ("serving", "no_wait_health", "expect_relaunch"),
    [
        (True, False, False),
        (False, False, True),
        # The launch opted out of waiting, so a terminal-OK driver proves
        # nothing and the probe is not consulted.
        (True, True, True),
    ],
)
def test_resume_needs_a_cluster_that_still_serves(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    serving: bool,
    no_wait_health: bool,
    expect_relaunch: bool,
) -> None:
    """A terminal-OK driver means this cluster served once, not that it still does.

    The driver waits for the servers it spawned before exiting, so SUCCEEDED is
    a claim about a moment that has passed. Resuming on it alone returned 0 with
    nothing serving after a crash. Asking the endpoint settles it -- and where
    the launch waived that wait, there is no claim to check, so no resume.
    """
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.2.1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "rayjob")
    monkeypatch.setenv("PD_MODE", "aggregated")
    _write_disk_state(
        backend="rayjob",
        head_pod_ip="10.0.2.1",
        last_restart_model="/models/test",
        last_restart_tp=8,
        last_restart_ep=1,
        last_restart_pd_mode="aggregated",
        last_restart_extra_args="",
    )

    submitted: list[str] = []
    killed: list[str] = []
    probed: list[str] = []

    class _FakeRay:
        def get_job(self, _sub: str) -> dict[str, str]:
            return {"status": "SUCCEEDED"}

        def submit_job(self, _ep: str, runtime_env: object = None) -> str:
            submitted.append("launch")
            return "launch-2"

    @contextlib.contextmanager
    def _fake_client(_state: object):
        yield _FakeRay()

    monkeypatch.setattr(mncli, "_ray_dashboard_client", _fake_client)
    monkeypatch.setattr(mncli, "_build_multinode_kill_entrypoint", lambda *_a, **_k: "kill-ep")
    monkeypatch.setattr(mncli, "_build_multinode_launch_entrypoint", lambda *_a, **_k: "launch-ep")
    monkeypatch.setattr(mncli, "_exec_kill_submission", lambda *_a, **_k: killed.append("kill") or "kill-9")
    monkeypatch.setattr(mncli, "_short_poll", lambda **_k: {"status": "SUCCEEDED"})
    monkeypatch.setattr(mncli, "cluster_is_serving", lambda *_a, **_k: probed.append("probe") or serving)

    args = argparse.Namespace(
        framework="sglang",
        model="/models/test",
        tp=8,
        ep=1,
        pd_mode="aggregated",
        extra_args="",
        pid_file=None,
        log_file=None,
        poll_interval=1,
        poll_timeout=5,
        print_logs=False,
        no_wait_health=no_wait_health,
    )

    assert mncli.cmd_restart_server(args) == 0
    assert bool(killed) is expect_relaunch
    assert bool(submitted) is expect_relaunch
    # Nothing to verify when the launch never made the claim.
    assert probed == ([] if no_wait_health else ["probe"])


def test_an_infera_handoff_left_on_the_default_backend_is_rejected(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Forgetting --mn-backend infera must fail here, not minutes later.

    --mn-backend defaults to rayjob and overwrites whatever the hand-off's shape
    implies, so an infera cluster whose operator omitted the flag was adopted as
    rayjob. Nothing caught it: server_control reads yes, because SSH control is
    real -- it is simply not the control this backend uses. The run then died
    inside a per-round restart on a head_pod_ip nobody had asked for, well after
    the adoption line said the cluster was fine.
    """
    monkeypatch.delenv("HYPERLOOM_MN_EXT_HEAD_IP", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "rayjob")

    args = argparse.Namespace(nodes=2, mn_backend="rayjob", model="/models/test", no_kernel=True)
    with pytest.raises(SystemExit) as excinfo:
        _prepare_multi_node_state(args)

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "it is an infera cluster" in err
    assert "--mn-backend infera" in err


def test_a_genuine_benchmark_only_handoff_still_proceeds(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No HEAD_IP and no SSH is the documented benchmark-only mode, not an error.

    The new guard keys on the hand-off carrying SSH control instead, so it
    separates "you meant infera" from "this cluster genuinely cannot be
    restarted".
    """
    from hyperloom.inference_optimizer.cli import multi_node as mn
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    monkeypatch.delenv("HYPERLOOM_MN_EXT_HEAD_IP", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_SSH_KEY", raising=False)
    for role in ("PREFILL", "DECODE", "WORKER"):
        monkeypatch.delenv(f"HYPERLOOM_MN_EXT_{role}_IPS", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "rayjob")
    monkeypatch.setattr(mncli, "install_geak_on_pods_best_effort", lambda: 0)
    monkeypatch.setattr(mn, "_replay_kernel_patches_for_multi_node", lambda _a: None)

    args = argparse.Namespace(nodes=2, mn_backend="rayjob", model="/models/test", no_kernel=True)
    _prepare_multi_node_state(args)

    assert "benchmark-only" in capsys.readouterr().out


def test_benchmark_only_rayjob_handoff_skips_bootstrap_instead_of_aborting(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Benchmark-only must survive the adoption it was just promised.

    Adoption printed "Continuing in benchmark-only mode" and then submitted the
    head-pod bootstrap anyway. ``cmd_bootstrap`` opens with
    ``_require_state("head_pod_ip")``, which is exactly the field this mode
    omits, so the run died on the line that said it would continue.

    ``cmd_bootstrap`` is deliberately left unstubbed: stubbing it to return 0 is
    what hid the abort from the existing adoption tests.
    """
    from hyperloom.inference_optimizer.cli import multi_node as mn
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    monkeypatch.delenv("HYPERLOOM_MN_EXT_HEAD_IP", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_SSH_KEY", raising=False)
    for role in ("PREFILL", "DECODE", "WORKER"):
        monkeypatch.delenv(f"HYPERLOOM_MN_EXT_{role}_IPS", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "rayjob")
    calls: list[str] = []
    monkeypatch.setattr(mncli, "install_geak_on_pods_best_effort", lambda: calls.append("geak") or 0)
    monkeypatch.setattr(mn, "_replay_kernel_patches_for_multi_node", lambda _a: calls.append("replay"))

    args = argparse.Namespace(nodes=2, mn_backend="rayjob", model="/models/test", no_kernel=False)
    _prepare_multi_node_state(args)

    assert calls == ["geak", "replay"]
    assert "skipping the head-pod bootstrap" in capsys.readouterr().err


def test_kill_inference_invalidates_the_launch_it_terminated(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kill must drop the launch id it just invalidated.

    ``last_restart_submission_id`` names the fan-out driver, which reaches
    SUCCEEDED once the ranks are spawned and stays there whether the detached
    servers live or die. A kill ends exactly that launch, so keeping the id would
    leave the state claiming a launch this cluster no longer has.
    """
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.2.1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "rayjob")
    _write_disk_state(
        backend="rayjob",
        head_pod_ip="10.0.2.1",
        last_server_pid_dir="/tmp/multi_node_pids",
    )
    monkeypatch.setattr(mncli, "_exec_kill_submission", lambda *_a, **_k: "kill-9")

    assert (
        mncli.cmd_kill_inference(argparse.Namespace(pid_file=None, print_logs=False, poll_interval=1, poll_timeout=5))
        == 0
    )

    after = ext.load_multi_node_state()
    assert "last_restart_submission_id" not in after
    assert after["last_kill_submission_id"] == "kill-9"
    # The rest of the bookkeeping is untouched: only the launch identity died.
    assert after["last_restart_framework"] == "sglang"


def test_adopting_a_rayjob_bootstraps_and_replays_patches(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adoption still owes the cluster its session-side setup.

    Regression guard. Removing the create path took the three steps that lived
    inside it with it -- none of them provision anything, so a handed-over
    RayJob silently came up without the BYOI bootstrap (no framework venv on
    PATH for later Ray Dashboard jobs) and without its applied patches.
    """
    from hyperloom.inference_optimizer.cli import multi_node as mn
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.2.1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "rayjob")
    calls: list[str] = []
    monkeypatch.setattr(mncli, "cmd_bootstrap", lambda _ns: calls.append("bootstrap") or 0)
    monkeypatch.setattr(mncli, "install_geak_on_pods_best_effort", lambda: calls.append("geak") or 0)
    monkeypatch.setattr(mn, "_replay_kernel_patches_for_multi_node", lambda _a: calls.append("replay"))

    args = argparse.Namespace(nodes=2, mn_backend="rayjob", model="/models/test", no_kernel=False)
    _prepare_multi_node_state(args)

    assert calls == ["bootstrap", "geak", "replay"]


def test_adopting_an_infera_cluster_skips_bootstrap_but_installs_geak(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap is a Ray Dashboard call, so it is rayjob-only; GEAK is the infera half."""
    from hyperloom.inference_optimizer.cli import multi_node as mn
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    calls: list[str] = []
    monkeypatch.setattr(mncli, "cmd_bootstrap", lambda _ns: calls.append("bootstrap") or 0)
    monkeypatch.setattr(mncli, "install_geak_on_pods_best_effort", lambda: calls.append("geak") or 0)
    monkeypatch.setattr(mn, "_replay_kernel_patches_for_multi_node", lambda _a: calls.append("replay"))

    args = argparse.Namespace(nodes=2, mn_backend="infera", model="/models/test", no_kernel=False)
    _prepare_multi_node_state(args)

    assert calls == ["geak", "replay"]


def test_adopting_with_no_kernel_skips_the_geak_install(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--no-kernel`` opts out of the kernel phase, so the pods need no GEAK."""
    from hyperloom.inference_optimizer.cli import multi_node as mn
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    calls: list[str] = []
    monkeypatch.setattr(mncli, "install_geak_on_pods_best_effort", lambda: calls.append("geak") or 0)
    monkeypatch.setattr(mn, "_replay_kernel_patches_for_multi_node", lambda _a: calls.append("replay"))

    args = argparse.Namespace(nodes=2, mn_backend="infera", model="/models/test", no_kernel=True)
    _prepare_multi_node_state(args)

    assert calls == ["replay"]


def test_missing_handoff_exits_config_error_not_transient(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unset hand-off is not something a retry can fix.

    The guard used to raise a bare RuntimeError, and ``main`` classifies those by
    message substring; this one matched nothing and fell through to
    EXIT_TRANSIENT, so the caller retried a permanently misconfigured run. The
    message is asserted too: the later ``missing required keys`` failure also
    maps to EXIT_CONFIG_ERROR, so the code alone does not prove the guard fired.
    """
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    session = tmp_path / "session"
    session.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    monkeypatch.delenv("HYPERLOOM_MN_EXT_SERVICE_URL", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")

    with pytest.raises(mncli.ConfigurationError, match="HYPERLOOM_MN_EXT_SERVICE_URL is unset"):
        mncli._load_state()
    assert mncli.main(["verify"]) == mncli.EXIT_CONFIG_ERROR
    assert "HYPERLOOM_MN_EXT_SERVICE_URL is unset" in capsys.readouterr().err


def test_bootstrap_accepts_handed_over_rayjob_without_rayjob_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A handed-over RayJob has head_pod_ip but no rayjob_id; bootstrap must run.

    Nothing writes rayjob_id now that the platform owns cluster creation, so
    demanding it in the state guard rejected every external RayJob before the
    first bootstrap could submit anything.
    """
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    session = tmp_path / "session"
    session.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://head:8888")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.2.1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "rayjob")

    state = ext.load_multi_node_state()
    assert state["head_pod_ip"] == "10.0.2.1"
    assert not state.get("rayjob_id")

    # Stop right after the guard: reaching the Ray Dashboard proves it passed.
    class _Reached(RuntimeError):
        pass

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise _Reached("reached the dashboard client")

    monkeypatch.setattr(mncli, "_ray_dashboard_client", _boom)
    args = argparse.Namespace(script=None, force=False, print_logs=False)
    with pytest.raises(_Reached):
        mncli.cmd_bootstrap(args)
