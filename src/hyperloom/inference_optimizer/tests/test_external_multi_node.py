# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cluster hand-off: env-synthesized multi-node state and its adoption guards."""

from __future__ import annotations

import argparse
import json
import os
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


def test_unset_nodes_falls_back_to_the_handed_over_pod_count(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subcommand that did not inherit NODES must not degrade to one pod.

    ``hyperloom-mn restart-server`` invoked on its own got ``nodes=1`` and took
    the single-pod path, which starts one detached server per pod instead of one
    cluster across them. Nothing states an intent here, so the two GPU pod IPs
    the platform handed over are the only evidence of the cluster's shape.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)

    assert ext.build_external_state_from_env()["nodes"] == 2


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


def test_rayjob_handoff_without_pod_ips_keeps_the_stated_node_count(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rayjob hand-off carries no pod IPs, so there is nothing to cross-check."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_PREFILL_IPS", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_DECODE_IPS", raising=False)
    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.2.1")

    assert ext.build_external_state_from_env()["nodes"] == 1


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
    """A kill must drop the launch id, or resume treats a dead cluster as healthy.

    ``last_restart_submission_id`` names the fan-out driver, which reaches
    SUCCEEDED once the ranks are spawned and stays there whether the detached
    servers live or die. ``cmd_restart_server``'s resume fast path reads that
    terminal-OK status as proof of health, so a kill that left the id behind let
    the next restart with an unchanged config skip KILL+LAUNCH and report
    success onto nothing.
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
