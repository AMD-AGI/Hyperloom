# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ray cluster lifecycle helpers for the Ray execution backend."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# Minimum soft RLIMIT_NOFILE the Ray raylet needs to stay up.
DEFAULT_MIN_NOFILE = 65536
DEFAULT_RAY_STATUS_TIMEOUT_SEC = 5.0
DEFAULT_RAY_STOP_TIMEOUT_SEC = 30.0
#: Wall-clock bound on the ``ray.init`` connect handshake. Registration with the
#: raylet has no timeout of its own, so a raylet that accepts the socket and then
#: never replies blocks the caller forever. Observed: a coordinator sat in this
#: call for 44 minutes, holding tick 1, until the container was torn down. The
#: bound fires only while the connect releases the GIL, and the registration does
#: not (see ``DEFAULT_RAY_CONNECT_PROBE_TIMEOUT_SEC``).
DEFAULT_RAY_INIT_TIMEOUT_SEC = 300.0
#: Wall-clock bound on the driver connect this process tries in a child before it
#: connects in-process. In Ray 2.44 the in-process connect cannot be bounded from
#: inside: ``CCoreWorkerProcess.Initialize`` runs with the GIL held
#: (``_raylet.pyx:3037``) and waits for the raylet's ``RegisterClientReply`` with
#: no timeout (``core_worker.cc:285``), so no thread, timer or event loop of the
#: process runs again until the raylet answers.
DEFAULT_RAY_CONNECT_PROBE_TIMEOUT_SEC = 60.0

# Tail of a failed `ray start` carried in the exception, bounding its length.
_RAY_START_OUTPUT_TAIL_CHARS = 1500

# Custom Ray resource declared on the single-node head so serving-family work (serving / benchmark / profile /
# gpu_research) can hold a whole-machine ``serving_slot`` as the authoritative physical mutex.
RAY_SERVING_SLOT = "serving_slot"
_HEAD_CUSTOM_RESOURCES = {RAY_SERVING_SLOT: 1}


#: Held for the whole life of a ``ray.init`` attempt and released by the thread that
#: made it -- including an abandoned one, on whatever schedule it finishes. Only one
#: connect may be outstanding in this process at a time, because ``ray.init`` is
#: process-global and uncancellable: a second attempt overlapping an abandoned first
#: is how a late connect lands between the new attempt's shutdown and its own init
#: and hands it the OLD cluster, silently, through ``ignore_reinit_error=True`` --
#: which is precisely what the version-mismatch retry exists to escape.
_INIT_GATE = threading.Semaphore(1)

#: Set when an attempt timed out, so the next one clears a session its runner may
#: have created. Read only after the gate is held, i.e. after that runner is done.
_STALE_CONNECT_POSSIBLE = threading.Event()


def _resources_start_args() -> list[str]:
    """Return the ``ray start`` argv for the head node's custom resources."""
    return ["--resources", json.dumps(_HEAD_CUSTOM_RESOURCES)]


# --- Local-head port isolation (spur host-network co-location) --------------- Many optimizer sessions can be
# co-scheduled on ONE compute node (SLURM packs sub-node ``--gpus`` requests), and spur runs the container on the host
# network stack (the bridge has no egress).
_HL_RAY_HEAD_PORT_ENV = "HL_RAY_HEAD_PORT"
# The Ray address that means "discover the cluster this host already runs".
_RAY_ADDRESS_AUTO = "auto"


def _free_tcp_port() -> int:
    """Reserve and return a currently-free loopback TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _isolated_head_port_args() -> Tuple[int, list[str]]:
    """Return ``(gcs_port, extra_start_args)`` bound to FREE probed ports."""
    override = os.environ.get(_HL_RAY_HEAD_PORT_ENV, "").strip()
    port = int(override) if override.isdigit() else 0
    gcs_port = port if 1 <= port <= 65535 else _free_tcp_port()
    return gcs_port, [
        f"--dashboard-port={_free_tcp_port()}",
        f"--ray-client-server-port={_free_tcp_port()}",
    ]


def _fd_limit_warn(msg: str) -> None:
    """Emit an fd-limit warning to stderr with a stable prefix."""
    print(f"[kernel-agent WARN] {msg}", file=sys.stderr)


def _min_nofile_target() -> int:
    """Return the target soft RLIMIT_NOFILE value."""
    raw = os.environ.get("RAY_MIN_NOFILE", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_MIN_NOFILE


def _positive_float_env(name: str, default: float) -> float:
    """Return a positive float env override, else ``default``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _ray_status_timeout_sec() -> float:
    """Return the ``ray status`` probe timeout in seconds."""
    return _positive_float_env("HYPERLOOM_RAY_STATUS_TIMEOUT_SEC", DEFAULT_RAY_STATUS_TIMEOUT_SEC)


def _ray_stop_timeout_sec() -> float:
    """Return the ``ray stop --force`` timeout in seconds."""
    return _positive_float_env("HYPERLOOM_RAY_STOP_TIMEOUT_SEC", DEFAULT_RAY_STOP_TIMEOUT_SEC)


def _ray_init_timeout_sec() -> float:
    """Return the ``ray.init`` connect timeout in seconds."""
    return _positive_float_env("HYPERLOOM_RAY_INIT_TIMEOUT_SEC", DEFAULT_RAY_INIT_TIMEOUT_SEC)


def _ray_connect_probe_timeout_sec() -> float:
    """Return the bound on the child-process driver connect in seconds."""
    return _positive_float_env("HYPERLOOM_RAY_CONNECT_PROBE_TIMEOUT_SEC", DEFAULT_RAY_CONNECT_PROBE_TIMEOUT_SEC)


def ensure_fd_limit(min_soft: Optional[int] = None) -> Tuple[int, int]:
    """Raise this process's RLIMIT_NOFILE soft limit before Ray starts."""
    if min_soft is None:
        min_soft = _min_nofile_target()
    inf = resource.RLIM_INFINITY
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # RLIM_INFINITY (-1) means "unlimited"; don't treat it as a tiny number.
    if soft == inf or soft >= min_soft:
        return soft, hard
    target = min_soft if hard == inf else min(min_soft, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        soft = target
    except (ValueError, OSError) as exc:  # pragma: no cover - defensive
        _fd_limit_warn(
            f"could not raise RLIMIT_NOFILE soft limit to {target} "
            f"(soft={soft}, hard={hard}): {exc}; Ray raylet may be unstable "
            f"(issue #433). Launch the container with --ulimit nofile=1048576."
        )
        return soft, hard
    if hard != inf and hard < min_soft:
        _fd_limit_warn(
            f"RLIMIT_NOFILE hard cap {hard} is below the raylet target "
            f"{min_soft}; raised soft to {soft} but this may still be too low "
            f"(issue #433). Launch the container with --ulimit nofile=1048576 "
            f"(>= {min_soft})."
        )
    return soft, hard


def ray_status_ok() -> bool:
    """Check whether a Ray cluster is currently reachable."""
    try:
        proc = subprocess.run(
            ["ray", "status"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=_ray_status_timeout_sec(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _stop_ray_force() -> None:
    """Run ``ray stop --force`` with a bounded timeout; never raises."""
    try:
        subprocess.run(
            ["ray", "stop", "--force"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_ray_stop_timeout_sec(),
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


#: ``ray start`` leaves its daemons (gcs_server, raylet, dashboard, ...) running after it exits, and they inherit its
#: session, process group and stdio. Left in the caller's group they make a shell that tracks background jobs by
#: process group (an agent's sandbox bash) see the caller as still running long after it returned, and an inherited
#: pipe keeps that shell's output stream open. A new session plus non-pipe stdio detaches them the way a daemon is.
_RAY_START_DETACH_KWARGS = {"stdin": subprocess.DEVNULL, "start_new_session": True}


def _ray_start_evidence(captured: str) -> str:
    """Carry the tail of the captured ``ray start`` output into an error message."""
    if captured:
        return f"ray start output: {captured[-_RAY_START_OUTPUT_TAIL_CHARS:]}"
    return "ray start produced no output"


def ensure_ray_cluster(num_gpus: Optional[int] = None) -> None:
    """Ensure a Ray cluster is reachable, starting a head node if needed."""
    if ray_status_ok():
        return
    # Clears stale Ray discovery state before the local head starts.
    _stop_ray_force()
    ensure_fd_limit()
    _start_local_head(num_gpus)


def _start_local_head(num_gpus: Optional[int]) -> None:
    """Run ``ray start --head`` and require the new cluster to be reachable."""
    # The head started here has answered no driver yet; whatever was probed was the head it replaces.
    forget_ray_head_verdict()
    gcs_port, iso_args = _isolated_head_port_args()
    # Dashboard bound to loopback: keeps the unauthenticated Ray Jobs endpoint off the pod network.
    cmd = ["ray", "start", "--head", f"--port={gcs_port}", "--dashboard-host=127.0.0.1"]
    if num_gpus is not None:
        cmd.append(f"--num-gpus={num_gpus}")
    cmd.extend(iso_args)
    # serving_slot: whole-machine mutex so serving-family tasks serialise GPU access.
    cmd.extend(_resources_start_args())
    # This output is the only evidence a failure leaves. It is captured through an anonymous file, not a pipe: a
    # daemon that inherits a pipe holds it open and a pipe reader waits on it forever.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as sink:
        proc = subprocess.run(cmd, stdout=sink, stderr=subprocess.STDOUT, text=True, **_RAY_START_DETACH_KWARGS)
        sink.seek(0)
        captured = sink.read().strip()
    if proc.returncode != 0:
        raise RuntimeError(f"failed to start Ray (rc={proc.returncode}); {_ray_start_evidence(captured)}")
    if not ray_status_ok():
        raise RuntimeError(f"ray start exited 0 but cluster is not reachable; {_ray_start_evidence(captured)}")


def _is_ray_version_mismatch(text: str) -> bool:
    """Detect Ray's version-mismatch banner in captured output."""
    return "version mismatch" in (text or "").lower()


def force_restart_local_cluster(num_gpus: Optional[int] = None) -> None:
    """Tear down any reachable Ray cluster and start a fresh local head."""
    ensure_fd_limit()
    _stop_ray_force()
    _start_local_head(num_gpus)


def _ray_address() -> str:
    """Return the address this process connects its Ray driver to."""
    return os.environ.get("RAY_ADDRESS", _RAY_ADDRESS_AUTO)


#: One driver connect and a clean disconnect, as ``quiet_ray_init`` makes in-process, in a child that can be killed.
#: The child first has the kernel SIGKILL it when the probing process dies (``PR_SET_PDEATHSIG``, 1), and gives up if
#: that process is already gone, so a parent killed outright mid-probe -- by a SIGTERM it does not handle, say -- does
#: not leave a driver waiting on the head.
_CONNECT_PROBE_SOURCE = (
    "import ctypes, os, signal, sys\n"
    "prctl = getattr(ctypes.CDLL(None), 'prctl', None)\n"
    "if prctl is not None:\n"
    "    prctl(1, signal.SIGKILL)\n"
    "if os.getppid() != int(sys.argv[2]):\n"
    "    sys.exit('the probing process is gone')\n"
    "import ray\n"
    "ray.init(address=sys.argv[1], log_to_driver=False, logging_level='error')\n"
    "ray.shutdown()\n"
)
# Tail of a failed probe's output carried into its outcome.
_CONNECT_PROBE_OUTPUT_TAIL_CHARS = 600


class RayHeadUnresponsiveError(RuntimeError):
    """The Ray head did not answer a driver connect probe, so this process does not connect to it in-process."""


@dataclass(frozen=True)
class RayConnectProbe:
    """The outcome of one driver connect tried in a child process.

    Attributes:
        answered: The head registered the driver, or answered it with a Ray version mismatch, which the in-process
            connect recovers from by replacing the head.
        outcome: What the probe did, for the log.
    """

    answered: bool
    outcome: str


def probe_ray_connect(address: str, timeout: float) -> RayConnectProbe:
    """Connect a throwaway driver to ``address`` in a child process and report whether the head answered in time.

    The child leads a session of its own, so a probe still running when the wait ends -- at the bound, or because
    this process was interrupted mid-wait (``KeyboardInterrupt``, the ``SystemExit`` of a signal handler) -- has its
    whole process group SIGKILLed and is reaped, and leaves no driver behind.
    """
    started = time.monotonic()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as sink:
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", _CONNECT_PROBE_SOURCE, address, str(os.getpid())],
                stdout=sink,
                stderr=subprocess.STDOUT,
                **_RAY_START_DETACH_KWARGS,
            )
        except OSError as exc:
            return RayConnectProbe(False, f"could not be started: {exc!r}")
        try:
            returncode: Optional[int] = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            returncode = None
        finally:
            # However the wait ended, a probe still running goes now, with its whole group.
            if proc.returncode is None:
                _kill_process_group(proc)
        if returncode is None:
            return RayConnectProbe(False, f"did not connect within {timeout:g}s and was killed")
        sink.seek(0)
        output = sink.read().strip()
    elapsed = time.monotonic() - started
    if returncode == 0:
        return RayConnectProbe(True, f"connected in {elapsed:.1f}s")
    if _is_ray_version_mismatch(output):
        return RayConnectProbe(True, f"was answered with a Ray version mismatch in {elapsed:.1f}s")
    tail = output[-_CONNECT_PROBE_OUTPUT_TAIL_CHARS:]
    return RayConnectProbe(False, f"exited {returncode} after {elapsed:.1f}s: {tail}")


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL the process group ``proc`` leads, then reap it."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_ray_stop_timeout_sec())


class _HeadVerdict:
    """Whether the Ray head answers a driver: probed once, then kept until :func:`forget_ray_head_verdict`."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.probe: Optional[RayConnectProbe] = None
        # No head answered ``ray status`` when routing first asked; the probe then runs before the connect, against
        # the head ``ensure_ray_cluster`` has started by that time.
        self.deferred = False


_HEAD_VERDICT = _HeadVerdict()


def forget_ray_head_verdict() -> None:
    """Drop the verdict on the Ray head, so the next routing question or in-process connect probes it again.

    Called whenever the head this process would connect to may no longer be the one probed: the backend was marked
    unhealthy (an actor died, and a raylet reaped after an OOM can leave the head accepting drivers it never
    registers), or this process started a head. Each of those is followed by an in-process ``ray.init``, which freezes
    the process on a head that does not answer.
    """
    verdict = _HEAD_VERDICT
    with verdict.lock:
        verdict.probe = None
        verdict.deferred = False


def ray_head_answers_driver() -> bool:
    """Whether this process may connect a Ray driver to the head, probing a running head once per verdict.

    Asked when work is routed, before any lease exists, so a head that does not answer sends every caller down the
    local path -- the one ``INFERENCE_OPTIMIZER_RAY_EXEC=0`` selects -- before anything connects in-process. With no
    head running yet the answer is yes: this process starts one, and :func:`quiet_ray_init` probes it before
    connecting.
    """
    verdict = _HEAD_VERDICT
    with verdict.lock:
        if verdict.probe is None and not verdict.deferred:
            if ray_status_ok():
                verdict.probe = _probe_and_report(_ray_address())
            else:
                verdict.deferred = True
        return verdict.probe is None or verdict.probe.answered


def _refuse_unless_head_answers(address: str) -> None:
    """Probe the head before the first in-process connect unless routing already did; raise if it did not answer."""
    verdict = _HEAD_VERDICT
    with verdict.lock:
        if verdict.probe is None:
            verdict.probe = _probe_and_report(address)
        probe = verdict.probe
    if not probe.answered:
        raise RayHeadUnresponsiveError(
            f"not connecting to the Ray head in-process: a driver connect probe {probe.outcome}"
        )


def _probe_and_report(address: str) -> RayConnectProbe:
    """Probe the head at ``address`` and log the outcome: one WARNING when it did not answer."""
    timeout = _ray_connect_probe_timeout_sec()
    probe = probe_ray_connect(address, timeout)
    if probe.answered:
        log.info("Ray head at address %r answered a driver connect probe: it %s", address, probe.outcome)
        return probe
    log.warning(
        "Ray head at address %r did not answer a driver: ray.init in a child process %s (bound %gs, "
        "HYPERLOOM_RAY_CONNECT_PROBE_TIMEOUT_SEC). Not connecting in-process, where the same wait holds the GIL and "
        "freezes this process: serving and GPU work run on the local subprocess path from here on, as "
        "with INFERENCE_OPTIMIZER_RAY_EXEC=0. To use Ray again, run `ray stop --force` before the next session so it "
        "starts a fresh head; to skip Ray and this probe, export INFERENCE_OPTIMIZER_RAY_EXEC=0.",
        address,
        probe.outcome,
        timeout,
    )
    return probe


#: Ray resource keys that describe capacity rather than something a task or actor holds.
_NON_HELD_RESOURCE_PREFIXES = ("node:", "memory", "object_store_memory")


#: The per-node Ray daemons ``ray stop --force`` would stop, and the argument that says which cluster each serves:
#: a head's GCS server by its port, a node's raylet by its node id. Every other Ray process belongs to one of these.
_RAY_DAEMON_IDENTITY_ARGS = {"gcs_server": "--gcs_server_port=", "raylet": "--node_id="}


def _visible_ray_daemons() -> Optional[dict[str, list[str]]]:
    """List the Ray GCS servers and raylets in this PID namespace by identity (see ``_RAY_DAEMON_IDENTITY_ARGS``).

    A daemon whose identity cannot be read is listed as ``""``; ``None`` means ``/proc`` itself was unreadable.
    """
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    found: dict[str, list[str]] = {name: [] for name in _RAY_DAEMON_IDENTITY_ARGS}
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/comm", encoding="utf-8") as comm:
                name = comm.read().strip()
            if name not in found:
                continue
            with open(f"/proc/{entry}/cmdline", "rb") as cmdline:
                argv = cmdline.read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        prefix = _RAY_DAEMON_IDENTITY_ARGS[name]
        found[name].append(next((a[len(prefix) :] for a in argv if a.startswith(prefix)), ""))
    return found


def _cluster_activity() -> list[str]:
    """Describe what is running on the connected cluster besides this driver; empty when nothing is.

    Read from the GCS tables directly so it works without a dashboard: actors not yet
    dead (including ones that hold no resources), and jobs whose driver is another
    live process (which also covers tasks that driver has pending).
    """
    from ray._private import state as ray_state

    import ray

    own_job = str(ray.get_runtime_context().get_job_id() or "")
    activity: list[str] = []
    live_actors = [a for a in ray_state.actors().values() if str(a.get("State") or "") != "DEAD"]
    if live_actors:
        activity.append(f"{len(live_actors)} actor(s) not dead")
    # By job id alone: drivers in other PID namespaces can share this process's pid.
    other_drivers = [j for j in ray_state.jobs() if not j.get("IsDead") and str(j.get("JobID") or "") != own_job]
    if other_drivers:
        pids = ", ".join(str(j.get("DriverPid")) for j in other_drivers)
        activity.append(f"other live driver(s) attached (pid {pids})")
    return activity


def local_head_restartable() -> Tuple[bool, str]:
    """Whether the connected Ray cluster is a lone local head this process may restart.

    Restarting means ``ray stop --force``, which stops every Ray process visible on
    this host, so it is allowed only when that is exactly the connected head and
    nothing else depends on it: no explicit cluster address, not a multi-node run,
    exactly one Ray head visible on the host, exactly one live node and it is this
    host's raylet, no resource held, no actor that is not dead and no other live
    driver. Anything that cannot be inspected counts as a reason not to restart.

    Returns:
        ``(True, "")`` when a restart is safe, else ``(False, why)``.
    """
    address = os.environ.get("RAY_ADDRESS", "").strip()
    if address and address.lower() != _RAY_ADDRESS_AUTO:
        return False, f"RAY_ADDRESS={address!r} names an explicit cluster"
    from ._multi_node_env import is_multi_node

    if is_multi_node():
        return False, "this is a multi-node run"
    import ray

    RayError = getattr(getattr(ray, "exceptions", None), "RayError", RuntimeError)

    try:
        # Every node record, dead ones included: a multi-node cluster whose worker is down is still multi-node.
        nodes = list(ray.nodes())
        context = ray.get_runtime_context()
        local_node = context.get_node_id()
        gcs_port = str(context.gcs_address or "").rsplit(":", 1)[-1]
        totals = ray.cluster_resources()
        available = ray.available_resources()
        activity = _cluster_activity()
    except (RayError, OSError, ImportError, AttributeError) as exc:
        # An unreadable cluster is not a safe one to stop. ``OSError`` covers a GCS that
        # refuses or times out; ``ImportError``/``AttributeError`` a private ``ray._private.state``
        # API that moved. Anything else is a bug here and raises.
        log.warning("could not inspect the Ray cluster before a serving_slot repair: %r", exc)
        return False, f"could not inspect the cluster: {exc!r}"
    if len(nodes) != 1:
        return False, f"the cluster has {len(nodes)} nodes"
    if not nodes[0].get("Alive") or str(nodes[0].get("NodeID") or "") != str(local_node or ""):
        return False, "its only node is not this host's live raylet"
    # ``ray stop --force`` stops every Ray process on the host, so the connected head's GCS server and raylet must
    # be the only ones there: another cluster's head, or a worker node of one, would be stopped with it.
    daemons = _visible_ray_daemons()
    if daemons is None:
        return False, "the Ray processes on this host cannot be listed"
    if daemons["gcs_server"] != [gcs_port] or daemons["raylet"] != [str(local_node or "")]:
        return False, (
            f"{len(daemons['gcs_server'])} Ray head(s) and {len(daemons['raylet'])} raylet(s) are visible on this "
            "host, not only the connected cluster's"
        )
    held = sorted(
        key
        for key, total in totals.items()
        if not key.startswith(_NON_HELD_RESOURCE_PREFIXES) and available.get(key, 0.0) < total
    )
    if held:
        return False, f"resources are in use on it ({', '.join(held)})"
    if activity:
        return False, f"it has work on it ({'; '.join(activity)})"
    return True, ""


def restart_local_head_with_serving_slot(num_gpus: Optional[int] = None) -> None:
    """Replace a local head that lacks ``serving_slot`` with one that declares it, then reconnect.

    The same repair ``install_kernel_tools.sh`` applies when it finds such a head:
    ``ray stop --force`` and a fresh ``ray start --head --resources=...``.
    """
    import ray
    from ray.exceptions import RayError

    try:
        ray.shutdown()
    except RayError as exc:
        # ``ray stop --force`` below ends the cluster this driver was attached to either way.
        log.warning("ray.shutdown() failed before restarting the local head; continuing: %r", exc)
    force_restart_local_cluster(num_gpus=num_gpus)
    quiet_ray_init(num_gpus=num_gpus)


# Env vars safe to forward to Ray workers; excludes *_VISIBLE_DEVICES (Ray-owned; forcing them triggers set_visible_accelerator_ids IndexError on ROCm).
SAFE_ENV_KEYS = (
    "HYPERLOOM_BENCHMARK_CONFIG",
    "PATH",
    "HOME",
    "LD_LIBRARY_PATH",
    # Single artefact root others default under.
    "USER_DATA_PATH",
    "HYPERLOOM_RUNTIME_DIR",
    "KERNEL_AGENT_ENV",
    "MAGPIE_PATH",
    "INFERENCEX_PATH",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    # Gateway auth headers travel with their endpoint, so a worker that gets the URL and key but not the header is
    # rejected by a header-authenticated gateway (an AMD APIM subscription key, for one).
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CUSTOM_HEADERS",
    "LLM_API_BASE",
    # Operator overrides that point GEAK at an endpoint other than the one preflight resolved; never derived here.
    "GEAK_API_KEY",
    "GEAK_BASE_URL",
    # e2e optimizer runner path + repo root so a Ray worker can locate interface/run_e2e.py and the e2e_workflow/
    # checkout.
    "GEAK_ROOT",
    "GEAK_E2E_RUNNER",
    "GEAK_CLAUDE_EFFORT",
    "GEAK_CLAUDE_MODEL",
    "GEAK_E2E_TIMEOUT_S",
    # Scoring/profiler/run knobs read by GEAK itself; stripped at the Ray boundary without this allowlist entry.
    "GEAK_SCORE_TARGET",
    "GEAK_SKIP_PROFILE",
    "GEAK_MAX_BENCHMARK_SHAPES",
    "GEAK_RUN_MODE",
)


def safe_runtime_env() -> dict:
    """Build a Ray ``runtime_env`` from the allowlisted environment keys."""
    env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
    # CLAUDE_CODE_OAUTH_TOKEN is forwarded verbatim, never mirrored into these: either key var switches the Claude CLI
    # out of subscription mode.
    anthropic_key = env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")
    if anthropic_key:
        env.setdefault("ANTHROPIC_API_KEY", anthropic_key)
        env.setdefault("ANTHROPIC_AUTH_TOKEN", anthropic_key)
    openai_url = env.get("OPENAI_BASE_URL")
    if openai_url:
        env.setdefault("LLM_API_BASE", openai_url)
    return {"env_vars": env}


def quiet_ray_init(num_gpus: Optional[int] = None) -> None:
    """Initialize ray while suppressing the connect banner on stdout."""
    import contextlib
    import io
    import ray

    runtime_env = safe_runtime_env()

    def _connect(address: str) -> None:
        """Call ``ray.init`` with standard options.

        Deliberately does NOT redirect stdout. ``sys.stdout`` is process-global and
        this runs on a thread the caller abandons on timeout, so a redirection held
        here would never be unwound -- every later write, from every thread, would
        vanish into a buffer nobody reads. The caller suppresses the banner instead,
        around a join it always completes.
        """
        ray.init(
            address=address,
            ignore_reinit_error=True,
            log_to_driver=False,
            logging_level="error",
            runtime_env=runtime_env,
        )

    def _init(address: str) -> None:
        """Connect under a wall-clock bound.

        ``ray.init`` takes no timeout: the driver registers with the raylet over a
        socket and waits for a reply that a wedged raylet never sends, so the call
        blocks indefinitely with nothing to observe -- no child process, no log, no
        failure. A daemon thread makes that state reportable. The call itself cannot
        be cancelled, so the thread is abandoned rather than joined; it cannot keep
        the interpreter alive. What it could do is finish late and leave this
        process connected after the caller was told the cluster is unusable, so the
        next attempt shuts that session back down before it connects.
        """
        timeout = _ray_init_timeout_sec()
        # Wait out any abandoned runner rather than racing it. It releases the
        # gate when it finishes, so by the time this attempt holds it, every
        # connect that could still land has landed and the cleanup below is
        # deterministic. A runner that never returns leaves this reporting an
        # unusable cluster, which is what it is, instead of a second connect
        # interleaved with the first.
        if not _INIT_GATE.acquire(timeout=timeout):
            raise TimeoutError(
                f"a previous ray.init(address={address!r}) is still outstanding after "
                f"{timeout:g}s; the raylet never answered it and this process cannot "
                "safely start a second connect while it is unresolved."
            )
        if _STALE_CONNECT_POSSIBLE.is_set():
            # A previous attempt timed out and its runner has since finished; it
            # may have connected. Clear what it left before taking the process.
            _STALE_CONNECT_POSSIBLE.clear()
            with contextlib.suppress(RuntimeError, OSError):
                ray.shutdown()
        outcome: dict[str, BaseException] = {}
        # One lock decides, for a connect that lands near the deadline, whether
        # the caller got it or the next attempt has to undo it. The caller marks
        # a possible stale connect under the same lock the runner reports
        # through, and the runner releases the gate only after reporting, so
        # the next attempt cannot clear the marker before this one resolves.
        verdict_lock = threading.Lock()
        verdict: dict[str, bool] = {"done": False}

        def _runner() -> None:
            try:
                _connect(address)
            except BaseException as exc:  # re-raised on the caller's thread
                log.debug("ray.init(address=%r) raised on the connect thread", address, exc_info=True)
                outcome["error"] = exc
            with verdict_lock:
                verdict["done"] = True
            # Released last, and by this thread rather than the caller: the
            # caller may already have given up, and no second connect may start
            # while this one is unresolved.
            _INIT_GATE.release()

        thread = threading.Thread(target=_runner, name="ray-init", daemon=True)
        # The banner suppression is held here, across a join that always returns, so
        # stdout is restored whether the connect succeeded, failed or was abandoned.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            thread.start()
            thread.join(timeout)
        with verdict_lock:
            # ``done`` rather than ``thread.is_alive()``: a runner that returned
            # a microsecond ago is still alive to that check and has already
            # connected. Claimed under the same lock the runner reports through,
            # so exactly one of the two sides owns the outcome.
            timed_out = not verdict["done"]
            if timed_out:
                _STALE_CONNECT_POSSIBLE.set()
        if timed_out:
            raise TimeoutError(
                f"ray.init(address={address!r}) did not complete within {timeout:g}s; "
                "the raylet accepted the connection but never finished registration. "
                "Raise HYPERLOOM_RAY_INIT_TIMEOUT_SEC if the cluster is merely slow to start."
            )
        error = outcome.get("error")
        if error is not None:
            raise error

    address = _ray_address()
    _refuse_unless_head_answers(address)
    try:
        _init(address)
    except Exception as exc:
        if not _is_ray_version_mismatch(str(exc)):
            raise
        # Foreign cluster: replace with a local head, then retry exactly once.
        try:
            ray.shutdown()
        except (RuntimeError, OSError):
            pass
        force_restart_local_cluster(num_gpus=num_gpus)
        # The restart dropped the verdict on the old head; the new one is probed before this process connects to it.
        _refuse_unless_head_answers(_RAY_ADDRESS_AUTO)
        _init(_RAY_ADDRESS_AUTO)
