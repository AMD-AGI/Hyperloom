"""Reliable subprocess-tree teardown for Magpie-launched servers
(Hyperloom ``bugs.md`` §B).

Why this exists
---------------

``BaselineExecutor`` / ``ProfileExecutor`` shell out to Magpie, which in
turn ``execve`` s a shell wrapper that starts a vLLM / SGLang server,
then runs the benchmark client. Many of those wrapper scripts:

* ``trap`` on EXIT but ``exit`` the *script*, leaving the server child
  reparented to PID 1 when the wrapper itself dies first.
* ``setsid`` themselves to escape the wrapper's controlling terminal.
* ``nohup`` the server so SIGHUP from a closing stdin doesn't kill it.

The cumulative effect is that on every Hyperloom-side timeout (subprocess
``TimeoutExpired``) or even on graceful Magpie exit-with-error, the
server child remains running, holds GPU memory, and — most importantly
for ``bugs.md`` §C #1 — keeps a ``bash`` interpreter alive that will
later re-source ``vllm_mi300x.sh`` while the *next* Magpie invocation
is mid-``shutil.copy2`` over the same file.

Closing the leak
----------------

The two-line fix is:

1. Launch Magpie inside its own POSIX session
   (``start_new_session=True`` / ``preexec_fn=os.setsid``). Every
   descendant inherits the session id; ``os.killpg(pgid, signal)``
   reaches them all in one syscall, regardless of how many ``setsid``
   layers a script tried to peel away.

2. From every BaselineExecutor / ProfileExecutor exit path call
   :func:`kill_my_spawned_server` with the launched Popen handle. The
   helper is idempotent and never raises — safe to slap into a
   ``finally:`` even when the launch itself failed.

Compared to ``recover``'s pattern-based killer (which is scoped to a
robustness-action surface and can match unrelated processes), this
helper is **scoped strictly to the PID tree we created**. That makes it
safe to run on every benchmark call, not only during recovery.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time

log = logging.getLogger(__name__)


# Grace window between SIGTERM and SIGKILL. Magpie / vLLM / SGLang
# servers normally flush their request buffer + GPU allocator state
# within ~1–2 s of receiving SIGTERM; 5 s leaves headroom for the slow
# graphs-capture teardown path without dragging out the cleanup window
# (operators have already given up on the call by this point).
_TERM_GRACE_SECONDS = 5.0


# Subprocess-launch kwargs that ensure every descendant lives in the
# same POSIX session. Importing this from the launch site keeps the
# os-specific dance in one place — Windows installs don't run Magpie
# but we leave the branch in to keep the helper portable for tests.
def new_session_kwargs() -> dict:
    """``kwargs`` to feed ``subprocess.Popen`` so the child gets its
    own POSIX session (=> killable via ``os.killpg(pid, signal)``).

    Returns ``{}`` on non-POSIX where ``setsid`` is meaningless; the
    helpers below detect that and fall through to ``proc.terminate``.

    Returns:
        dict: ``{"start_new_session": True}`` on POSIX, else ``{}``.
    """
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def _process_group_alive(pgid: int) -> bool:
    """Return True iff at least one process is still in ``pgid``.

    ``killpg(pgid, 0)`` is the standard "does this group exist?" probe:
    it sends no signal but raises ``ProcessLookupError`` once the group
    is empty (no member processes remain). Any other ``OSError``
    (typically ``EPERM`` from a sandbox) is treated as "still alive"
    since we can't prove otherwise.

    Args:
        pgid (int): The POSIX process-group id to probe.

    Returns:
        bool: True iff at least one process remains in the group (and on
            non-POSIX always False).
    """
    if os.name != "posix":
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _signal_group(pgid: int, sig: int) -> None:
    """Send ``sig`` to every member of ``pgid``; swallow ``ESRCH``.

    Args:
        pgid (int): The POSIX process-group id to signal.
        sig (int): The signal number to send.
    """
    if os.name != "posix":
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except OSError as exc:
        log.warning(
            "_subprocess_kill: killpg(%d, %d) failed: %s",
            pgid, sig, exc,
        )


def kill_my_spawned_server(
    proc: subprocess.Popen | None,
    *,
    grace_seconds: float = _TERM_GRACE_SECONDS,
) -> None:
    """Tear down the entire process tree rooted at ``proc``.

    Contract:

    * No-op when ``proc`` is ``None`` or has already exited (``poll() ==
      <int>``). Callers don't have to guard with ``if proc is not None``.
    * SIGTERM to the process group, wait up to ``grace_seconds`` for the
      group to drain, then SIGKILL whatever survived. This is the only
      ordering that reliably reaps grandchildren on Linux while still
      giving co-operating children a chance to flush state (e.g. close
      GPU contexts cleanly).
    * Never raises. We log a warning if signalling failed for an
      unexpected reason (``EPERM`` etc.) but proceed — the alternative
      is letting the exception bubble out of a ``finally:`` and mask
      the executor's real return value.

    Requires the child to have been launched with
    :func:`new_session_kwargs` (or ``preexec_fn=os.setsid`` /
    ``start_new_session=True``) so ``os.getpgid(proc.pid)`` returns a
    pgid distinct from Hyperloom's own — otherwise we would risk
    killing the Coordinator. We assert this defensively below.

    Args:
        proc (subprocess.Popen | None): The launched process handle (root of
            the tree); ``None`` or already-exited is a no-op.
        grace_seconds (float): Seconds to wait after SIGTERM before SIGKILL.
    """
    if proc is None:
        return
    if proc.poll() is not None:
        return
    if os.name != "posix":
        try:
            proc.terminate()
            proc.wait(timeout=grace_seconds)
        except (subprocess.TimeoutExpired, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        return

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    except OSError as exc:
        log.warning(
            "_subprocess_kill: getpgid(%d) failed: %s; falling back to "
            "single-pid terminate", proc.pid, exc,
        )
        try:
            proc.terminate()
        except OSError:
            pass
        return

    own_pgid = os.getpgid(0)
    if pgid == own_pgid:
        log.error(
            "_subprocess_kill: refusing to killpg own session (pgid=%d, "
            "child pid=%d). The child was almost certainly launched "
            "without start_new_session=True — that is a Hyperloom bug, "
            "fix the launch site instead of widening this helper's "
            "scope.",
            pgid, proc.pid,
        )
        return

    _signal_group(pgid, signal.SIGTERM)

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _process_group_alive(pgid):
            break
        time.sleep(0.05)

    if _process_group_alive(pgid):
        _signal_group(pgid, signal.SIGKILL)

    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        log.warning(
            "_subprocess_kill: proc.wait() did not return within 1s "
            "after SIGKILL'ing pgid=%d (pid=%d). The reaper may be "
            "wedged; leaving the zombie for init to collect.",
            pgid, proc.pid,
        )
    except OSError:
        pass


# Sentinel ``returncode`` used when ``run_with_session_kill`` reaps a
# child because its ``soft_deadline_sec`` elapsed (vs the legacy
# ``timeout=`` hard cap which still raises ``TimeoutExpired``).
# Callers that opted into the soft deadline detect this by checking
# ``returncode == OVERTIME_KILL_RETURNCODE``; the ExploreExecutor in
# particular turns this into the ``KILLED_OVERTIME`` per-variant
# outcome (no tput, no fingerprint promotion).
#
# Value chosen so it cannot collide with a real POSIX signal-based
# returncode (which is encoded as ``-N`` for signal ``N``; the largest
# defined signal on Linux is well under 100). 909 is the canonical
# Hyperloom "soft overtime kill" sentinel — grep this constant first
# when triaging a mystery returncode.
OVERTIME_KILL_RETURNCODE: int = -909


def run_with_session_kill(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: int | float | None = None,
    text: bool = True,
    soft_deadline_sec: float | None = None,
) -> subprocess.CompletedProcess:
    """``subprocess.run``-compatible call that ALSO tears down the entire
    descendant tree on every exit path (success, nonzero, timeout,
    exception).

    Why a wrapper instead of just ``subprocess.run(...)``:

    * ``subprocess.run`` returns a ``CompletedProcess`` and discards the
      Popen handle. Once it returns, there is no portable way for the
      caller to reach the (possibly-leaked) grandchildren — and the
      grandchildren are exactly what ``bugs.md`` §B is about: Magpie's
      shell wrapper exits cleanly, the vLLM/SGLang server it spawned
      via ``nohup`` / ``setsid`` survives, and the leaked bash
      interpreter is what later re-sources a half-truncated benchmark
      script in ``bugs.md`` §C #1.
    * We therefore launch via ``Popen(start_new_session=True)`` so the
      whole tree shares a pgid we control, ``communicate()`` for the
      same return shape ``subprocess.run`` provides, and
      ``kill_my_spawned_server`` in a ``finally:`` so the tree is
      guaranteed to be gone before the function returns.

    Returns a ``CompletedProcess`` (so callers can keep their existing
    ``.returncode / .stdout / .stderr`` access) and re-raises
    ``subprocess.TimeoutExpired`` exactly like ``subprocess.run`` does.

    ``soft_deadline_sec``: optional secondary deadline that fires
    BEFORE the (typically much larger) ``timeout=`` hard cap. When
    set, the helper polls the child every 0.5 s; once the deadline
    elapses, the process tree is reaped and the function returns a
    :class:`subprocess.CompletedProcess` with
    ``returncode = OVERTIME_KILL_RETURNCODE`` (does NOT raise). This
    lets the ExploreExecutor implement the per-variant
    "wall-clock > baseline × ratio" early-kill rule (Fix E) while
    preserving the legacy ``TimeoutExpired`` semantics for any
    caller that doesn't opt in. Pass ``None`` (or any value ≤ 0) to
    keep the legacy behaviour.

    Tests that previously patched ``subprocess.run`` should patch this
    function instead (e.g.
    ``patch("inference_optimizer.orchestrator.action_executors._subprocess_kill.run_with_session_kill")``)
    — the call shape and return type are intentionally identical.

    Args:
        cmd (list[str]): The command (argv) to launch.
        env (dict[str, str] | None): Environment for the child process.
        cwd (str | None): Working directory for the child process.
        timeout (int | float | None): Hard timeout (raises ``TimeoutExpired``).
        text (bool): Whether to decode stdout/stderr as text.
        soft_deadline_sec (float | None): Optional soft wall-clock deadline; on
            elapse the tree is reaped and a sentinel result is returned (no
            raise). ``None`` / ≤ 0 keeps legacy behaviour.

    Returns:
        subprocess.CompletedProcess: The completed process result, with
            ``returncode == OVERTIME_KILL_RETURNCODE`` when the soft deadline
            fired.

    Raises:
        subprocess.TimeoutExpired: When the hard ``timeout`` elapses.
    """
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(  # noqa: S603 — cmd is caller's responsibility
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            env=env,
            cwd=cwd,
            **new_session_kwargs(),
        )
        try:
            stdout, stderr = _communicate_with_soft_deadline(
                proc,
                hard_timeout=timeout,
                soft_deadline_sec=soft_deadline_sec,
            )
        except subprocess.TimeoutExpired:
            # Reap before re-raising so the caller's `finally:` /
            # `except:` doesn't see a still-running tree.
            kill_my_spawned_server(proc)
            try:
                # Drain whatever the pipes have so the exception carries
                # the partial output (subprocess.run does the same).
                stdout, stderr = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            raise
        except _SoftDeadlineExceeded as exc:
            kill_my_spawned_server(proc)
            try:
                stdout, stderr = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            log.info(
                "_subprocess_kill: soft_deadline_sec=%.1fs exceeded "
                "(elapsed=%.1fs); reaped tree with sentinel returncode=%d.",
                exc.deadline_sec, exc.elapsed_sec, OVERTIME_KILL_RETURNCODE,
            )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=OVERTIME_KILL_RETURNCODE,
                stdout=stdout if stdout is not None else ("" if text else b""),
                stderr=stderr if stderr is not None else ("" if text else b""),
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout if stdout is not None else ("" if text else b""),
            stderr=stderr if stderr is not None else ("" if text else b""),
        )
    finally:
        kill_my_spawned_server(proc)


class _SoftDeadlineExceeded(Exception):
    """Internal sentinel raised by :func:`_communicate_with_soft_deadline`
    when the soft (Fix-E) wall-clock deadline elapses. NEVER bubbles
    past :func:`run_with_session_kill` — converted into a sentinel
    ``CompletedProcess`` instead.
    """

    def __init__(self, *, deadline_sec: float, elapsed_sec: float) -> None:
        """Record the deadline and actual elapsed time on the sentinel.

        Args:
            deadline_sec (float): The soft deadline that was exceeded.
            elapsed_sec (float): The actual wall-clock elapsed at trip time.
        """
        super().__init__(
            f"soft deadline {deadline_sec:.1f}s elapsed "
            f"(actual={elapsed_sec:.1f}s)"
        )
        self.deadline_sec = float(deadline_sec)
        self.elapsed_sec = float(elapsed_sec)


def _communicate_with_soft_deadline(
    proc: subprocess.Popen,
    *,
    hard_timeout: int | float | None,
    soft_deadline_sec: float | None,
) -> tuple[str | bytes, str | bytes]:
    """``proc.communicate`` shim that also enforces ``soft_deadline_sec``.

    No-op fast path (``soft_deadline_sec`` falsy or ≤ 0) delegates
    straight to ``proc.communicate(timeout=hard_timeout)`` so callers
    that did not opt in see byte-identical behaviour.

    When ``soft_deadline_sec`` is positive, polls ``proc.communicate``
    in 0.5 s slices, raising :class:`_SoftDeadlineExceeded` once the
    monotonic wall-clock exceeds the deadline. The legacy
    ``hard_timeout`` is still enforced via ``communicate``'s own
    timeout argument on the final slice so a stuck child can't dodge
    both gates.

    Args:
        proc (subprocess.Popen): The running child process to communicate with.
        hard_timeout (int | float | None): The legacy hard timeout passed
            through to ``communicate``.
        soft_deadline_sec (float | None): The soft wall-clock deadline; falsy
            / ≤ 0 delegates straight to ``communicate``.

    Returns:
        tuple[str | bytes, str | bytes]: The ``(stdout, stderr)`` captured.

    Raises:
        _SoftDeadlineExceeded: When ``soft_deadline_sec`` elapses first.
        subprocess.TimeoutExpired: When the hard timeout elapses first.
    """
    if soft_deadline_sec is None or float(soft_deadline_sec) <= 0.0:
        return proc.communicate(timeout=hard_timeout)

    deadline_sec = float(soft_deadline_sec)
    poll_interval = 0.5
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        remaining_soft = deadline_sec - elapsed
        if remaining_soft <= 0.0:
            raise _SoftDeadlineExceeded(
                deadline_sec=deadline_sec, elapsed_sec=elapsed,
            )
        # The slice we wait this iteration: bounded above by the soft
        # remaining AND the hard remaining (so ``TimeoutExpired`` still
        # fires at the right wall-clock if the soft deadline is huge).
        slice_sec = min(poll_interval, remaining_soft)
        if hard_timeout is not None:
            hard_remaining = float(hard_timeout) - elapsed
            if hard_remaining <= 0.0:
                # Let proc.communicate's own TimeoutExpired path fire.
                return proc.communicate(timeout=0.0)
            slice_sec = min(slice_sec, hard_remaining)
        try:
            return proc.communicate(timeout=slice_sec)
        except subprocess.TimeoutExpired:
            # Not yet done; loop to re-evaluate both deadlines.
            continue


__all__ = [
    "OVERTIME_KILL_RETURNCODE",
    "kill_my_spawned_server",
    "new_session_kwargs",
    "run_with_session_kill",
]
