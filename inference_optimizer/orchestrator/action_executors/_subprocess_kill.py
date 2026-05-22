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
    """Send ``sig`` to every member of ``pgid``; swallow ``ESRCH``."""
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


def run_with_session_kill(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: int | float | None = None,
    text: bool = True,
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

    Tests that previously patched ``subprocess.run`` should patch this
    function instead (e.g.
    ``patch("inference_optimizer.orchestrator.action_executors._subprocess_kill.run_with_session_kill")``)
    — the call shape and return type are intentionally identical.
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
            stdout, stderr = proc.communicate(timeout=timeout)
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
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout if stdout is not None else ("" if text else b""),
            stderr=stderr if stderr is not None else ("" if text else b""),
        )
    finally:
        kill_my_spawned_server(proc)


__all__ = [
    "kill_my_spawned_server",
    "new_session_kwargs",
    "run_with_session_kill",
]
