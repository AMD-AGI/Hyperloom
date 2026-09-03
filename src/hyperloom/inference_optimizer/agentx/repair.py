# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime repair of the AgentX aiperf dependency.

``HYPERLOOM_AGENTX`` declares aiperf as a required, version-pinned dependency,
and ``install.sh`` already knows how to install it (``ensure_aiperf``, pinned to
``AIPERF_REF`` in lockstep with ``INFERENCEX_REF``). What was missing is that the
install was conditioned on a *runtime* mode flag being true in the *installer's*
process, with nothing keeping the two moments consistent: provision without
``HYPERLOOM_AGENTX``/``INSTALL_AIPERF``, turn AgentX on later, and the box has no
aiperf. Both halves behaved as designed; the combination did not.

Measured: that gap sent a known, self-declared dependency into the enablement
lane, where an LLM specialist re-derived the install from scratch and had its
commands rejected by the setup-command allowlist -- spending the run's budget on
a problem this repository could already fix in one call. The preflight's own
error text names the fix ("install the pinned build via install.sh"), but it is
written for an operator, and on that path there is no operator.

So resolve it where the requirement is actually known. The runtime flag is the
single source of truth for "this box needs aiperf"; install-time opt-in stays a
pre-warm optimisation. Repair is attempted at most once per process (a second
attempt cannot succeed where the first failed, and a grid re-preflights every
round), and a failed repair is never silent -- it is folded into the preflight
error the caller surfaces, alongside the original diagnosis.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Mapping, Optional

log = logging.getLogger(__name__)

#: Entry point ``install.sh`` exposes for "run ensure_aiperf and nothing else".
#: A full install re-clones Magpie/InferenceX and chains the kernel-agent
#: installer -- far too much to run mid-session for one missing package.
ONLY_AIPERF_FLAG = "--only-aiperf"

#: The install is a pip install from a git ref (build included). 30 min matches
#: the per-command bound ``integrate_patch`` allows for the same class of work.
REPAIR_TIMEOUT_SEC = 1800

#: Outcome of this process's single repair attempt: absent = never attempted,
#: ``None`` = installed, ``str`` = why it failed. A dict rather than a rebound
#: module global so tests can reset it the way ``_PREFLIGHTED_BINS`` is reset.
_REPAIR_RESULT: dict[str, Optional[str]] = {}

_REPAIR_KEY = "aiperf"

#: Installer output kept in the error summary. Enough to show the failing pip /
#: git line without pasting a whole install log into a benchmark result.
_OUTPUT_TAIL_LINES = 12

#: Lines outside the tail window are dropped UNLESS they announce a failure --
#: the installer's own ``[... ERROR]`` prefix, or a bare ``ERROR:`` from pip.
_ERROR_LINE_RE = re.compile(r"(?:^|\W)(?:ERROR|FATAL)\b[: ]", re.IGNORECASE)

#: Cap on those rescued lines, so a build that fails a hundred times cannot turn
#: this one-line summary back into a log dump.
_ERROR_LINE_BUDGET = 4


def install_script_path() -> Path:
    """Return the packaged ``assets/install.sh``.

    Resolved from this module rather than from ``$REPO_ROOT`` so a wheel install
    repairs itself with the installer it actually shipped with -- the pin only
    means anything when the installer and the running code are the same vintage.
    """
    return Path(__file__).resolve().parent.parent / "assets" / "install.sh"


def ensure_aiperf_installed(
    *,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = REPAIR_TIMEOUT_SEC,
) -> Optional[str]:
    """Install the pinned aiperf via the packaged installer.

    Idempotent by construction: ``ensure_aiperf`` skips when the recorded ref
    already matches and force-reinstalls when it does not, so this is safe to
    call for a missing build and for a stale one alike.

    Args:
        env: Environment for the installer subprocess. Defaults to this
            process's environment.
        timeout_sec: Wall-clock bound on the installer.

    Returns:
        ``None`` when aiperf is installed, else a one-line summary of why the
        repair did not land (never raises: the caller folds this into the
        preflight error it was already about to report).
    """
    if _REPAIR_KEY in _REPAIR_RESULT:
        prior = _REPAIR_RESULT[_REPAIR_KEY]
        if prior is not None:
            log.debug("AgentX: reusing this process's failed aiperf repair (%s)", prior)
        return prior
    result = _install_aiperf(env=env, timeout_sec=timeout_sec)
    _REPAIR_RESULT[_REPAIR_KEY] = result
    return result


def _install_aiperf(*, env: Optional[Mapping[str, str]], timeout_sec: int) -> Optional[str]:
    """Run ``install.sh --only-aiperf`` once and classify the outcome."""
    script = install_script_path()
    if not script.is_file():
        return f"the packaged installer is missing at {script}"

    child_env = dict(os.environ if env is None else env)
    # ``--only-aiperf`` already bypasses the opt-in gate, but state the opt-in so
    # the installer's own log says why it ran, and so a future refactor that
    # re-routes this through the ordinary gate keeps working.
    child_env["INSTALL_AIPERF"] = "1"
    # The installer runs under ``set -u`` and expands ``${HOME}`` for its state
    # dir. The benchmark child env this inherits does not always carry HOME, and
    # the resulting "HOME: unbound variable" reads like a packaging bug rather
    # than a missing variable, so supply this process's own.
    child_env.setdefault("HOME", os.path.expanduser("~"))

    log.warning(
        "AgentX: aiperf is missing or is not the pinned build; installing it via "
        "%s %s. This is the same install the preflight tells operators to run, "
        "and it is pinned by AIPERF_REF -- a mismatched build measures the corpus "
        "under different invariants.",
        script,
        ONLY_AIPERF_FLAG,
    )
    try:
        proc = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, packaged installer.
            ["bash", str(script), ONLY_AIPERF_FLAG],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return f"{script.name} {ONLY_AIPERF_FLAG} did not finish within {timeout_sec}s"
    except OSError as exc:
        return f"could not run {script.name} {ONLY_AIPERF_FLAG}: {type(exc).__name__}: {exc}"

    if proc.returncode != 0:
        return f"{script.name} {ONLY_AIPERF_FLAG} exited {proc.returncode}: {_output_tail(proc.stdout, proc.stderr)}"
    log.info("AgentX: aiperf install completed; re-running the capability preflight")
    return None


def _output_tail(stdout: Optional[str], stderr: Optional[str]) -> str:
    """The installer lines worth keeping, redacted, flattened onto one line.

    The installer inherits the session environment, so its output can echo a
    credential; this lands in a benchmark result that is written to disk.

    Lines that name a failure are kept even when they fall outside the tail
    window. Measured against a Python 3.10 box, where the line that gives the
    actual cause --

        ERROR: Package 'aiperf' requires a different Python: 3.10.12 not in ...

    -- landed second-from-last behind nine lines of torch-gate warnings. A plain
    tail would have dropped exactly the sentence this summary exists to carry,
    and the reader would have been left with "install failed" and no reason.
    """
    from hyperloom.common.env_safety import redact_secret_values

    combined = ((stdout or "") + (stderr or "")).strip()
    if not combined:
        return "(no output)"
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    tail = lines[-_OUTPUT_TAIL_LINES:]
    # Anything earlier that announces a failure, in the order it was printed.
    kept = [line for line in lines[:-_OUTPUT_TAIL_LINES] if _ERROR_LINE_RE.search(line)]
    return redact_secret_values(" | ".join(kept[-_ERROR_LINE_BUDGET:] + tail))
