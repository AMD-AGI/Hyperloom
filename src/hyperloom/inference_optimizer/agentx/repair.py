# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime repair of the AgentX aiperf dependency."""

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
    """Return the packaged ``assets/install.sh``."""
    return Path(__file__).resolve().parent.parent / "assets" / "install.sh"


def ensure_aiperf_installed(
    *,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: int = REPAIR_TIMEOUT_SEC,
) -> Optional[str]:
    """Install the pinned aiperf via the packaged installer."""
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
    # ``--only-aiperf`` already bypasses the opt-in gate, but state the opt-in so the installer's own log says why it
    # ran, and so a future refactor that re-routes this through the ordinary gate keeps working.
    child_env["INSTALL_AIPERF"] = "1"
    # The installer runs under ``set -u`` and expands ``${HOME}`` for its state dir.
    if not child_env.get("HOME"):
        # setdefault would leave an empty-string HOME in place, and the installer expands ${HOME}/.hyperloom into an
        # unwritable /.hyperloom -- the stamp write then fails and every later provision redoes the install this one
        # was supposed to record.
        child_env["HOME"] = os.path.expanduser("~")

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
    """The installer lines worth keeping, redacted, flattened onto one line."""
    from hyperloom.common.env_safety import redact_secret_values

    # Joined rather than concatenated: a stdout tail without a trailing newline would otherwise fuse into the first
    # stderr line, and that first stderr line is usually the pip error this summary exists to carry.
    combined = "\n".join(part for part in ((stdout or "").strip(), (stderr or "").strip()) if part)
    if not combined:
        return "(no output)"
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    tail = lines[-_OUTPUT_TAIL_LINES:]
    # Anything earlier that announces a failure, in the order it was printed.
    kept = [line for line in lines[:-_OUTPUT_TAIL_LINES] if _ERROR_LINE_RE.search(line)]
    return redact_secret_values(" | ".join(kept[-_ERROR_LINE_BUDGET:] + tail))
