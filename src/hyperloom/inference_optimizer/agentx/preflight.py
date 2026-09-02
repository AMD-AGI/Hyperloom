# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX preflight: resolve and capability-check the aiperf binary.

``HYPERLOOM_AGENTX`` needs an aiperf build with AgentX (``weka-trace``) support.
This module verifies *capability*, not mere existence, so a plain mainline
aiperf on ``PATH`` fails loud with actionable guidance instead of erroring deep
inside a benchmark run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional


class AgentXPreflightError(RuntimeError):
    """Raised when the aiperf binary is missing or not AgentX-capable."""


def resolve_aiperf_bin(env: Mapping[str, str]) -> Optional[str]:
    """Return ``AIPERF_BIN`` (operator override) else a PATH lookup else None."""
    override = (env.get("AIPERF_BIN") or "").strip()
    if override:
        return override
    # Resolve against the SAME PATH the benchmark subprocess will use (the child
    # env), not this process's os.environ, so preflight probes the binary that
    # actually runs. Falls back to os.environ PATH when env has none.
    return shutil.which("aiperf", path=env.get("PATH"))


SCENARIO_NAME = "inferencex-agentx-mvp"

# Corpora aiperf_client.sh can select on its own (the model-family whitelist
# picks one of these two). An operator override is checked on top of them.
_DEFAULT_CORPORA = (
    "semianalysis_cc_traces_weka_062126",
    "semianalysis_cc_traces_weka_062126_256k",
)

_ALLOWLIST_SNIPPET = (
    "import json;"
    "from aiperf.common.scenario import get_scenario;"
    f"print(json.dumps(list(get_scenario({SCENARIO_NAME!r}).require_loader)))"
)


def _default_probe(aiperf_bin: str) -> str:
    """Run ``aiperf profile --help`` and return combined stdout+stderr."""
    out = subprocess.run(
        [aiperf_bin, "profile", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return (out.stdout or "") + (out.stderr or "")


def _interpreters_for(aiperf_bin: str) -> list[str]:
    """Interpreters that might have the probed aiperf importable.

    install.sh pips aiperf into Hyperloom's own environment, so ``sys.executable``
    is the usual hit; an ``AIPERF_BIN`` pointing at another venv is served by the
    python sitting next to it.
    """
    import sys

    sibling = Path(aiperf_bin).resolve().parent / "python"
    return [str(sibling), sys.executable]


def _default_loader_probe(aiperf_bin: str) -> Optional[list[str]]:
    """Return the scenario's loader allowlist, or None if it cannot be read.

    Read from the *scenario registry* of the aiperf that will actually run, not
    from help text: the allowlist is the thing that decides whether this build
    measures the corpus we are about to hand it.
    """
    import json

    for interp in _interpreters_for(aiperf_bin):
        try:
            out = subprocess.run(
                [interp, "-c", _ALLOWLIST_SNIPPET],
                capture_output=True,
                text=True,
                timeout=60,
            )
        # SubprocessError as well as OSError: subprocess.run(timeout=...) raises
        # TimeoutExpired, which descends from SubprocessError, not OSError. Left
        # uncaught it escapes check_aiperf_capability entirely -- turning the
        # documented "degrade to the flag probe with a warning" into a hard
        # preflight failure, on the one input (a hung interpreter) the timeout
        # exists to handle. The sibling probes in cli/preflight.py already catch
        # both.
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode != 0:
            continue
        try:
            loaders = json.loads((out.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            continue
        if isinstance(loaders, list):
            return [str(x) for x in loaders]
    return None


def check_aiperf_capability(
    aiperf_bin: Optional[str],
    *,
    require_progress_api: bool = False,
    probe: Optional[Callable[[str], str]] = None,
    loader_probe: "Optional[Callable[[str], Optional[list[str]]]]" = None,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Raise :class:`AgentXPreflightError` unless ``aiperf_bin`` can run AgentX.

    The check that matters is not "does this build have the flags" -- measured:
    the previous pin (aiperf 0.8.0) carries ``weka-trace``, ``--scenario`` and
    ``--benchmark-duration``, and even defines a scenario by the same name, yet
    its invariants differ (no ``require_streaming``, a 60s trace idle-gap cap
    against the current 10s system cap, no profile-metric coverage floor). What
    separates the two is the scenario's own loader allowlist: the current corpus
    is simply not in the old one.

    So this asks the aiperf that will actually run whether its scenario admits
    the corpus we are about to hand it -- the invariant itself rather than a
    proxy for it. That also closes the path a flag probe cannot see: an operator
    pinning an OLDER corpus via ``WEKA_LOADER_OVERRIDE`` (upstream's own H100
    recipes do) lands in the stale build's allowlist and would otherwise replay
    it under the wrong invariants and stamp the result submittable.

    Falls back to the flag probe, loudly, when the allowlist cannot be read at
    all -- refusing outright would break setups that work today over what may be
    nothing worse than an unusual install layout.

    Args:
        aiperf_bin: Resolved aiperf path (None/empty means "not found").
        require_progress_api: Require the local phase-progress API used by
            AgentX trace capture.
        probe: Injectable help-text probe, used for the fallback path.
        loader_probe: Injectable allowlist probe; returns the scenario's
            permitted loaders, or None when they cannot be determined.
        env: Environment the benchmark will run with; read for an operator
            corpus pin. Defaults to the current process environment.
    """
    if not aiperf_bin:
        raise AgentXPreflightError(
            "HYPERLOOM_AGENTX is on but aiperf was not found. Install the pinned "
            "SemiAnalysisAI/aiperf build via install.sh (AIPERF_REF), or set "
            "AIPERF_BIN to an aiperf with AgentX (weka-trace) support."
        )

    runtime_env = os.environ if env is None else env
    override = (runtime_env.get("AGENTX_DATASET") or "").strip() or (
        runtime_env.get("WEKA_LOADER_OVERRIDE") or ""
    ).strip()

    loaders = (loader_probe or _default_loader_probe)(aiperf_bin)
    if loaders is not None:
        # Two distinct questions, and only asking the second one leaves the
        # silent path open. Measured: with WEKA_LOADER_OVERRIDE pointing at an
        # older corpus -- which upstream's own H100/H200 recipes do -- the stale
        # build admits it, so a run-scoped check alone waves the stale build
        # through and it replays under the wrong invariants.
        #
        # (1) Is this build current? The 062126 corpora were added alongside the
        #     current invariant set, so their presence dates the build. This
        #     holds regardless of which corpus the run happens to select.
        stale = [c for c in _DEFAULT_CORPORA if c not in loaders]
        if stale:
            raise AgentXPreflightError(
                f"aiperf at {aiperf_bin!r} is a stale build: its {SCENARIO_NAME!r} "
                f"scenario predates {', '.join(stale)}, so it locks a different set of "
                f"invariants (the previous pin, 0.8.0, has no require_streaming and a "
                f"60s trace idle-gap cap against the current 10s system cap) and its "
                f"results are not comparable. Note that it carries the same AgentX "
                f"flags and a scenario of the same name, which is why a flag check "
                f"passes it. Reinstall the pinned build (AIPERF_REF in install.sh), or "
                f"point AIPERF_BIN at one."
            )
        # (2) Will this run's corpus be admitted? Catches a typo or a corpus this
        #     scenario does not permit, before a server boot rather than after.
        if override and override not in loaders:
            raise AgentXPreflightError(
                f"the corpus pin {override!r} is not in the {SCENARIO_NAME!r} loader "
                f"allowlist of the aiperf at {aiperf_bin!r}. Permitted: "
                f"{', '.join(sorted(loaders))}."
            )
        if not require_progress_api:
            return

    if loaders is None:
        # Allowlist unreadable: fall back to the old flag probe, and say that the
        # real check did not run so a stale build is not silently blessed.
        print(
            f"WARNING: could not read the {SCENARIO_NAME!r} loader allowlist from "
            f"{aiperf_bin!r}; falling back to a flag-presence check, which cannot "
            f"tell the pinned build from an older one carrying the same flags"
            + (
                f", and cannot confirm that the pinned corpus {override!r} is one this "
                f"scenario admits -- an unpermitted or misspelled name will now surface "
                f"only after the server boots"
                if override
                else ""
            )
            + ".",
            file=sys.stderr,
        )

    probe = probe or _default_probe
    try:
        help_text = probe(aiperf_bin)
    except Exception as exc:  # noqa: BLE001 — surface as a structured preflight error
        raise AgentXPreflightError(f"aiperf capability probe failed for {aiperf_bin!r}: {exc}") from exc

    if loaders is None:
        scenario_flags = [
            flag for flag in ("weka-trace", "--scenario", "--benchmark-duration") if flag not in (help_text or "")
        ]
        if scenario_flags:
            raise AgentXPreflightError(
                f"aiperf at {aiperf_bin!r} is not AgentX-capable (missing: "
                f"{', '.join(scenario_flags)}); install the pinned SemiAnalysisAI/aiperf "
                "build via install.sh (AIPERF_REF) or point AIPERF_BIN at one."
            )

    if require_progress_api:
        api_flags = [flag for flag in ("--api-host", "--api-port") if flag not in (help_text or "")]
        if api_flags:
            raise AgentXPreflightError(
                f"aiperf at {aiperf_bin!r} cannot expose phase progress "
                f"(missing: {', '.join(api_flags)}); install the pinned "
                "SemiAnalysisAI/aiperf build via install.sh."
            )
