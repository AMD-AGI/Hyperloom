# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Capability probe: which arguments does an aiter tuner script actually accept?

aiter moves scripts and changes their argparse surface between versions. forge
kept sending a flag the script at the old path never accepted, and 14 runs died
on ``unrecognized arguments: --libtype hipblaslt`` while producing nothing.

Probing ``--help`` before the real call catches that class of breakage, but only
if the outcome of a rejected flag is chosen carefully. Dropping the flag and
running anyway is the worst option: the run completes, writes something or
nothing, and reports no gain -- which is indistinguishable from "this path has
no headroom". That is exactly how the original breakage stayed invisible for a
week. So rejected arguments are split three ways:

======================  ==========================================================
required (see below)    fail the run immediately, with the rejected flag named
known-droppable         drop it and warn -- it only affects speed or log detail
anything else           keep it and let the script reject it, so the call-time
                        guard reports the exact argparse error as the failure
======================  ==========================================================

A probe that cannot run (missing interpreter, timeout, unparseable help) never
vetoes anything: ``ScriptSurface.supports`` answers True for everything, which
leaves behaviour identical to the pre-probe code path.

Probing costs ~6-7s per script -- the expense is ``import aiter`` pulling in JIT
modules, not argparse -- so results are cached by ``(path, sha256)`` on disk. A
tuning session touching 9 scripts would otherwise pay a minute every time.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .utils import sha256_file

log = logging.getLogger(__name__)

# argparse lists every option it accepts in --help, but also wraps long lines and
# mentions flags in prose/epilogs. Matching permissively is the safe direction:
# an over-accepted flag is still caught at call time by the "unrecognized
# arguments" guard, whereas an under-accepted one would veto a working call.
_FLAG_RE = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")

# Arguments that decide whether the candidate set is non-empty at all. Losing one
# of these does not degrade the run, it invalidates it.
#   --libtype / --with-hipblaslt : both gate hipblaslt (gemm_a16w16_tune.py:992);
#                                  without them the candidate set is empty and
#                                  the tuner exits in seconds having tuned 0.
#   --splitK                     : without it the tuner sets maxsplitK=0 and
#                                  never evaluates split-K>0 at all.
#   --mxfp4-flydsl               : the only way into the FlyDSL dtype path; the
#                                  CK path rejects b16 x fp4x2 outright.
REQUIRED_FLAGS = frozenset(
    {
        "--libtype",
        "--with-hipblaslt",
        "--splitK",
        "--mxfp4-flydsl",
    }
)

# Safe to drop: these change how long the run takes or how much it prints, not
# what it searches.
DROPPABLE_FLAGS = frozenset(
    {
        "-v",
        "--verbose",
        "--iters",
        "--warmup",
        "--mp",
        "--timeout",
        "--min_improvement_pct",
    }
)

_PROBE_TIMEOUT_S = 120

# Every real invocation of a tuner script goes through ``["python3", ...]``, so
# the probe has to ask that interpreter what the script accepts. See probe_script.
_TUNER_PYTHON = "python3"

# (resolved path, content digest) -> surface, so repeated lookups inside one
# session do not even hit the disk cache.
_MEMO: dict[tuple[str, str], "ScriptSurface"] = {}


@dataclass(frozen=True)
class ScriptSurface:
    """The argparse surface of one tuner script."""

    script: str
    flags: frozenset[str]
    # False when --help could not be run or produced nothing parseable. The
    # surface is then permissive rather than restrictive.
    probed: bool
    reason: str = ""

    def supports(self, flag: str) -> bool:
        """True when the script accepts ``flag`` -- or when we could not tell."""
        return not self.probed or flag in self.flags


def _cache_root() -> Path:
    override = os.environ.get("FORGE_SCRIPT_PROBE_CACHE", "").strip()
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".cache"
    return root / "kernelforge.gemm_tune" / "script_probe"


def _cache_path(digest: str) -> Path:
    return _cache_root() / f"{digest}.json"


def _read_cache(digest: str) -> frozenset[str] | None:
    try:
        raw = _cache_path(digest).read_text(encoding="utf-8")
        payload = json.loads(raw)
        # A truncated or hand-edited cache file can decode to a list or a string
        # just as validly as to a dict, and .get on those raises rather than
        # missing. A corrupt cache must cost a re-probe, not the whole run.
        flags = payload.get("flags") if isinstance(payload, dict) else None
    except (OSError, ValueError):
        return None
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        return None
    return frozenset(flags)


def _write_cache(digest: str, script: Path, flags: frozenset[str]) -> None:
    path = _cache_path(digest)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The digest alone identifies the entry; the path is recorded only so a
        # human reading the cache can tell what it belongs to.
        path.write_text(
            json.dumps({"script": str(script), "flags": sorted(flags)}, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:  # a cold cache is a slowdown, never a failure
        log.debug("script probe cache write failed for %s: %s", script, exc)


def parse_help_flags(text: str) -> frozenset[str]:
    """Extract the flags named anywhere in an argparse --help dump."""
    return frozenset(_FLAG_RE.findall(text or ""))


def probe_script(
    script: Path | str,
    *,
    timeout_s: int = _PROBE_TIMEOUT_S,
    use_cache: bool = True,
) -> ScriptSurface:
    """Return the argparse surface of ``script``, running ``--help`` if needed."""
    path = Path(script)
    digest = sha256_file(path)
    if not digest:
        return ScriptSurface(str(path), frozenset(), False, "script unreadable")

    key = (str(path), digest)
    if use_cache:
        memo = _MEMO.get(key)
        if memo is not None:
            return memo
        cached = _read_cache(digest)
        if cached is not None:
            surface = ScriptSurface(str(path), cached, True)
            _MEMO[key] = surface
            return surface

    try:
        proc = subprocess.run(
            # The same interpreter the tuner is launched with. Probing under
            # sys.executable instead is a silent no-op in the usual deployment
            # -- forge in a venv, aiter on the system python3: the probe fails
            # to import aiter, parses no flags, and a surface with probed=False
            # says every flag is supported. The one guard that exists to catch a
            # rejected --libtype then never fires. The reverse is just as bad: a
            # venv carrying a different aiter can report a flag as unsupported
            # that the real interpreter accepts, failing a run that would work.
            [_TUNER_PYTHON, str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(path.parent),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Includes TimeoutExpired. Stay permissive: see module docstring.
        log.warning("script probe failed for %s: %r", path, exc)
        return ScriptSurface(str(path), frozenset(), False, repr(exc))

    # Some scripts print usage to stderr, and a non-zero rc from --help is not
    # unusual once aiter's import side effects are involved. Judge by whether we
    # recovered flags, not by the exit code -- the same lesson as the tuner's own
    # exit code being unusable.
    flags = parse_help_flags(f"{proc.stdout or ''}\n{proc.stderr or ''}")
    if not flags:
        log.warning("script probe parsed no flags from %s --help (rc=%d)", path, proc.returncode)
        return ScriptSurface(str(path), frozenset(), False, f"no flags in --help (rc={proc.returncode})")

    if use_cache:
        _write_cache(digest, path, flags)
    surface = ScriptSurface(str(path), flags, True)
    _MEMO[key] = surface
    return surface


@dataclass(frozen=True)
class ArgFilter:
    """Result of checking an argv tail against a script's surface."""

    args: list[str]
    rejected_required: list[str]
    dropped: list[str]

    @property
    def ok(self) -> bool:
        return not self.rejected_required


_NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _is_flag(token: str) -> bool:
    """Whether a token starts an option rather than being a value.

    Negative numbers are values. Testing ``isdigit()`` alone called ``-1.0``
    and ``-1e-3`` flags, which splits an option from its own argument: the
    number then looks like an unsupported flag and the option looks like it was
    passed nothing.
    """
    if not token.startswith("-") or token == "-":
        return False
    return not _NUMERIC.match(token)


def filter_args(args: list[str], surface: ScriptSurface) -> ArgFilter:
    """Drop unsupported droppable flags; report unsupported required ones.

    Flags are consumed together with their values, so removing ``--iters 20``
    does not leave a stray ``20`` behind. Unsupported flags that are neither
    required nor droppable are kept on purpose: letting the script reject them
    yields a precise argparse error, which beats guessing here.
    """
    kept: list[str] = []
    rejected_required: list[str] = []
    dropped: list[str] = []

    i = 0
    while i < len(args):
        token = args[i]
        if not _is_flag(token):
            kept.append(token)
            i += 1
            continue

        values: list[str] = []
        j = i + 1
        while j < len(args) and not _is_flag(args[j]):
            values.append(args[j])
            j += 1

        if surface.supports(token):
            kept.append(token)
            kept.extend(values)
        elif token in REQUIRED_FLAGS:
            rejected_required.append(token)
            kept.append(token)
            kept.extend(values)
        elif token in DROPPABLE_FLAGS:
            dropped.append(token)
        else:
            # Unknown-and-unsupported: keep it so the failure is explicit.
            kept.append(token)
            kept.extend(values)
        i = j

    if dropped:
        log.warning(
            "%s does not accept %s; dropped (affects speed/verbosity only)",
            surface.script,
            ", ".join(dropped),
        )
    return ArgFilter(kept, rejected_required, dropped)
