# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The start-up script of the sealed measurement runner template (design 4.8.1).

A hosted runner is not started from a copy of Hyperloom that the image ships. The template's start command is this
script. It prepares the runner's software, then replaces itself with ``python -m hyperloom.measure.runner run``, the
command the local sandbox starts too (:mod:`hyperloom.measure.local_templates`), so the sandbox's own process is the
runner. The steps, each of which ends the script with ``runner_start_failed`` when it fails:

1. The cache. A measurement sandbox has no network: there is no git, no package index and no pip configuration in
   play. The commit's source and every wheel ``requirements/runner.lock`` pins are in the runner cache, a directory on
   shared storage that the template mounts read-only (``HYPERLOOM_RUNNER_CACHE``, ``/runner-cache`` when unset; layout
   and builder in :mod:`hyperloom.measure.runner.cache_layout` and :mod:`hyperloom.measure.runner.cache_build`).
   ``HYPERLOOM_SOURCE_SHA`` (a full commit hash) names the entry. The script first checks it with the sandbox's own
   interpreter in isolated mode (``-I``): the entry has a manifest (a build that did not finish has none), the source is
   the tree the manifest recorded, and every pinned wheel is there with its recorded size. What is missing is named in
   the failure. The same step writes ``$HYPERLOOM_SCRATCH_DIR/runner-wheels.lock``, the manifest's wheels each pinned to
   its recorded hash.
2. A virtual environment, ``$HYPERLOOM_SCRATCH_DIR/hl-venv``, made by ``HYPERLOOM_BASE_PYTHON`` (``python3``). In the
   framework image ``python3`` is ``/opt/venv``, and a venv made with ``--system-site-packages`` does not see what is
   installed there (torch, sglang, aiter), so the script drops a ``.pth`` file into the new venv that adds the source
   (``<entry>/source/src``) and then the base interpreter's site directories, with their own ``.pth`` files processed, as
   ``site.addsitedir`` does. The venv's own packages come first.
3. ``pip install --isolated --no-index --find-links <entry>/wheels --only-binary=:all: --require-hashes --no-deps -r
   runner-wheels.lock`` into that venv, after the ``.pth`` is in place: a pin the base already has is satisfied and left
   alone, every other one is installed from the cache and must match its hash. ``--isolated`` makes pip ignore
   ``PIP_INDEX_URL``, ``PIP_FIND_LINKS`` and the like, so the template's environment cannot point it anywhere else.
4. A check that the runner imports, so a broken install ends the script instead of leaving the application waiting. Python
   writes its bytecode below ``$HYPERLOOM_SCRATCH_DIR/pycache``, never into the cache's source tree.
5. ``HYPERLOOM_CODE_REVISION`` is set to the commit and the runner replaces the script. The runner's remaining
   arguments are the script's own (``--spec-wait-sec`` and so on); ``--in``, ``--out`` and ``--scratch`` come from
   ``HYPERLOOM_IN_DIR``, ``HYPERLOOM_OUT_DIR`` and ``HYPERLOOM_SCRATCH_DIR`` (``/in``, ``/out``, ``/scratch``).
   The unix times the script began and ended go to the runner as ``HYPERLOOM_BOOTSTRAP_STARTED_UNIX`` and
   ``HYPERLOOM_BOOTSTRAP_ENDED_UNIX`` (empty when the image's ``date`` cannot say them), so the runner can state how long
   the sandbox spent on the script, and how long the interpreter took to reach the runner, as phases of the test.

A failing step writes ``/out/status.json`` itself (``state`` failed, ``error_class`` ``runner_start_failed``, the step
and the last of its output in ``message``) and exits 2, as the runner does when it cannot start, so the application
reads the failure instead of waiting for a runner that never came. Every step runs without
``HYPERLOOM_SIGNING_KEY_FILE``: the key file exists before the script starts, and nothing the script installs or builds
needs it; the runner takes it from its own environment.

When the template sets ``HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256``, only the entry whose manifest file has that SHA-256 is
accepted: the host takes the value from the builder's result and so binds the sandbox to the build it made. Without it the
manifest vouches only for the entry it sits in, so the cache directory must then be writable by the builder alone.

``BOOTSTRAP_SCRIPT`` is POSIX ``sh`` with a ``#!/bin/sh`` line, so it runs the same whether a template hands it over as
a file or as the argument of ``sh -c``; one text serves the hosted template and the stand-in the contract suite runs.

Starting it from the cache entry. A template whose step cannot carry an argv (:func:`runner_start_command` passes the
whole script as one argument) runs the copy the builder writes into every entry, ``<entry>/bootstrap.sh``, whose
SHA-256 the entry's manifest records (:mod:`hyperloom.measure.runner.cache_build`)::

    sh $HYPERLOOM_RUNNER_CACHE/$HYPERLOOM_SOURCE_SHA/bootstrap.sh <runner arguments>

What that trusts, for operators: the template pins the manifest (``HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256``) and the
manifest pins ``bootstrap.sh``, but the script is what checks the manifest, so it can check nothing before it runs: a
``bootstrap.sh`` replaced in the cache runs, pin or no pin. That plain step is therefore sound only when the cache is
writable by its builder alone (the sandbox mounts it read-only; no other account on the shared storage may write it).
Where that cannot be promised, the step checks the chain first, with the one line :func:`runner_start_shell_line`
returns: it copies the manifest and ``bootstrap.sh`` out of the cache, checks the copied manifest against the pin and
the copied script against the hash that manifest records (``sha256sum -c``), and only then runs the checked copy, so a
file swapped after the check is never the one that runs. A missing pin, a missing file or a hash that differs ends the
step with the same ``runner_start_failed`` status and exit 2 as a failing step of the script.

A template step must return. A sandbox template's steps run before the sandbox service starts its own agent in the
sandbox, the one that answers readiness and through which the application delivers the spec: a step that runs the runner
in the foreground never ends, so the agent never starts, the sandbox never turns ready and the spec never arrives. Use
:func:`runner_template_step`, which makes the directories, runs the template's own preparation and then starts the
checked line in the background (``nohup ... &``) with its output in ``runner-boot.log`` next to the scratch directory,
and returns at once. The runner then waits for the spec for ``--spec-wait-sec`` seconds, :data:`TEMPLATE_SPEC_WAIT_SEC`
unless the step's arguments name another.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from hyperloom.measure.runner import cache_layout

__all__ = [
    "BOOTSTRAP_SCRIPT",
    "TEMPLATE_SPEC_WAIT_SEC",
    "runner_start_command",
    "runner_start_shell_line",
    "runner_template_step",
]

#: ``$0`` of ``sh -c`` is a name for messages, ``$@`` are the runner's own arguments.
_NAME = "hyperloom-runner-bootstrap"

#: How long a runner a template step starts waits for its spec, unless the step's arguments say. The step returns before
#: the sandbox is ready, and the application writes the spec only after the create, which a quota can hold for minutes,
#: and the readiness wait; the runner's own default (600 s) is the wait of a sandbox whose spec is already on its way.
#: Half an hour covers a create queued behind others (the joint runs used it, with no spec missed); a runner whose spec
#: never comes ends the sandbox no later than that, and the application deletes a sandbox it gave up on anyway.
TEMPLATE_SPEC_WAIT_SEC = 1800
_SPEC_WAIT_ARG = "--spec-wait-sec"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SCRIPT = r"""#!/bin/sh
set -eu

SCRATCH=${HYPERLOOM_SCRATCH_DIR:-/scratch}
IN_DIR=${HYPERLOOM_IN_DIR:-/in}
OUT_DIR=${HYPERLOOM_OUT_DIR:-/out}
BASE_PY=${HYPERLOOM_BASE_PYTHON:-python3}
CACHE=${@CACHE_ENV@:-@DEFAULT_CACHE@}
SHA=${HYPERLOOM_SOURCE_SHA:-}
REQS=$SCRATCH/@REQUIREMENTS_FILE@
VENV=$SCRATCH/hl-venv
LOG=${TMPDIR:-/tmp}/hl-bootstrap.$$.log
STEP=start

# now: the unix time, with fractions where this image's date has them, else whole seconds, else nothing.
now() {
    t=$(date +%s.%N 2>/dev/null) || t=
    case $t in '' | *[!0-9.]*) t=$(date +%s 2>/dev/null) || t= ;; esac
    printf '%s' "$t"
}
STARTED=$(now)

# fail RC: leave the application a status.json that says what failed, then end like a runner that could not start.
fail() {
    detail=$(tail -c 600 "$LOG" 2>/dev/null | LC_ALL=C tr -cd ' -~' | tr -d '"\\') || detail=
    mkdir -p "$OUT_DIR" 2>/dev/null || true
    {
        printf '{"error_class": "runner_start_failed", "message": "runner bootstrap, step %s, exit %s: %s", ' \
            "$STEP" "$1" "$detail"
        printf '"state": "failed"}\n'
    } >"$OUT_DIR/.status.json.tmp" && mv "$OUT_DIR/.status.json.tmp" "$OUT_DIR/status.json" ||
        echo "bootstrap: cannot write $OUT_DIR/status.json" >&2
    exit 2
}

# refuse STEP MESSAGE: a failure of the script's own check.
refuse() {
    STEP=$1
    echo "$2" >"$LOG"
    cat "$LOG" >&2
    fail 1
}

# step NAME COMMAND...: run one command with its output kept for the message of a failure and without the signing key.
step() {
    STEP=$1
    shift
    echo "bootstrap: $STEP" >&2
    (unset HYPERLOOM_SIGNING_KEY_FILE; "$@") >"$LOG" 2>&1 || {
        rc=$?
        cat "$LOG" >&2
        fail "$rc"
    }
}

[ -n "$SHA" ] || refuse cache "HYPERLOOM_SOURCE_SHA is not set"
step cache "$BASE_PY" -I - "$CACHE" "$SHA" "$REQS" <<'PY'
@CACHE_CHECK@
PY
SRC=$CACHE/$SHA/source
WHEELS=$CACHE/$SHA/wheels
export HYPERLOOM_CODE_REVISION=$SHA
export PYTHONPYCACHEPREFIX=$SCRATCH/pycache

step scratch mkdir -p "$SCRATCH"
step venv "$BASE_PY" -m venv --clear "$VENV"
step base-packages "$VENV/bin/python" - "$BASE_PY" "$SRC/src" <<'PY'
import json, os, subprocess, sys, sysconfig

base_python, source = sys.argv[1:3]
probe = (
    "import json, site, sysconfig; paths = sysconfig.get_paths(); "
    "print(json.dumps([*site.getsitepackages(), paths['purelib'], paths['platlib']]))"
)
found = subprocess.run([base_python, "-c", probe], check=True, capture_output=True, text=True).stdout
base_dirs = [path for path in dict.fromkeys(json.loads(found)) if os.path.isdir(path)]
if not base_dirs:
    sys.exit("the base interpreter reports no site-packages directory")
lines = [source, *(f"import site; site.addsitedir({path!r})" for path in base_dirs)]
with open(os.path.join(sysconfig.get_path("purelib"), "hyperloom-runner.pth"), "w") as pth:
    pth.write("\n".join(lines) + "\n")
PY
step lock "$VENV/bin/python" -m pip --isolated install --no-index --find-links "$WHEELS" --only-binary=:all: \
    --require-hashes --no-deps --disable-pip-version-check --no-input -r "$REQS"
step import "$VENV/bin/python" -c "import hyperloom.measure.runner.__main__"

rm -f "$LOG"
export HYPERLOOM_BOOTSTRAP_STARTED_UNIX=$STARTED HYPERLOOM_BOOTSTRAP_ENDED_UNIX=$(now)
exec "$VENV/bin/python" -m hyperloom.measure.runner run --in "$IN_DIR" --out "$OUT_DIR" --scratch "$SCRATCH" "$@"
"""

BOOTSTRAP_SCRIPT = (
    _SCRIPT.replace("@CACHE_ENV@", cache_layout.CACHE_ENV)
    .replace("@DEFAULT_CACHE@", cache_layout.DEFAULT_CACHE_DIR)
    .replace("@REQUIREMENTS_FILE@", cache_layout.REQUIREMENTS_FILE)
    .replace("@CACHE_CHECK@", Path(cache_layout.__file__).read_text(encoding="utf-8").rstrip("\n"))
)


_VERIFY_FAILED = json.dumps(
    {
        "error_class": "runner_start_failed",
        "message": (
            f"runner bootstrap, step verify, exit 1: {cache_layout.BOOTSTRAP_NAME} or {cache_layout.MANIFEST_NAME} of the "
            f"cache entry is missing or not the one {cache_layout.MANIFEST_PIN_ENV} pins, or that or "
            "HYPERLOOM_SOURCE_SHA is not set"
        ),
        "state": "failed",
    }
)

#: The checked start of :func:`runner_start_shell_line`: ``e`` the entry, ``d`` a private copy, ``o`` the outputs.
_VERIFIED_START = (
    'e="${@CACHE_ENV@:-@DEFAULT_CACHE@}/${HYPERLOOM_SOURCE_SHA:-}" d="${TMPDIR:-/tmp}/hl-start.$$" '
    'o="${HYPERLOOM_OUT_DIR:-/out}"; '
    'if [ -n "${HYPERLOOM_SOURCE_SHA:-}" ] && [ -n "${@PIN_ENV@:-}" ] && mkdir -m 700 "$d" '
    '&& cp "$e/@MANIFEST@" "$e/@SCRIPT@" "$d/" '
    '&& printf \'%s  %s\\n\' "$@PIN_ENV@" "$d/@MANIFEST@" | sha256sum -c - >&2 '
    '&& h=$(sed -n \'s/^  "bootstrap_sha256": "\\([0-9a-f]\\{64\\}\\)",*$/\\1/p\' "$d/@MANIFEST@") '
    '&& [ -n "$h" ] && printf \'%s  %s\\n\' "$h" "$d/@SCRIPT@" | sha256sum -c - >&2; '
    'then exec sh "$d/@SCRIPT@"@ARGS@; fi; '
    'echo "@NAME@: @SCRIPT@ or @MANIFEST@ of $e is missing or not the one @PIN_ENV@ pins" >&2; '
    'mkdir -p "$o" && printf \'%s\\n\' @STATUS@ >"$o/.status.json.tmp" && mv "$o/.status.json.tmp" "$o/status.json"; '
    "exit 2"
)


def runner_start_command(*runner_args: str) -> tuple[str, ...]:
    """The start argv of a template that runs the script as ``sh -c``; ``runner_args`` are passed on to the runner.

    A template that hands the script over as a file uses :data:`BOOTSTRAP_SCRIPT` itself.
    """
    return ("/bin/sh", "-c", BOOTSTRAP_SCRIPT, _NAME, *runner_args)


def runner_start_shell_line(*runner_args: str) -> str:
    """One POSIX ``sh`` line for a template step that runs the entry's ``bootstrap.sh`` only after checking it.

    The line checks the entry's manifest against ``HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256`` and ``bootstrap.sh``
    against the hash that manifest records, then runs the checked copy with ``runner_args`` (each quoted for ``sh``);
    see the module's documentation for what each start form trusts.
    """
    line = (
        _VERIFIED_START.replace("@CACHE_ENV@", cache_layout.CACHE_ENV)
        .replace("@DEFAULT_CACHE@", cache_layout.DEFAULT_CACHE_DIR)
        .replace("@PIN_ENV@", cache_layout.MANIFEST_PIN_ENV)
        .replace("@MANIFEST@", cache_layout.MANIFEST_NAME)
        .replace("@SCRIPT@", cache_layout.BOOTSTRAP_NAME)
        .replace("@NAME@", _NAME)
        .replace("@STATUS@", shlex.quote(_VERIFY_FAILED))
    )
    return line.replace("@ARGS@", "".join(f" {shlex.quote(arg)}" for arg in runner_args))


def _with_spec_wait(runner_args: tuple[str, ...]) -> tuple[str, ...]:
    named = any(arg == _SPEC_WAIT_ARG or arg.startswith(f"{_SPEC_WAIT_ARG}=") for arg in runner_args)
    return runner_args if named else (*runner_args, _SPEC_WAIT_ARG, str(TEMPLATE_SPEC_WAIT_SEC))


def runner_template_step(
    *runner_args: str,
    pre: str = "",
    env: Mapping[str, str] | None = None,
    dirs: Sequence[str] = ("/in", "/out", "/scratch"),
    boot_log: str = "/runner-boot.log",
) -> str:
    """The one-line start step of a measurement template; it starts the runner in the background and returns at once.

    The line makes ``dirs`` (the template's ``HYPERLOOM_IN_DIR``, ``HYPERLOOM_OUT_DIR`` and ``HYPERLOOM_SCRATCH_DIR``,
    whose defaults the script's are), runs ``pre`` (the template's own preparation, one line of ``sh``, run in the
    foreground and only when the directories were made), then starts :func:`runner_start_shell_line` with
    ``runner_args`` under ``nohup`` in the background, with ``env`` set for it alone and its output in ``boot_log``
    (``runner-boot.log`` next to the scratch directory, never inside the runner's own). A step that never returned
    would keep the sandbox from ever turning ready (see the module's documentation). ``runner_args`` get
    ``--spec-wait-sec`` :data:`TEMPLATE_SPEC_WAIT_SEC` unless they name it.

    Raises:
        ValueError: ``pre`` spans lines, or a name in ``env`` is not a shell variable name.
    """
    if "\n" in pre or "\r" in pre:
        raise ValueError("pre is one line of sh: a template step is a single line")
    bad = sorted(name for name in env or {} if not _ENV_NAME.match(name))
    if bad:
        raise ValueError(f"not shell variable names: {', '.join(map(repr, bad))}")
    assignments = "".join(f"{name}={shlex.quote(value)} " for name, value in (env or {}).items())
    start = shlex.quote(runner_start_shell_line(*_with_spec_wait(runner_args)))
    background = f"({assignments}nohup sh -c {start} >{shlex.quote(boot_log)} 2>&1 &)"
    made = "mkdir -p " + " ".join(shlex.quote(directory) for directory in dirs)
    return " && ".join(part for part in (made, pre, background) if part)
