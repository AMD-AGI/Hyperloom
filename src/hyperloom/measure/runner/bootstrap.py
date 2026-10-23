# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The start-up script of the sealed measurement runner template (design 4.8.1).

A hosted runner is not started from a copy of Hyperloom that the image ships. The template's start command is this
script. It prepares the runner's software, then replaces itself with ``python -m hyperloom.measure.runner run``, the
command the local sandbox starts too (:mod:`hyperloom.measure.local_templates`), so the sandbox's own process is the
runner. The steps, each of which ends the script with ``runner_start_failed`` when it fails:

1. The source. With ``HYPERLOOM_SOURCE_SHA`` (a full commit hash) and ``HYPERLOOM_SOURCE_REPO`` it fetches that commit
   into ``$HYPERLOOM_SCRATCH_DIR/hl-src``; a fetch by full hash returns that commit or fails, so git, not the network,
   vouches for the content. Without a hash, ``HYPERLOOM_SOURCE_DIR`` names a checkout the image or the test already
   has.
2. A virtual environment, ``$HYPERLOOM_SCRATCH_DIR/hl-venv``, made by ``HYPERLOOM_BASE_PYTHON`` (``python3``). In the
   framework image ``python3`` is ``/opt/venv``, and a venv made with ``--system-site-packages`` does not see what is
   installed there (torch, sglang, aiter), so the script drops a ``.pth`` file into the new venv that adds the source
   (``<source>/src``) and then the base interpreter's site directories, with their own ``.pth`` files processed, as
   ``site.addsitedir`` does. The venv's own packages come first.
3. ``pip install --require-hashes --no-deps -r requirements/runner.lock`` into that venv, after the ``.pth`` is in
   place: a pin the base already has is satisfied and left alone, every other one is installed and must match its hash,
   so nothing is resolved on the sandbox. ``PIP_INDEX_URL``, ``PIP_FIND_LINKS`` and the like are the template's
   environment; the script sets no index.
4. A check that the runner imports, so a broken install ends the script instead of leaving the application waiting.
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

``BOOTSTRAP_SCRIPT`` is POSIX ``sh`` with a ``#!/bin/sh`` line, so it runs the same whether a template hands it over as
a file or as the argument of ``sh -c``; one text serves the hosted template and the stand-in the contract suite runs.
"""

from __future__ import annotations

__all__ = ["BOOTSTRAP_SCRIPT", "runner_start_command"]

#: ``$0`` of ``sh -c`` is a name for messages, ``$@`` are the runner's own arguments.
_NAME = "hyperloom-runner-bootstrap"

BOOTSTRAP_SCRIPT = r"""#!/bin/sh
set -eu

SCRATCH=${HYPERLOOM_SCRATCH_DIR:-/scratch}
IN_DIR=${HYPERLOOM_IN_DIR:-/in}
OUT_DIR=${HYPERLOOM_OUT_DIR:-/out}
BASE_PY=${HYPERLOOM_BASE_PYTHON:-python3}
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

if [ -n "${HYPERLOOM_SOURCE_SHA:-}" ]; then
    SRC=$SCRATCH/hl-src
    case $HYPERLOOM_SOURCE_SHA in
        *[!0-9a-f]*) refuse source "HYPERLOOM_SOURCE_SHA is not a lowercase hexadecimal commit hash" ;;
    esac
    [ ${#HYPERLOOM_SOURCE_SHA} -eq 40 ] || [ ${#HYPERLOOM_SOURCE_SHA} -eq 64 ] ||
        refuse source "HYPERLOOM_SOURCE_SHA is not a full commit hash"
    [ -n "${HYPERLOOM_SOURCE_REPO:-}" ] || refuse source "HYPERLOOM_SOURCE_REPO is not set"
    step clean-source rm -rf "$SRC"
    step git-init git init -q "$SRC"
    step git-fetch git -C "$SRC" fetch -q --depth 1 "$HYPERLOOM_SOURCE_REPO" "$HYPERLOOM_SOURCE_SHA"
    step git-checkout git -C "$SRC" checkout -q --detach FETCH_HEAD
    export HYPERLOOM_CODE_REVISION=$HYPERLOOM_SOURCE_SHA
else
    SRC=${HYPERLOOM_SOURCE_DIR:-}
    [ -n "$SRC" ] || refuse source "neither HYPERLOOM_SOURCE_SHA nor HYPERLOOM_SOURCE_DIR is set"
fi
[ -f "$SRC/requirements/runner.lock" ] || refuse source "the source has no requirements/runner.lock"

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
step lock "$VENV/bin/python" -m pip install --require-hashes --no-deps --disable-pip-version-check --no-input \
    -r "$SRC/requirements/runner.lock"
step import "$VENV/bin/python" -c "import hyperloom.measure.runner.__main__"

rm -f "$LOG"
export HYPERLOOM_BOOTSTRAP_STARTED_UNIX=$STARTED HYPERLOOM_BOOTSTRAP_ENDED_UNIX=$(now)
exec "$VENV/bin/python" -m hyperloom.measure.runner run --in "$IN_DIR" --out "$OUT_DIR" --scratch "$SCRATCH" "$@"
"""


def runner_start_command(*runner_args: str) -> tuple[str, ...]:
    """The start argv of a template that runs the script as ``sh -c``; ``runner_args`` are passed on to the runner.

    A template that hands the script over as a file uses :data:`BOOTSTRAP_SCRIPT` itself.
    """
    return ("/bin/sh", "-c", BOOTSTRAP_SCRIPT, _NAME, *runner_args)
