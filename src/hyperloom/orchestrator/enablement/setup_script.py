# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render ``reports/enablement/enablement_setup.sh``: the enablement's setup, without a launch.

The script re-runs the accepted setup commands verbatim and in order, writes
each contributing root's captured final-state files to the absolute path that
root resolved to in the session, and ends by checking that every package
version the enablement changed came out the same. It never starts a server;
``enablement_setting.sh`` calls it and then launches.

A stack the session did not capture in full cannot be reproduced from final
state, so the script then refuses before it changes anything, naming what is
missing. That refusal is part of the contract: :func:`setup_script_record`
reads it back so a recipe never declares a refusing script standalone.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from hyperloom.common.io import atomic_write_text
from hyperloom.inference_optimizer.session.session_paths import enablement_dir

from .recipe.projections import root_id_for

if TYPE_CHECKING:
    from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

SETUP_SCRIPT_NAME = "enablement_setup.sh"

#: Final-state copies the script installs, next to it so the directory travels as one unit.
SETUP_FILES_DIR = "setup_files"

#: Opens the refusal block; its presence is what makes a script non-standalone.
REFUSAL_MARKER = "# This enablement cannot be reproduced from this session's records:"

_HELPERS = """\
check_sha256() {
  local want="$1" path="$2" got
  if [ ! -f "$path" ] || [ -L "$path" ]; then
    echo "ERROR: $path is not a regular file" >&2
    exit 1
  fi
  got="$(sha256sum -- "$path" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    echo "ERROR: sha256 of $path is $got, expected $want" >&2
    exit 1
  fi
}

inside_root() {
  local root parent
  root="$(realpath -m -- "$1")"
  parent="$(realpath -m -- "$(dirname -- "$2")")"
  case "$parent/" in
    "$root"/*) ;;
    *)
      echo "ERROR: $2 resolves outside its root $1" >&2
      exit 1
      ;;
  esac
}

place_file() {
  local want="$1" mode="$2" src="$SCRIPT_DIR/$3" dst="$4/$5"
  check_sha256 "$want" "$src"
  inside_root "$4" "$dst"
  # The final state is a regular file: a symlink there is replaced, never followed.
  if [ -L "$dst" ]; then rm -f -- "$dst"; fi
  install -D -T -m "$mode" -- "$src" "$dst"
  check_sha256 "$want" "$dst"
}

remove_file() {
  local dst="$1/$2"
  inside_root "$1" "$dst"
  rm -f -- "$dst"
  if [ -e "$dst" ] || [ -L "$dst" ]; then
    echo "ERROR: could not remove $dst" >&2
    exit 1
  fi
}"""

_VERSION_CHECK = """\
import importlib.metadata as metadata
import json
import sys

want = json.loads({payload})
bad = []
for name, version in sorted(want.items()):
    try:
        got = metadata.version(name)
    except metadata.PackageNotFoundError:
        got = None
    if got != version:
        bad.append(f"{{name}}: expected {{version or 'absent'}}, found {{got or 'absent'}}")
for line in bad:
    print(f"ERROR: package version differs from the accepted enablement: {{line}}", file=sys.stderr)
sys.exit(1 if bad else 0)"""


@dataclass
class _Plan:
    """What the script does, and why it cannot when ``refusals`` is non-empty."""

    places: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    removes: list[tuple[str, str]] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_inside(rel: str) -> PurePosixPath | None:
    """``rel`` as a relative path that cannot climb out of its root, else ``None``."""
    path = PurePosixPath(str(rel or ""))
    if not str(rel or "") or path.is_absolute() or ".." in path.parts:
        return None
    return path


def _coverage_refusals(enablement: "EnablementRound", captured: Mapping[str, set[str]]) -> list[str]:
    """Name every kept change the per-root captures do not contain.

    A patch is covered once the KEEP recorded the targets it declares and the
    capture of their root holds each of them; one stacked after the last KEEP,
    or whose targets could not be proven, has no final state to install.
    """
    refusals = [
        f"kept patch {Path(str(patch)).name} has no captured final state"
        for patch in enablement.kept_patches
        if str(patch) not in (enablement.patch_targets or {})
    ]
    for root_id, declared in (enablement.accepted_stack_targets or {}).items():
        refusals += [
            f"{rel} in root {root_id} was declared but not captured"
            for rel in sorted(declared or {})
            if str(rel) not in captured.get(str(root_id), set())
        ]
    for artifact in enablement.kept_artifacts:
        root = str(artifact.get("root") or enablement.framework_root or "")
        rel = str(artifact.get("rel_target") or "")
        if not rel or rel not in captured.get(root_id_for(root), set()):
            refusals.append(f"installed file {Path(str(artifact.get('target') or rel)).name} was not captured")
    return refusals


def _plan_files(session_dir: Path, enablement: "EnablementRound", script_dir: Path) -> _Plan:
    """Copy every captured file next to the script and plan where it goes.

    Each root's capture is written to the absolute path that root resolved to
    in the session, so a stack spanning several trees lands each file in its
    own tree rather than under one framework root.
    """
    plan = _Plan()
    roots = {str(r.get("id")): str(r.get("path") or "") for r in enablement.roots if isinstance(r, Mapping)}
    captured: dict[str, set[str]] = {}
    for capture in sorted(enablement.source_snapshots, key=lambda s: str(s.get("root_id"))):
        root_id = str(capture.get("root_id") or "")
        root = roots.get(root_id, "")
        overlay = _relative_inside(str(capture.get("snapshot_ref") or ""))
        if not PurePosixPath(root).is_absolute() or overlay is None:
            plan.refusals.append(f"root {root_id} has no absolute path or capture to install from")
            continue
        for row in capture.get("files") or []:
            rel = _relative_inside(str(row.get("rel") or ""))
            op = str(row.get("op") or "")
            if rel is None or op not in ("upsert", "delete"):
                plan.refusals.append(f"{row.get('rel')} in root {root_id} was not captured")
                continue
            captured.setdefault(root_id, set()).add(str(rel))
            if op == "delete":
                plan.removes.append((root, str(rel)))
                continue
            source = session_dir / str(overlay) / "files" / str(rel)
            if not source.is_file() or source.is_symlink():
                plan.refusals.append(f"captured {rel} in root {root_id} is not in the session")
                continue
            copy_rel = f"{SETUP_FILES_DIR}/{root_id}/{rel}"
            copy = script_dir / copy_rel
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copy)
            mode = f"{stat.S_IMODE(source.stat().st_mode):04o}"
            plan.places.append((_sha256(copy), mode, copy_rel, root, str(rel)))
    plan.refusals.extend(_coverage_refusals(enablement, captured))
    return plan


def changed_versions(enablement: "EnablementRound") -> tuple[str, dict[str, str | None]] | None:
    """Return ``(interpreter, {package: version})`` the enablement changed, or ``None``.

    The change is the accepted closure against the one observed before the
    first setup command ran, through the same interpreter; ``None`` marks a
    removed package. The versions the KEEP asserted are checked as well. ``None``
    when either closure or the interpreter is unrecorded, or the two were read
    through different interpreters -- no diff then says what changed.
    """
    accepted = enablement.environment_closure or {}
    baseline = enablement.environment_closure_baseline or {}
    interpreter = str(accepted.get("interpreter") or "")
    after = accepted.get("distributions") or {}
    before = baseline.get("distributions") or {}
    if not (interpreter and after and before) or str(baseline.get("interpreter") or "") != interpreter:
        return None
    want: dict[str, str | None] = {name: version for name, version in after.items() if before.get(name) != version}
    want.update({name: None for name in before if name not in after})
    want.update({str(k): str(v) for k, v in (enablement.installed_versions_at_keep or {}).items()})
    return interpreter, want


def _version_check(interpreter: str, want: Mapping[str, str | None]) -> list[str]:
    quoted = shlex.quote(interpreter)
    missing = shlex.quote(f"ERROR: interpreter {interpreter} is not present")
    return [
        "# Package versions this enablement changed, as observed when it was accepted.",
        f"[ -x {quoted} ] || {{ echo {missing} >&2; exit 1; }}",
        # From / so the script's own directory is not on the probe's sys.path.
        f"(cd / && {quoted} - <<'PY'",
        _VERSION_CHECK.format(payload=repr(json.dumps(dict(want), sort_keys=True))),
        "PY",
        ")",
    ]


def _refusal(reasons: list[str]) -> list[str]:
    lines = [REFUSAL_MARKER, 'echo "ERROR: this enablement cannot be reproduced from its records:" >&2']
    lines += [f"echo {shlex.quote('  - ' + reason)} >&2" for reason in reasons]
    return [*lines, "exit 1"]


def render_setup_script(session_dir: Path, enablement: "EnablementRound") -> str:
    """Write the setup's file copies and return the script text that installs them."""
    script_dir = enablement_dir(session_dir)
    files_dir = script_dir / SETUP_FILES_DIR
    shutil.rmtree(files_dir, ignore_errors=True)
    plan = _plan_files(session_dir, enablement, script_dir)
    commands = [str(c) for c in enablement.setup_commands or [] if str(c).strip()]
    versions = changed_versions(enablement)
    if versions is None and commands:
        plan.refusals.append("the package versions before and after the setup commands were not both recorded")
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by hyperloom: reproduces this enablement's setup. Starts no server.",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "$SCRIPT_DIR"',
    ]
    if plan.refusals:
        shutil.rmtree(files_dir, ignore_errors=True)
        return "\n".join([*lines, *_refusal(plan.refusals)]) + "\n"
    runtime = str((enablement.active_runtime or {}).get("venv_root") or "")
    if runtime:
        lines.append(
            f"# NOTE: the accepted server ran from the isolated runtime {runtime!r}, which this does not create."
        )
    # The same defaults the session's setup replay ran under.
    lines.append('export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"')
    lines.append('export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"')
    if commands:
        lines += ["", "# Setup commands, verbatim and in the order the session ran them.", *commands]
    if plan.places or plan.removes:
        lines += ["", _HELPERS, "", "# Final state of every file the enablement changed, per root."]
        lines += [f"place_file {' '.join(shlex.quote(part) for part in place)}" for place in plan.places]
        lines += [f"remove_file {' '.join(shlex.quote(part) for part in remove)}" for remove in plan.removes]
    if versions is not None:
        lines += ["", *_version_check(*versions)]
    return "\n".join(lines) + "\n"


def write_setup_script(session_dir: str | Path, enablement: "EnablementRound") -> str:
    """Write ``enablement_setup.sh`` and its file copies; return its session-relative path."""
    root = Path(session_dir)
    out = enablement_dir(root) / SETUP_SCRIPT_NAME
    atomic_write_text(out, render_setup_script(root, enablement), make_parents=True, mode=0o700)
    return str(out.relative_to(root))


def setup_script_record(session_dir: Path, *, sufficient: bool) -> dict[str, Any] | None:
    """Name the setup script by path and sha256, and whether it reproduces the enablement on its own.

    ``standalone`` is claimed only for a sufficient recipe whose script does
    not refuse: the verdict judges the recorded stack, the refusal judges what
    this script was able to carry.
    """
    path = enablement_dir(session_dir) / SETUP_SCRIPT_NAME
    if not path.is_file():
        return None
    body = path.read_bytes()
    return {
        "path": str(path.relative_to(session_dir)),
        "sha256": hashlib.sha256(body).hexdigest(),
        "standalone": sufficient and REFUSAL_MARKER.encode() not in body,
    }


__all__ = [
    "REFUSAL_MARKER",
    "SETUP_SCRIPT_NAME",
    "changed_versions",
    "render_setup_script",
    "setup_script_record",
    "write_setup_script",
]
