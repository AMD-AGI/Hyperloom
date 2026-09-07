# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``setup_executions``: one durable row per attempted setup execution.

Recorded where the commands run rather than where the round reports, because
several exits after setup has already mutated the shared venv return without the
outcome lists -- and one returns without the enablement flag at all, which the
lane's rearm ignores entirely.

No row stores a command verbatim: this is the one field that records a *failed*
execution, whose text is in no durable field today, so keeping it verbatim would
open a credential sink the pre-existing ``setup.cmd`` exposure does not cover.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from .credentials import (
    classify_credential_class,
    detect_credential_channels,
    installer_class,
    option_operands,
    sanitize_command_text,
    split_env_assignments,
)
from .steps import command_digest

#: Clip for the sanitized form, mirroring the executor's own rejection clip so a
#: single install naming a hundred packages cannot crowd out its neighbours.
_CMD_SANITIZED_CHARS = 160

_REQUIREMENT_OPTIONS: frozenset[str] = frozenset({"-r", "--requirement", "--constraint"})

#: pip's short spelling of ``--constraint``; admitted only for that family,
#: because the same flag names a channel to conda.
_PIP_CONSTRAINT_SHORT = "-c"
_ARCHIVE_SUFFIXES: tuple[str, ...] = (".whl", ".tar.gz", ".tgz", ".zip", ".tar.bz2")
_VCS_PREFIXES: tuple[str, ...] = ("git+", "hg+", "svn+")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: The word after which an installer's bare operands are the things installed
#: rather than the program or its subcommand.
_INSTALL_VERBS: frozenset[str] = frozenset({"install", "add", "get", "i", "ci", "reinstall", "update", "upgrade"})

#: pip's hash-checking mode, which is the one way a bare requirement names bytes.
_HASH_OPTION = "--hash"

#: Options whose next token is their value rather than a thing being installed.
#: An option missing from this set costs an over-refusal, never a false identity.
_VALUE_OPTIONS: frozenset[str] = frozenset(
    {
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-i",
        "--index-url",
        "--extra-index-url",
        "-f",
        "--find-links",
        "--hash",
        "-e",
        "--editable",
        "--registry",
        "--channel",
        "--target",
        "--prefix",
        "--root",
        "--python-version",
        "--platform",
        "--abi",
        "--implementation",
        "--upgrade-strategy",
    }
)

OUTCOMES: tuple[str, ...] = ("applied", "failed", "skipped")


def _file_identity(path: Path, *, root: Path) -> dict[str, str] | None:
    """Digest one input file, named by where the delivery would carry it."""
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        # Outside the session, so no bundle can carry it; the name is all a
        # consumer gets, and the delivery rule refuses on the absence.
        rel = path.name
    return {"rel": rel, "sha256": hashlib.sha256(payload).hexdigest()}


def _vcs_identity(operand: str) -> tuple[dict[str, Any] | None, str]:
    """Identify a VCS requirement, or name it unresolved when its ref moves."""
    _, _, tail = operand.partition("://")
    ref = tail.rsplit("@", 1)[1] if "@" in tail else ""
    if ref and _COMMIT_SHA_RE.match(ref):
        return {"kind": "vcs_url", "resolved_ref": ref}, ""
    return None, "vcs_ref"


def _looks_local(operand: str) -> bool:
    if "://" in operand:
        return False
    return operand.startswith((".", "/")) or operand.endswith(_ARCHIVE_SUFFIXES)


def _package_operands(tokens: list[str]) -> list[str]:
    """Return the things an install names directly, after its own verb.

    A token following an option this module does not know to take a value reads
    as a package, which over-refuses rather than inventing an identity.
    """
    out: list[str] = []
    installing = False
    previous = ""
    for token in tokens:
        if token.startswith("-"):
            previous = token
            continue
        if not installing:
            installing = token in _INSTALL_VERBS
            previous = ""
            continue
        was = previous
        previous = ""
        if was in _VALUE_OPTIONS:
            continue
        if not (token.startswith(_VCS_PREFIXES) or "://" in token or _looks_local(token)):
            out.append(token)
    return out


def _requirement_digests(pairs: list[tuple[str, str]]) -> list[str]:
    """Return the digests a hash-checking install pinned its requirements to."""
    return [operand.rsplit(":", 1)[-1] for option, operand in pairs if option == _HASH_OPTION and operand]


def setup_input_identity(cmd: str, *, cwd: Path | str) -> tuple[list[dict[str, Any]], list[str]]:
    """Identify every input an install consumes that decides what it installs.

    An installer allowlisted by prefix runs verbatim, so a local wheel, a
    requirements file, a branch-pinned VCS ref or a plain package spec can
    decide what gets installed while nothing durable names the bytes it
    installed. Each is either captured by content digest or reported unresolved,
    so ``sufficient`` is never reported over an install the recipe cannot
    reproduce.

    A version pin is not an identity: the same ``name==version`` against the
    same index resolves to different bytes over time, and the install runs
    against whatever that index then holds. Only pip's hash-checking mode names
    the bytes, so a bare requirement is identified when the command carries a
    ``--hash`` and unresolved when it does not.

    Returns:
        The identified inputs, and the sorted kinds that could not be identified.
    """
    root = Path(cwd)
    _, tokens = split_env_assignments(cmd)
    requirement_options = set(_REQUIREMENT_OPTIONS)
    if installer_class(cmd) == "pip":
        requirement_options.add(_PIP_CONSTRAINT_SHORT)
    pairs = option_operands(tokens)
    digests = _requirement_digests(pairs)
    identities: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for option, operand in pairs:
        if not operand:
            continue
        if option in requirement_options:
            identity = _file_identity(root / operand, root=root)
            if identity is None:
                unresolved.append("requirements_file")
            else:
                identities.append({"kind": "requirements_file", **identity})
        elif operand.startswith(_VCS_PREFIXES):
            identity, missing = _vcs_identity(operand)
            if identity is None:
                unresolved.append(missing)
            else:
                identities.append(identity)
        elif "://" in operand:
            unresolved.append("remote_artifact")
        elif _looks_local(operand):
            identity = _file_identity(root / operand, root=root)
            if identity is None:
                unresolved.append("local_file")
            else:
                identities.append({"kind": "local_file", **identity})
    for spec in _package_operands(tokens):
        if digests:
            identities.append({"kind": "pinned_package", "spec": spec, "digests": sorted(set(digests))})
        else:
            unresolved.append("mutable_package")
    return identities, sorted(set(unresolved))


def build_execution_row(
    *,
    seq: int,
    round_task_id: str,
    cmd_index: int,
    cmd: str,
    source: str,
    outcome: str,
    env: Mapping[str, str] | None,
    cwd: Path | str,
    fs_root: Path | str = "/",
) -> dict[str, Any]:
    """Build one durable ledger row for one attempted execution.

    Args:
        seq: Monotonic execution identity; a row is never matched by its text.
        round_task_id: The round the execution belongs to.
        cmd_index: Position within the round's resolved command list.
        cmd: The verbatim command, used only for its digest and sanitized form.
        source: ``"proposed"`` (this round's own payload) or ``"inherited"``.
        outcome: One of :data:`OUTCOMES`.
        env: The environment the command was given, classified by channel name.
        cwd: The directory the command ran in, for resolving its file inputs.
        fs_root: Filesystem root the ambient channel locations are probed under.

    Raises:
        ValueError: When ``outcome`` is outside the recorded vocabulary.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    identities, unresolved = setup_input_identity(cmd, cwd=cwd) if outcome == "applied" else ([], [])
    return {
        "seq": int(seq),
        "round_task_id": str(round_task_id or ""),
        "cmd_index": int(cmd_index),
        "cmd_sanitized": sanitize_command_text(cmd, clip=_CMD_SANITIZED_CHARS),
        "cmd_digest": command_digest(cmd),
        "source": str(source),
        "outcome": outcome,
        "round_disposition": "unreported",
        "present_at_final_launch": False,
        "replayed_at_final_launch": False,
        "credential_class": classify_credential_class(cmd),
        "credential_channels": detect_credential_channels(env, fs_root=fs_root),
        "installer": installer_class(cmd),
        "input_identity": identities,
        "unresolved_inputs": unresolved,
    }


def mark_round_disposition(
    ledger: list[dict[str, Any]],
    *,
    round_task_id: str,
    disposition: str,
    accepted: bool,
) -> list[dict[str, Any]]:
    """Record a round's outcome onto its rows, and presence at the graded launch.

    Presence is per occurrence and succession is a separate fact: an entry *is*
    an entry of the accepted round, while another occurrence of the same digest
    merely reproduces its effect. Keying both off the command string would let a
    failed occurrence be certified by a later successful one.

    Only one round terminates the lane, so an accepted round takes presence away
    from whichever earlier round held it: leaving it standing would let a stale
    row answer for a command the validated launch never ran.
    """
    rows = [dict(row) for row in ledger]
    for row in rows:
        own = str(row.get("round_task_id") or "") == str(round_task_id or "")
        if own:
            row["round_disposition"] = str(disposition)
        if own or accepted:
            row["present_at_final_launch"] = bool(own and accepted and str(row.get("outcome")) == "applied")
    if not accepted:
        return rows
    replayed = {str(row.get("cmd_digest") or "") for row in rows if row.get("present_at_final_launch")}
    for row in rows:
        row["replayed_at_final_launch"] = str(row.get("cmd_digest") or "") in replayed
    return rows
