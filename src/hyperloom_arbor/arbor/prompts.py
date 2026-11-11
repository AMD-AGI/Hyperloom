"""The agent's and the critic's instructions as versioned inputs: which template each turn reads, and the digests that
tie every verdict to the instructions, pack, config and Arbor that produced it.

**Which template.** Each role reads one file, the first of:
1. ``prompt_overrides`` in the campaign config: ``{"main": FILE, "critic": FILE}``, a path relative to
   ``prompt_profile`` when that is set, else to the config file's directory (or absolute);
2. ``prompt_profile`` in the config: a directory holding ``claude_loop_prompt.md`` and/or ``critic_prompt.md``;
3. ``<session>/prompt_profile/``, the same two names: a fork's own profile (``arbor checkpoint fork
   --prompt-profile``), so a branch never edits a file another branch reads;
4. the built-in ``arbor/policy/claude_loop_prompt.md`` and ``critic_prompt.md``.

A role the profile has no file for reads the next place's. A named file that is missing is an error, not a fallback:
a branch that meant to change its instructions must not run the built-in ones unseen.

**The digests** (``provenance``), on every lesson row: ``prompt_digest`` (both templates in use), ``pack_digest``
(every file of the pack's directory), ``config_digest`` (the campaign config), ``arbor_commit``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROLES = ("main", "critic")
BUILTIN = {"main": Path(__file__).parent / "policy" / "claude_loop_prompt.md",
           "critic": Path(__file__).parent / "policy" / "critic_prompt.md"}
#: A profile directory's file for each role: the built-in's own name, so a profile starts as a copy of the built-ins.
PROFILE_FILES = {role: path.name for role, path in BUILTIN.items()}
#: <session>/: a fork's own profile.
SESSION_PROFILE = "prompt_profile"


class PromptError(ValueError):
    """A prompt the config names that cannot be read."""


def template(config: dict | None, role: str, session_dir=None, base_dir=None) -> Path:
    """The file ``role`` (``main`` or ``critic``) reads, by the order in the module's docstring."""
    if role not in ROLES:
        raise PromptError(f"unknown prompt role {role!r}: one of {', '.join(ROLES)}")
    config = config or {}
    profile = config.get("prompt_profile")
    base = Path(base_dir) if base_dir else Path.cwd()
    profile_dir = (Path(profile) if Path(str(profile)).is_absolute() else base / str(profile)) if profile else None
    overrides = config.get("prompt_overrides") or {}
    if not isinstance(overrides, dict):
        raise PromptError("prompt_overrides must be an object: {\"main\": FILE, \"critic\": FILE}")
    named = overrides.get(role) or (overrides.get("agent") if role == "main" else None)
    if named:
        path = Path(str(named))
        if not path.is_absolute():
            path = (profile_dir or base) / path
        if not path.is_file():
            raise PromptError(f"prompt_overrides.{role} names {path}, which is not a file")
        return path
    if profile_dir is not None:
        if not profile_dir.is_dir():
            raise PromptError(f"prompt_profile {profile_dir} is not a directory")
        path = profile_dir / PROFILE_FILES[role]
        if path.is_file():
            return path
    if session_dir:
        path = Path(session_dir) / SESSION_PROFILE / PROFILE_FILES[role]
        if path.is_file():
            return path
    return BUILTIN[role]


def base_dir(campaign) -> Path | None:
    """The directory a campaign's relative prompt paths resolve against: its config file's."""
    where = getattr(campaign, "config_path", "") or ""
    return Path(where).parent if where else None


def read(config: dict | None, role: str, session_dir=None, base_dir=None) -> str:
    path = template(config, role, session_dir, base_dir)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def prompt_digest(config: dict | None, session_dir=None, base_dir=None) -> str:
    """The digest of the templates in use, both roles in order."""
    h = hashlib.sha256()
    for role in ROLES:
        path = template(config, role, session_dir, base_dir)
        h.update(role.encode() + b"\0" + (path.read_bytes() if path.is_file() else b"") + b"\0")
    return "sha256:" + h.hexdigest()


def tree_digest(directory) -> str | None:
    """The digest of every file under ``directory`` (names and bytes, in sorted order); None when it is not there."""
    root = Path(directory) if directory else None
    if root is None or not root.is_dir():
        return None
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(path.relative_to(root)).replace("\\", "/").encode() + b"\0" + path.read_bytes() + b"\0")
    return "sha256:" + h.hexdigest()


def config_digest(config: dict | None) -> str:
    """The digest of the campaign config as loaded (keys sorted; the loader's own ``_`` keys left out)."""
    doc = {k: v for k, v in (config or {}).items() if not str(k).startswith("_")}
    return _sha(json.dumps(doc, sort_keys=True, default=str).encode())


def arbor_commit(root=None) -> str | None:
    """The Arbor checkout's commit, with ``-dirty`` when it has changes; None outside a git checkout."""
    root = Path(root) if root else Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return head.stdout.strip() + ("-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else "")


def provenance(campaign, session_dir=None, arbor_root=None) -> dict:
    """The four digests a lesson row carries. Never raises: a part that cannot be read is None."""
    config = getattr(campaign, "config", None) or {}
    out: dict = {}
    try:
        out["prompt_digest"] = prompt_digest(config, session_dir, base_dir(campaign))
    except (OSError, PromptError):
        out["prompt_digest"] = None
    pack = getattr(campaign, "pack", None)
    try:
        out["pack_digest"] = tree_digest(getattr(pack, "pack_dir", None))
    except OSError:
        out["pack_digest"] = None
    out["config_digest"] = config_digest(config)
    out["arbor_commit"] = arbor_commit(arbor_root)
    return out
