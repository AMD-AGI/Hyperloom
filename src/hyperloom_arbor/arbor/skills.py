"""How an agent gets a family's skills (``arbor/skillcheck.py`` says what a family is).

Claude Code finds a project's skills one folder deep: it finds the flat ones in ``.claude/skills/`` itself, and none of
a family's. Arbor hands each family a campaign loads to its agent as a Claude Code plugin built for the session,
``<session>/skills/plugins/<family>/``:

- **``.claude-plugin/plugin.json``,** written from the family's ``family.json``;
- **``skills/``,** the family's folder copied whole, so its skills keep their names (``family:name``) and their links
  (``../shared/<file>``).

The SDK loads them through its ``plugins`` option. A plugin follows the tree: a family that changed since it was built
is built again, at the next turn that asks for it.

For people developing Arbor, ``python3 -m arbor.skills plugin <family> [--out DIR]`` builds the same plugin from a
family (``arbor-dev``, say) and prints the ``claude --plugin-dir`` line that loads it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

from arbor import skillcheck

PLUGIN_MANIFEST = Path(".claude-plugin") / "plugin.json"
#: Where a session's plugins are built, under the session.
SESSION_PLUGINS = Path("skills") / "plugins"
#: What a built plugin records of its source, so an unchanged family is not built again.
STAMP = ".arbor-source.json"
#: The keys of ``family.json`` a plugin's manifest carries.
MANIFEST_KEYS = ("name", "version", "description", "author")
#: Where ``plugin`` builds a family for a person, unless told otherwise.
DEV_OUT = Path.home() / ".cache" / "arbor" / "plugins"
_SKIP = ("__pycache__", ".pytest_cache")


def digest(folder: Path) -> str:
    """A family folder's content, as one hash over its files' paths and bytes."""

    found = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and not set(p.parts) & set(_SKIP)):
        found.update(path.relative_to(folder).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return found.hexdigest()


def _built(out: Path, source: str) -> bool:
    try:
        return json.loads((out / STAMP).read_text(encoding="utf-8")).get("digest") == source
    except (OSError, ValueError, AttributeError):
        return False


def build(family: Path, out: Path, overlay: dict[str, Path] | None = None) -> Path:
    """Build the family at ``family`` as a Claude Code plugin at ``out``, unless ``out`` is already that build; ``out``.
    ``overlay`` puts a folder in place of a skill of the family, by the skill's name: a session's merged skill
    (``arbor/conflicts.py``).

    Several processes may build one session's plugin at once (a trigger skill's, the main agent's): each builds in a
    folder of its own and moves it into place, and one that finds the same build there already keeps it."""

    overlay = overlay or {}
    source = digest(family)
    if overlay:
        source = hashlib.sha256("\0".join([source, *(f"{name}={digest(path)}" for name, path
                                                     in sorted(overlay.items()))]).encode()).hexdigest()
    if _built(out, source):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.with_name(f".{out.name}.{os.getpid()}")
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(family, work / "skills", ignore=shutil.ignore_patterns(skillcheck.FAMILY_FILE, *_SKIP))
    for name, path in overlay.items():
        shutil.rmtree(work / "skills" / name, ignore_errors=True)
        shutil.copytree(path, work / "skills" / name, ignore=shutil.ignore_patterns(*_SKIP))
    found = skillcheck.manifest(family)
    plugin = {key: found[key] for key in MANIFEST_KEYS if key in found}
    plugin["name"] = family.name
    (work / PLUGIN_MANIFEST).parent.mkdir(parents=True)
    (work / PLUGIN_MANIFEST).write_text(json.dumps(plugin, indent=2) + "\n", encoding="utf-8")
    (work / STAMP).write_text(json.dumps({"source": str(family), "digest": source}) + "\n", encoding="utf-8")
    shutil.rmtree(out, ignore_errors=True)
    try:
        os.replace(work, out)
    except OSError:
        shutil.rmtree(work, ignore_errors=True)
        if not _built(out, source):
            raise
    return out


def session_plugins(root, session_dir, families: Iterable[str],
                    overlays: dict[str, dict[str, Path]] | None = None) -> list[str]:
    """The plugins of ``families``, built for the session from the families under ``root``, each with its
    ``overlays`` (``build``); a name with no family is left out (the pack's checks report it)."""

    found = skillcheck.families(Path(root))
    base = Path(session_dir) / SESSION_PLUGINS
    return [str(build(found[name], base / name, (overlays or {}).get(name)))
            for name in dict.fromkeys(families) if name in found]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m arbor.skills",
                                     description="Build a skill family as a Claude Code plugin.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    plugin = sub.add_parser("plugin", help="build a family (arbor-dev, say) for `claude --plugin-dir`")
    plugin.add_argument("family")
    plugin.add_argument("--out", type=Path, default=DEV_OUT, help=f"where to build it (default {DEV_OUT})")
    plugin.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="the Arbor checkout (default: this one)")
    args = parser.parse_args(argv)
    found = skillcheck.families(args.root)
    if args.family not in found:
        print(f"no family {args.family!r} under {args.root / skillcheck.SKILLS_DIR} "
              f"(families: {', '.join(found) or 'none'})", file=sys.stderr)
        return 1
    out = build(found[args.family], args.out / args.family)
    print(f"built {args.family} at {out}; load it with:\n  claude --plugin-dir {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
