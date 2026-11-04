"""Arbor's skills and skill families as Arbor finds them, and what is wrong with them.

- **A flat skill** is ``<root>/.claude/skills/<name>/SKILL.md``, named ``<name>``: today's skills.
- **A family** is a Claude Code plugin, ``<root>/plugins/<family>/``. It has ``.claude-plugin/plugin.json`` and
  ``skills/<name>/SKILL.md``, and its skills are named ``<family>:<name>``, as Claude Code names a plugin's skills.
  Contributed skills come as families.

Claude Code finds a skill one directory deep only, and reads its front matter. A missing or unparseable front matter
loads the skill with no fields, so Arbor refuses it. Arbor's own key is ``metadata.arbor``: where the skill can take
part (``modes``, ``chapter``, ``triggers``), what it ``produces``, ``after`` and ``before`` for its order in a chapter,
``reuse_key`` and ``gpu`` (``arbor/lifecycle.py``).

A skill for developing Arbor declares the ``develop`` mode alone, and none of the keys that place a skill in a
campaign. It lives in a family that holds only such skills (``develop_families``), which a campaign never loads.

``python3 -m arbor.skillcheck [ROOT]`` prints every error and warning, and exits 1 on an error. The tests run it over
the repository.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from arbor import lifecycle

SKILLS_DIR = Path(".claude") / "skills"
PLUGINS_DIR = Path("plugins")
PLUGIN_MANIFEST = Path(".claude-plugin") / "plugin.json"
SKILL_FILE = "SKILL.md"
#: Claude Code truncates a skill's description and ``when_to_use`` together past this, in the listing every agent reads.
DESCRIPTION_CHARS = 1536
#: Claude Code's guidance for a ``SKILL.md``; what is longer goes to files it links.
SKILL_LINES = 500
#: The keys of ``metadata.arbor`` that hold lists of names or paths.
LIST_KEYS = ("modes", "triggers", "after", "before", "produces", "reuse_key")
ARBOR_KEYS = (*LIST_KEYS, "chapter", "gpu", "blocking")
FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*(\n|\Z)", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    """A skill as found: its name (``family:name`` in a family), its directory, its front matter and length."""

    name: str
    path: Path
    front: dict
    lines: int
    family: str | None = None

    @property
    def arbor(self) -> dict:
        """Its ``metadata.arbor``, ``{}`` when it has none."""

        metadata = self.front.get("metadata")
        found = metadata.get("arbor") if isinstance(metadata, dict) else None
        return found if isinstance(found, dict) else {}

    @property
    def modes(self) -> tuple[str, ...]:
        """The ways it takes part; ``use`` alone when it declares none, as every skill is used today."""

        modes = self.arbor.get("modes")
        return tuple(modes) if isinstance(modes, list) and modes else ("use",)

    @property
    def develop(self) -> bool:
        """Whether it is for developing Arbor, never a campaign's."""

        return lifecycle.DEVELOP in self.modes


def front_matter(text: str) -> tuple[dict | None, str]:
    """A ``SKILL.md``'s front matter, and why it could not be read (``""`` when it could)."""

    found = FRONT.match(text)
    if found is None:
        return None, "no front matter (a --- block at the top)"
    try:
        front = yaml.safe_load(found.group(1))
    except yaml.YAMLError as exc:
        return None, f"front matter is not YAML: {str(exc).splitlines()[0]}"
    if not isinstance(front, dict):
        return None, "front matter is not a mapping"
    return front, ""


def description(path: Path) -> str:
    """A skill's description on one line, read as YAML (a folded ``>-`` included); ``""`` without one."""

    try:
        front, _ = front_matter((path / SKILL_FILE).read_text(encoding="utf-8"))
    except OSError:
        return ""
    text = (front or {}).get("description")
    return " ".join(text.split()) if isinstance(text, str) else ""


def _skill(path: Path, name: str, family: str | None = None) -> Skill:
    text = (path / SKILL_FILE).read_text(encoding="utf-8")
    front, _ = front_matter(text)
    return Skill(name, path, front or {}, len(text.splitlines()), family)


def plugins(root: Path) -> dict[str, Path]:
    """The families under ``root``, by name: each directory of ``plugins/`` with a plugin manifest."""

    base = root / PLUGINS_DIR
    if not base.is_dir():
        return {}
    return {path.name: path for path in sorted(base.iterdir()) if (path / PLUGIN_MANIFEST).is_file()}


def find(root: Path) -> dict[str, Skill]:
    """Every skill under ``root``, by name: the flat ones and each family's."""

    found: dict[str, Skill] = {}
    flat = root / SKILLS_DIR
    if flat.is_dir():
        for path in sorted(flat.iterdir()):
            if (path / SKILL_FILE).is_file():
                found[path.name] = _skill(path, path.name)
    for family, path in plugins(root).items():
        skills = path / "skills"
        for skill in sorted(skills.iterdir()) if skills.is_dir() else ():
            if (skill / SKILL_FILE).is_file():
                found[f"{family}:{skill.name}"] = _skill(skill, f"{family}:{skill.name}", family)
    return found


def skill_problems(skill: Skill) -> tuple[list[str], list[str]]:
    """A skill's errors and warnings."""

    where = skill.name
    front, why = front_matter((skill.path / SKILL_FILE).read_text(encoding="utf-8"))
    if front is None:
        return [f"{where}: {why}"], []
    errors, warnings = [], []
    if "name" in front and front["name"] != skill.path.name:
        errors.append(f"{where}: name {front['name']!r} is not its directory's, {skill.path.name!r}")
    text = front.get("description")
    if not isinstance(text, str) or not text.strip():
        errors.append(f"{where}: no description, which is what an agent chooses it by")
    elif len(text) + len(str(front.get("when_to_use") or "")) > DESCRIPTION_CHARS:
        warnings.append(f"{where}: description and when_to_use pass {DESCRIPTION_CHARS} characters; the rest is cut")
    if skill.lines > SKILL_LINES:
        warnings.append(f"{where}: SKILL.md has {skill.lines} lines, over {SKILL_LINES}; move detail to linked files")
    errors += _arbor_errors(skill, front)
    if "run" in skill.modes and front.get("disable-model-invocation") is not True:
        warnings.append(f"{where}: a run skill without disable-model-invocation: true, so agents may pick it")
    return errors, warnings


def _arbor_errors(skill: Skill, front: dict) -> list[str]:
    metadata = front.get("metadata")
    if metadata is None or not isinstance(metadata, dict) or "arbor" not in metadata:
        return []
    arbor = metadata["arbor"]
    where = f"{skill.name}: metadata.arbor"
    if not isinstance(arbor, dict):
        return [f"{where} must be a mapping"]
    errors = [f"{where}: unknown key {key!r}" for key in arbor if key not in ARBOR_KEYS]
    for key in LIST_KEYS:
        value = arbor.get(key)
        if value is not None and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            errors.append(f"{where}.{key} must be a list of strings: {value!r}")
    errors += [
        f"{where}.modes: unknown mode {mode!r} (modes: {', '.join(lifecycle.SKILL_MODES)})"
        for mode in arbor.get("modes") or []
        if mode not in lifecycle.SKILL_MODES
    ]
    if skill.develop:
        if set(skill.modes) != {lifecycle.DEVELOP}:
            errors.append(f"{where}.modes: develop is a skill's only mode; a campaign's modes go in another skill")
        errors += [f"{where}.{key}: a develop skill takes no part in a campaign" for key in ARBOR_KEYS
                   if key != "modes" and key in arbor]
        if skill.family is None:
            errors.append(f"{skill.name}: a develop skill belongs in a family under plugins/, which a campaign never "
                          f"loads; a flat one loads with the project's skills")
    chapter = arbor.get("chapter")
    if chapter is not None and chapter not in lifecycle.CHAPTER_NAMES:
        errors.append(f"{where}.chapter: unknown chapter {chapter!r}")
    errors += [
        f"{where}.triggers: unknown event {event!r}"
        for event in arbor.get("triggers") or []
        if event not in lifecycle.EVENT_NAMES
    ]
    errors += [f"{where}.{key} must be true or false" for key in ("gpu", "blocking")
               if key in arbor and not isinstance(arbor[key], bool)]
    return errors


def plugin_problems(path: Path) -> list[str]:
    """A family's errors: its manifest, its name, and that it holds a skill."""

    where = f"plugins/{path.name}"
    try:
        manifest = json.loads((path / PLUGIN_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{where}: {PLUGIN_MANIFEST.as_posix()} cannot be read: {exc}"]
    errors = []
    if not isinstance(manifest, dict) or manifest.get("name") != path.name:
        errors.append(f"{where}: plugin.json's name must be {path.name!r}, its directory's")
    elif not str(manifest.get("description") or "").strip():
        errors.append(f"{where}: plugin.json has no description")
    skills = path / "skills"
    if not skills.is_dir() or not any((skill / SKILL_FILE).is_file() for skill in skills.iterdir()):
        errors.append(f"{where}: no skills/<name>/SKILL.md")
        return errors
    kinds = {_skill(skill, skill.name, path.name).develop
             for skill in skills.iterdir() if (skill / SKILL_FILE).is_file()}
    if len(kinds) > 1:
        errors.append(f"{where}: mixes develop skills with a campaign's; a family is for one or the other")
    return errors


def develop_families(root: Path) -> set[str]:
    """The families for developing Arbor: those holding a ``develop`` skill. A campaign never loads one."""

    return {skill.family for skill in find(root).values() if skill.family is not None and skill.develop}


def check(root: Path) -> tuple[list[str], list[str]]:
    """Every error and warning of the skills and families under ``root``."""

    errors: list[str] = []
    warnings: list[str] = []
    for path in plugins(root).values():
        errors += plugin_problems(path)
    for skill in find(root).values():
        found, warned = skill_problems(skill)
        errors += found
        warnings += warned
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]) if args else Path(__file__).resolve().parent.parent
    errors, warnings = check(root)
    for line in warnings:
        print(f"warning: {line}")
    for line in errors:
        print(f"error: {line}")
    print(f"{len(find(root))} skills, {len(plugins(root))} families: {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
