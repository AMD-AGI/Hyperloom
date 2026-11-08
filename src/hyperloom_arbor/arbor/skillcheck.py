"""Arbor's skills and skill families as Arbor finds them, and what is wrong with them.

- **A flat skill** is ``<root>/.claude/skills/<name>/SKILL.md``, named ``<name>``.
- **A family** is ``<root>/.claude/skills/<family>/``, marked by its ``family.json`` (``name``, ``description``,
  ``version``, ``author``). Its skills are ``<family>/<name>/SKILL.md``, named ``<family>:<name>``; a folder of it
  without a ``SKILL.md`` (``shared/``) holds files its skills share, which they link as ``../shared/<file>``.
  Contributed skills come as families.
- **A name resolves** to its ``SKILL.md`` by its colon (``resolve``): ``dossier:discover`` is
  ``.claude/skills/dossier/discover/SKILL.md``, ``profiler`` is ``.claude/skills/profiler/SKILL.md``.

Claude Code finds a skill one directory deep only, so it finds the flat skills itself and none of a family's: Arbor
hands a family to the agent as a Claude Code plugin it builds for the session (``arbor/skills.py``). Claude Code reads
a skill's front matter; a missing or unparseable one loads the skill with no fields, so Arbor refuses it.

Arbor's own key is ``metadata.arbor`` (``arbor/lifecycle.py``):

- **``type``:** ``runtime`` (the default) or ``development``;
- **how a runtime skill takes part:** ``modes`` (``use`` alone when it names none), and where: ``chapter`` for ``run``,
  ``chapters`` for ``use`` (every chapter when it names none), ``triggers`` and ``blocking`` for ``trigger``;
- **its order in a chapter:** ``after`` and ``before``;
- **what it writes and needs:** ``produces``, ``reuse_key`` and ``gpu``.

A skill for developing Arbor declares ``type: development`` and nothing that places a skill in a campaign. It lives in a
family that holds only such skills (``development_families``), which a campaign never loads.

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
#: What marks a folder of ``SKILLS_DIR`` as a family, and what it says about it.
FAMILY_FILE = "family.json"
SKILL_FILE = "SKILL.md"
#: Claude Code truncates a skill's description and ``when_to_use`` together past this, in the listing every agent reads.
DESCRIPTION_CHARS = 1536
#: Claude Code's guidance for a ``SKILL.md``; what is longer goes to files it links.
SKILL_LINES = 500
#: The keys of ``metadata.arbor`` that hold lists of names or paths.
LIST_KEYS = ("modes", "chapters", "triggers", "after", "before", "produces", "reuse_key", "overlaps")
ARBOR_KEYS = ("type", *LIST_KEYS, "chapter", "gpu", "blocking")
#: Each mode's placement keys: a skill naming one without its mode is refused.
PLACEMENT = {"chapter": "run", "chapters": "use", "triggers": "trigger", "blocking": "trigger"}
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
    def development(self) -> bool:
        """Whether it is for developing Arbor, never a campaign's (``type: development``)."""

        return self.arbor.get("type") == lifecycle.DEVELOPMENT

    @property
    def modes(self) -> tuple[str, ...]:
        """The ways it takes part in a campaign: ``use`` alone when it declares none, as every skill was used before
        chapters; none for a development skill."""

        if self.development:
            return ()
        modes = self.arbor.get("modes")
        return tuple(modes) if isinstance(modes, list) and modes else ("use",)


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


def families(root: Path) -> dict[str, Path]:
    """The families under ``root``, by name: each folder of ``.claude/skills/`` with a ``family.json``."""

    base = root / SKILLS_DIR
    if not base.is_dir():
        return {}
    return {path.name: path for path in sorted(base.iterdir()) if (path / FAMILY_FILE).is_file()}


def family_skills(path: Path) -> list[Path]:
    """A family's skill folders: each folder of it with a ``SKILL.md``."""

    return [skill for skill in sorted(path.iterdir()) if (skill / SKILL_FILE).is_file()]


def resolve(root: Path, name: str) -> Path:
    """The ``SKILL.md`` a skill's name stands for, whether or not it exists: a family's skill for ``family:name``, a
    flat skill otherwise."""

    family, _, skill = name.partition(":")
    folder = Path(family, skill) if skill else Path(family)
    return Path(root) / SKILLS_DIR / folder / SKILL_FILE


def find(root: Path) -> dict[str, Skill]:
    """Every skill under ``root``, by name: the flat ones and each family's."""

    found: dict[str, Skill] = {}
    flat = root / SKILLS_DIR
    if flat.is_dir():
        for path in sorted(flat.iterdir()):
            if (path / SKILL_FILE).is_file():
                found[path.name] = _skill(path, path.name)
    for family, path in families(root).items():
        for skill in family_skills(path):
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
    kind = arbor.get("type", lifecycle.RUNTIME)
    if kind not in lifecycle.TYPES:
        errors.append(f"{where}.type: unknown type {kind!r} (types: {', '.join(lifecycle.TYPES)})")
    errors += [
        f"{where}.modes: unknown mode {mode!r} (modes: {', '.join(lifecycle.MODES)})"
        + ("; a skill for developing Arbor declares type: development" if mode == "develop" else "")
        for mode in arbor.get("modes") or []
        if mode not in lifecycle.MODES
    ]
    if skill.development:
        errors += [f"{where}.{key}: a development skill takes no part in a campaign" for key in ARBOR_KEYS
                   if key != "type" and key in arbor]
        if skill.family is None:
            errors.append(f"{skill.name}: a development skill belongs in a family, which a campaign never "
                          f"loads; a flat one loads with the project's skills")
        return errors
    errors += [f"{where}.{key} places a {mode} skill, and it does not declare the {mode} mode"
               for key, mode in PLACEMENT.items() if key in arbor and mode not in skill.modes]
    chapter = arbor.get("chapter")
    if chapter is not None and chapter not in lifecycle.CHAPTER_NAMES:
        errors.append(f"{where}.chapter: unknown chapter {chapter!r}")
    elif chapter is None and "run" in skill.modes:
        errors.append(f"{where}.chapter: a run skill names the chapter it runs in")
    errors += [f"{where}.chapters: unknown chapter {name!r}" for name in arbor.get("chapters") or []
               if isinstance(name, str) and name not in lifecycle.CHAPTER_NAMES]
    errors += [
        f"{where}.triggers: unknown event {event!r}"
        for event in arbor.get("triggers") or []
        if event not in lifecycle.EVENT_NAMES
    ]
    errors += [f"{where}.{key} must be true or false" for key in ("gpu", "blocking")
               if key in arbor and not isinstance(arbor[key], bool)]
    return errors


def manifest(path: Path) -> dict:
    """A family's ``family.json``; ``{}`` when it cannot be read."""

    try:
        found = json.loads((path / FAMILY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def family_problems(path: Path) -> list[str]:
    """A family's errors: its ``family.json``, its name, that it holds a skill and is not a skill itself."""

    where = (SKILLS_DIR / path.name).as_posix()
    try:
        found = json.loads((path / FAMILY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{where}: {FAMILY_FILE} cannot be read: {exc}"]
    errors = []
    if not isinstance(found, dict) or found.get("name") != path.name:
        errors.append(f"{where}: {FAMILY_FILE}'s name must be {path.name!r}, its folder's")
    elif not str(found.get("description") or "").strip():
        errors.append(f"{where}: {FAMILY_FILE} has no description")
    if (path / SKILL_FILE).is_file():
        errors.append(f"{where}: a family is not a skill; its {SKILL_FILE} belongs in a folder of its own")
    skills = family_skills(path)
    if not skills:
        errors.append(f"{where}: no <name>/{SKILL_FILE}")
        return errors
    if len({_skill(skill, skill.name, path.name).development for skill in skills}) > 1:
        errors.append(f"{where}: mixes development skills with runtime ones; a family is one type or the other")
    return errors


def development_families(root: Path) -> set[str]:
    """The families for developing Arbor: those holding a development skill. A campaign never loads one."""

    return {skill.family for skill in find(root).values() if skill.family is not None and skill.development}


def check(root: Path) -> tuple[list[str], list[str]]:
    """Every error and warning of the skills and families under ``root``."""

    errors: list[str] = []
    warnings: list[str] = []
    for path in families(root).values():
        errors += family_problems(path)
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
    print(f"{len(find(root))} skills, {len(families(root))} families: {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
