"""Pack loading, config loading, validation.

Pack = contract (shared across campaigns of one type).
Config = instantiation (one model, one env, one campaign).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from arbor import component

VALID_OBJECTIVE_KINDS = {"maximize_perf", "find_witness"}
VALID_WORKLOAD_KINDS = {"serving", "kernel", "repro", "document"}
ALLOWED_PACK_DIRS = {"knowledge", "validator", "eval"}

# A pack's results/ is refused unless the user opts in (`arbor run --allow-pack-results`;
# operator, 2026-10-30: a colleague's Qwen3.8 packs keep their results there).
PACK_RESULTS_DIR = "results"
PACK_RESULTS_FLAG = "--allow-pack-results"
PACK_RESULTS_CONCERNS = (
    "a pack is the contract every campaign of its kind shares, so results kept in it reach every "
    "campaign that uses the pack, where an agent may read another model's, node's or image's "
    "numbers as evidence about its own; they also grow the pack and its repository")

# Skills every campaign agent gets unless its pack lists them in
# `opt-out-skills:` (names under <arbor_root>/.claude/skills/). Empty on
# 2026-10-06 (operator): campaigns run as before until a skill is added here.
DEFAULT_ON_SKILLS: tuple[str, ...] = ()


@dataclass
class Pack:
    name: str
    description: str
    objective: dict
    validator: dict
    workload: dict
    constraints: list[str]
    knowledge_dir: Path
    pack_dir: Path
    # Optional pace-threshold overrides (arbor.state.Policy field names).
    # Absent from every pack written before 2026-09-25, which is why it
    # defaults rather than being required.
    policy: dict | None = None
    # Skills the campaign agent may use, by name: <arbor_root>/.claude/skills/<name>/.
    # The agent gets DEFAULT_ON_SKILLS minus `opt-out-skills:`, plus `skills:`
    # (agent_skills). Listing "profiler" makes every profile go through that
    # skill (render.py states the precedence); a pack without it profiles its
    # own way. With nothing declared and no default-on skills, the agent runs
    # as before 2026-10-06.
    skills: list[str] | None = None
    opt_out_skills: list[str] | None = None
    # Where skills take part (arbor/lifecycle.py): the skill families the pack loads, as
    # <arbor_root>/plugins/<family>/; each chapter's run and use lists; each event's skills.
    # None when the pack does not declare them, as every pack before M10.
    plugins: list[str] | None = None
    lifecycle: dict | None = None
    triggers: dict | None = None
    # pack.yaml has a key YAML read as true: an `on:` meant as `triggers:`.
    on_key: bool = False
    # What changes under a host (component mode): its `lifecycle:` replaces the chapters it
    # names (arbor/lifecycle.py declared).
    component: dict | None = None


@dataclass
class Campaign:
    pack: Pack
    config: dict
    identity: str
    goal: str
    space: dict
    session_root: Path
    # "standalone" (Arbor owns the node, its container and its clock) or "component"
    # (a host owns them: arbor/component.py, docs/component-mode.md). Read once here;
    # the driver records it in session.json and refuses a config that changes it.
    mode: str = "standalone"


def load_pack(pack_dir: str | Path) -> Pack:
    pack_dir = Path(pack_dir)
    manifest = yaml.safe_load((pack_dir / "pack.yaml").read_text())
    return Pack(
        name=manifest["name"],
        description=manifest.get("description", ""),
        objective=manifest["objective"],
        validator=manifest["validator"],
        workload=manifest["workload"],
        constraints=manifest.get("constraints", []),
        knowledge_dir=pack_dir / "knowledge",
        pack_dir=pack_dir,
        policy=manifest.get("policy") or None,
        skills=manifest.get("skills"),
        opt_out_skills=manifest.get("opt-out-skills"),
        plugins=manifest.get("plugins"),
        lifecycle=manifest.get("lifecycle"),
        triggers=manifest.get("triggers"),
        on_key=True in manifest,
        component=manifest.get("component"),
    )


def agent_skills(pack: Pack, default_on: tuple[str, ...] | None = None) -> list[str] | None:
    """The skills a campaign agent may invoke: default-on minus opt-outs, plus opt-ins.

    None only when the pack declares neither field and nothing is default-on:
    the agent's SDK options then stay as they were before packs listed skills.
    Otherwise a list, possibly empty, which is the whole set: a pack that opts
    out of a skill must not get it back through the SDK's own default.
    """
    if default_on is None:
        default_on = DEFAULT_ON_SKILLS
    opt_in = getattr(pack, "skills", None)
    opt_out = getattr(pack, "opt_out_skills", None)
    if opt_in is None and opt_out is None and not default_on:
        return None
    opted_out = set(opt_out or [])
    names: list[str] = []
    for name in [*(s for s in default_on if s not in opted_out), *(opt_in or [])]:
        if name not in names:
            names.append(name)
    return names


def load_campaign(config_path: str | Path, arbor_root: str | Path) -> Campaign:
    config_path = Path(config_path)
    arbor_root = Path(arbor_root)
    config = json.loads(config_path.read_text())
    pack = load_pack(arbor_root / "packs" / config["pack"])
    return Campaign(
        pack=pack,
        config=config,
        identity=config.get("identity", ""),
        goal=config.get("goal", ""),
        space=config.get("space", {}),
        session_root=Path(os.path.expandvars(os.path.expanduser(config.get("session_root", "")))),
        mode=component.mode_of(config),
    )


def _skill_names_ok(value) -> bool:
    return isinstance(value, list) and all(
        isinstance(s, str) and s and s.strip() == s for s in value)


def validate_pack(pack: Pack, default_on: tuple[str, ...] | None = None,
                  allow_results: bool = False) -> list[str]:
    """The pack's errors. ``allow_results``: a results/ directory is accepted (the user
    passed --allow-pack-results); without it, it is refused with why and how to opt in."""
    if default_on is None:
        default_on = DEFAULT_ON_SKILLS
    errors = []

    if not pack.knowledge_dir.is_dir():
        errors.append("knowledge/ directory missing")

    validator_dir = pack.pack_dir / "validator"
    if not validator_dir.is_dir():
        errors.append("validator/ directory missing")

    bench = pack.validator.get("bench", "")
    if bench and not (pack.pack_dir / bench).exists():
        errors.append(f"bench script not found: {bench}")

    correctness = pack.validator.get("correctness", "")
    if correctness and not (pack.pack_dir / correctness).exists():
        errors.append(f"correctness script not found: {correctness}")

    kind = pack.objective.get("kind", "")
    if kind not in VALID_OBJECTIVE_KINDS:
        errors.append(f"unknown objective.kind: {kind!r}")

    if not pack.objective.get("metric"):
        errors.append("objective.metric missing or empty")

    if not pack.objective.get("direction"):
        errors.append("objective.direction missing or empty")

    wk = pack.workload.get("kind", "")
    if wk not in VALID_WORKLOAD_KINDS:
        errors.append(f"unknown workload.kind: {wk!r}")

    for entry in pack.pack_dir.iterdir():
        if entry.name in ("pack.yaml",):
            continue
        if entry.is_dir() and entry.name == PACK_RESULTS_DIR:
            if not allow_results:
                errors.append(
                    f"{PACK_RESULTS_DIR}/ inside the pack is refused by default: "
                    f"{PACK_RESULTS_CONCERNS}. Keep results in the session or a results repository; "
                    f"to keep this pack's {PACK_RESULTS_DIR}/ anyway, pass {PACK_RESULTS_FLAG} to "
                    f"`arbor run`")
        elif entry.is_dir() and entry.name not in ALLOWED_PACK_DIRS:
            errors.append(f"unrecognized directory: {entry.name}/")

    skill_errors = []
    if pack.skills is not None and not _skill_names_ok(pack.skills):
        skill_errors.append(f"skills must be a list of skill names: {pack.skills!r}")
    if pack.opt_out_skills is not None and not _skill_names_ok(pack.opt_out_skills):
        skill_errors.append(f"opt-out-skills must be a list of skill names: {pack.opt_out_skills!r}")
    if not skill_errors:
        # Opting out of a skill that is not default-on is a typo or a stale pack.
        for name in pack.opt_out_skills or []:
            if name not in default_on:
                skill_errors.append(f"opt-out-skills: {name} is not default-on "
                                    f"(default-on: {list(default_on)})")
            if name in (pack.skills or []):
                skill_errors.append(f"skill {name} is in both skills and opt-out-skills")
        # Packs live in <arbor_root>/packs/<name> (load_campaign).
        skills_dir = pack.pack_dir.parent.parent / ".claude" / "skills"
        for name in agent_skills(pack, default_on) or []:
            if not (skills_dir / name / "SKILL.md").is_file():
                skill_errors.append(f"skill not found: {name} (no .claude/skills/{name}/SKILL.md)")
    errors.extend(skill_errors)
    errors.extend(lifecycle_errors(pack))

    return errors


def lifecycle_errors(pack: Pack) -> list[str]:
    """What is wrong with where the pack's skills take part (arbor/lifecycle.py): the sections' shape, families the
    pack loads that are not there, and skills it names that no flat skill or listed family holds, or that do not
    declare the way the pack has them take part."""
    from arbor import lifecycle, skillcheck

    errors = []
    if pack.on_key:
        errors.append("pack.yaml has an `on:` key, which YAML reads as true: list event skills under `triggers:`")
    if pack.lifecycle is not None and (pack.skills is not None or pack.opt_out_skills is not None):
        errors.append("a pack gives its agent's skills with skills: (every chapter) or lifecycle: (per chapter), "
                      "not both")
    if pack.plugins is not None and not _skill_names_ok(pack.plugins):
        return [*errors, f"plugins must be a list of family names: {pack.plugins!r}"]
    shape = lifecycle.errors(pack.lifecycle, pack.triggers) + _component_errors(pack)
    if shape:
        return [*errors, *shape]
    root = pack.pack_dir.parent.parent  # packs live in <arbor_root>/packs/<name>
    families = skillcheck.plugins(root)
    errors += [f"plugin not found: {name} (no plugins/{name}/{skillcheck.PLUGIN_MANIFEST.as_posix()})"
               for name in pack.plugins or [] if name not in families]
    develop = skillcheck.develop_families(root)
    errors += [f"plugin {name} is for developing Arbor, not for a campaign (its skills declare the develop mode)"
               for name in pack.plugins or [] if name in develop]
    found = skillcheck.find(root)
    plans = [lifecycle.from_pack(pack.lifecycle, pack.triggers, None)]
    hosted = (pack.component or {}).get("lifecycle")
    if hosted:
        plans.append(lifecycle.from_pack({**pack.lifecycle, **hosted}, pack.triggers, None))
    for name in dict.fromkeys(name for plan in plans for name in plan.named()):
        family = name.split(":", 1)[0] if ":" in name else None
        if family is not None and family not in (pack.plugins or []):
            errors.append(f"skill {name}: its family {family} is not in the pack's plugins")
        elif name not in found:
            errors.append(f"skill not found: {name}")
    return errors + list(dict.fromkeys(error for plan in plans for error in _mode_errors(plan, found)))


def _component_errors(pack: Pack) -> list[str]:
    """What is wrong with the pack's ``component:`` section, which Arbor reads only under a host."""
    from arbor import lifecycle

    section = pack.component
    if section is None:
        return []
    if not isinstance(section, dict):
        return [f"component must be a mapping: {section!r}"]
    errors = [f"component.{key} is not built yet; Arbor reads only "
              f"{', '.join(lifecycle.COMPONENT_KEYS)}" for key in section if key not in lifecycle.COMPONENT_KEYS]
    if "lifecycle" in section:
        if pack.lifecycle is None:
            errors.append("component.lifecycle replaces chapters of lifecycle:, which the pack does not declare")
        else:
            errors += [f"component.{error}" for error in lifecycle.errors(section["lifecycle"], None)]
    return errors


def _order_errors(chapter: str, names: list[str], found: dict) -> list[str]:
    """Run skills a chapter lists against the order they declare: one listed before a skill it must come ``after``,
    or after one it must come ``before``."""
    errors = []
    for at, name in enumerate(names):
        meta = found[name].arbor if name in found else {}
        errors += [f"lifecycle: {chapter}.run lists {name} before {other}, which it must come after"
                   for other in meta.get("after") or [] if other in names[at + 1:]]
        errors += [f"lifecycle: {chapter}.run lists {name} after {other}, which it must come before"
                   for other in meta.get("before") or [] if other in names[:at]]
    return errors


def _mode_errors(declared, found: dict) -> list[str]:
    """Skills the pack runs in a chapter or triggers on an event that do not declare that they can."""
    errors = []
    for chapter, names in declared.run.items():
        for skill in (found[name] for name in names if name in found):
            if "run" not in skill.modes:
                errors.append(f"lifecycle: {chapter}.run lists {skill.name}, which does not declare the run mode")
            elif skill.arbor.get("chapter") not in (None, chapter):
                errors.append(f"lifecycle: {chapter}.run lists {skill.name}, whose chapter is "
                              f"{skill.arbor['chapter']}")
        errors += _order_errors(chapter, list(names), found)
    for event, names in declared.triggers.items():
        for skill in (found[name] for name in names if name in found):
            if "trigger" not in skill.modes:
                errors.append(f"triggers: {event} lists {skill.name}, which does not declare the trigger mode")
            elif skill.arbor.get("triggers") and event not in skill.arbor["triggers"]:
                errors.append(f"triggers: {event} lists {skill.name}, whose events are "
                              f"{', '.join(skill.arbor['triggers'])}")
    for chapter, names in declared.use.items():
        errors += [f"lifecycle: {chapter}.use lists {name}, which does not declare the use mode"
                   for name in names if name in found and "use" not in found[name].modes]
    return errors


def validate_campaign(campaign: Campaign, allow_results: bool = False) -> list[str]:
    errors = validate_pack(campaign.pack, allow_results=allow_results)

    if not campaign.config.get("pack"):
        errors.append("config missing 'pack' field")

    if not campaign.config.get("docker_container"):
        errors.append("config missing 'docker_container'")

    if not campaign.config.get("workspace_root"):
        errors.append("config missing 'workspace_root'")

    if not campaign.config.get("bench_script"):
        errors.append("config missing 'bench_script'")

    if not campaign.config.get("correctness_script"):
        errors.append("config missing 'correctness_script'")

    # The mode and its component block (arbor/component.py). The driver refuses to
    # start on these, and on a component config's other refusals, with exit 2.
    errors.extend(component.config_errors(campaign.config))

    return errors
