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
# operator, 2026-10-30: Ravi's Qwen3.8 packs keep their results there, ArborDevelopment #25).
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
