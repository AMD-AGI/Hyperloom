# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Round configuration, loaded from YAML and validated before any step runs.

Every location is explicit: the driver has no default that points at a particular machine.
``round.example.yaml`` next to this package shows the full shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

AGENT_STEPS = ("findings", "implement", "diagnose", "report")
ARM_TREES = ("base", "candidate")
ARM_ROLES = ("claude", "specialist", "geak")
DEFAULT_PYTEST_ARGS = ("-q", "-m", "not critic_agent_e2e and not targeted_build_e2e", "-p", "no:cacheprovider")


class ConfigError(ValueError):
    """The round configuration lacks a value, or holds one the driver cannot use."""


@dataclass(frozen=True)
class Target:
    """The repository the round changes: a candidate branch is cut from ``base_ref``."""

    repo: Path
    base_ref: str
    branch: str


@dataclass(frozen=True)
class Data:
    """Where session data comes from and which sessions count as recent."""

    bundles_dir: Path
    recent_since: str
    last_days: int = 15
    models_dir: Path | None = None
    local_sessions: tuple[Path, ...] = ()
    exclude_roots: tuple[str, ...] = ()
    key_file: Path | None = None
    socks: str = ""
    api_base: str = ""


@dataclass(frozen=True)
class Agent:
    """Claude Code settings for the judgment steps."""

    model: str
    env_file: Path | None = None
    total_budget_usd: float = 100.0
    budget_usd: dict[str, float] = field(default_factory=dict)
    max_turns: dict[str, int] = field(default_factory=dict)
    max_levers: int = 4
    implement_attempts: int = 3


@dataclass(frozen=True)
class Checks:
    """How the target repository's tests and linter run."""

    python: Path
    pytest_args: tuple[str, ...] = DEFAULT_PYTEST_ARGS
    lint: tuple[str, ...] = ("ruff", "check")
    allow_new_failures: bool = False


@dataclass(frozen=True)
class Arm:
    """One A/B arm: which tree it runs and which models override the agent default."""

    name: str
    tree: str
    models: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AB:
    """GPU validation: arms run one after another from the same framework snapshot."""

    python: Path
    sessions_dir: Path
    arms: tuple[Arm, ...]
    scenario: Path | None = None
    extra_args: tuple[str, ...] = ()
    hours: float = 6.0
    snapshot_dir: Path | None = None
    snapshot_paths: tuple[Path, ...] = ()
    clear_caches: tuple[Path, ...] = ()
    kernel_agent_env: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    min_free_gb: float = 50.0
    on_interrupt: str = "rerun"
    poll_sec: float = 300.0
    stop_ray: bool = False


@dataclass(frozen=True)
class RoundConfig:
    round_dir: Path
    target: Target
    data: Data
    agent: Agent
    checks: Checks
    ab: AB
    replay_commands: tuple[tuple[str, ...], ...] = ()
    router_log: Path | None = None
    publish_push: bool = False
    publish_remote: str = "origin"


def _path(value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty path")
    return Path(value).expanduser()


def _opt_path(value: Any, where: str) -> Path | None:
    return None if value in (None, "") else _path(value, where)


def _section(raw: dict, name: str, required: bool = True) -> dict:
    value = raw.get(name)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    return value


def _require(section: dict, key: str, where: str) -> Any:
    if section.get(key) in (None, ""):
        raise ConfigError(f"{where}.{key} is required")
    return section[key]


def _step_map(value: Any, where: str, cast: type) -> dict:
    mapping = value or {}
    if not isinstance(mapping, dict):
        raise ConfigError(f"{where} must map step names to values")
    unknown = sorted(set(mapping) - set(AGENT_STEPS))
    if unknown:
        raise ConfigError(f"{where} names unknown agent steps: {', '.join(unknown)}")
    return {k: cast(v) for k, v in mapping.items()}


def _target(raw: dict) -> Target:
    s = _section(raw, "target")
    return Target(
        repo=_path(_require(s, "repo", "target"), "target.repo"),
        base_ref=str(_require(s, "base_ref", "target")),
        branch=str(_require(s, "branch", "target")),
    )


def _data(raw: dict) -> Data:
    s = _section(raw, "data")
    return Data(
        bundles_dir=_path(_require(s, "bundles_dir", "data"), "data.bundles_dir"),
        recent_since=str(_require(s, "recent_since", "data")),
        last_days=int(s.get("last_days", 15)),
        models_dir=_opt_path(s.get("models_dir"), "data.models_dir"),
        local_sessions=tuple(_path(p, "data.local_sessions") for p in s.get("local_sessions") or ()),
        exclude_roots=tuple(str(p) for p in s.get("exclude_roots") or ()),
        key_file=_opt_path(s.get("key_file"), "data.key_file"),
        socks=str(s.get("socks") or ""),
        api_base=str(s.get("api_base") or ""),
    )


def _agent(raw: dict) -> Agent:
    s = _section(raw, "agent")
    agent = Agent(
        model=str(_require(s, "model", "agent")),
        env_file=_opt_path(s.get("env_file"), "agent.env_file"),
        total_budget_usd=float(s.get("total_budget_usd", 100.0)),
        budget_usd=_step_map(s.get("budget_usd"), "agent.budget_usd", float),
        max_turns=_step_map(s.get("max_turns"), "agent.max_turns", int),
        max_levers=int(s.get("max_levers", 4)),
        implement_attempts=int(s.get("implement_attempts", 3)),
    )
    if agent.total_budget_usd <= 0 or any(v <= 0 for v in agent.budget_usd.values()):
        raise ConfigError("agent budgets must be positive")
    if agent.max_levers < 1 or agent.implement_attempts < 1:
        raise ConfigError("agent.max_levers and agent.implement_attempts must be at least 1")
    return agent


def _checks(raw: dict) -> Checks:
    s = _section(raw, "checks")
    return Checks(
        python=_path(_require(s, "python", "checks"), "checks.python"),
        pytest_args=tuple(str(a) for a in s.get("pytest_args") or DEFAULT_PYTEST_ARGS),
        lint=tuple(str(a) for a in s.get("lint") or ("ruff", "check")),
        allow_new_failures=bool(s.get("allow_new_failures", False)),
    )


def _arms(value: Any) -> tuple[Arm, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ConfigError("ab.arms needs at least two arms; the first one is the control")
    arms = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConfigError(f"ab.arms[{i}] must be a mapping")
        name, tree = str(_require(item, "name", f"ab.arms[{i}]")), str(_require(item, "tree", f"ab.arms[{i}]"))
        models = {str(k): str(v) for k, v in (item.get("models") or {}).items()}
        if tree not in ARM_TREES:
            raise ConfigError(f"ab.arms[{i}].tree must be one of {', '.join(ARM_TREES)}")
        if set(models) - set(ARM_ROLES):
            raise ConfigError(f"ab.arms[{i}].models may only set {', '.join(ARM_ROLES)}")
        arms.append(Arm(name=name, tree=tree, models=models))
    if len({a.name for a in arms}) != len(arms):
        raise ConfigError("ab.arms names must be unique")
    return tuple(arms)


def _ab(raw: dict) -> AB:
    s = _section(raw, "ab")
    snap = s.get("snapshot") or {}
    ab = AB(
        python=_path(_require(s, "python", "ab"), "ab.python"),
        sessions_dir=_path(_require(s, "sessions_dir", "ab"), "ab.sessions_dir"),
        arms=_arms(s.get("arms")),
        scenario=_opt_path(s.get("scenario"), "ab.scenario"),
        extra_args=tuple(str(a) for a in s.get("extra_args") or ()),
        hours=float(s.get("hours", 6.0)),
        snapshot_dir=_opt_path(snap.get("dir"), "ab.snapshot.dir"),
        snapshot_paths=tuple(_path(p, "ab.snapshot.paths") for p in snap.get("paths") or ()),
        clear_caches=tuple(_path(p, "ab.clear_caches") for p in s.get("clear_caches") or ()),
        kernel_agent_env=_opt_path(s.get("kernel_agent_env"), "ab.kernel_agent_env"),
        env={str(k): str(v) for k, v in (s.get("env") or {}).items()},
        min_free_gb=float(s.get("min_free_gb", 50.0)),
        on_interrupt=str(s.get("on_interrupt", "rerun")),
        poll_sec=float(s.get("poll_sec", 300.0)),
        stop_ray=bool(s.get("stop_ray", False)),
    )
    if ab.on_interrupt not in ("rerun", "keep"):
        raise ConfigError("ab.on_interrupt must be 'rerun' or 'keep'")
    if bool(ab.snapshot_dir) != bool(ab.snapshot_paths):
        raise ConfigError("ab.snapshot needs both 'dir' and 'paths', or neither")
    return ab


def parse_config(raw: Any) -> RoundConfig:
    """Validate a parsed YAML document; raises ConfigError naming the first bad value."""
    if not isinstance(raw, dict):
        raise ConfigError("the round configuration must be a mapping")
    replay = _section(raw, "replay", required=False)
    publish = _section(raw, "publish", required=False)
    commands = replay.get("commands") or []
    if not all(isinstance(c, list) and c and all(isinstance(a, str) for a in c) for c in commands):
        raise ConfigError("replay.commands must be a list of argument lists")
    return RoundConfig(
        round_dir=_path(raw.get("round_dir"), "round_dir"),
        target=_target(raw),
        data=_data(raw),
        agent=_agent(raw),
        checks=_checks(raw),
        ab=_ab(raw),
        replay_commands=tuple(tuple(c) for c in commands),
        router_log=_opt_path(_section(raw, "compare", required=False).get("router_log"), "compare.router_log"),
        publish_push=bool(publish.get("push", False)),
        publish_remote=str(publish.get("remote") or "origin"),
    )


def load_config(path: Path) -> RoundConfig:
    with open(path) as fh:
        return parse_config(yaml.safe_load(fh))
