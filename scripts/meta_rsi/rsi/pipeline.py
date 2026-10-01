# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rsi pipeline: ordered steps over one round, resumable from the round's state file."""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from meta_rsi.rsi.config import RoundConfig
from meta_rsi.rsi.state import FINISHED, RoundState

if TYPE_CHECKING:
    from meta_rsi.rsi.agent import AgentResult, AgentSpec

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
LOG_TAIL_LINES = 30
ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


class StepFailed(RuntimeError):
    """A step could not produce its outputs; the message says why and what to change."""


@dataclass(frozen=True)
class Step:
    """One pipeline step. ``needs`` must be finished before the step may run on its own."""

    name: str
    kind: str
    run: Callable[[RoundContext], dict]
    needs: tuple[str, ...] = ()


class CommandRunner:
    """Runs external commands, appends their output to a log file, and turns failures into StepFailed."""

    def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        log: Path | None = None,
        check: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            proc = subprocess.run(
                list(cmd),
                cwd=cwd,
                env=dict(env) if env is not None else None,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise StepFailed(f"{cmd[0]} could not run: {exc}") from exc
        output = (proc.stdout or "") + (proc.stderr or "")
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as fh:
                fh.write(f"$ {' '.join(cmd)}\n{output}\n")
        if check and proc.returncode != 0:
            tail = "\n".join(output.splitlines()[-LOG_TAIL_LINES:])
            raise StepFailed(f"{' '.join(cmd[:3])} exited {proc.returncode}:\n{tail}")
        return proc


def _log(message: str) -> None:
    print(f"[rsi] {message}", file=sys.stderr, flush=True)


def read_env_file(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines of an env file (optionally ``export``-prefixed and quoted). ``$NAME`` and
    ``${NAME}`` resolve to a key defined earlier in the file, else to this process's environment."""
    values: dict[str, str] = {}

    def resolve(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        return values.get(name, os.environ.get(name, match.group(0)))

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        values[key.strip()] = ENV_REF.sub(resolve, value.strip().strip("'\""))
    return values


@dataclass
class RoundContext:
    """What every step gets: configuration, state, and the runners for commands and agents."""

    config: RoundConfig
    state: RoundState
    runner: CommandRunner = field(default_factory=CommandRunner)
    agent: Callable[[AgentSpec], AgentResult] | None = None
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = _log

    @property
    def round_dir(self) -> Path:
        return self.config.round_dir

    def path(self, *parts: str) -> Path:
        """A path under the round directory, with its parent created."""
        p = self.round_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def worktree(self, which: str) -> Path:
        """The round's checkout of the target repository: ``base`` or ``candidate``."""
        return self.round_dir / "worktrees" / which

    def script_env(self, **extra: str) -> dict[str, str]:
        """Environment for the round's analysis scripts (see ``round_env.py``)."""
        d = self.config.data
        env = dict(os.environ)
        env.update(
            PULSE_ROUND_DIR=str(self.round_dir),
            PULSE_BUNDLES=str(d.bundles_dir),
            PULSE_RECENT_SINCE=d.recent_since,
        )
        optional = {
            "PULSE_MODELS_DIR": str(d.models_dir) if d.models_dir else "",
            "PULSE_LOCAL_SESSIONS": ":".join(str(p) for p in d.local_sessions),
            "PULSE_EXCLUDE_ROOTS": ":".join(d.exclude_roots),
            "PULSE_KEY_FILE": str(d.key_file) if d.key_file else "",
            "PULSE_SOCKS": d.socks,
            "PULSE_API_BASE": d.api_base,
        }
        env.update({k: v for k, v in optional.items() if v})
        env.update(extra)
        return env

    def agent_env(self) -> dict[str, str]:
        """Environment for agent sessions: this process's, plus the configured credentials file."""
        env = dict(os.environ)
        if self.config.agent.env_file:
            env.update(read_env_file(self.config.agent.env_file))
        return env

    def git(self, *args: str, cwd: Path) -> str:
        """Stdout of a git command, stripped; a non-zero exit raises StepFailed."""
        return self.runner.run(["git", *args], cwd=cwd, log=self.path("logs", "git.log")).stdout.strip()

    def run_agent(self, spec: AgentSpec) -> AgentResult:
        """Run a Claude Code agent inside the round's remaining budget and record what it spent."""
        if self.agent is None:
            raise StepFailed("no agent runner is configured")
        remaining = self.config.agent.total_budget_usd - self.state.agent_cost_usd
        if remaining <= 0:
            raise StepFailed("the round's agent budget is spent; raise agent.total_budget_usd to continue")
        spec = dataclasses.replace(spec, budget_usd=min(spec.budget_usd, remaining))
        result = self.agent(spec)
        self.state.agent_cost_usd += result.cost_usd
        self.state.save()
        self.log(f"agent {spec.name}: {result.turns} turns, ${result.cost_usd:.2f}")
        return result


def select_steps(steps: Sequence[Step], *, until: str = "", only: Sequence[str] = (), start: str = "") -> list[Step]:
    """The steps to run, in pipeline order: ``only`` wins; otherwise ``start`` through ``until``."""
    names = [s.name for s in steps]
    for name in (*only, until, start):
        if name and name not in names:
            raise SystemExit(f"unknown step '{name}'; steps are: {', '.join(names)}")
    if only:
        return [s for s in steps if s.name in only]
    first = names.index(start) if start else 0
    last = names.index(until) if until else len(names) - 1
    return list(steps[first : last + 1])


def _missing_needs(ctx: RoundContext, step: Step) -> list[str]:
    return [n for n in step.needs if ctx.state.step(n).status not in FINISHED]


def run_pipeline(ctx: RoundContext, steps: Sequence[Step], *, rerun: bool = False) -> bool:
    """Run ``steps`` in order, skipping finished ones unless ``rerun``; stops at the first failure."""
    for step in steps:
        rec = ctx.state.step(step.name)
        if rec.status in FINISHED and not rerun:
            ctx.log(f"{step.name}: already {rec.status}")
            continue
        missing = _missing_needs(ctx, step)
        if missing:
            ctx.log(f"{step.name}: needs {', '.join(missing)} first")
            return False
        rec.start()
        ctx.state.save()
        ctx.log(f"{step.name}: running ({step.kind})")
        outputs: dict | None = None
        try:
            outputs = step.run(ctx)
        except StepFailed as exc:
            rec.fail(str(exc))
            ctx.state.save()
            ctx.log(f"{step.name}: failed: {exc}")
            return False
        finally:
            if outputs is None and rec.status == "running":
                rec.fail("stopped by an unexpected error; see the driver log")
                ctx.state.save()
        rec.finish(outputs)
        ctx.state.save()
        ctx.log(f"{step.name}: {rec.status}")
    return True
