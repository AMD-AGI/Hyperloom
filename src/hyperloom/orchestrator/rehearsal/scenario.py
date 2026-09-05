# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A bring-up round written down: what each attempt does, in order.

An attempt states what the server's own log contained, that log being what the
classifier reads; naming a ladder stage synthesises one for it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.bringup import LadderStage

__all__ = [
    "DIED_SILENTLY",
    "HANG",
    "READY",
    "STAGE_FAILED",
    "LaunchAttempt",
    "LaunchScenario",
    "ScenarioError",
    "boot_log_for",
]

#: The attempt booted and served.
READY = "ready"

#: The attempt stopped at a named ladder stage.
STAGE_FAILED = "stage_failed"

#: The attempt never returned and was reaped on its hard timeout.
HANG = "hang"

#: The attempt's process died leaving no server log at all.
DIED_SILENTLY = "died_silently"

_OUTCOMES = frozenset({READY, STAGE_FAILED, HANG, DIED_SILENTLY})

#: The milestone line a server prints on reaching each stage, in ladder order.
_STAGE_PROGRESS: Mapping[LadderStage, str] = {
    LadderStage.CONFIG_VALIDATE: "INFO server_args=ServerArgs(model_path=<model>, tp_size=1)",
    LadderStage.WEIGHTS_LOADING: "INFO Loading weights from safetensors checkpoint shards",
    LadderStage.WEIGHTS_LOADED: "INFO Loading weights took 41.20 seconds",
    LadderStage.ENGINE_INIT: "INFO KV Cache is allocated. max_total_num_tokens=262144",
    LadderStage.GRAPH_CAPTURE: "INFO Capture cuda graph begin. This can take up to several minutes.",
    LadderStage.HTTP_READY: "INFO Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)",
    LadderStage.GENERATES: "INFO The server is fired up and ready to roll!",
}

#: The failure line synthesised for a stage that was not reached, worded so the
#: enablement rule table maps it back. Stages absent here get a generic one.
_STAGE_FAILURE: Mapping[LadderStage, str] = {
    LadderStage.ARGV_PARSE: "usage: server [-h]\nserver: error: unrecognized arguments: --speculative-config",
    LadderStage.IMPORT: 'ModuleNotFoundError: No module named "flashinfer"',
    LadderStage.CONFIG_VALIDATE: ("ValueError: Architectures ['ScriptedForCausalLM'] are not supported for now."),
    LadderStage.WEIGHTS_LOADING: "KeyError: 'model.layers.0.mlp.gate_proj.weight'",
    LadderStage.ENGINE_INIT: "NotImplementedError: paged attention v2 has no ROCm path yet",
    LadderStage.HTTP_READY: "RuntimeError: HIP out of memory. Tried to allocate 4.00 GiB",
    LadderStage.ACCURACY_OK: "ERROR baseline accuracy 0.02 did not meet the accuracy floor 0.30",
}

#: Stages reached before the server owns a log file, so a failure here goes to
#: the launcher's stderr instead.
_PRE_LOG_STAGES = frozenset({LadderStage.ARGV_PARSE, LadderStage.IMPORT})

#: What a stage with no rule of its own fails with. The frame is what makes an
#: unrecognised wall placeable and dedupable.
_GENERIC_FAILURE = "\n".join(
    (
        "Traceback (most recent call last):",
        '  File "engine/core.py", line 214, in _advance',
        "    raise RuntimeError(reason)",
        "RuntimeError: bring-up stopped before the next milestone",
    )
)

#: Spacing between synthesised log stamps, which is where elapsed time is read.
_LINE_SPACING_SEC = 2.0

#: Where a synthesised log's first line is stamped. Fixed, so replays match.
_LOG_EPOCH = (2026, 3, 14, 9, 0, 0)


class ScenarioError(ValueError):
    """A scenario that cannot be played as written."""


@dataclass(frozen=True)
class LaunchAttempt:
    """One launch attempt: what it did, how long it took, what it left behind.

    Attributes:
        name: Label carried into the recorded call.
        outcome: One of the module's four outcome constants.
        stage: Ladder stage the boot stopped at, for :data:`STAGE_FAILED`.
        message: Failure line to write instead of the stage's canonical one.
        server_log: Whole server log, replacing what the stage would synthesise.
        returncode: Exit code; defaults to what the outcome implies.
        duration_sec: Seconds the attempt took, charged to the clock.
        wrapper_stdout: Launcher stdout the attempt returns.
        wrapper_stderr: Launcher stderr the attempt returns; defaults to the
            failure line for a boot that stopped before it owned a log.
        artifacts: Files to write under the round directory, keyed by relative
            path; a mapping or list is written as JSON.
    """

    name: str = ""
    outcome: str = READY
    stage: str = ""
    message: str = ""
    server_log: str = ""
    returncode: int | None = None
    duration_sec: float = 30.0
    wrapper_stdout: str = ""
    wrapper_stderr: str = ""
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject an attempt whose outcome or stage cannot be played.

        Raises:
            ScenarioError: On an unknown outcome, or a stage failure naming no
                stage.
            ValueError: On a stage name the ladder does not define.
        """
        if self.outcome not in _OUTCOMES:
            raise ScenarioError(f"unknown attempt outcome {self.outcome!r}; expected one of {sorted(_OUTCOMES)}")
        if self.outcome == STAGE_FAILED and not (self.stage or self.server_log):
            raise ScenarioError("a stage_failed attempt must name a stage or supply a server_log")
        if self.stage:
            LadderStage.from_name(self.stage)

    @property
    def failed_stage(self) -> LadderStage | None:
        """LadderStage | None: The stage named by this attempt, when it names one."""
        return LadderStage.from_name(self.stage) if self.stage else None

    def rendered_log(self) -> str:
        """Return the server log this attempt writes.

        Returns:
            str: The explicit ``server_log``, else the log synthesised for the
            named stage, else ``""`` for a process that died before opening it.
        """
        if self.server_log:
            return self.server_log
        if self.outcome == DIED_SILENTLY:
            return ""
        if self.outcome == READY:
            return boot_log_for(None)
        return boot_log_for(self.failed_stage, message=self.message)

    def rendered_stderr(self) -> str:
        """Return the launcher stderr this attempt reports.

        Returns:
            str: The explicit ``wrapper_stderr``, else the failure line for a
            stage in :data:`_PRE_LOG_STAGES`, else ``""``.
        """
        if self.wrapper_stderr:
            return self.wrapper_stderr
        stage = self.failed_stage
        if self.outcome != STAGE_FAILED or stage not in _PRE_LOG_STAGES:
            return ""
        return self.message or _STAGE_FAILURE.get(stage, _GENERIC_FAILURE)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LaunchAttempt":
        """Build an attempt from its serialized form.

        Args:
            raw: One entry of a scenario's ``attempts`` list.

        Returns:
            LaunchAttempt: The parsed attempt.

        Raises:
            ScenarioError: On a key the format does not define.
        """
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ScenarioError(f"unknown attempt keys: {unknown}")
        return cls(**dict(raw))


@dataclass(frozen=True)
class LaunchScenario:
    """An ordered round: the attempts a scripted backend plays, one per launch.

    Attributes:
        attempts: The attempts, in the order they are served.
        name: Label for the scenario as a whole.
    """

    attempts: tuple[LaunchAttempt, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        """Reject a scenario with nothing to play.

        Raises:
            ScenarioError: When ``attempts`` is empty.
        """
        if not self.attempts:
            raise ScenarioError("a scenario needs at least one attempt")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LaunchScenario":
        """Build a scenario from its serialized form.

        Exactly one of two spellings is declared: ``attempts`` lists them, or
        ``blockers`` lists stages peeled one per attempt, with a clean boot
        appended unless ``clean_after`` is false.

        Args:
            raw: The scenario mapping.

        Returns:
            LaunchScenario: The parsed scenario.

        Raises:
            ScenarioError: When the mapping declares neither spelling, or both,
                or when the spelling it declares is not a list.
        """
        name = str(raw.get("name", ""))
        has_attempts = "attempts" in raw
        has_blockers = "blockers" in raw
        if has_attempts == has_blockers:
            raise ScenarioError("a scenario declares exactly one of 'attempts' or 'blockers'")
        if has_attempts:
            entries = _entry_list(raw, "attempts")
            return cls(attempts=tuple(LaunchAttempt.from_dict(e) for e in entries), name=name)
        blockers = _entry_list(raw, "blockers")
        return cls(attempts=_peel(blockers, clean_after=bool(raw.get("clean_after", True))), name=name)

    @classmethod
    def from_file(cls, path: str | Path) -> "LaunchScenario":
        """Read a scenario from a JSON file.

        Args:
            path: The scenario file.

        Returns:
            LaunchScenario: The parsed scenario.

        Raises:
            OSError: When the file cannot be read.
            json.JSONDecodeError: When it is not JSON.
            ScenarioError: When it is JSON but not an object, or not a scenario.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ScenarioError(f"scenario {path} is not a JSON object")
        return cls.from_dict(raw)


def _entry_list(raw: Mapping[str, Any], key: str) -> Sequence[Any]:
    """Return the list ``key`` declares, rejecting anything else."""
    value = raw[key]
    if not isinstance(value, (list, tuple)):
        raise ScenarioError(f"a scenario's {key!r} is a list, not {type(value).__name__}")
    return value


def _peel(blockers: Sequence[Any], *, clean_after: bool) -> tuple[LaunchAttempt, ...]:
    """Expand a stack of stage names, or mappings naming one, into one attempt each."""
    if not blockers:
        raise ScenarioError("a blocker stack needs at least one blocker")
    attempts: list[LaunchAttempt] = []
    for index, blocker in enumerate(blockers):
        entry: dict[str, Any] = {"stage": blocker} if isinstance(blocker, str) else dict(blocker)
        entry["outcome"] = STAGE_FAILED
        entry.setdefault("name", f"blocker-{index}-{entry.get('stage', '')}")
        attempts.append(LaunchAttempt.from_dict(entry))
    if clean_after:
        attempts.append(LaunchAttempt(name="clean-boot", outcome=READY))
    return tuple(attempts)


def boot_log_for(stage: LadderStage | None, *, message: str = "") -> str:
    """Render a server log for a boot that stopped at ``stage``.

    Every milestone strictly below ``stage`` is announced, then the failure --
    the shape the classifier's progress scan is written against.

    Args:
        stage: The stage the boot stopped at, or ``None`` for a clean boot.
        message: Failure line to print instead of the stage's canonical one.

    Returns:
        str: The rendered log, one timestamped line per event.
    """
    lines: list[str] = []
    for milestone, text in _STAGE_PROGRESS.items():
        if stage is not None and milestone >= stage:
            break
        lines.append(text)
    if stage is not None:
        lines.append(message or _STAGE_FAILURE.get(stage, _GENERIC_FAILURE))
    return _stamped("\n".join(lines))


def _stamped(text: str) -> str:
    """Prefix every line of ``text`` with an increasing timestamp."""
    from datetime import datetime, timedelta

    base = datetime(*_LOG_EPOCH)
    out: list[str] = []
    for index, line in enumerate(text.splitlines()):
        stamp = (base + timedelta(seconds=index * _LINE_SPACING_SEC)).strftime("%Y-%m-%d %H:%M:%S")
        out.append(f"[{stamp}] {line}")
    return "\n".join(out) + ("\n" if out else "")
