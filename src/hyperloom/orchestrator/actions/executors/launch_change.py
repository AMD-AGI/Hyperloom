# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a candidate changes about the launch the stack is running, judged on that launch's own record.

Every measurement records the server args and env it requested (``launch_evidence``), whichever executor
materialized it. A candidate is composed onto the current best's record exactly as a launch composes it, and
the change is whatever differs: settings it adds or rewrites, settings it removes, envs it sets or unsets. A
setting the engine already resolved to the requested value is not a change either.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from hyperloom.common.coerce import to_str_list
from hyperloom.common.launch_log_evidence import launch_flag_setting_name, settings_the_engine_rewrote
from hyperloom.inference_optimizer.grid_server_args import compose_server_args


@dataclass(frozen=True)
class LaunchChange:
    """The part of a candidate's launch that differs from the stack's."""

    settings: Mapping[str, list[str]]
    removed: tuple[str, ...]
    envs: Mapping[str, str]
    #: The env a launch inherits from its image is not recorded, so every unset counts.
    unset_envs: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.settings or self.removed or self.envs or self.unset_envs)

    @property
    def key(self) -> str:
        """One string per distinct change, so candidates with the same effect compare equal."""
        return json.dumps(
            [self.settings, list(self.removed), self.envs, list(self.unset_envs)],
            sort_keys=True,
        )


def stack_launch_evidence(state: Any) -> Mapping[str, Any]:
    """The launch record of the measurement ``current_best`` holds; empty before anything was measured."""
    measurement = getattr(state, "current_best_measurement", None)
    evidence = measurement.get("launch_evidence") if isinstance(measurement, Mapping) else None
    return evidence if isinstance(evidence, Mapping) else {}


def launch_change(
    evidence: Mapping[str, Any],
    framework: str,
    *,
    extra_args: str,
    extra_envs: Mapping[str, Any],
    remove_args: Iterable[str] = (),
    unset_envs: Iterable[str] = (),
    args_mode: str = "append",
) -> LaunchChange:
    """Compose the candidate onto ``evidence``'s requested launch and return what differs from it."""
    stack_args = str(evidence.get("requested_server_args") or "")
    stack = _named_settings(compose_server_args(inherited_args=stack_args), framework)
    launch = _named_settings(
        compose_server_args(
            inherited_args=stack_args,
            variant_extra_args=extra_args,
            remove_args=to_str_list(remove_args),
            args_mode=args_mode,
        ),
        framework,
    )
    resolved = _resolved_settings(evidence)
    settings = {
        name: values
        for name, values in launch.items()
        if stack.get(name) != values and not (name in resolved and _setting_already_holds(values, resolved[name]))
    }
    raw_env = evidence.get("requested_server_env")
    stack_env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, Mapping) else {}
    envs = {str(k): str(v) for k, v in extra_envs.items() if stack_env.get(str(k)) != str(v)}
    return LaunchChange(
        settings=settings,
        removed=tuple(sorted(name for name in stack if name not in launch)),
        envs=envs,
        unset_envs=tuple(sorted(set(to_str_list(unset_envs)))),
    )


def _named_settings(argv_text: str, framework: str) -> dict[str, list[str]]:
    """Map each setting ``argv_text`` names to its value tokens, last occurrence winning like the engine."""
    try:
        tokens = shlex.split(argv_text)
    except ValueError:
        return {}
    settings: dict[str, list[str]] = {}
    current = ""
    for token in tokens:
        if token.startswith("-"):
            name, separator, attached = token.partition("=")
            current = launch_flag_setting_name(name, framework)
            settings[current] = [attached] if separator else []
        elif current:
            settings[current].append(token)
    return settings


def _resolved_settings(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """The engine's resolved config, without the settings it rewrote after parsing them."""
    raw = evidence.get("observed_server_config")
    config = dict(raw) if isinstance(raw, Mapping) else {}
    adjusted = evidence.get("engine_adjusted_settings")
    for name in settings_the_engine_rewrote(config, adjusted if isinstance(adjusted, Mapping) else None):
        config.pop(name, None)
    return config


def _setting_already_holds(proposed: list[str], observed: Any) -> bool:
    """Whether ``observed`` is already the value ``proposed`` asks for."""
    if not proposed:
        # A bare flag asks for the feature on; anything else is a change.
        return observed is True
    if isinstance(observed, (list, tuple)):
        return [str(item) for item in observed] == proposed
    if len(proposed) != 1:
        return False
    text = proposed[0]
    if isinstance(observed, bool):
        return text.lower() == str(observed).lower()
    if isinstance(observed, (int, float)):
        try:
            return float(text) == float(observed)
        except ValueError:
            return False
    if observed is None:
        return False
    return text == str(observed)


__all__ = ["LaunchChange", "launch_change", "stack_launch_evidence"]
