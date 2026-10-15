# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a candidate changes about the launch it is composed onto.

A candidate is composed onto its base exactly as a grid variant launches (:func:`compose_variant_args`), and
the change is whatever differs: settings it adds or rewrites, settings it removes, envs it sets or unsets.
Every measurement records the args and env it requested (``launch_evidence``), so the current best's record
is a base, and a round composing its own base is judged on that. A setting the engine resolved to the
requested value is not a change, but only on the launch the engine was observed running.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from hyperloom.common.coerce import to_str_list
from hyperloom.common.launch_log_evidence import launch_flag_setting_name, settings_the_engine_rewrote
from hyperloom.inference_optimizer.grid_server_args import compose_server_args, server_args_env_name


#: A candidate whose args add to its base rather than replace it.
APPEND = "append"


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


def compose_variant_args(
    inherited_args: str,
    base_extra_args: str,
    *,
    extra_args: str,
    remove_args: Iterable[str],
    args_mode: str,
    base_remove_args: Iterable[str],
    base_args_mode: str,
) -> str:
    """The server args a variant launches with: the inherited args, then the round's base args, then its own."""
    replacing = str(base_args_mode).strip().lower() == "replace"
    variant_remove = to_str_list(remove_args)
    # A replacing base drops the inherited string wholesale, so only the
    # variant's own removals still name flags that survive to be stripped.
    effective_remove = (
        variant_remove if replacing else list(dict.fromkeys(to_str_list(base_remove_args) + variant_remove))
    )
    return compose_server_args(
        inherited_args="" if replacing else inherited_args,
        base_extra_args=base_extra_args,
        variant_extra_args=extra_args,
        remove_args=effective_remove,
        args_mode=args_mode,
    )


@dataclass(frozen=True)
class BaseLaunch:
    """The launch candidates are composed onto, and what the engine resolved for it when that was observed."""

    inherited_args: str
    extra_args: str = ""
    remove_args: tuple[str, ...] = ()
    args_mode: str = APPEND
    env: Mapping[str, str] = field(default_factory=dict)
    resolved: Mapping[str, Any] = field(default_factory=dict)

    def compose(self, extra_args: str = "", remove_args: Iterable[str] = (), args_mode: str = APPEND) -> str:
        """The args a candidate with these fields launches with on this base."""
        return compose_variant_args(
            self.inherited_args,
            self.extra_args,
            extra_args=extra_args,
            remove_args=remove_args,
            args_mode=args_mode,
            base_remove_args=self.remove_args,
            base_args_mode=self.args_mode,
        )


def evidence_launch(evidence: Mapping[str, Any]) -> BaseLaunch:
    """The launch one measurement recorded: the args and env it requested, and what the engine resolved."""
    raw_env = evidence.get("requested_server_env")
    return BaseLaunch(
        inherited_args=str(evidence.get("requested_server_args") or ""),
        env={str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, Mapping) else {},
        resolved=_resolved_settings(evidence),
    )


def recorded_launch(state: Any) -> BaseLaunch:
    """The launch ``current_best``'s measurement recorded; an empty base before anything was measured."""
    measurement = getattr(state, "current_best_measurement", None)
    evidence = measurement.get("launch_evidence") if isinstance(measurement, Mapping) else None
    return evidence_launch(evidence if isinstance(evidence, Mapping) else {})


def round_base(
    recorded: BaseLaunch,
    framework: str,
    *,
    config_envs: Mapping[str, Any],
    extra_args: str,
    extra_envs: Mapping[str, Any],
    remove_args: Iterable[str],
    unset_envs: Iterable[str],
    args_mode: str,
) -> BaseLaunch:
    """The base a round composes its variants onto: its config, then its own base args and envs.

    The engine's record of ``recorded`` describes it only when both launch the same args.
    """
    args_env = server_args_env_name(framework)
    unset = set(to_str_list(unset_envs))
    env = {str(k): str(v) for k, v in config_envs.items() if str(k) != args_env and str(k) not in unset}
    env.update({str(k): str(v) for k, v in extra_envs.items()})
    base = BaseLaunch(
        inherited_args=str(config_envs.get(args_env) or ""),
        extra_args=extra_args,
        remove_args=tuple(to_str_list(remove_args)),
        args_mode=args_mode,
        env=env,
    )
    same = _named_settings(base.compose(), framework) == _named_settings(recorded.compose(), framework)
    return replace(base, resolved=recorded.resolved) if same else base


def launch_change(
    base: BaseLaunch,
    framework: str,
    *,
    extra_args: str,
    extra_envs: Mapping[str, Any],
    remove_args: Iterable[str] = (),
    unset_envs: Iterable[str] = (),
    args_mode: str = APPEND,
) -> LaunchChange:
    """Compose the candidate onto ``base`` and return what differs from ``base`` itself."""
    stack = _named_settings(base.compose(), framework)
    launch = _named_settings(base.compose(extra_args, remove_args, args_mode), framework)
    settings = {
        name: values
        for name, values in launch.items()
        if stack.get(name) != values
        and not (name in base.resolved and _setting_already_holds(values, base.resolved[name]))
    }
    envs = {str(k): str(v) for k, v in extra_envs.items() if base.env.get(str(k)) != str(v)}
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


__all__ = [
    "BaseLaunch",
    "LaunchChange",
    "compose_variant_args",
    "evidence_launch",
    "launch_change",
    "recorded_launch",
    "round_base",
]
