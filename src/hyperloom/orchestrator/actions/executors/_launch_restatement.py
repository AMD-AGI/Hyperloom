# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a candidate launch changes about the launch it is composed onto.

Both launches are read back from configs the launch's own materializer built, so everything the server is
started with -- argv, envs, removals, overlays, AgentX launch overrides -- is on one side or the other, and a
candidate whose launch equals the stack's measures the stack again.
"""

from __future__ import annotations

import json
import logging
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.common.coerce import to_str_list
from hyperloom.common.launch_log_evidence import launch_flag_setting_name
from hyperloom.inference_optimizer.grid_server_args import server_args_env_name

from ._server_argv import ConfigUnreadable, config_server_argv

log = logging.getLogger(__name__)


def config_envs(config_path: Path) -> dict[str, str]:
    """Read ``benchmark.envs`` out of a built config file as a ``str -> str`` dict.

    Drops the framework's server-args key (``EXTRA_SGLANG_ARGS`` and friends),
    which holds the launch argv rather than an env value, and a null value,
    which the launch leaves unset. Returns an empty dict when the file carries
    no ``benchmark.envs`` mapping.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    bench = cfg.get("benchmark") if isinstance(cfg, Mapping) else None
    envs = bench.get("envs") if isinstance(bench, Mapping) else None
    if not isinstance(envs, Mapping):
        return {}
    args_env = server_args_env_name(str(bench.get("framework") or ""))
    return {str(k): str(v) for k, v in envs.items() if str(k) != args_env and v is not None}


_NEGATION = "no_"
_NEGATED = "false"


def named_settings(argv_text: str, framework: str) -> dict[str, list[str]]:
    """Map each setting ``argv_text`` names to its value tokens as the engine resolves them.

    Raises:
        ValueError: When the argv does not parse or opens on a positional token.
    """
    return _settings_of_tokens(shlex.split(argv_text), framework)


def _settings_of_tokens(tokens: Sequence[str], framework: str) -> dict[str, list[str]]:
    # Last occurrence wins, and a bare ``--no-x`` sets ``x`` to false, so both spellings land on one setting.
    if tokens and not str(tokens[0]).startswith("-"):
        raise ValueError(f"server argv opens on a positional token: {tokens[0]!r}")
    settings: dict[str, list[str]] = {}
    current = ""
    for token in map(str, tokens):
        if token.startswith("-"):
            name, separator, attached = token.partition("=")
            current = launch_flag_setting_name(name, framework)
            if current.startswith(_NEGATION) and not separator:
                current = current[len(_NEGATION) :]
                settings[current] = [_NEGATED]
            else:
                settings[current] = [attached] if separator else []
        else:
            settings[current].append(token)
    return settings


def observed_launch_after(
    observed_config: Mapping[str, Any],
    observed_env: Mapping[str, str],
    opening: Mapping[str, Any],
    now: Mapping[str, Any],
    framework: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """The part of the observed launch that still describes the stack once it has moved from ``opening`` to ``now``.

    The observed launch is what the server ran when the round opened. A KEEP that changed a setting, an env, a removal
    or an unset (added or cleared) leaves the observed value of that one, and of its negated spelling, stale; every
    other setting is still what the server runs. Nothing is known when either stack's args cannot be read. Both stacks are ``measured_against`` shaped, with their args
    composed as launched, so a change of args mode alone moves nothing.
    """
    removed = set(now.get("remove_args") or ()) ^ set(opening.get("remove_args") or ())
    try:
        before = named_settings(str(opening.get("extra_server_args") or ""), framework)
        after = named_settings(str(now.get("extra_server_args") or ""), framework)
        removed_settings = named_settings(" ".join(sorted(removed)), framework).keys()
    except ValueError:
        return {}, {}
    stale = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    stale |= removed_settings
    stale |= {_opposite_setting(name) for name in stale}
    envs_before, envs_after = opening.get("extra_envs") or {}, now.get("extra_envs") or {}
    stale_envs = {
        name for name in envs_before.keys() | envs_after.keys() if envs_before.get(name) != envs_after.get(name)
    }
    stale_envs |= set(now.get("unset_envs") or ()) ^ set(opening.get("unset_envs") or ())
    return (
        {name: value for name, value in observed_config.items() if name not in stale},
        {name: value for name, value in observed_env.items() if name not in stale_envs},
    )


def _opposite_setting(name: str) -> str:
    """The setting a boolean flag's negated spelling sets: ``no_x`` for ``x`` and back."""
    return name[3:] if name.startswith("no_") else f"no_{name}"


@dataclass(frozen=True)
class Launch:
    """The server a materialized config starts: its settings, its envs, and the controls neither of them shows."""

    settings: Mapping[str, list[str]]
    envs: Mapping[str, str]
    controls: Mapping[str, Any]

    @property
    def identity(self) -> str:
        """One string per distinct server launch."""
        return json.dumps([self.settings, self.envs, self.controls], sort_keys=True)


def read_launch(config_path: Path, unset_envs: Iterable[str] = ()) -> Launch:
    """Read the launch a config built by the grid materializer starts, whichever layout it carries.

    ``unset_envs`` are the names the run also pops from the inherited process env, which no config records.

    Raises:
        ConfigUnreadable, OSError, ValueError, yaml.YAMLError: When the config or its argv cannot be read.
    """
    bench = yaml.safe_load(config_path.read_text(encoding="utf-8"))["benchmark"]
    framework = str(bench.get("framework") or "")
    controls: dict[str, Any] = {"unset_envs": sorted(set(to_str_list(unset_envs)))}
    if not is_magpie_agentx(bench):
        argv = config_server_argv(config_path).text
        return Launch(named_settings(argv, framework), config_envs(config_path), controls)
    overrides = (bench.get("agentx") or {}).get("launch_overrides") or {}
    envs = {**config_envs(config_path), **{str(k): str(v) for k, v in (overrides.get("env") or {}).items()}}
    controls.update(
        remove_args=sorted(overrides.get("remove_args") or []),
        replace_args=bool(overrides.get("replace_args")),
        unset_env=sorted(overrides.get("unset_env") or []),
        executable=str(overrides.get("executable") or ""),
        source_files=dict(overrides.get("source_files") or {}),
    )
    return Launch(_settings_of_tokens(list(overrides.get("append_args") or []), framework), envs, controls)


def launch_change(
    base: Launch, launch: Launch, observed: tuple[Mapping[str, Any], Mapping[str, str]]
) -> dict[str, Any]:
    """What ``launch`` changes about ``base``, by kind; empty when both start the same server.

    A setting or env set to the value the observed server already runs with is not a change, since the engine
    resolves defaults no argv names. Removals and controls always are.
    """
    observed_config, observed_env = observed
    change = {
        "settings": {
            name: values
            for name, values in launch.settings.items()
            if base.settings.get(name) != values and not _already_resolved(name, values, observed_config)
        },
        "removed": sorted(base.settings.keys() - launch.settings.keys()),
        "envs": {
            name: value
            for name, value in launch.envs.items()
            if base.envs.get(name) != value and observed_env.get(name) != value
        },
        "unset_envs": sorted(base.envs.keys() - launch.envs.keys()),
        "controls": sorted(
            name
            for name in base.controls.keys() | launch.controls.keys()
            if base.controls.get(name) != launch.controls.get(name)
        ),
    }
    return {kind: diff for kind, diff in change.items() if diff}


def _already_resolved(name: str, values: list[str], observed_config: Mapping[str, Any]) -> bool:
    """Whether the observed server already resolved setting ``name`` to ``values``."""
    if name in observed_config and setting_already_holds(values, observed_config[name]):
        return True
    return values == [_NEGATED] and observed_config.get(_NEGATION + name) is True


def setting_already_holds(proposed: list[str], observed: Any) -> bool:
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


def _names_only_flags(argv_text: str) -> bool:
    """Whether ``argv_text`` parses and opens on a flag, so ``named_settings`` accounts for every token."""
    try:
        tokens = shlex.split(argv_text)
    except ValueError:
        return False
    return not tokens or tokens[0].startswith("-")


def restates_the_baseline(
    argv_text: str,
    envs: Mapping[str, Any],
    base_argv: str,
    base_envs: Mapping[str, str],
    framework: str,
) -> bool:
    """Whether appending ``argv_text`` and ``envs`` to the baseline launch would change nothing.

    True only when every setting the request names is already in ``base_argv``
    with the same value tokens, and every env already carries that value in
    ``base_envs``. A request this cannot read in full -- an unparsable argv, a
    positional token, a flag whose negated spelling the baseline also names --
    is never a restatement, and neither is an empty one.
    """
    text = str(argv_text or "").strip()
    requested_envs = {str(k): str(v) for k, v in (envs or {}).items()}
    if (not text and not requested_envs) or not _names_only_flags(text):
        return False
    base_settings = named_settings(base_argv, framework)
    for name, values in named_settings(text, framework).items():
        if base_settings.get(name) != values or _opposite_setting(name) in base_settings:
            return False
    return all(base_envs.get(name) == value for name, value in requested_envs.items())


def config_launch_restated_by(config_path: str, argv_text: str, envs: Mapping[str, Any], framework: str) -> bool:
    """Whether appending ``argv_text`` and ``envs`` to the config at ``config_path`` would change nothing.

    False when the config is absent or unreadable: the launch to compare
    against is then unknown.
    """
    if not str(config_path or "").strip():
        return False
    try:
        base_argv = config_server_argv(config_path).text
        base_envs = config_envs(Path(config_path))
    except (ConfigUnreadable, OSError, ValueError, yaml.YAMLError) as exc:
        log.warning("baseline config %s unreadable for the restatement check (%s)", config_path, exc)
        return False
    return restates_the_baseline(argv_text, envs, base_argv, base_envs, framework)
