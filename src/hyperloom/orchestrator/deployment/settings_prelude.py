# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render accepted runtime settings into single-file deployment modules."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from hyperloom.common.env import is_truthy
from hyperloom.common.env_safety import BLOCKED_EXTERNAL_ENV_NAMES, is_secret_shaped_env_name

# Env settings reach torch when this module is imported before the GPU is initialized and
# before torch._inductor/_dynamo read their config; the import-timing guard below fails
# loudly otherwise. Variables only the process start reads (LD_*, PYTHON*) never get here:
# they are excluded as benchmark controls.
# PYTORCH_TUNABLEOP_RECORD_UNTUNED has no API equivalent, so it falls through to tier 1 untranslated.
_TORCH_API_TRANSLATIONS: Mapping[str, tuple[str, Callable[[object], bool | int]]] = {
    "PYTORCH_TUNABLEOP_ENABLED": ("torch.cuda.tunable.enable", is_truthy),
    "PYTORCH_TUNABLEOP_TUNING": ("torch.cuda.tunable.tuning_enable", is_truthy),
    "PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS": ("torch.cuda.tunable.set_max_tuning_duration", int),
    "PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS": ("torch.cuda.tunable.set_max_tuning_iterations", int),
}

# Modules that freeze config from os.environ at first import -- if either is already in
# sys.modules, or the GPU is already initialized, an env setting may have been read too late.
_IMPORT_SENSITIVE_MODULES = ("torch._inductor", "torch._dynamo")

_PRELUDE_MODULE_ERROR = "Settings prelude did not produce a valid module"


@dataclass(frozen=True)
class SettingsPlan:
    applied_env: dict[str, str] = field(default_factory=dict)
    applied_api: list[str] = field(default_factory=list)
    unset_env: list[str] = field(default_factory=list)
    excluded: list[dict[str, str]] = field(default_factory=list)
    untranslated: list[dict[str, str]] = field(default_factory=list)


def plan_settings(
    extra_envs: Mapping[str, Any] | None,
    *,
    unset_envs: Sequence[str] = (),
) -> SettingsPlan:
    """Classify accepted env settings into source-renderable and manifest-only buckets."""
    plan = SettingsPlan()
    unset_canonical = {str(name).strip().upper() for name in unset_envs}
    for raw_name, raw_value in (extra_envs or {}).items():
        name = str(raw_name).strip()
        canonical = name.upper()
        value = str(raw_value)
        if canonical in unset_canonical:
            plan.unset_env.append(name)
        elif is_secret_shaped_env_name(name):
            raise ValueError(f"Accepted settings require an external credential: {name}")
        elif canonical in BLOCKED_EXTERNAL_ENV_NAMES:
            plan.excluded.append({"name": name, "value": value, "reason": "benchmark_control"})
        elif canonical in _TORCH_API_TRANSLATIONS:
            # Also set the env var: torch caches it on first read, so the API call isn't always enough.
            plan.applied_api.append(_render_torch_api_call(canonical, raw_value))
            plan.applied_env[name] = value
        elif isinstance(raw_value, str) and raw_value.startswith("/"):
            plan.untranslated.append({"name": name, "value": value, "reason": "unportable_absolute_path"})
        else:
            plan.applied_env[name] = value
    added_canonical = {n.upper() for n in plan.unset_env}
    for raw_name in unset_envs:
        name = str(raw_name).strip()
        canonical = name.upper()
        if canonical not in added_canonical:
            plan.unset_env.append(name)
            added_canonical.add(canonical)
    return plan


def render_prelude(plan: SettingsPlan) -> str:
    """Render a deterministic Python prelude for the settings that can be applied in-process."""
    if not plan.applied_env and not plan.unset_env and not plan.applied_api:
        return ""

    lines = [
        "# --- hyperloom accepted settings (generated at export; do not edit) ---",
        "import os as _hl_os",
        "",
    ]

    # An applied_env entry with no _TORCH_API_TRANSLATIONS backup is the only kind actually
    # at risk if this module's first import happens too late to matter.
    has_unprotected_env = any(name.upper() not in _TORCH_API_TRANSLATIONS for name in plan.applied_env)
    if has_unprotected_env:
        lines.extend(_render_import_timing_guard())

    emitted_env_loop = False
    emitted_unset_loop = False
    if plan.applied_env:
        entries = [
            _tuple_literal((json.dumps(name), json.dumps(value))) for name, value in sorted(plan.applied_env.items())
        ]
        lines.extend(
            [
                f"for _hl_k, _hl_v in {_tuple_literal(entries)}:",
                "    _hl_os.environ[_hl_k] = _hl_v",
            ]
        )
        emitted_env_loop = True
    if plan.unset_env:
        entries = [json.dumps(name) for name in sorted(set(plan.unset_env))]
        lines.extend(
            [
                f"for _hl_k in {_tuple_literal(entries)}:",
                "    _hl_os.environ.pop(_hl_k, None)",
            ]
        )
        emitted_unset_loop = True
    if emitted_env_loop:
        lines.append("del _hl_k, _hl_v")
    elif emitted_unset_loop:
        lines.append("del _hl_k")

    if plan.applied_api:
        lines.extend(
            [
                "",
                "",
                "def _hl_apply_runtime_settings() -> None:",
                '    """Settings an environment variable can no longer reach once torch is loaded."""',
                "    try:",
                "        import torch",
                "    except Exception:",
                "        return",
            ]
        )
        for call in sorted(plan.applied_api):
            lines.extend(
                [
                    "    try:",
                    f"        {call}",
                    "    except Exception:",
                    "        pass",
                ]
            )
        lines.extend(["", "", "_hl_apply_runtime_settings()"])
    lines.append("# --- end hyperloom accepted settings ---")
    return "\n".join(lines)


def apply_settings_prelude(source: bytes, plan: SettingsPlan, function: str) -> bytes:
    """Splice the rendered settings prelude into a single-file optimization module."""
    prelude = render_prelude(plan)
    if not prelude:
        return source

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(_PRELUDE_MODULE_ERROR) from exc

    insertion_line = _insertion_line(tree)
    text = source.decode()
    lines = text.splitlines(keepends=True)
    before = "".join(lines[:insertion_line])
    after = "".join(lines[insertion_line:])
    prefix = "\n" if before and before.endswith("\n") else ("\n\n" if before else "")
    rendered = f"{before}{prefix}{prelude}\n\n{after}"
    out = rendered.encode()

    try:
        new_tree = ast.parse(out)
    except SyntaxError as exc:
        raise ValueError(_PRELUDE_MODULE_ERROR) from exc
    if not any(isinstance(node, ast.FunctionDef) and node.name == function for node in new_tree.body):
        raise ValueError(_PRELUDE_MODULE_ERROR)
    return out


def _render_import_timing_guard() -> list[str]:
    """Lines that raise if this module's first import happened too late to matter."""
    return [
        "import sys as _hl_sys",
        "_hl_torch = _hl_sys.modules.get('torch')",
        f"if any(_hl_m in _hl_sys.modules for _hl_m in {_IMPORT_SENSITIVE_MODULES!r}) or (",
        "    _hl_torch is not None and _hl_torch.cuda.is_initialized()",
        "):",
        "    raise RuntimeError(",
        "        f'{__name__} was imported after the GPU was initialized or torch._inductor/_dynamo '",
        "        'were loaded, so its settings cannot take effect. Move its import above `import torch`.'",
        "    )",
        "del _hl_sys, _hl_torch",
        "",
    ]


def _render_torch_api_call(name: str, value: object) -> str:
    function_name, coerce = _TORCH_API_TRANSLATIONS[name]
    return f"{function_name}({coerce(value)!r})"


def _tuple_literal(entries: Sequence[str]) -> str:
    suffix = "," if len(entries) == 1 else ""
    return f"({', '.join(entries)}{suffix})"


def _insertion_line(tree: ast.Module) -> int:
    safe_lines: list[int] = []
    if tree.body:
        first = tree.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
        ):
            safe_lines.append(first.end_lineno)
    safe_lines.extend(
        node.end_lineno
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__" and node.end_lineno is not None
    )
    return max(safe_lines, default=0)


__all__ = [
    "SettingsPlan",
    "apply_settings_prelude",
    "plan_settings",
    "render_prelude",
]
