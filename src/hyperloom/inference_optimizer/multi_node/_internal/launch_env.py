# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Immutable launch environment shared by reuse checks and remote submission."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class LaunchEnv:
    """Resolved launch inputs; the digest is request identity, not readiness."""

    forward_env: Mapping[str, str] = field(repr=False)
    unset: tuple[str, ...] = ()
    profiler_dir: str = ""
    log_dir: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "forward_env", MappingProxyType(dict(self.forward_env)))
        object.__setattr__(self, "unset", tuple(sorted(set(self.unset))))

    @property
    def digest(self) -> str:
        payload = {
            "env": dict(self.forward_env),
            "unset": self.unset,
            "profiler_dir": self.profiler_dir,
            "log_dir": self.log_dir,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def expand_env_vars(value: str, source: Mapping[str, str]) -> str:
    """Expand POSIX $VAR/${VAR} paths using only the captured controller env."""
    return re.sub(
        r"\$(\w+|\{[^}]*\})",
        lambda match: source.get(match.group(1).strip("{}"), match.group(0)),
        value,
        flags=re.ASCII,
    )
