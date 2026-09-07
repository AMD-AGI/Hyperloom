# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Typed data model for attempt-scoped runtime acquisition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


# Acquisition-method vocabulary accepted by ``from_state``; compiled builds are deferred to the targeted-build path.
_ACQUISITION_METHODS: frozenset[str] = frozenset({"wheel", "editable_ref", "local_tree", "package_source", "none"})


@dataclass(frozen=True)
class FrameworkRuntime:
    """The explicit runtime the bench subprocess must resolve to."""

    bin_path: str = ""
    python_path: str = ""
    venv_root: str = ""
    pythonpath_prefix: str = ""
    server_args: str = ""
    envs: Mapping[str, str] = field(default_factory=dict)
    pythonpath_prefixes: tuple[str, ...] = ()
    ld_library_path_prefix: tuple[str, ...] = ()
    runtime_env: Mapping[str, str] = field(default_factory=dict)
    entrypoint_bin_dir: str = ""
    runtime_python_exe: str = ""
    source_root: str = ""
    attempt_root: str = ""

    def to_runtime_override(self) -> dict[str, Any]:
        """Project onto the dict consumed by ``apply_runtime_override``."""
        out: dict[str, Any] = {}
        if self.bin_path:
            out["path_prefix"] = self.bin_path
        if self.pythonpath_prefix:
            out["pythonpath_prefix"] = self.pythonpath_prefix
        if self.bin_path:
            out["framework_bin"] = self.bin_path
        if self.python_path:
            out["framework_python"] = self.python_path
        if self.venv_root:
            out["framework_venv_root"] = self.venv_root
        if self.pythonpath_prefixes:
            out["pythonpath_prefixes"] = list(self.pythonpath_prefixes)
        if self.ld_library_path_prefix:
            out["ld_library_path_prefix"] = list(self.ld_library_path_prefix)
        if self.runtime_env:
            out["runtime_env"] = dict(self.runtime_env)
        if self.entrypoint_bin_dir:
            out["entrypoint_bin_dir"] = self.entrypoint_bin_dir
        if self.runtime_python_exe:
            out["runtime_python_exe"] = self.runtime_python_exe
        return out

    def to_state(self) -> dict[str, Any]:
        """Serialize to a plain dict for shared-state / params."""
        return {
            "bin_path": self.bin_path,
            "python_path": self.python_path,
            "venv_root": self.venv_root,
            "pythonpath_prefix": self.pythonpath_prefix,
            "server_args": self.server_args,
            "envs": dict(self.envs),
            "pythonpath_prefixes": list(self.pythonpath_prefixes),
            "ld_library_path_prefix": list(self.ld_library_path_prefix),
            "runtime_env": dict(self.runtime_env),
            "entrypoint_bin_dir": self.entrypoint_bin_dir,
            "runtime_python_exe": self.runtime_python_exe,
            "source_root": self.source_root,
            "attempt_root": self.attempt_root,
        }

    @classmethod
    def from_state(cls, d: Mapping[str, Any] | None) -> "FrameworkRuntime":
        """Rehydrate from a plain dict; missing keys default to empty."""
        d = d or {}

        def _str_map(key: str) -> dict[str, str]:
            raw = d.get(key)
            return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}

        def _str_tuple(key: str) -> tuple[str, ...]:
            raw = d.get(key)
            return tuple(str(x) for x in raw) if isinstance(raw, (list, tuple)) else ()

        return cls(
            bin_path=str(d.get("bin_path") or ""),
            python_path=str(d.get("python_path") or ""),
            venv_root=str(d.get("venv_root") or ""),
            pythonpath_prefix=str(d.get("pythonpath_prefix") or ""),
            server_args=str(d.get("server_args") or ""),
            envs=_str_map("envs"),
            pythonpath_prefixes=_str_tuple("pythonpath_prefixes"),
            ld_library_path_prefix=_str_tuple("ld_library_path_prefix"),
            runtime_env=_str_map("runtime_env"),
            entrypoint_bin_dir=str(d.get("entrypoint_bin_dir") or ""),
            runtime_python_exe=str(d.get("runtime_python_exe") or ""),
            source_root=str(d.get("source_root") or ""),
            attempt_root=str(d.get("attempt_root") or ""),
        )


@dataclass(frozen=True)
class EnablementStackAction:
    """A candidate attempt-runtime acquisition the enablement loop may run."""

    kind: str
    framework: str
    gap_id: str
    capability: str
    reason: str = ""
    acquisition_method: str = "none"
    repo_url: str = ""
    ref: str = ""
    index_url: str = ""
    packages: tuple[str, ...] = ()
    expected_symbols: tuple[str, ...] = ()
    expected_files: tuple[str, ...] = ()
    server_args: str = ""
    envs: Mapping[str, str] = field(default_factory=dict)
    attempt_venv_root: str = ""
    pr_number: int = 0
    localized_paths: tuple[str, ...] = ()

    def to_state(self) -> dict[str, Any]:
        """Serialize to a plain dict for task params / shared state."""
        return {
            "kind": self.kind,
            "framework": self.framework,
            "gap_id": self.gap_id,
            "capability": self.capability,
            "reason": self.reason,
            "acquisition_method": self.acquisition_method,
            "repo_url": self.repo_url,
            "ref": self.ref,
            "index_url": self.index_url,
            "packages": list(self.packages),
            "expected_symbols": list(self.expected_symbols),
            "expected_files": list(self.expected_files),
            "server_args": self.server_args,
            "envs": dict(self.envs),
            "attempt_venv_root": self.attempt_venv_root,
            "pr_number": self.pr_number,
            "localized_paths": list(self.localized_paths),
        }

    @classmethod
    def from_state(cls, d: Mapping[str, Any] | None) -> "EnablementStackAction":
        """Rehydrate from a plain dict; missing keys default sensibly."""
        d = d or {}
        raw_envs = d.get("envs")
        envs = {str(k): str(v) for k, v in raw_envs.items()} if isinstance(raw_envs, dict) else {}

        def _tuple(key: str) -> tuple[str, ...]:
            raw = d.get(key)
            return tuple(str(x) for x in raw) if isinstance(raw, (list, tuple)) else ()

        method = str(d.get("acquisition_method") or "none")
        if method not in _ACQUISITION_METHODS:
            method = "none"
        try:
            pr_number = int(d.get("pr_number") or 0)
        except (TypeError, ValueError):
            pr_number = 0
        return cls(
            kind=str(d.get("kind") or "runtime_candidate"),
            framework=str(d.get("framework") or "").strip().lower(),
            gap_id=str(d.get("gap_id") or ""),
            capability=str(d.get("capability") or ""),
            reason=str(d.get("reason") or ""),
            acquisition_method=method,
            repo_url=str(d.get("repo_url") or ""),
            ref=str(d.get("ref") or ""),
            index_url=str(d.get("index_url") or ""),
            packages=_tuple("packages"),
            expected_symbols=_tuple("expected_symbols"),
            expected_files=_tuple("expected_files"),
            server_args=str(d.get("server_args") or ""),
            envs=envs,
            attempt_venv_root=str(d.get("attempt_venv_root") or ""),
            pr_number=pr_number,
            localized_paths=_tuple("localized_paths"),
        )


@dataclass(frozen=True)
class ProvisionResult:
    """Outcome of provisioning one :class:`EnablementStackAction`."""

    ok: bool
    runtime: FrameworkRuntime = field(default_factory=FrameworkRuntime)
    installed_versions: Mapping[str, str] = field(default_factory=dict)
    log_path: str = ""
    error: str = ""

    def to_state(self) -> dict[str, Any]:
        """Serialize to a plain dict for shared state / observability."""
        return {
            "ok": self.ok,
            "runtime": self.runtime.to_state(),
            "installed_versions": dict(self.installed_versions),
            "log_path": self.log_path,
            "error": self.error,
        }

    @classmethod
    def from_state(cls, d: Mapping[str, Any] | None) -> "ProvisionResult":
        """Rehydrate from a plain dict."""
        d = d or {}
        raw_versions = d.get("installed_versions")
        versions = {str(k): str(v) for k, v in raw_versions.items()} if isinstance(raw_versions, dict) else {}
        return cls(
            ok=bool(d.get("ok")),
            runtime=FrameworkRuntime.from_state(d.get("runtime")),
            installed_versions=versions,
            log_path=str(d.get("log_path") or ""),
            error=str(d.get("error") or ""),
        )


__all__ = [
    "EnablementStackAction",
    "FrameworkRuntime",
    "ProvisionResult",
]
