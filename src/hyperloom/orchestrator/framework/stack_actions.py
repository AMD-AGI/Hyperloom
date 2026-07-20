# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Typed data model for Rung 3 attempt-scoped runtime acquisition (M1).

An :class:`EnablementStackAction` describes an isolated runtime the enablement
loop may acquire to provide a missing framework capability (a wheel, an editable
checkout at a ref, or a local tree). A :class:`ProvisionResult` carries the
outcome of provisioning one action into an attempt venv, and the resolved
:class:`FrameworkRuntime` the bench subprocess must use.

All three types provide an explicit ``to_state`` / ``from_state`` boundary so
they cross the typed <-> dict (shared-state / task-params) seam deliberately,
never via ``dataclasses.asdict``. Pure-Python: no network, subprocess, or
filesystem access here (that lives in ``adapters.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


# Acquisition methods the M1 adapters support. Compiled builds are Rung 5
# (deferred): only wheel / editable-ref / local-tree / package-source here.
_ACQUISITION_METHODS: frozenset[str] = frozenset(
    {"wheel", "editable_ref", "local_tree", "package_source", "none"}
)
_ACTION_KINDS: frozenset[str] = frozenset({"runtime_candidate", "pr_backport", "vendor_files"})


@dataclass(frozen=True)
class FrameworkRuntime:
    """The explicit runtime the bench subprocess must resolve to.

    Attributes:
        bin_path: Attempt-local bin dir prepended to the YAML PATH (holds the
            server console script, e.g. ``.../venv/bin``).
        python_path: Attempt-local interpreter (``.../venv/bin/python``).
        venv_root: Attempt venv root (``$SESSION_DIR/enablement/stacks/...``).
        pythonpath_prefix: Optional single dir prepended to PYTHONPATH (an
            editable checkout's package dir); empty when a wheel install.
        server_args: Extra server args to route into ``EXTRA_{FW}_ARGS``.
        envs: Extra benchmark envs to merge (never mutates os.environ).
    """

    bin_path: str = ""
    python_path: str = ""
    venv_root: str = ""
    pythonpath_prefix: str = ""
    server_args: str = ""
    envs: Mapping[str, str] = field(default_factory=dict)

    def to_runtime_override(self) -> dict[str, str]:
        """Project onto the dict consumed by ``apply_runtime_override``.

        Maps this runtime onto the exact keys ``apply_runtime_override``
        recognizes (``path_prefix`` / ``pythonpath_prefix`` /
        ``framework_bin`` / ``framework_python`` / ``framework_venv_root``) so
        the attempt runtime lands in the materialized YAML ``benchmark.envs``.

        Returns:
            dict[str, str]: The runtime-override dict (empty values omitted).
        """
        out: dict[str, str] = {}
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
        }

    @classmethod
    def from_state(cls, d: Mapping[str, Any] | None) -> "FrameworkRuntime":
        """Rehydrate from a plain dict; missing keys default to empty."""
        d = d or {}
        raw_envs = d.get("envs")
        envs = {str(k): str(v) for k, v in raw_envs.items()} if isinstance(raw_envs, dict) else {}
        return cls(
            bin_path=str(d.get("bin_path") or ""),
            python_path=str(d.get("python_path") or ""),
            venv_root=str(d.get("venv_root") or ""),
            pythonpath_prefix=str(d.get("pythonpath_prefix") or ""),
            server_args=str(d.get("server_args") or ""),
            envs=envs,
        )


@dataclass(frozen=True)
class EnablementStackAction:
    """A candidate attempt-runtime acquisition the enablement loop may run.

    Attributes:
        kind: One of ``runtime_candidate`` / ``pr_backport`` / ``vendor_files``.
        framework: Target framework (``vllm`` / ``sglang`` / ...).
        gap_id: Canonical gap id (``gap.enablement.<failure_kind>``).
        capability: The missing capability being repaired (e.g. ``deepseek_v4``).
        reason: Human-readable justification / evidence summary.
        acquisition_method: How the runtime is acquired (see
            :data:`_ACQUISITION_METHODS`).
        repo_url: Origin-allowlisted git URL (empty unless editable/source).
        ref: Pinned ref (hash recorded); empty unless editable/source.
        index_url: Host-allowlisted pip index (empty unless wheel).
        packages: Pinned package specs to install.
        expected_symbols: Symbols the adapter probe must find post-provision.
        expected_files: Files the adapter probe must find post-provision.
        server_args: Extra server args routed to ``EXTRA_{FW}_ARGS``.
        envs: Extra benchmark envs to merge.
        attempt_venv_root: Attempt venv root; filled by the executor stage.
    """

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
        )


@dataclass(frozen=True)
class ProvisionResult:
    """Outcome of provisioning one :class:`EnablementStackAction`.

    Attributes:
        ok: Whether provision + probe succeeded.
        runtime: The resolved runtime the bench must use (empty on failure).
        installed_versions: Package -> version map recorded post-install.
        log_path: Path to the provision log (for observability).
        error: Failure reason when ``ok`` is False.
    """

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
        versions = (
            {str(k): str(v) for k, v in raw_versions.items()} if isinstance(raw_versions, dict) else {}
        )
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
