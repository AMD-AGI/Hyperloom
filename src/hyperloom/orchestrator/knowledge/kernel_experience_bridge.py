# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hyperloom-side control-plane bridge for KernelForge experience knowledge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from .config import KnowledgeConfig


@dataclass(frozen=True)
class KernelExperienceStatus:
    """Secret-free capability/status projection for one KernelForge launch."""

    mode: str
    backend: str
    capability_expected: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "capability_expected": self.capability_expected,
            "status": self.status,
        }


@dataclass
class KernelExperienceBridge:
    """Configure KernelForge and collect its already-produced result metadata.

    The bridge intentionally owns no kernel-experience CRUD, ranking, or local
    knowledge implementation. KernelForge remains the data-plane owner.
    """

    config: KnowledgeConfig
    audit_hook: Any = None

    @property
    def status(self) -> KernelExperienceStatus:
        expected = self.config.mode.value in {"local", "remote"}
        return KernelExperienceStatus(
            mode=self.config.mode.value,
            backend=self.config.backend,
            capability_expected=expected,
            status="configured" if expected else "disabled",
        )

    def configure_child_env(self, env: MutableMapping[str, str]) -> KernelExperienceStatus:
        """Validate and configure a KernelForge subprocess environment."""

        self.config.apply_to_child_env(env)
        status = self.status
        self._emit(
            {
                "op": "kernel_experience_passthrough",
                "method": "configure_child_env",
                "mode": status.mode,
                "backend": status.backend,
                "resolution": status.status,
                "success": True,
                "provenance": {"component": "hyperloom", "target": "kernelforge"},
            }
        )
        return status

    def collect_result(self, result: Mapping[str, Any] | None) -> dict[str, Any]:
        """Collect a bounded, secret-free provenance projection from KernelForge."""

        payload = result if isinstance(result, Mapping) else {}
        experience = payload.get("kb_experience")
        exp = experience if isinstance(experience, Mapping) else {}
        read = exp.get("read") if isinstance(exp.get("read"), Mapping) else {}
        write = exp.get("write") if isinstance(exp.get("write"), Mapping) else {}
        collected = {
            **self.status.to_dict(),
            "result_present": bool(exp),
            "read_applied": bool(read.get("applied")),
            "write_attempted": bool(write) or bool(exp.get("write_attempted")),
            "experience_id": str(
                exp.get("experience_id")
                or read.get("experience_id")
                or write.get("experience_id")
                or ""
            ),
            "provenance": {
                "component": "kernelforge",
                "transport": "child_result",
            },
        }
        self._emit(
            {
                "op": "kernel_experience_passthrough",
                "method": "collect_result",
                "mode": self.config.mode.value,
                "backend": self.config.backend,
                "resolution": "collected" if exp else "not_reported",
                "success": True,
                "result": collected,
                "provenance": collected["provenance"],
            }
        )
        return collected

    def _emit(self, event: dict[str, Any]) -> None:
        if not callable(self.audit_hook):
            return
        try:
            self.audit_hook(event)
        except Exception:
            # Audit is observational and cannot break a forge attempt.
            return


__all__ = ["KernelExperienceBridge", "KernelExperienceStatus"]
