"""The build-attempt projection the enablement section reports."""

from __future__ import annotations

from typing import Any


def build_attempt_summary(manifest_entry: dict[str, Any]) -> dict[str, Any]:
    """Project a ``BuildResult.to_state()`` entry onto a build-attempt summary."""
    installed = manifest_entry.get("installed_versions") or {}
    probes = manifest_entry.get("build_probes") or []
    return {
        "component": str(manifest_entry.get("component") or ""),
        "ref": str(installed.get("aiter_ref") or installed.get("vllm_ref") or installed.get("sgl_kernel_ref") or ""),
        "gpu_arch": str(installed.get("arch") or ""),
        "max_jobs": 0,
        "ok": bool(manifest_entry.get("ok")),
        "failure_class": str(manifest_entry.get("failure_class") or "ok"),
        "failure_summary": str(manifest_entry.get("failure_summary") or ""),
        "installed_versions": {str(k): str(v) for k, v in installed.items()} if isinstance(installed, dict) else {},
        "build_probes": [str(p) for p in probes[:8]],
        "build_log_path": str(manifest_entry.get("build_log_path") or ""),
        "attempt_root": str(manifest_entry.get("attempt_root") or ""),
    }
