"""The build-attempt projection the recipe replays a build from.

This projection used to live in the Session Breakdown's read side, which
recorded the breakdown by re-reading state and disk. That side was retired
when the breakdown moved to recording at author time (#1455), and the recipe
is now its only consumer -- so it lives here, with the machinery that needs
it, rather than in a module that no longer has a reason to hold it.
"""

from __future__ import annotations

from typing import Any


def build_attempt_summary(manifest_entry: dict[str, Any]) -> dict[str, Any]:
    """Project a ``BuildResult.to_state()`` entry onto a build-attempt summary.

    Injected into :func:`build_recipe_steps`, which reads ``component``,
    ``ref``, ``gpu_arch`` and ``max_jobs`` off the result. Renaming one of
    those keys does not raise -- it silently empties the build step and the
    verdict then reports ``build_inputs_incomplete``.
    """
    action = manifest_entry.get("action") or {}
    installed = manifest_entry.get("installed_versions") or {}
    probes = manifest_entry.get("build_probes") or []
    return {
        "component": str(action.get("component") or manifest_entry.get("component") or ""),
        "ref": str(
            installed.get("aiter_ref")
            or installed.get("vllm_ref")
            or installed.get("sgl_kernel_ref")
            or action.get("ref")
            or ""
        ),
        "gpu_arch": str(installed.get("arch") or action.get("gpu_arch") or ""),
        "max_jobs": int(action.get("max_jobs") or 0),
        "ok": bool(manifest_entry.get("ok")),
        "failure_class": str(manifest_entry.get("failure_class") or "ok"),
        "failure_summary": str(manifest_entry.get("failure_summary") or ""),
        "installed_versions": {str(k): str(v) for k, v in installed.items()} if isinstance(installed, dict) else {},
        "build_probes": [str(p) for p in probes[:8]],
        "build_log_path": str(manifest_entry.get("build_log_path") or ""),
        "attempt_root": str(manifest_entry.get("attempt_root") or ""),
    }


#: ``EnablementRound`` fields the recipe projection and its decision read.
