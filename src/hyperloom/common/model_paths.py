# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared model-path resolution: a ``--model`` value -> a local model directory.

The CLI ``--model`` accepts a local path OR a HuggingFace repo id and is
persisted verbatim into ``state.model_path``. In-process metadata readers
(roofline ceiling, model-config summary, KB tags, model-class inference, fp8
detection) need a real directory; a bare repo id makes them silently degrade.
This is the single source of truth for turning either form into a local dir,
reusing the serving engine's own HF hub cache for repo ids -- never a hardcoded
models root.

Zero first-party dependency (stdlib + optional ``huggingface_hub``), so any
package may import it without a cycle. Standalone kernel-agent tools that cannot
import ``hyperloom.common`` (the Ray/subprocess ``sys.path`` contract documented
in ``hyperloom.common.__init__``) mirror this same strategy independently.
"""

from __future__ import annotations

from pathlib import Path


def resolve_local_model_dir(model: str | Path | None) -> Path | None:
    """Resolve a ``--model`` value (local path OR HF repo id) to a local dir.

    * **Local path** -- an existing directory is returned unchanged.
    * **HF repo id** (e.g. ``Qwen/Qwen3-0.6B``) -- resolved to the serving
      engine's HF hub cache via ``huggingface_hub.try_to_load_from_cache``
      (honoring ``HF_HOME`` / ``HF_HUB_CACHE``). The snapshot dir carries a
      non-derivable commit-hash segment, so ``huggingface_hub`` locates it
      rather than string-building a path -- and no models root is hardcoded.

    Args:
        model: A model directory path or a HuggingFace repo id.

    Returns:
        The resolved local model directory, or ``None`` when unresolved (repo id
        neither a dir nor cached, or ``huggingface_hub`` unavailable), so callers
        keep their existing degrade path.
    """
    raw = ("" if model is None else str(model)).strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    try:
        is_dir = p.is_dir()
    except OSError:
        # Permission denied or other OS error — treat as not a local dir and
        # fall through to the HF hub cache probe.
        is_dir = False
    if is_dir:
        return p
    # Repo id: reuse the engine's HF hub cache. Lazy import keeps this module
    # dependency-light -- a missing huggingface_hub just degrades to None.
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 -- optional dep; degrade to no-resolution.
        return None
    try:
        hit = try_to_load_from_cache(repo_id=raw, filename="config.json")
    except Exception:  # noqa: BLE001 -- cache probe is best-effort.
        return None
    if isinstance(hit, str) and Path(hit).is_file():
        return Path(hit).parent
    return None
