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


def _identity_leaf(seg: str) -> str:
    """Reduce one identity segment to its ``org/repo`` (or bare ``repo``) form.

    Strips a ``models--`` cache prefix and canonicalizes the ``org--repo`` cache
    separator to ``org/repo``; a segment without org info stays bare.
    """
    s = seg.strip().strip("/").split("/")[-1]
    if s.startswith("models--"):
        s = s[len("models--"):]
    return s.replace("--", "/")


def _identity_bare(full: str) -> str:
    """Return the repo-only tail of an ``org/repo`` identity (or the value)."""
    return full.rsplit("/", 1)[-1] if "/" in full else full


def model_identity_candidates(model: str | Path | None) -> tuple[set[str], set[str]]:
    """Return ``(full, bare)`` casefolded identity candidate sets.

    ``full`` keeps the ``org/repo`` qualification (e.g. ``qwen/qwen2.5-7b``);
    ``bare`` is the repo tail only (``qwen2.5-7b``). Covers flat dirs, bare
    names, HF repo ids, and HF hub cache paths whose ``snapshots/<hash>``
    basename hides the repo name in an upstream ``models--org--repo`` segment.
    """
    raw = ("" if model is None else str(model)).strip()
    if not raw:
        return set(), set()
    p = Path(raw)
    segments = [p.name]
    # HF hub cache hides the repo name in a mid-path models--org--repo segment;
    # the snapshots/<hash> basename alone cannot recover it.
    segments += [part for part in p.parts if part.startswith("models--")]
    full = {_identity_leaf(s) for s in segments if s}
    full = {f for f in full if f}
    bare = {_identity_bare(f) for f in full}
    return {f.casefold() for f in full}, {b.casefold() for b in bare}


def model_identities_match(declared: str, *launched: str) -> bool:
    """Whether ``declared`` names the same model as any ``launched`` value.

    Prefers a fully-qualified ``org/repo`` match. Falls back to the repo-only
    (bare) name ONLY when at least one side lacks org info — so two different
    orgs sharing a repo name (``a/Llama-8B`` vs ``b/Llama-8B``) do NOT match,
    while a declared clean name (``Llama-8B``, no org) still matches its
    launched ``org/repo`` form.
    """
    d_full, d_bare = model_identity_candidates(declared)
    l_full: set[str] = set()
    l_bare: set[str] = set()
    for value in launched:
        vf, vb = model_identity_candidates(value)
        l_full |= vf
        l_bare |= vb
    if d_full & l_full:
        return True
    # Both sides carry org qualification but no full match -> distinct models.
    if any("/" in x for x in d_full) and any("/" in x for x in l_full):
        return False
    return bool(d_bare & l_bare)


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
