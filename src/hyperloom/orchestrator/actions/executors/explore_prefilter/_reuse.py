# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Skip an EXPLORE variant whose exact launch was already measured and reverted.

A decision round measures one launch on one stack. When the performance gate
reverts it, a later round proposing the same launch on the same stack -- in
this session or another -- would boot the same server to reach the same
verdict. The store keeps one record per reverted launch, keyed by everything
that decides what the server ran and how it was graded:

* the variant and the stack under it, as launched (args, envs, removals,
  modes, the inherited recipe args);
* the workload contract and the base benchmark block it was materialized from;
* model, engine, GPU, and the image digest / engine / ROCm / AITER versions
  the session recorded at boot;
* the content of every kernel overlay on the server's ``PYTHONPATH``;
* the decision protocol (warm or cold decision round), the grading axes and
  the KEEP threshold.

A hit only ever removes the variant from the round. Nothing is replayed, so a
stored number never reaches KEEP/REVERT, the anchor, the journal or the KB,
and a KEEP or a stack revalidation always rests on a measurement taken now.
Rounds that kept, failed, or reverted on accuracy are not stored. Anything
that cannot be pinned is not stored or matched: no image or engine version on
record, a runtime override, or an overlay too large to hash.

Off unless ``HYPERLOOM_MEASUREMENT_STORE`` names a directory. Records older
than ``HYPERLOOM_MEASUREMENT_MAX_AGE_DAYS`` (14) are not used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ENV_STORE = "HYPERLOOM_MEASUREMENT_STORE"
ENV_MAX_AGE_DAYS = "HYPERLOOM_MEASUREMENT_MAX_AGE_DAYS"
DEFAULT_MAX_AGE_DAYS = 14.0
SCHEMA = 2
OUTCOME_REVERT = "REVERT"
# A launch digest when an overlay is too large to hash; it matches nothing recorded.
LAUNCH_UNPINNED = "unpinned"

_OVERLAY_MAX_FILES = 5000
_OVERLAY_MAX_BYTES = 256 * 1024 * 1024
# Envs that name where things are rather than what runs.
_LOCATION_ENV = re.compile(r"(DIR|PATH|LOG|FILE|PORT|HOST|URL|TOKEN|SECRET|PASSWORD)")
_RUNTIME_KEYS = (("image_digest", "image_digest"), ("rocm", "rocm"), ("aiter", "aiter"))
_PYTHONPATH = "PYTHONPATH"
# Envs the benchmark harness reads for the workload and parallelism, as opposed to launch levers.
HARNESS_ENVS = frozenset({"MODEL", "TP", "EP", "PP", "CONC", "ISL", "OSL", "NUM_PROMPTS"})


def store_root() -> Path | None:
    raw = str(os.environ.get(ENV_STORE) or "").strip()
    return Path(raw) if raw else None


def reuse_enabled() -> bool:
    return store_root() is not None


def _max_age_sec() -> float:
    try:
        days = float(os.environ.get(ENV_MAX_AGE_DAYS, DEFAULT_MAX_AGE_DAYS))
    except (TypeError, ValueError):
        days = DEFAULT_MAX_AGE_DAYS
    return max(0.0, days) * 86400.0


def runtime_identity(stack_meta: dict[str, Any] | None, framework: str) -> dict[str, str]:
    """Image digest and versions from ``SharedState.stack_fingerprint_meta``."""
    meta = stack_meta if isinstance(stack_meta, dict) else {}
    out: dict[str, str] = {}
    version = str(meta.get((framework or "").lower()) or "").strip()
    if version and version != "unknown":
        out["framework_version"] = version
    for src, dst in _RUNTIME_KEYS:
        value = str(meta.get(src) or "").strip()
        if value and value != "unknown":
            out[dst] = value
    return out


def _hash_tree(root: Path) -> str | None:
    """Content digest of a directory; None when it is too large to pin.

    Read every time rather than memoised on file stats: an overlay rewritten
    in place within the filesystem's mtime granularity would keep its digest.
    """
    listing: list[str] = []
    size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(filenames):
                if name.endswith((".pyc", ".lock")):
                    continue
                path = Path(dirpath) / name
                size += path.stat().st_size
                listing.append(str(path.relative_to(root)))
                if len(listing) > _OVERLAY_MAX_FILES or size > _OVERLAY_MAX_BYTES:
                    return None
        digest = hashlib.sha256()
        for rel in listing:
            digest.update(rel.encode())
            digest.update((root / rel).read_bytes())
    except OSError:
        return None
    return digest.hexdigest()


def overlay_identity(pythonpath_entries: list[str]) -> list[str] | None:
    """Content digests of the directories on the server's ``PYTHONPATH``, in order.

    Digests, not paths: the same overlay authored into another session's
    directory is the same kernel.
    """
    out: list[str] = []
    for entry in dict.fromkeys(e.strip() for e in pythonpath_entries if e and e.strip()):
        path = Path(entry)
        if not path.is_dir():
            continue
        digest = _hash_tree(path)
        if digest is None:
            return None
        out.append(digest)
    return out


def _is_location(key: Any, value: Any, roots: list[str]) -> bool:
    text = str(value)
    return bool(_LOCATION_ENV.search(str(key).upper())) or any(r in text for r in roots if r)


def launch_digest(envs: dict[str, Any], priced: frozenset[str]) -> str:
    """Digest of a launch's envs and kernel overlays, leaving out ``priced`` envs and locations."""
    overlays = overlay_identity(str(envs.get(_PYTHONPATH) or "").split(":"))
    if overlays is None:
        return LAUNCH_UNPINNED
    kept = {
        str(k): str(v)
        for k, v in sorted(envs.items())
        if str(k).upper() not in priced and not _is_location(k, v, []) and not str(v).startswith("/")
    }
    return hashlib.sha256(json.dumps([kept, overlays], sort_keys=True).encode()).hexdigest()[:16]


def benchmark_identity(bench: dict[str, Any], *, volatile_roots: list[str]) -> dict[str, Any]:
    """The base benchmark block minus where it writes."""
    envs = {
        str(key): str(value)
        for key, value in sorted((bench.get("envs") or {}).items())
        if not _is_location(key, value, volatile_roots)
    }
    return {
        "framework": str(bench.get("framework") or "").lower(),
        "model": str(bench.get("model") or ""),
        "precision": str(bench.get("precision") or ""),
        "envs": envs,
    }


def measurement_identity(
    session: dict[str, Any], stack: dict[str, Any], variant: dict[str, Any], pythonpath_entries: list[str]
) -> dict[str, Any] | None:
    """Everything a measurement depends on, or None when it cannot be pinned.

    ``session`` carries what every launch in the round shares: model, engine,
    GPU, workload signature, runtime versions, base benchmark block, inherited
    args, decision protocol and grading.
    """
    runtime = session.get("runtime") or {}
    if not (runtime.get("image_digest") and runtime.get("framework_version")):
        return None
    overlays = overlay_identity(pythonpath_entries)
    if overlays is None:
        return None
    return {"schema": SCHEMA, **session, "stack": stack, "variant": variant, "overlays": overlays}


def measurement_key(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def _record_path(root: Path, key: str) -> Path:
    return root / key[:2] / f"{key}.json"


def lookup_revert(key: str, *, now: float | None = None) -> dict[str, Any] | None:
    """The fresh stored REVERT for ``key``, or None."""
    root = store_root()
    if root is None or not key:
        return None
    try:
        record = json.loads(_record_path(root, key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("schema") != SCHEMA or record.get("outcome") != OUTCOME_REVERT:
        return None
    age = (now if now is not None else time.time()) - float(record.get("ts_epoch") or 0.0)
    if age < 0 or age > _max_age_sec():
        return None
    return record


def record_revert(key: str, identity: dict[str, Any], fields: dict[str, Any]) -> None:
    """Store a reverted decision round under ``key``; a re-measurement replaces it."""
    root = store_root()
    if root is None or not key:
        return
    doc = {
        **fields,
        "schema": SCHEMA,
        "key": key,
        "identity": identity,
        "outcome": OUTCOME_REVERT,
        "ts_epoch": time.time(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = _record_path(root, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("explore reuse: could not record %s (%s)", path, exc)
