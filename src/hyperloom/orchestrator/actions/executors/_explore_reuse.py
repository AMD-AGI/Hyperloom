# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Answer an EXPLORE variant from an exact past measurement instead of rerunning it.

A decision round measures one launch on one stack, and the gain it reports is
paired: the variant against the anchor it was graded on. When a later round
proposes the same launch on the same stack -- in this session or another --
that measurement already answers it. The store keeps one record per
measurement, keyed by everything that decides what the server ran and how it
was measured:

* the variant and the stack under it, as launched (args, envs, removals,
  modes, the inherited recipe args);
* the workload contract and the base benchmark block it was materialized from;
* model, engine, GPU, and the image digest / engine / ROCm / AITER versions
  the session recorded at boot;
* the content of every kernel overlay on the server's ``PYTHONPATH``;
* the decision protocol (warm or cold decision round).

Anything that cannot be pinned refuses reuse rather than guessing: no image or
engine version on record, a runtime override, or an overlay too large to hash.

A hit is replayed as the paired gain, not the raw throughput: the reused
throughput is the current anchor scaled by the recorded gain, so it goes
through the same KEEP/REVERT gates a fresh measurement would against the
anchor that is current now. Only throughput-graded sessions reuse; an
interactivity verdict depends on two axes the gain does not carry.

Off unless ``HYPERLOOM_MEASUREMENT_STORE`` names a directory. Records older
than ``HYPERLOOM_MEASUREMENT_MAX_AGE_DAYS`` (14) are not reused.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._grid_base import VariantResult

log = logging.getLogger(__name__)

ENV_STORE = "HYPERLOOM_MEASUREMENT_STORE"
ENV_MAX_AGE_DAYS = "HYPERLOOM_MEASUREMENT_MAX_AGE_DAYS"
DEFAULT_MAX_AGE_DAYS = 14.0
SCHEMA = 1

_OVERLAY_MAX_FILES = 5000
_OVERLAY_MAX_BYTES = 256 * 1024 * 1024
# Envs that name where things are rather than what runs.
_LOCATION_ENV = re.compile(r"(DIR|PATH|LOG|FILE|PORT|HOST|URL|TOKEN|SECRET|PASSWORD)")
_RUNTIME_KEYS = (("image_digest", "image_digest"), ("rocm", "rocm"), ("aiter", "aiter"))

_SCALED_FIELDS = ("output_throughput", "input_throughput", "total_token_throughput", "request_throughput")
_CARRIED_FIELDS = (
    "ttft_mean_ms",
    "tpot_mean_ms",
    "tpot_p90_ms",
    "e2el_mean_ms",
    "intvty_p90",
    "intvty_p50",
    "duration_seconds",
    "request_error_rate",
    "completed_requests",
)


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


def benchmark_identity(bench: dict[str, Any], *, volatile_roots: list[str]) -> dict[str, Any]:
    """The base benchmark block minus where it writes."""
    roots = [r for r in volatile_roots if r]
    envs = {}
    for key, value in sorted((bench.get("envs") or {}).items()):
        text = str(value)
        if _LOCATION_ENV.search(str(key).upper()) or any(r in text for r in roots):
            continue
        envs[str(key)] = text
    return {
        "framework": str(bench.get("framework") or "").lower(),
        "model": str(bench.get("model") or ""),
        "precision": str(bench.get("precision") or ""),
        "envs": envs,
    }


def measurement_identity(
    *,
    model: str,
    framework: str,
    gpu: str,
    workload_signature: str,
    runtime: dict[str, str],
    benchmark: dict[str, Any],
    inherited_args: str,
    stack: dict[str, Any],
    variant: dict[str, Any],
    pythonpath_entries: list[str],
    decision_protocol: str,
    runtime_override: Any = None,
) -> dict[str, Any] | None:
    """Everything a measurement depends on, or None when it cannot be pinned."""
    if runtime_override:
        return None
    if not (runtime.get("image_digest") or runtime.get("framework_version")):
        return None
    overlays = overlay_identity(pythonpath_entries)
    if overlays is None:
        return None
    return {
        "schema": SCHEMA,
        "model": model,
        "framework": (framework or "").lower(),
        "gpu": (gpu or "").lower(),
        "workload_signature": workload_signature,
        "runtime": dict(sorted(runtime.items())),
        "benchmark": benchmark,
        "inherited_args": inherited_args,
        "stack": stack,
        "variant": variant,
        "overlays": overlays,
        "decision_protocol": decision_protocol,
    }


def measurement_key(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def _record_path(root: Path, key: str) -> Path:
    return root / key[:2] / f"{key}.json"


@dataclass
class ReusedMeasurement:
    """A stored decision round that answers this variant."""

    key: str
    record: dict[str, Any]

    @property
    def gain_pct(self) -> float:
        return float(self.record["gain_pct"])

    @property
    def accuracy(self) -> float | None:
        value = self.record.get("accuracy")
        return float(value) if isinstance(value, (int, float)) else None

    def provenance(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "measured_at": self.record.get("ts"),
            "session_dir": self.record.get("session_dir"),
            "round_id": self.record.get("round_id"),
            "variant_name": self.record.get("variant_name"),
            "workspace": self.record.get("workspace"),
            "gain_pct": self.gain_pct,
            "base_tput": self.record.get("base_tput"),
            "tput": self.record.get("output_throughput"),
        }

    def as_result(self, *, name: str, extra_server_args: str, extra_envs: dict[str, str], anchor_tput: float) -> Any:
        """The stored round replayed against ``anchor_tput``: throughputs by the paired gain."""
        recorded_base = float(self.record.get("base_tput") or 0.0)
        factor = anchor_tput / recorded_base if recorded_base > 0 else 0.0
        fields: dict[str, Any] = {}
        for f in _SCALED_FIELDS:
            value = self.record.get(f)
            fields[f] = float(value) * factor if isinstance(value, (int, float)) else None
        for f in _CARRIED_FIELDS:
            fields[f] = self.record.get(f)
        return VariantResult(
            name=name,
            extra_server_args=extra_server_args,
            extra_envs=dict(extra_envs),
            status="succeeded",
            workspace=self.record.get("workspace"),
            note="reused_measurement",
            **fields,
        )


def lookup(
    key: str, *, accuracy_required: bool, keep_threshold_pct: float, now: float | None = None
) -> ReusedMeasurement | None:
    """The stored measurement for ``key`` when it is fresh and complete enough.

    A session that gates on accuracy needs the recorded score, except for a
    gain below ``keep_threshold_pct``: that reverts before accuracy is read.
    """
    root = store_root()
    if root is None:
        return None
    try:
        record = json.loads(_record_path(root, key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return None
    age = (now if now is not None else time.time()) - float(record.get("ts_epoch") or 0.0)
    if age < 0 or age > _max_age_sec():
        return None
    if not isinstance(record.get("gain_pct"), (int, float)) or not float(record.get("base_tput") or 0.0) > 0:
        return None
    if (
        accuracy_required
        and not isinstance(record.get("accuracy"), (int, float))
        and float(record["gain_pct"]) >= keep_threshold_pct
    ):
        return None
    return ReusedMeasurement(key=key, record=record)


def record(key: str, identity: dict[str, Any], result: Any, *, anchor_tput: float, **context: Any) -> None:
    """Store a fresh decision round under ``key``; a re-measurement replaces it."""
    root = store_root()
    tput = float(getattr(result, "output_throughput", 0.0) or 0.0)
    if root is None or anchor_tput <= 0 or tput <= 0:
        return
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "key": key,
        "identity": identity,
        "ts_epoch": time.time(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_tput": anchor_tput,
        "gain_pct": (tput / anchor_tput - 1.0) * 100.0,
    }
    for f in (*_SCALED_FIELDS, *_CARRIED_FIELDS):
        doc[f] = getattr(result, f, None)
    doc.update(context)
    path = _record_path(root, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("explore reuse: could not record %s (%s)", path, exc)
