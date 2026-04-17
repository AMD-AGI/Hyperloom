"""cross_session_kb.py — persistent cross-session knowledge base with semantic fingerprinting.

Stores optimization knowledge that persists across sessions and transfers
across models. Uses semantic kernel fingerprints to identify similar kernels
across different model architectures.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_KB_DIR = "/shared_nfs/nehaprakriya/TBO/global_kb"


@dataclass
class KernelFingerprint:
    """Semantic fingerprint for a GPU kernel — identifies similar kernels across models."""
    op_type: str = ""
    shape_class: str = ""
    backend: str = ""
    precision: str = ""
    scheduling: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        key = f"{self.op_type}|{self.shape_class}|{self.backend}|{self.precision}|{self.scheduling}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @staticmethod
    def from_kernel_info(info: dict[str, Any]) -> "KernelFingerprint":
        """Extract fingerprint from kernel analysis metadata."""
        name = info.get("kernel_name", "").lower()
        source = info.get("source_file", "").lower()
        shapes = info.get("shapes", [])

        op_type = "unknown"
        if "gemm" in name or "matmul" in name:
            op_type = "gemm"
        elif "attention" in name or "mha" in name or "mla" in name:
            op_type = "attention"
        elif "norm" in name or "rmsnorm" in name or "layernorm" in name:
            op_type = "normalization"
        elif "moe" in name or "expert" in name or "gate" in name:
            op_type = "moe-dispatch"
        elif "reduce" in name or "allreduce" in name:
            op_type = "collective"
        elif "quant" in name:
            op_type = "quantization"
        elif "softmax" in name:
            op_type = "softmax"
        elif "embedding" in name:
            op_type = "embedding"

        shape_class = _classify_shape(shapes)

        backend = "unknown"
        if "triton" in source:
            backend = "triton"
        elif "ck" in source or "composable_kernel" in source:
            backend = "ck"
        elif "hip" in source or "cuda" in source:
            backend = "hip"
        elif "aiter" in source:
            backend = "aiter"

        precision = info.get("precision", "fp16")
        scheduling = info.get("scheduling", "")

        return KernelFingerprint(
            op_type=op_type,
            shape_class=shape_class,
            backend=backend,
            precision=precision,
            scheduling=scheduling,
            tags=info.get("tags", []),
        )


def _classify_shape(shapes: list[str]) -> str:
    """Classify GEMM shapes into categories for cross-model matching."""
    if not shapes:
        return "unknown"
    for s in shapes:
        parts = s.replace("x", ",").split(",")
        try:
            dims = [int(p.strip()) for p in parts if p.strip().isdigit()]
        except ValueError:
            continue
        if len(dims) >= 2:
            m, n = dims[0], dims[1]
            if m <= 16:
                return "decode-small-m"
            elif m <= 128:
                return "decode-medium-m"
            elif m <= 1024:
                return "prefill-medium"
            else:
                return "prefill-large"
    return "unknown"


@dataclass
class KBEntry:
    """A knowledge base entry with semantic fingerprint."""
    id: str = ""
    fingerprint: str = ""
    session_id: str = ""
    model_name: str = ""
    gpu_type: str = ""
    action_type: str = ""
    target_kernel: str = ""
    outcome: str = ""
    gain_pct: float = 0.0
    approach: str = ""
    lesson: str = ""
    failure_reason: str = ""
    confidence: float = 0.5
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CrossSessionKB:
    """Persistent knowledge base that transfers across sessions and models."""

    def __init__(self, kb_dir: str = DEFAULT_KB_DIR):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self._entries_path = self.kb_dir / "entries.jsonl"
        self._fingerprint_index: dict[str, list[KBEntry]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load fingerprint index from disk."""
        self._fingerprint_index.clear()
        if not self._entries_path.exists():
            return
        for line in self._entries_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                entry = KBEntry(**{k: v for k, v in raw.items() if k in KBEntry.__dataclass_fields__})
                fp = entry.fingerprint
                self._fingerprint_index.setdefault(fp, []).append(entry)
            except (json.JSONDecodeError, TypeError):
                continue

    def record(
        self,
        kernel_info: dict[str, Any],
        session_id: str,
        model_name: str,
        gpu_type: str,
        action_type: str,
        outcome: str,
        gain_pct: float = 0.0,
        approach: str = "",
        lesson: str = "",
        failure_reason: str = "",
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> KBEntry:
        """Record an optimization outcome to the KB."""
        fp = KernelFingerprint.from_kernel_info(kernel_info)
        entry = KBEntry(
            id=f"kb_{hashlib.sha256(f'{time.time()}{session_id}'.encode()).hexdigest()[:12]}",
            fingerprint=fp.fingerprint,
            session_id=session_id,
            model_name=model_name,
            gpu_type=gpu_type,
            action_type=action_type,
            target_kernel=kernel_info.get("kernel_name", ""),
            outcome=outcome,
            gain_pct=gain_pct,
            approach=approach,
            lesson=lesson,
            failure_reason=failure_reason,
            confidence=confidence,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        line = json.dumps(asdict(entry), default=str) + "\n"
        with open(self._entries_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        self._fingerprint_index.setdefault(fp.fingerprint, []).append(entry)
        return entry

    def query_similar(
        self,
        kernel_info: dict[str, Any],
        *,
        min_confidence: float = 0.3,
        max_results: int = 10,
    ) -> list[KBEntry]:
        """Find KB entries for semantically similar kernels."""
        fp = KernelFingerprint.from_kernel_info(kernel_info)

        exact = self._fingerprint_index.get(fp.fingerprint, [])
        results = [e for e in exact if e.confidence >= min_confidence]

        if len(results) < max_results:
            for other_fp, entries in self._fingerprint_index.items():
                if other_fp == fp.fingerprint:
                    continue
                for e in entries:
                    if e.confidence < min_confidence:
                        continue
                    if _fingerprints_related(fp, e, kernel_info):
                        results.append(e)
                        if len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break

        results.sort(key=lambda e: (-e.confidence, -e.gain_pct))
        return results[:max_results]

    def get_transfer_candidates(
        self,
        model_name: str,
        gpu_type: str,
        min_gain: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Find successful optimizations from other models that might transfer."""
        candidates: list[dict[str, Any]] = []
        seen_fps: set[str] = set()

        for fp, entries in self._fingerprint_index.items():
            successes = [
                e for e in entries
                if e.outcome == "success"
                and e.gain_pct >= min_gain
                and e.model_name != model_name
                and e.gpu_type == gpu_type
            ]
            if not successes:
                continue
            already_tried = any(
                e for e in entries
                if e.model_name == model_name
            )
            if already_tried:
                continue

            best = max(successes, key=lambda e: e.gain_pct)
            if fp not in seen_fps:
                candidates.append({
                    "fingerprint": fp,
                    "source_model": best.model_name,
                    "target": best.target_kernel,
                    "approach": best.approach,
                    "expected_gain": best.gain_pct,
                    "confidence": best.confidence,
                    "lesson": best.lesson,
                    "pattern": best.action_type,
                    "success_rate": len(successes) / len(entries),
                })
                seen_fps.add(fp)

        candidates.sort(key=lambda c: -c["expected_gain"])
        return candidates

    def success_rate_for(self, kernel_info: dict[str, Any]) -> float:
        """Historical success rate for this kernel fingerprint."""
        fp = KernelFingerprint.from_kernel_info(kernel_info)
        entries = self._fingerprint_index.get(fp.fingerprint, [])
        if not entries:
            return 0.5
        successes = sum(1 for e in entries if e.outcome == "success")
        return successes / len(entries)

    def format_for_prompt(self, entries: list[KBEntry], max_entries: int = 5) -> str:
        """Format KB entries for LLM prompt context."""
        if not entries:
            return "[No prior knowledge for this kernel type]\n"
        lines = ["Prior knowledge from cross-session KB:"]
        for e in entries[:max_entries]:
            lines.append(
                f"  - [{e.model_name}/{e.gpu_type}] {e.action_type} on {e.target_kernel}: "
                f"{e.outcome} ({e.gain_pct:+.1f}%) — {e.lesson[:100]}"
            )
        return "\n".join(lines) + "\n"

    @property
    def entry_count(self) -> int:
        return sum(len(v) for v in self._fingerprint_index.values())


def _fingerprints_related(
    query_fp: KernelFingerprint,
    entry: KBEntry,
    kernel_info: dict[str, Any],
) -> bool:
    """Check if a KB entry is related to the query (fuzzy match)."""
    entry_meta = entry.metadata or {}
    if query_fp.op_type and entry_meta.get("op_type") == query_fp.op_type:
        return True
    entry_kernel = entry.target_kernel.lower()
    query_kernel = kernel_info.get("kernel_name", "").lower()
    if entry_kernel and query_kernel:
        shared_tokens = set(entry_kernel.split("_")) & set(query_kernel.split("_"))
        if len(shared_tokens) >= 2:
            return True
    return False
