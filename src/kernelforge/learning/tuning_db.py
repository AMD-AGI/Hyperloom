# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tuning Database — config→performance lookup that grows with every benchmark.

The biggest time sink in SLA kernel development was trial-and-error on
tile configurations. A tuning DB eliminates repeated exploration:

  "For attention_backward with seq_len=8192, head_dim=128 on gfx950,
   the best CK config is BLOCK_M=128, BLOCK_N=64, wpe=2 → 80.2 ms"

The DB grows automatically:
  1. Every bench_wallclock() call logs {operation, shape, backend, config, wall_ms}
  2. Every successful experiment adds its best config to the "golden configs" table
  3. When starting a new task, the agent queries: "what config worked for similar shapes?"
  4. Transfer rules capture cross-operation learnings (e.g., "wpe=2 for ALL sparse kernels")

This is the single highest-leverage learning mechanism. The SLA work took
~50 iterations across fwd/bwd. With a tuning DB, bwd would have started
from fwd's best config and saved ~20 iterations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Persisting tuning results to the on-disk tuning DB (tuning_entries.jsonl,
# golden_configs.json, transfer_rules.json) is disabled so runs do not mutate
# the repo's knowledge_base. This will be redesigned as a dedicated feature
# later; flip to True to re-enable persistence.
_TUNING_DB_WRITE_ENABLED = False


@dataclass
class TuningEntry:
    """A single data point: config → performance for a specific context."""

    operation: str  # "attention_fwd", "attention_bwd", "gemm", "moe"
    backend: str  # "ck", "flydsl", "triton"
    gpu_target: str  # "gfx950"
    dtype: str  # "bf16", "fp16", "fp8"

    # Shape (normalized to canonical keys)
    shape: dict[str, int]  # {"M": 4096, "N": 4096, "K": 4096} or {"seq_len": 8192, ...}

    # Configuration that was tested
    config: dict[str, Any]  # {"BLOCK_M": 128, "BLOCK_N": 64, "wpe": 2, ...}

    # Results
    wall_ms: float
    snr_db: float | None = None
    passed_correctness: bool = True

    # PMC diagnosis
    pmc_diagnosis: str = ""  # "COMPUTE-BOUND", "BALANCED", "MEMORY-BOUND"
    wait_mfma_ratio: float | None = None
    vgpr: int | None = None

    # Metadata
    experiment_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def shape_key(self) -> str:
        """Normalized shape string for grouping."""
        return "|".join(f"{k}={v}" for k, v in sorted(self.shape.items()))

    def context_key(self) -> str:
        """Unique key for operation+backend+gpu+dtype+shape."""
        return f"{self.operation}|{self.backend}|{self.gpu_target}|{self.dtype}|{self.shape_key()}"

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> TuningEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TransferRule:
    """A rule that transfers knowledge across operations/shapes.

    Example:
      "For ALL sparse attention kernels on gfx950, wpe=2 beats wpe=3.
       Evidence: SLA fwd (8.86 vs 13.40 ms), SLA bwd (80.2 vs 105 ms)."
    """

    rule_id: str
    description: str
    scope: str  # "all_sparse", "attention_*", "gemm_large", etc.
    parameter: str  # "wpe", "BLOCK_M", "num_stages", etc.
    recommended_value: Any
    anti_value: Any = None  # Value to AVOID
    evidence: list[str] = field(default_factory=list)  # experiment IDs
    confidence: float = 0.0  # 0-1 based on evidence count

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> TransferRule:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TuningDatabase:
    """Persistent tuning database — grows with every experiment.

    Usage:
        db = TuningDatabase("knowledge_base/tuning_db")

        # Log a result (called automatically by bench tool)
        db.log(operation="attention_bwd", backend="ck", gpu_target="gfx950",
               dtype="bf16", shape={"seq_len": 8192, "head_dim": 128},
               config={"BLOCK_M": 128, "wpe": 2}, wall_ms=80.2, snr_db=35.0)

        # Query: what config works best for this shape?
        best = db.best_config(operation="attention_bwd", backend="ck",
                              shape={"seq_len": 8192, "head_dim": 128})

        # Query: what worked for SIMILAR shapes?
        suggestions = db.suggest_configs(operation="attention_bwd", backend="ck",
                                          shape={"seq_len": 4096, "head_dim": 128})

        # Context for agent prompt
        context = db.context_for_task(operation="attention_bwd", backend="ck",
                                       shape={"seq_len": 8192, "head_dim": 128})
    """

    def __init__(self, db_dir: str | Path):
        self.db_dir = Path(db_dir)
        self._entries_path = self.db_dir / "tuning_entries.jsonl"
        self._golden_path = self.db_dir / "golden_configs.json"
        self._rules_path = self.db_dir / "transfer_rules.json"

    def _ensure_db_dir(self) -> None:
        """Materialize the DB directory, but only on the way to an actual write.

        Constructing a ``TuningDatabase`` used to mkdir unconditionally, which
        created an empty tree under whatever root was handed in even though
        ``_TUNING_DB_WRITE_ENABLED`` is False and nothing is ever written.
        """
        self.db_dir.mkdir(parents=True, exist_ok=True)

    # ─── Logging ───

    def log(self, **kwargs) -> TuningEntry:
        """Log a tuning result. Called after every benchmark."""
        entry = TuningEntry(**kwargs)

        if not _TUNING_DB_WRITE_ENABLED:
            return entry

        # Append to JSONL (append-only, no read-modify-write)
        self._ensure_db_dir()
        with open(self._entries_path, "a") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")

        # Update golden config if this is the best for its context
        self._update_golden(entry)

        return entry

    def _update_golden(self, entry: TuningEntry) -> None:
        """Update golden configs if this entry is the best for its context."""
        if not entry.passed_correctness:
            return

        golden = self._load_golden()
        key = entry.context_key()

        if key not in golden or entry.wall_ms < golden[key]["wall_ms"]:
            golden[key] = {
                "config": entry.config,
                "wall_ms": entry.wall_ms,
                "snr_db": entry.snr_db,
                "pmc_diagnosis": entry.pmc_diagnosis,
                "vgpr": entry.vgpr,
                "experiment_id": entry.experiment_id,
                "timestamp": entry.timestamp,
            }
            self._save_golden(golden)

    def _load_golden(self) -> dict:
        if self._golden_path.exists():
            return json.loads(self._golden_path.read_text())
        return {}

    def _save_golden(self, golden: dict) -> None:
        if not _TUNING_DB_WRITE_ENABLED:
            return
        self._ensure_db_dir()
        self._golden_path.write_text(json.dumps(golden, indent=2, default=str))

    # ─── Querying ───

    def all_entries(self) -> list[TuningEntry]:
        """Load all tuning entries."""
        if not self._entries_path.exists():
            return []
        entries = []
        with open(self._entries_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(TuningEntry.from_dict(json.loads(line)))
        return entries

    def best_config(
        self,
        operation: str,
        backend: str,
        shape: dict[str, int] | None = None,
        gpu_target: str = "gfx950",
        dtype: str = "bf16",
    ) -> dict | None:
        """Get the best-known config for an exact operation+shape+backend.

        Returns dict with config, wall_ms, etc. or None if no data.
        """
        golden = self._load_golden()

        # Try exact match first
        if shape:
            shape_key = "|".join(f"{k}={v}" for k, v in sorted(shape.items()))
            key = f"{operation}|{backend}|{gpu_target}|{dtype}|{shape_key}"
            if key in golden:
                return golden[key]

        # Fall back to any matching operation+backend
        matches = []
        prefix = f"{operation}|{backend}|{gpu_target}|{dtype}|"
        for key, val in golden.items():
            if key.startswith(prefix):
                matches.append(val)

        return min(matches, key=lambda x: x["wall_ms"]) if matches else None

    def suggest_configs(
        self,
        operation: str,
        backend: str,
        shape: dict[str, int],
        gpu_target: str = "gfx950",
        dtype: str = "bf16",
        max_suggestions: int = 5,
    ) -> list[dict]:
        """Suggest configs based on similar shapes and operations.

        Similarity is based on:
          1. Exact match (same operation + shape) — highest confidence
          2. Same operation, similar shape (within 2× on each dimension)
          3. Same operation class (e.g., attention_fwd → attention_bwd)
          4. Transfer rules (cross-operation learnings)
        """
        suggestions = []

        # Level 1: Exact match
        exact = self.best_config(operation, backend, shape, gpu_target, dtype)
        if exact:
            suggestions.append(
                {
                    "source": "exact_match",
                    "confidence": 1.0,
                    **exact,
                }
            )

        # Level 2: Similar shapes (within 2× on each dimension)
        entries = self.all_entries()
        similar = []
        for entry in entries:
            if (
                entry.operation == operation
                and entry.backend == backend
                and entry.gpu_target == gpu_target
                and entry.passed_correctness
            ):
                if self._shape_similar(shape, entry.shape, factor=2.0):
                    similar.append(entry)

        # Rank by wall_ms, deduplicate by config
        similar.sort(key=lambda e: e.wall_ms)
        seen_configs = set()
        for entry in similar:
            config_key = json.dumps(entry.config, sort_keys=True)
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                suggestions.append(
                    {
                        "source": f"similar_shape ({entry.shape_key()})",
                        "confidence": 0.7,
                        "config": entry.config,
                        "wall_ms": entry.wall_ms,
                    }
                )

        # Level 3: Same operation class
        op_class = operation.rsplit("_", 1)[0]  # "attention_bwd" → "attention"
        for entry in entries:
            if (
                entry.operation.startswith(op_class)
                and entry.backend == backend
                and entry.gpu_target == gpu_target
                and entry.passed_correctness
                and entry.operation != operation
            ):
                config_key = json.dumps(entry.config, sort_keys=True)
                if config_key not in seen_configs:
                    seen_configs.add(config_key)
                    suggestions.append(
                        {
                            "source": f"related_op ({entry.operation})",
                            "confidence": 0.4,
                            "config": entry.config,
                            "wall_ms": entry.wall_ms,
                        }
                    )

        # Level 4: Transfer rules
        rules = self._load_rules()
        for rule in rules:
            if self._rule_applies(rule, operation):
                suggestions.append(
                    {
                        "source": f"transfer_rule ({rule['rule_id']})",
                        "confidence": rule["confidence"],
                        "config": {rule["parameter"]: rule["recommended_value"]},
                        "note": rule["description"],
                    }
                )

        return suggestions[:max_suggestions]

    def _shape_similar(self, a: dict, b: dict, factor: float = 2.0) -> bool:
        """Check if two shapes are within factor× on shared dimensions."""
        shared_keys = set(a.keys()) & set(b.keys())
        if not shared_keys:
            return False
        for key in shared_keys:
            ratio = max(a[key], b[key]) / max(min(a[key], b[key]), 1)
            if ratio > factor:
                return False
        return True

    # ─── Transfer Rules ───

    def _load_rules(self) -> list[dict]:
        if self._rules_path.exists():
            return json.loads(self._rules_path.read_text())
        return []

    def _save_rules(self, rules: list[dict]) -> None:
        if not _TUNING_DB_WRITE_ENABLED:
            return
        self._ensure_db_dir()
        self._rules_path.write_text(json.dumps(rules, indent=2, default=str))

    def _rule_applies(self, rule: dict, operation: str) -> bool:
        """Check if a transfer rule applies to an operation."""
        scope = rule.get("scope", "")
        if scope == "all":
            return True
        if "*" in scope:
            prefix = scope.replace("*", "")
            return operation.startswith(prefix)
        return scope in operation

    def add_transfer_rule(
        self,
        rule_id: str,
        description: str,
        scope: str,
        parameter: str,
        recommended_value: Any,
        anti_value: Any = None,
        evidence: list[str] | None = None,
    ) -> None:
        """Add a cross-operation transfer rule.

        Example:
            db.add_transfer_rule(
                rule_id="sparse_wpe2",
                description="For ALL sparse attention on gfx950, wpe=2 beats wpe=3",
                scope="all_sparse",
                parameter="wpe",
                recommended_value=2,
                anti_value=3,
                evidence=["exp_sla_fwd_001", "exp_sla_bwd_002"],
            )
        """
        rules = self._load_rules()

        # Update existing or add new
        existing = next((r for r in rules if r["rule_id"] == rule_id), None)
        if existing:
            existing["description"] = description
            existing["recommended_value"] = recommended_value
            existing["anti_value"] = anti_value
            if evidence:
                existing["evidence"] = list(set(existing.get("evidence", []) + evidence))
            existing["confidence"] = min(1.0, len(existing["evidence"]) * 0.2)
        else:
            rules.append(
                TransferRule(
                    rule_id=rule_id,
                    description=description,
                    scope=scope,
                    parameter=parameter,
                    recommended_value=recommended_value,
                    anti_value=anti_value,
                    evidence=evidence or [],
                    confidence=min(1.0, len(evidence or []) * 0.2),
                ).to_dict()
            )

        self._save_rules(rules)

    # ─── Auto-discovery of transfer rules ───

    def discover_transfer_rules(self) -> list[TransferRule]:
        """Analyze the tuning DB to discover cross-operation patterns.

        Finds parameters that consistently have the same optimal value
        across multiple operations/shapes.
        """
        entries = [e for e in self.all_entries() if e.passed_correctness]
        if len(entries) < 5:
            return []

        discovered = []

        # Group by (backend, parameter)
        param_values: dict[tuple[str, str], list[tuple[Any, float, str]]] = {}
        for entry in entries:
            for param, value in entry.config.items():
                key = (entry.backend, param)
                param_values.setdefault(key, []).append((value, entry.wall_ms, entry.operation))

        # Find parameters where one value consistently wins
        for (backend, param), value_perf_ops in param_values.items():
            # Group by value
            by_value: dict[Any, list[float]] = {}
            by_value_ops: dict[Any, set[str]] = {}
            for value, wall_ms, op in value_perf_ops:
                by_value.setdefault(value, []).append(wall_ms)
                by_value_ops.setdefault(value, set()).add(op)

            if len(by_value) < 2:
                continue  # need at least 2 values to compare

            # Find the value with lowest median wall_ms
            medians = {}
            for value, times in by_value.items():
                sorted_times = sorted(times)
                medians[value] = sorted_times[len(sorted_times) // 2]

            best_value = min(medians, key=medians.get)
            worst_value = max(medians, key=medians.get)

            # Check if it wins across multiple operations
            if len(by_value_ops.get(best_value, set())) >= 2:
                speedup = medians[worst_value] / medians[best_value]
                if speedup > 1.1:  # at least 10% better
                    rule = TransferRule(
                        rule_id=f"auto_{backend}_{param}_{best_value}",
                        description=(
                            f"For {backend} kernels, {param}={best_value} is "
                            f"{speedup:.2f}× faster than {param}={worst_value} "
                            f"across {len(by_value_ops[best_value])} operations"
                        ),
                        scope="all",
                        parameter=param,
                        recommended_value=best_value,
                        anti_value=worst_value,
                        evidence=list(by_value_ops[best_value])[:5],
                        confidence=min(1.0, len(by_value_ops[best_value]) * 0.2),
                    )
                    discovered.append(rule)

        return discovered

    # ─── Context for agent prompts ───

    def context_for_task(
        self,
        operation: str,
        backend: str,
        shape: dict[str, int],
        gpu_target: str = "gfx950",
        dtype: str = "bf16",
    ) -> str:
        """Generate tuning context for an agent starting a new task.

        This is the key accelerator — instead of starting from scratch,
        the agent starts with the best known config and nearby results.
        """
        lines = ["## Tuning Database"]

        # Best known config for exact match
        best = self.best_config(operation, backend, shape, gpu_target, dtype)
        if best:
            lines.append("\n### Best Known Config (exact match)")
            lines.append(f"  Config: {best['config']}")
            lines.append(f"  wall_ms: {best['wall_ms']}")
            if best.get("pmc_diagnosis"):
                lines.append(f"  PMC: {best['pmc_diagnosis']}")
            lines.append("  START FROM THIS CONFIG — don't explore from scratch")
        else:
            lines.append("\n### No exact match — querying similar shapes")

        # Suggestions from similar contexts
        suggestions = self.suggest_configs(operation, backend, shape, gpu_target, dtype)
        if suggestions:
            lines.append(f"\n### Suggested Starting Configs ({len(suggestions)})")
            for i, s in enumerate(suggestions):
                conf = s.get("confidence", 0)
                lines.append(
                    f"  {i + 1}. [{conf:.0%} confidence] from {s['source']}: "
                    f"{s.get('config', {})} → {s.get('wall_ms', '?')} ms"
                )
                if s.get("note"):
                    lines.append(f"     Note: {s['note']}")

        # Transfer rules
        rules = self._load_rules()
        applicable = [r for r in rules if self._rule_applies(r, operation)]
        if applicable:
            lines.append(f"\n### Transfer Rules ({len(applicable)})")
            for r in applicable:
                lines.append(f"  - {r['parameter']}={r['recommended_value']}: {r['description']}")
                if r.get("anti_value") is not None:
                    lines.append(f"    AVOID: {r['parameter']}={r['anti_value']}")

        # Stats
        total = len(self.all_entries())
        golden = self._load_golden()
        lines.append(f"\n### DB Stats: {total} entries, {len(golden)} golden configs")

        return "\n".join(lines)
