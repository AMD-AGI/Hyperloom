# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn a serving log into a tuning demand list and an apply verdict.

Shapes used to be derived from ``config.json``. Measured against 42 real log
arms, that derivation served **0.4%** of the lookups the runtime actually made
(1.5% in thorough mode). It is not a precision problem: observed M values
include 15 and 17, which no stepping rule produces, and the fp8 arms' only
(N, K) pair is the lm_head one -- which has empty intersection with anything
derived from hidden_size/intermediate_size. forge's own code already said so, in
the TunableOp skip reason: "Cannot reliably infer all shapes from config.json
alone." That conclusion was used to skip one tuner; it applies to all of them.

So the shape list comes from the log: every lookup the runtime made, and which
ones missed. The same parse also answers "was the artifact ever read?", because
both facts come from the same lines -- one parser, two consumers.

Two properties matter more than they look:

* **The extended key columns are optional.** The bf16 op logs
  dtype/otype/bias/scaleAB/bpreshuffle; the a8w8_blockscale op logs M/N/K alone.
  A parser that requires the wide form silently drops the narrow one -- the
  first version did exactly that and lost 252 of 440 misses.
* **Zero hits is not the same as zero hit-logging.** Hit lines are gated behind
  ``AITER_LOG_TUNED_CONFIG=1``; miss lines are unconditional. Reading "no hit
  lines" as "the table was never used" would fail every arm that simply did not
  set the flag, so that case reports ``inconclusive_no_hit_logging``.
"""

from __future__ import annotations

import json
import os
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# aiter dense GEMM lookup. M/N/K always present; the wider key group is emitted
# by the bf16 op only, so it must stay optional (see module docstring).
DENSE_LOOKUP = re.compile(
    r"\[aiter\]\s+shape is\s+"
    r"M:(?P<M>\d+),\s*N:(?P<N>\d+),\s*K:(?P<K>\d+)"
    r"(?:\s+dtype='(?P<dtype>[^']*)'\s+otype='(?P<otype>[^']*)'\s+"
    r"bias=(?P<bias>True|False),\s*"
    r"scaleAB=(?P<scaleAB>True|False),\s*"
    r"bpreshuffle=(?P<bpreshuffle>True|False))?"
    r",?\s*"
    r"(?:not found tuned config in (?P<miss_table>[^,]+)"
    r"|found padded_M:\s*(?P<padded_M>\d+))"
)

# A hit line names the table it resolved in, after the padded-M part:
#   ... found padded_M: 8192, N:4096, K:4096 is tuned on cu_num = 256 in
#   /tmp/aiter_configs/bf16_tuned_gemm.csv, libtype is asm, kernel name is ...
# Parsed separately from DENSE_LOOKUP so the hit/miss branch above stays legible.
HIT_TABLE = re.compile(r"is tuned on cu_num\s*=\s*\d+\s+in\s+(?P<table>[^,]+)")

# Which tables the runtime actually loaded (os.pathsep-separated path list).
MERGE_TABLES = re.compile(
    r"\[aiter\]\s+merge tuned file under model_configs/ and configs/\s+(?P<paths>\S+)"
)

# aiter CK MoE dispatch; the tuple carries the dtype combination and token count.
FUSED_MOE = re.compile(
    r"\[aiter\]\s+\[fused_moe\]\s+using\s+(?P<stage>\S+)\s+(?P<tag>\S+)\s+for\s+\((?P<tuple>[^)]*)\)"
)

# vLLM Triton MoE: found vs not-found are two different lines.
VLLM_MOE_HIT = re.compile(r"Using configuration from (?P<path>\S+) for MoE layer")
VLLM_MOE_MISS = re.compile(r"Config file not found at (?P<path>\S+)")

# Per-table full key schema. The log prints whatever the op happens to print;
# the table identity still decides which columns the tuned CSV must be keyed on.
#
# ``q_dtype_w`` is listed for the a8w8 tables because their CSV is keyed on it,
# but the lookup line never prints it -- see KEY_FIELDS, which is what the
# parser can actually capture. So a demand entry for those tables carries
# (M, N, K) only, and the untuned CSV built from it fills q_dtype_w from the
# hardware's fp8 dtype exactly as the non-demand path does. That is a real
# limitation rather than a reconstruction: two runtime lookups differing only in
# q_dtype_w are indistinguishable in the log and collapse into one demand key.
#
# Read off the installed aiter on two MI355X boxes with independent installs (a
# sglang source checkout and the vLLM wheel), which agreed. The *untuned* table
# is the evidence that matters, since it is literally the tuner's input keys:
#
#   a8w8_blockscale_untuned_gemm.csv              M,N,K
#   a8w8_blockscale_bpreshuffle_untuned_gemm.csv  M,N,K
#   a4w4_blockscale_untuned_gemm.csv              M,N,K
#   a8w8_untuned_gemm.csv                         M,N,K,q_dtype_w
#   a8w8_bpreshuffle_untuned_gemm.csv             M,N,K,q_dtype_w
#   bf16_untuned_gemm.csv                         M,N,K,bias,dtype,outdtype,
#                                                 scaleAB,bpreshuffle
#
# This settles a documented disagreement: the RCA text claimed blockscale was
# additionally keyed on a scaling granularity and bpreshuffle on a preshuffle
# marker. Neither column exists. The RCA was wrong; this table is right.
TABLE_KEY_SCHEMA: dict[str, tuple[str, ...]] = {
    "bf16_tuned_gemm.csv": ("M", "N", "K", "dtype", "otype", "bias", "scaleAB", "bpreshuffle"),
    "a8w8_blockscale_tuned_gemm.csv": ("M", "N", "K"),
    "a8w8_blockscale_bpreshuffle_tuned_gemm.csv": ("M", "N", "K"),
    "a8w8_tuned_gemm.csv": ("M", "N", "K", "q_dtype_w"),
    "a8w8_bpreshuffle_tuned_gemm.csv": ("M", "N", "K", "q_dtype_w"),
    "a4w4_blockscale_tuned_gemm.csv": ("M", "N", "K"),
}

# Key columns the log actually exposes, so a demand entry can never claim one it
# did not observe. ``logged_fields`` on each Demand records which of these the
# line carried; anything in TABLE_KEY_SCHEMA beyond this set is supplied
# downstream from the hardware, not from evidence.
UNLOGGABLE_KEY_FIELDS = ("q_dtype_w",)

TABLE_TO_TUNER: dict[str, tuple[str, str]] = {
    "bf16_tuned_gemm.csv": ("sglang_dense_bf16", "AITER_CONFIG_GEMM_BF16"),
    "a8w8_tuned_gemm.csv": ("a8w8", "AITER_CONFIG_GEMM_A8W8"),
    "a8w8_blockscale_tuned_gemm.csv": ("a8w8_blockscale", "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"),
    "a8w8_bpreshuffle_tuned_gemm.csv": ("a8w8_bpreshuffle", "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE"),
    "a8w8_blockscale_bpreshuffle_tuned_gemm.csv": (
        "a8w8_blockscale_bpreshuffle", "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    ),
    "a4w4_blockscale_tuned_gemm.csv": ("a4w4_blockscale", "AITER_CONFIG_GEMM_A4W4"),
    "tuned_fmoe.csv": ("fmoe_ck", "AITER_CONFIG_FMOE"),
}

KEY_FIELDS = ("M", "N", "K", "dtype", "otype", "bias", "scaleAB", "bpreshuffle")

SCHEMA_VERSION = "gemm_demand/v1"

# Bounds on what one parse may consume. Every miss prints a line unconditionally
# and hit logging is now on for every serving run, so a long production run's
# server.log is a large file that this walks line by line while accumulating one
# entry per distinct key. Reading it is on the tuning path, so it has to stay
# bounded by something other than how long the server happened to run.
#
# Truncation is reported rather than silent: a demand list that stopped early is
# still the runtime's own shapes and still far better than config-derived ones,
# but a reader has to be able to tell it is a prefix. Both are overridable for
# an offline audit of a whole campaign.
_MAX_LINES_ENV = "FORGE_EVIDENCE_MAX_LINES"
_MAX_KEYS_ENV = "FORGE_EVIDENCE_MAX_KEYS"
DEFAULT_MAX_LINES = 2_000_000
# A run cannot tune more than a few dozen shapes in an hour (~74s each), so
# tens of thousands of distinct keys is already far past what any budget spends;
# what it does cost is memory, in the orchestrator's own process.
DEFAULT_MAX_KEYS_PER_TABLE = 50_000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass
class Demand:
    """Every key one tuned-config table was asked for and did not have."""

    table: str
    tuner: str | None
    env_var: str | None
    key_schema: list[str]
    logged_fields: list[str]
    miss_count: int = 0
    keys: list[dict[str, Any]] = field(default_factory=list)

    @property
    def distinct_keys(self) -> int:
        return len(self.keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "tuner": self.tuner,
            "env_var": self.env_var,
            "key_schema": list(self.key_schema),
            "logged_fields": list(self.logged_fields),
            "miss_count": self.miss_count,
            "distinct_keys": self.distinct_keys,
            "keys": self.keys,
        }


def _blank_key() -> dict[str, str | None]:
    return {f: None for f in KEY_FIELDS}


def parse_log(text: str) -> dict[str, Any]:
    """Parse a serving log into demands, an apply verdict and dispatch facts."""
    demands: dict[str, Demand] = {}
    key_counts: dict[str, dict[tuple, int]] = {}
    hits = 0
    misses = 0
    merged: list[str] = []
    # Tables the runtime named in a lookup. Stronger evidence than the merge
    # line for "did our artifact reach the server": when AITER_CONFIG_* is set,
    # aiter prints no merge line at all and simply resolves against the override,
    # so the lookup is the only place the path appears.
    consulted: set[str] = set()
    dispatch: dict[str, Any] = {}
    vllm_moe: dict[str, list[str]] = {"hit": [], "miss": []}

    max_lines = _env_int(_MAX_LINES_ENV, DEFAULT_MAX_LINES)
    max_keys = _env_int(_MAX_KEYS_ENV, DEFAULT_MAX_KEYS_PER_TABLE)
    truncated: dict[str, Any] = {}
    lines_read = 0

    for line in text.splitlines():
        lines_read += 1
        if lines_read > max_lines:
            truncated["lines"] = max_lines
            log.warning(
                "serving log exceeds %d lines; demand is derived from the first "
                "%d only (raise %s to read further)",
                max_lines, max_lines, _MAX_LINES_ENV,
            )
            break
        m = DENSE_LOOKUP.search(line)
        if m:
            if m.group("padded_M") is not None:
                hits += 1
                ht = HIT_TABLE.search(line)
                if ht:
                    consulted.add(ht.group("table").strip())
            else:
                misses += 1
                table_path = (m.group("miss_table") or "").strip()
                if table_path:
                    consulted.add(table_path)
                base = table_path.rsplit("/", 1)[-1]
                tuner, env = TABLE_TO_TUNER.get(base, (None, None))
                d = demands.get(base)
                if d is None:
                    d = Demand(
                        table=base,
                        tuner=tuner,
                        env_var=env,
                        key_schema=list(TABLE_KEY_SCHEMA.get(base, ("M", "N", "K"))),
                        logged_fields=[f for f in KEY_FIELDS if m.group(f) is not None],
                    )
                    demands[base] = d
                    key_counts[base] = {}
                d.miss_count += 1
                key = tuple(m.group(f) for f in KEY_FIELDS)
                counts = key_counts[base]
                # Keep counting repeats of keys already seen -- that ordering is
                # the only signal demand_shapes has -- but stop growing the set.
                if key in counts or len(counts) < max_keys:
                    counts[key] = counts.get(key, 0) + 1
                elif base not in truncated.setdefault("tables", {}):
                    truncated["tables"][base] = max_keys
                    log.warning(
                        "%s reached %d distinct demand keys; further new keys are "
                        "counted as misses but not listed (raise %s)",
                        base, max_keys, _MAX_KEYS_ENV,
                    )
            continue

        mm = MERGE_TABLES.search(line)
        if mm:
            merged.extend(p for p in re.split(r"[:;]", mm.group("paths")) if p)
            continue

        fm = FUSED_MOE.search(line)
        if fm:
            # One model dispatches DIFFERENT stages at different token counts, so
            # a single "saw 1stage" boolean collapses the decode range away and
            # suppresses tuning that 2stage would have covered.
            parts = [p.strip().strip("'") for p in fm.group("tuple").split(",")]
            moe = dispatch.setdefault("moe", {"impl": "aiter_ck", "by_stage": {}})
            stage_key = f"{fm.group('stage')}/{fm.group('tag')}"
            rec = moe["by_stage"].setdefault(stage_key, {"tokens": set(), "tuple": parts})
            # tuple layout: cu_num, token, model_dim, inter_dim, expert, topk, ...
            if len(parts) > 1 and parts[1].isdigit():
                rec["tokens"].add(int(parts[1]))
            continue

        vh = VLLM_MOE_HIT.search(line)
        if vh:
            vllm_moe["hit"].append(vh.group("path"))
            continue
        vm = VLLM_MOE_MISS.search(line)
        if vm:
            vllm_moe["miss"].append(vm.group("path"))

    moe = dispatch.get("moe")
    if moe and "by_stage" in moe:
        for rec in moe["by_stage"].values():
            rec["tokens"] = sorted(rec["tokens"])
        moe["stages_seen"] = sorted({k.split("/")[0] for k in moe["by_stage"]})
        # A stage that only covers large token counts must not suppress tuning
        # for the range the other stage serves.
        moe["tunable_ck_2stage"] = any(k.startswith("2stage") for k in moe["by_stage"])

    if vllm_moe["hit"] or vllm_moe["miss"]:
        moe_entry = dispatch.setdefault("moe", {"impl": "vllm_triton"})
        # A log carrying both aiter CK dispatch lines and vLLM Triton config
        # lines is a real shape (concatenated logs, or a framework switch inside
        # one run). Reporting impl="aiter_ck" while also reporting vllm_* counts
        # describes a runtime that does not exist, so say both were seen instead
        # of letting whichever arrived first define the answer.
        seen = {str(moe_entry.get("impl") or "")} | {"vllm_triton"}
        seen.discard("")
        if len(seen) > 1:
            moe_entry["impl"] = "mixed"
            moe_entry["impls_seen"] = sorted(seen)
        moe_entry["vllm_config_hit"] = len(vllm_moe["hit"])
        moe_entry["vllm_config_miss"] = len(vllm_moe["miss"])

    for base, d in demands.items():
        d.keys = [
            dict(zip(KEY_FIELDS, k, strict=True)) | {"requests": n}
            for k, n in sorted(key_counts[base].items(), key=lambda kv: -kv[1])
        ]

    ordered = sorted(demands.values(), key=lambda d: -d.miss_count)
    total = hits + misses
    return {
        "schema": SCHEMA_VERSION,
        "apply_verdict": {
            "hit": hits,
            "miss": misses,
            "hit_ratio": (hits / total) if total else None,
            "verdict": _apply_verdict(hits, misses),
        },
        "merged_tables": sorted(set(merged)),
        "consulted_tables": sorted(consulted),
        # Present only when a bound was hit, so its absence means the report
        # describes the whole log.
        **({"truncated": truncated} if truncated else {}),
        "dispatch": dispatch,
        "demands": [d.to_dict() for d in ordered],
    }


def _apply_verdict(hits: int, misses: int) -> str:
    if hits == 0 and misses > 0:
        # Hit lines need AITER_LOG_TUNED_CONFIG=1. Without it every arm looks
        # like a total miss, which would REVERT 42 of 42 arms.
        return "inconclusive_no_hit_logging"
    if hits == 0 and misses == 0:
        return "no_lookups"
    return "served" if hits > 0 else "unknown"


def parse_log_file(path: Path | str) -> dict[str, Any]:
    """Parse a log file; a missing/unreadable file yields an empty report."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("cannot read serving log %s: %s", path, exc)
        return parse_log("")
    return parse_log(text)


def load_demand(path: Path | str) -> dict[str, Any] | None:
    """Load a demand.json produced by :func:`parse_log`."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("cannot read demand file %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or "demands" not in data:
        log.warning("demand file %s is not a %s document", path, SCHEMA_VERSION)
        return None
    return data


def demand_for_tuner(report: dict[str, Any], tuner_name: str) -> dict[str, Any] | None:
    """The demand entry a given tuner is responsible for, if the log showed one."""
    for entry in report.get("demands") or []:
        if entry.get("tuner") == tuner_name:
            return entry
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def demand_shapes(
    entry: dict[str, Any],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Requested keys for one table, most-requested first.

    ``limit`` is a budget, not a filter: a shape costs ~74s to tune (58-155s
    measured), so an hour buys roughly 48 of them while a single arm can ask for
    492-849 distinct M values. Truncating by request count is the only ordering
    the log itself justifies -- see the caveat in the master doc about fp8 arms,
    where every key is requested exactly once and this ordering carries no
    information.
    """
    shapes: list[dict[str, Any]] = []
    for key in entry.get("keys") or []:
        m, n, k = _as_int(key.get("M")), _as_int(key.get("N")), _as_int(key.get("K"))
        if m is None or n is None or k is None:
            continue
        shape = {"M": m, "N": n, "K": k, "requests": _as_int(key.get("requests")) or 0}
        for extra in ("dtype", "otype", "bias", "scaleAB", "bpreshuffle"):
            if key.get(extra) is not None:
                shape[extra] = key[extra]
        shapes.append(shape)
    if limit is not None and limit > 0:
        shapes = shapes[:limit]
    return shapes


def write_demand(report: dict[str, Any], path: Path) -> Path:
    """Serialise a parsed report to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    return path
