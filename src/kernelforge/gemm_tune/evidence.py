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
MERGE_TABLES = re.compile(r"\[aiter\]\s+merge tuned file under model_configs/ and configs/\s+(?P<paths>\S+)")

# aiter CK MoE dispatch; the tuple carries the dtype combination and token count.
FUSED_MOE = re.compile(
    r"\[aiter\]\s+\[fused_moe\]\s+using\s+(?P<stage>\S+)\s+(?P<tag>\S+)\s+for\s+\((?P<tuple>[^)]*)\)"
)

# The same tuple, on the line that says the lookup MISSED. This is the MoE
# equivalent of DENSE_LOOKUP's miss branch and the only unambiguous "this key
# needs tuning" signal the MoE path emits -- the dispatch line above is printed
# whether or not a tuned row was found, so reading it alone cannot distinguish
# "tuned" from "fell back to a heuristic".
FUSED_MOE_MISS = re.compile(r"\[aiter\]\s+\[fused_moe\]\s+no tuned (?P<flavour>\S+) config for\s+\((?P<tuple>[^)]*)\)")

# Field order of the aiter fused-MoE dispatch tuple, read off a production log:
#
#   ('gfx950', 256, 1, 6144, 384, 128, 4, <ActivationType.Swiglu: 2>,
#    'torch.bfloat16', 'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2',
#    'QuantType.per_1x32', True, False)
#
# The first two entries are the architecture and the CU count, which are
# properties of the box rather than of the key; everything after them is, in
# this exact order, the twelve columns of aiter's untuned/tuned fmoe CSV. So
# MOE_TUPLE_FIELDS[2:] == the fmoe CSV header, and the slice is the whole
# conversion. An earlier version of this parser documented the layout as
# starting at ``cu_num`` and so read ``token`` out of the CU-count slot,
# reporting every model's token set as the constant [256].
MOE_TUPLE_FIELDS = (
    "arch",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)
# The subset that keys the CSV -- i.e. the fields that are not box properties.
MOE_KEY_FIELDS = MOE_TUPLE_FIELDS[2:]
# ``token`` varies per request; the rest of the key is fixed for a given model
# and parallelism layout, so it is what identifies "the MoE shape to tune".
MOE_SHAPE_FIELDS = tuple(f for f in MOE_KEY_FIELDS if f != "token")

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
        "a8w8_blockscale_bpreshuffle",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
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


def _moe_field(raw: str) -> str:
    """Normalise one dispatch-tuple entry to the spelling the fmoe CSV uses.

    The log prints Python ``repr``s; aiter's own untuned_fmoe.csv wants the bare
    values. Three shapes differ:

      ``'torch.bfloat16'``           -> ``torch.bfloat16``   (quotes)
      ``<ActivationType.Swiglu: 2>`` -> ``ActivationType.Swiglu``
      ``True`` / ``False``           -> ``1`` / ``0``
    """
    value = raw.strip()
    if value.startswith("<") and value.endswith(">"):
        # An enum repr: "<ActivationType.Swiglu: 2>". Keep the dotted name, drop
        # the numeric value -- the CSV is keyed on the name.
        value = value[1:-1].split(":", 1)[0].strip()
    value = value.strip("'\"")
    if value == "True":
        return "1"
    if value == "False":
        return "0"
    return value


def _moe_tuple(raw: str) -> list[str]:
    """Split a dispatch tuple into normalised fields."""
    return [_moe_field(p) for p in raw.split(",")]


def _blank_moe() -> dict[str, Any]:
    return {"impl": "aiter_ck", "by_stage": {}, "keys": {}}


def _moe_token_offset(parts: list[str]) -> int:
    """Index of ``token`` in a dispatch tuple.

    aiter prefixes the tuple with the architecture on the builds that print one
    ('gfx950', 256, 1, ...) and starts at the CU count on those that do not
    (304, 1, ...). Both forms put ``token`` immediately after the box-property
    prefix, and the arch is the only non-numeric field there, so its presence is
    what the offset keys on. Assuming one layout is what previously made every
    model report its token set as the constant [256] -- the CU count read out of
    the token slot.
    """
    return 2 if parts and _as_int(parts[0]) is None else 1


def _record_moe_key(moe: dict[str, Any], parts: list[str], *, miss: bool) -> int | None:
    """Fold one dispatch tuple into the observed-key table. Returns its token.

    Keys are grouped on everything except ``token``: one model serving at one
    parallelism layout dispatches a single MoE shape and simply varies the token
    count per batch, so collapsing on token is what turns thousands of log lines
    into the handful of rows a tuner is actually asked to produce.
    """
    offset = _moe_token_offset(parts)
    if len(parts) <= offset:
        return None
    token = _as_int(parts[offset])
    key_parts = parts[offset:]
    if len(key_parts) != len(MOE_KEY_FIELDS):
        # A truncated tuple still tells us which token counts were dispatched,
        # which is all the stage-coverage consumer needs. It cannot key a tuned
        # table, though, so no MoE key is recorded and fmoe_ck keeps refusing --
        # recording a short row would silently zero-fill the quantisation pair.
        moe["_unkeyed_tuple_count"] = moe.get("_unkeyed_tuple_count", 0) + 1
        if miss:
            moe["_unkeyed_miss_count"] = moe.get("_unkeyed_miss_count", 0) + 1
        moe.setdefault("_unkeyed_field_counts", set()).add(len(key_parts))
        return token
    fields = dict(zip(MOE_KEY_FIELDS, key_parts, strict=True))
    fields["arch"] = parts[0] if offset == 2 else ""
    fields["cu_num"] = parts[offset - 1]
    shape = tuple(fields[f] for f in MOE_SHAPE_FIELDS)
    rec = moe["keys"].get(shape)
    if rec is None:
        rec = {
            **{f: fields[f] for f in MOE_SHAPE_FIELDS},
            "arch": fields["arch"],
            "cu_num": fields["cu_num"],
            "tokens": set(),
            "untuned_tokens": set(),
            "miss_count": 0,
        }
        moe["keys"][shape] = rec
    if token is not None:
        rec["tokens"].add(token)
        if miss:
            rec["untuned_tokens"].add(token)
    if miss:
        rec["miss_count"] += 1
    return token


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
                "serving log exceeds %d lines; demand is derived from the first %d only (raise %s to read further)",
                max_lines,
                max_lines,
                _MAX_LINES_ENV,
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
                        base,
                        max_keys,
                        _MAX_KEYS_ENV,
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
            parts = _moe_tuple(fm.group("tuple"))
            moe = dispatch.setdefault("moe", _blank_moe())
            moe.setdefault("by_stage", {})
            moe.setdefault("keys", {})
            stage_key = f"{fm.group('stage')}/{fm.group('tag')}"
            rec = moe["by_stage"].setdefault(stage_key, {"tokens": set(), "tuple": parts})
            token = _record_moe_key(moe, parts, miss=False)
            if token is not None:
                rec["tokens"].add(token)
            continue

        fmm = FUSED_MOE_MISS.search(line)
        if fmm:
            # The dispatch line above says which stage ran, not whether a tuned
            # row was found -- it prints identically either way. This line is the
            # actual miss, and it is the MoE counterpart of the dense
            # "not found tuned config in ..." branch that drives dense demand.
            moe = dispatch.setdefault("moe", _blank_moe())
            moe.setdefault("by_stage", {})
            moe.setdefault("keys", {})
            moe["fallback_flavour"] = fmm.group("flavour")
            _record_moe_key(moe, _moe_tuple(fmm.group("tuple")), miss=True)
            continue

        vh = VLLM_MOE_HIT.search(line)
        if vh:
            vllm_moe["hit"].append(vh.group("path"))
            continue
        vm = VLLM_MOE_MISS.search(line)
        if vm:
            vllm_moe["miss"].append(vm.group("path"))

    moe = dispatch.get("moe")
    if moe:
        unkeyed_count = moe.pop("_unkeyed_tuple_count", 0)
        unkeyed_misses = moe.pop("_unkeyed_miss_count", 0)
        field_counts = sorted(moe.pop("_unkeyed_field_counts", set()))
        if unkeyed_count:
            # Short tuples are a supported aiter build variant and may appear
            # thousands of times in one serving log. Report the limitation once
            # per parse instead of emitting one warning for every dispatch.
            log.warning(
                "%d fused_moe tuple line(s) (%d misses) carry %s key fields; expected %d, recording tokens only",
                unkeyed_count,
                unkeyed_misses,
                field_counts,
                len(MOE_KEY_FIELDS),
            )
            moe["unkeyed_tuple_count"] = unkeyed_count
            moe["unkeyed_miss_count"] = unkeyed_misses
    if moe and "by_stage" in moe:
        for rec in moe["by_stage"].values():
            rec["tokens"] = sorted(rec["tokens"])
        moe["stages_seen"] = sorted({k.split("/")[0] for k in moe["by_stage"]})
        # A stage that only covers large token counts must not suppress tuning
        # for the range the other stage serves.
        moe["tunable_ck_2stage"] = any(k.startswith("2stage") for k in moe["by_stage"])
    if moe and isinstance(moe.get("keys"), dict):
        # Most-missed key first, so a consumer that can only afford one row tunes
        # the one the runtime asked for most.
        moe["keys"] = [
            {**rec, "tokens": sorted(rec["tokens"]), "untuned_tokens": sorted(rec["untuned_tokens"])}
            for rec in sorted(moe["keys"].values(), key=lambda r: (-r["miss_count"], -len(r["tokens"])))
        ]
        moe["miss_count"] = sum(r["miss_count"] for r in moe["keys"])

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


# Upper bound aiter clamps the gl=1 padding to; beyond it every M shares one row.
_PADDED_M_CAP = 8192


def padded_m(m: int) -> int:
    """The M a tuned row must be written at to serve ``m``.

    aiter resolves a lookup three times before giving up: the exact M, then
    ``get_padded_m(..., gl=0)``, then ``get_padded_m(..., gl=1)``. The gl=1 form
    is the next power of two, capped at 8192, and does not depend on N or K --
    verified against the installed aiter for every M in 1..4096 plus 5000, 8192,
    8193, 10000, 16384, 20000 and 100000, with zero mismatches. aiter's own
    shipped ``bf16_tuned_gemm.csv`` is keyed almost entirely on powers of two,
    which is the same statement from the other direction: rows are *meant* to sit
    at the padded M and serve the bucket below them.

    A row written at a raw observed M, by contrast, is reachable only by a
    request repeating that exact M.
    """
    if m <= 1:
        return 1
    return min(1 << (m - 1).bit_length(), _PADDED_M_CAP)


def demand_shapes(
    entry: dict[str, Any],
    *,
    limit: int | None = None,
    bucket: bool = True,
) -> list[dict[str, Any]]:
    """Requested keys for one table, most-requested first.

    ``limit`` is a budget, not a filter: the bf16 fast path costs ~93s per shape
    after including its torch baseline, so an hour buys roughly 37 of them while
    a single arm can ask for 492-849 distinct M values.

    With ``bucket`` (the default) the budget is spent on *lookup buckets* rather
    than on raw keys: keys are grouped by the M a tuned row must be written at
    (see ``padded_m``), the groups are ranked by total request count, and each
    chosen group contributes one row at its padded M. Ranking raw keys instead
    spends several slots inside one bucket and covers no more than one
    bucket-aware slot would have. Measured over the 17 models with a production
    serving log on /shared_nfs, as the share of logged misses a tuned table would
    actually serve:

        budget  24:  raw keys   1.1%   padded buckets  95.6%
        budget  48:  raw keys   2.2%   padded buckets  99.5%
        budget  96:  raw keys   4.2%   padded buckets 100.0%

    This also repairs the fp8 caveat noted below: where every key is requested
    exactly once the raw ordering carries no information, but summing those
    requests per bucket does.

    ``bucket=False`` restores the raw-key ordering, for a caller that wants the
    exact M values the runtime asked for rather than a tunable cover of them.
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

    if bucket:
        grouped: dict[tuple, dict[str, Any]] = {}
        for shape in shapes:
            padded = padded_m(shape["M"])
            rest = tuple(sorted((f, v) for f, v in shape.items() if f not in ("M", "requests")))
            got = grouped.get((padded, rest))
            if got is None:
                grouped[(padded, rest)] = {
                    **shape,
                    "M": padded,
                    "observed_M": [shape["M"]],
                }
            else:
                got["requests"] += shape["requests"]
                got["observed_M"].append(shape["M"])
        shapes = sorted(grouped.values(), key=lambda s: -s["requests"])
        for shape in shapes:
            shape["observed_M"] = sorted(set(shape["observed_M"]))

    if limit is not None and limit > 0:
        shapes = shapes[:limit]
    return shapes


def moe_dispatch_keys(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Runtime-observed MoE dispatch keys, most-missed first. Empty if none."""
    moe = ((report or {}).get("dispatch") or {}).get("moe") or {}
    keys = moe.get("keys")
    return list(keys) if isinstance(keys, list) else []


def moe_ck_missed_keys(report: dict[str, Any]) -> list[dict[str, Any]]:
    """MoE keys whose missed tokens were actually served by CK 2-stage.

    Miss lines do not name the dispatch stage, and keys deliberately collapse
    across token counts. Attribute misses by intersecting their tokens with the
    tokens observed on 2-stage dispatch lines. When an older report has no stage
    detail, retain the old fail-open behaviour; when stage detail exists, never
    hand CK a token observed only on 1-stage or another backend.
    """
    moe = ((report or {}).get("dispatch") or {}).get("moe") or {}
    by_stage = moe.get("by_stage") or {}
    ck_tokens = {
        token
        for stage, rec in by_stage.items()
        if str(stage).startswith("2stage")
        for value in (rec.get("tokens") or [])
        if (token := _as_int(value)) is not None
    }
    stage_tokens = {
        token
        for rec in by_stage.values()
        for value in (rec.get("tokens") or [])
        if (token := _as_int(value)) is not None
    }
    has_stage_detail = bool(stage_tokens)
    missed: list[dict[str, Any]] = []
    for key in moe_dispatch_keys(report):
        untuned = {token for value in (key.get("untuned_tokens") or []) if (token := _as_int(value)) is not None}
        if not untuned and (_as_int(key.get("miss_count")) or 0) > 0:
            # Compatibility with reports written before untuned_tokens was
            # persisted: all observed tokens are the best available bound.
            untuned = {token for value in (key.get("tokens") or []) if (token := _as_int(value)) is not None}
        if has_stage_detail:
            untuned &= ck_tokens
        if not untuned:
            continue
        missed.append({**key, "untuned_tokens": sorted(untuned)})
    return missed


def moe_untuned_csv_text(
    key: dict[str, Any],
    *,
    tokens: list[int] | None = None,
) -> str:
    """Render one observed MoE key as an aiter untuned-fmoe CSV.

    The twelve CSV columns are exactly the dispatch tuple minus its two
    box-property fields, in the same order, so this is a projection of what the
    runtime asked for rather than a reconstruction of it -- which is the whole
    point: the quantisation pair, the per-partition ``inter_dim`` and the EP
    path's extra masked expert slot are all chosen by the serving framework and
    cannot be recovered from the model config.

    ``tokens`` defaults to the token counts whose lookup actually missed, and
    falls back to every token seen for this key.
    """
    header = list(MOE_KEY_FIELDS)
    want = tokens or key.get("untuned_tokens") or key.get("tokens") or []
    lines = [",".join(header)]
    for token in sorted({int(t) for t in want}):
        row = [str(token)] + [str(key.get(f, "")) for f in header[1:]]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def write_demand(report: dict[str, Any], path: Path) -> Path:
    """Serialise a parsed report to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    return path
