# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Merge host-probe output into ranked framework-rewrite candidates.

The GPU kernel breakdown answers "which kernel is hot". A whole class of
framework-level inefficiency never reaches a kernel and is therefore invisible
to it: a collective that round-trips through the host to agree on a shape, a
pure function recomputed once per block per denoising step, a CPU-resident table
re-uploaded on every use. The host probe
(``inference_optimizer/assets/host_probe/hl_host_probe.py``) measures those
directly, once per rank; this module merges the per-rank reports and classifies
what it finds into the rewrite-pattern taxonomy the authoring specialist works
from.

Pure functions over already-read JSON plus one thin reader, so the classifier is
testable without a benchmark.

Taxonomy coverage
-----------------
The host probe can see five of the seven rewrite categories:

===========================  ========================================
``memoize_invariant``        a pure computation repeated with identical
                             arguments
``hoist_loop_invariant``     the same *logical* argument rebuilt every
                             iteration, so a cache would never hit until the
                             allocation moves out of the loop
``eliminate_host_round_trip`` an object collective agreeing on a value the
                             ranks could derive locally
``eliminate_host_sync``      a device-to-host read on the hot path
``fuse_collectives``         several adjacent same-shape collectives from one
                             enclosing call site
``keep_device_resident``     a host-to-device copy repeated for a value that
                             does not change
===========================  ========================================

Swapping in a vendor kernel and dropping no-op glue are not host-observable;
they come from the GPU breakdown and from reading the source, and the emitted
report says so rather than implying the list is exhaustive.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


log = logging.getLogger(__name__)


SCHEMA = "hyperloom.framework_rewrite_evidence/1"

# Directory names that hold every installed package rather than one framework. As a
# probe root each of these matches torch, so a collective attributes to a frame
# inside torch instead of to the framework helper that issued it.
_INTERPRETER_PACKAGE_DIRS: frozenset[str] = frozenset({"dist-packages", "site-packages"})

PROBE_FILE_GLOB = "hl_host_probe_rank*.json"

# Reports whose host-call table is empty carry no evidence but do carry a rank, so
# counting them would understate the per-rank averages. A launcher or a re-exec
# inherits ``RANK`` and installs its own probe, so these are expected.
_MIN_USEFUL_ROWS = 1

# Category ids. Stable strings: they cross into the specialist prompt, the
# switch manifest a specialist returns, and the KB, so renaming one is a
# contract change.
CATEGORY_MEMOIZE = "memoize_invariant"
CATEGORY_HOIST = "hoist_loop_invariant"
CATEGORY_HOST_ROUND_TRIP = "eliminate_host_round_trip"
CATEGORY_HOST_SYNC = "eliminate_host_sync"
CATEGORY_FUSE_COLLECTIVES = "fuse_collectives"
CATEGORY_DEVICE_RESIDENT = "keep_device_resident"

# Taxonomy letters, matching the specialist-facing reference so a candidate can
# be traced back to the pattern description it instantiates.
_CATEGORY_TAXONOMY: dict[str, str] = {
    CATEGORY_MEMOIZE: "a",
    CATEGORY_HOIST: "b",
    CATEGORY_HOST_ROUND_TRIP: "c",
    CATEGORY_HOST_SYNC: "c",
    CATEGORY_FUSE_COLLECTIVES: "d",
    CATEGORY_DEVICE_RESIDENT: "f",
}

# Per-category rewrite recipe surfaced with the evidence. The specialist still
# chooses the landing point and writes the code; this only states the shape of
# the fix so the evidence is actionable on its own.
_CATEGORY_RECIPE: dict[str, str] = {
    CATEGORY_MEMOIZE: (
        "Memoize the computation behind a module-level or instance-level cache "
        "keyed by the complete argument identity (tensor data_ptr, shape, dtype, "
        "device and _version; plus every scalar that changes the result). Pin the "
        "source tensors in the cache entry so a freed allocation cannot be "
        "recycled to the same address and produce a false hit."
    ),
    CATEGORY_HOIST: (
        "Hoist the allocation out of the loop first: the arguments are logically "
        "invariant but rebuilt every iteration, so a cache keyed on tensor "
        "identity can never hit. Compute the value once per outer iteration and "
        "pass it in. Only after hoisting does memoizing the callee pay, which "
        "makes this an enabler: measured on its own it will look like no gain."
    ),
    CATEGORY_HOST_ROUND_TRIP: (
        "Cache the rendezvous instead of repeating it. The ranks agree on a value "
        "that is deterministic for a given (local value, world size, process "
        "group), so exchange it once and reuse it; key the cache by the process "
        "group identity so separate groups cannot cross-contaminate."
    ),
    CATEGORY_HOST_SYNC: (
        "Remove the device-to-host read from the hot path: derive the value from "
        "metadata already known on the host, or read it once and cache it keyed by "
        "the source tensor's identity and _version."
    ),
    CATEGORY_FUSE_COLLECTIVES: (
        "Pack the payloads into one buffer and issue a single collective, then "
        "split the result. Guard the fast path on the shapes actually matching and "
        "fall back to the separate collectives otherwise."
    ),
    CATEGORY_DEVICE_RESIDENT: (
        "Keep the value resident on the device instead of rebuilding it on the "
        "host and re-uploading. Cache the device-side tensor keyed by the geometry "
        "that determines it."
    ),
}

# APIs that pickle through the host to agree on a value: each call is a host
# round-trip, and their payload is usually a shape or a length the ranks could
# compute locally.
_OBJECT_COLLECTIVE_APIS: frozenset[str] = frozenset(
    {
        "torch.distributed.all_gather_object",
        "torch.distributed.gather_object",
        "torch.distributed.scatter_object_list",
        "torch.distributed.broadcast_object_list",
    }
)

# APIs that read device memory back to the host, stalling the pipeline. The
# dunders are the implicit half: ``if scalar_tensor == 0``, ``float(t)``,
# ``int(t)`` and indexing a list with a tensor all sync, and none of them look
# like a transfer at the call site.
_HOST_SYNC_APIS: frozenset[str] = frozenset(
    {
        "torch.Tensor.item",
        "torch.Tensor.tolist",
        "torch.Tensor.cpu",
        "torch.Tensor.numpy",
        "torch.Tensor.__float__",
        "torch.Tensor.__int__",
        "torch.Tensor.__bool__",
        "torch.Tensor.__index__",
        "torch.cuda.synchronize",
    }
)

# APIs that move host memory onto the device.
_H2D_APIS: frozenset[str] = frozenset({"torch.Tensor.to", "torch.Tensor.cuda"})

# Tensor collectives, candidates for fusion when several adjacent call sites
# move identically shaped payloads.
_TENSOR_COLLECTIVE_APIS: frozenset[str] = frozenset(
    {
        "torch.distributed.all_gather",
        "torch.distributed.all_gather_into_tensor",
        "torch.distributed.all_to_all",
        "torch.distributed.all_to_all_single",
        "torch.distributed.all_reduce",
        "torch.distributed.reduce_scatter_tensor",
        "torch.distributed.broadcast",
    }
)

# Minimum per-rank call count before a host-API site is worth reporting. Below
# this the site is start-up or teardown work, not the hot path.
MIN_HOST_CALLS = 64

# Minimum per-rank call count before a framework function is worth reporting.
MIN_FRAMEWORK_CALLS = 32

# Minimum repeat rate for a memoization candidate. Half the sampled calls
# repeating an earlier argument identity already means half the work is dead.
MIN_STRICT_REPEAT_RATE = 0.5

# Minimum loose-repeat rate for a hoist candidate. Paired with a strict rate
# below :data:`MIN_STRICT_REPEAT_RATE`, the gap between the two rates is the
# whole signal: same-geometry arguments, freshly allocated each time.
MIN_LOOSE_REPEAT_RATE = 0.5

# Strict rate at or below which a hoist candidate is a *pure* enabler: the same
# object essentially never arrives twice, so memoizing the callee cannot pay
# until the allocation moves out of the loop. Above this the site has a mix of
# stable and rebuilt arguments, so part of the win is already reachable and
# calling it an enabler would overstate the dependency.
MAX_STRICT_REPEAT_FOR_PURE_ENABLER = 0.25

# Minimum distinct enclosing call sites sharing a shape signature before a
# collective is called fusable.
MIN_FUSION_SITES = 2

# Candidates emitted, worst-first truncated. A specialist gets a bounded prompt;
# an unbounded list would push the ranking work back onto the reader.
MAX_CANDIDATES = 40

# A site is set-up work when it stopped being called before the hot loop started.
#
# Why the split is needed at all: loading a multi-gigabyte checkpoint issues
# hundreds of host-to-device copies in a burst, and on absolute cost that outranks
# every genuine per-step inefficiency — one real ``create_pipeline`` site spent
# 29.7s moving 4.8 GiB, more wall time than the object collective that is the
# workload's single biggest lever. Rewriting it cannot move steady-state throughput,
# and a specialist's attention budget is the scarce resource.
#
# Two earlier anchors were tried and both were killed by real data, for the same
# underlying reason — they measured absolute position along the timeline, and both
# ends of that timeline are unpredictable:
#
#   1. "Does the site's calls span enough of the run?" Weight loading took 580s of a
#      644s process, collapsing the generation phase to 9.6% of wall clock, so a span
#      floor marked *every* real finding as set-up. The head of the run is long.
#   2. "Is the site's last call near the latest call in the process?" On the first
#      live orchestrator leg a ``barrier`` called *five* times spanned 518s to 1393s
#      while the denoising loop finished at 995s. Every candidate came back at
#      995/1393 = 0.714 and was demoted, the object collective included. The tail of
#      the run is long too.
#
# The property that actually distinguishes them is not when a site stopped relative
# to the clock, but whether it stopped *before the hot loop began*. The hottest site
# by call count is necessarily inside the innermost loop — the loop product is
# blocks x steps x chunks, while set-up is O(sub-modules) — so its first call marks
# that boundary. Measured hottest-to-median call-count ratios on three real runs:
# 2689x, 2808x and 99741x.
#
# Sites that stop before it are still reported, marked ``setup_phase``, because a
# slow load is worth knowing about — just not worth a rewrite. The label covers
# everything that is not steady-state work: weight loading, model construction
# (every ``__init__`` runs once per sub-module, which reads as a repeated argument
# identity and ranked as a memoization candidate on a real deep run), and work
# confined to an earlier phase.
#
# When the distribution is flat the anchor's assumption does not hold, so nothing is
# demoted: a wrong "this is set-up" note is worse than no note, because it tells the
# specialist to skip a site rather than to think about it. This ratio is the
# significance floor, two orders of magnitude below every ratio measured so far.
HOT_LOOP_DOMINANCE_RATIO = 20.0


# Env switch turning the whole probe off for a run. The probe is on by default
# for the profile action because tier 1 costs a wrapper call on a handful of
# APIs, but an operator chasing a profiler anomaly needs a way to take it out of
# the picture entirely.
ENABLE_ENV = "HYPERLOOM_FRAMEWORK_REWRITE_EVIDENCE"

# Env switch adding the tier-2 hook, which counts every framework call and
# fingerprints its arguments. Off by default: it inflates host time enough to
# skew a co-collected torch trace, so it belongs to a dedicated evidence leg.
DEEP_ENV = "HYPERLOOM_FRAMEWORK_REWRITE_EVIDENCE_DEEP"

# Subdirectory of the run workspace the per-rank probe reports are written into.
PROBE_SUBDIR = "host_probe"

# Merged evidence document filename, written next to the probe reports.
EVIDENCE_FILENAME = "framework_rewrite_evidence.json"


def probe_asset_dir() -> Path:
    """Return the bundled directory holding the probe and its import shim.

    Returns:
        The directory to prepend to the benchmark process's ``PYTHONPATH``. It
        contains ``sitecustomize.py``, which CPython auto-imports at start-up,
        so no change to the framework's own entrypoint is needed.
    """
    from hyperloom.inference_optimizer.session.paths import asset_root

    return asset_root() / "assets" / "host_probe"


def probe_enabled() -> bool:
    """Return True when the host probe should be installed for this run.

    Returns:
        True unless :data:`ENABLE_ENV` is explicitly set to a falsey token.
    """
    raw = str(os.environ.get(ENABLE_ENV, "")).strip().lower()
    return raw not in ("0", "false", "no", "off")


def deep_probe_enabled() -> bool:
    """Return True when the tier-2 (argument-fingerprinting) hook is requested.

    Returns:
        True when :data:`DEEP_ENV` is set to a truthy token.
    """
    raw = str(os.environ.get(DEEP_ENV, "")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _is_package_root(path: str) -> bool:
    """Return whether ``path`` is an interpreter package root rather than a package.

    PolicyGate's source-root allowlist legitimately contains ``dist-packages`` so a
    patch against an installed framework such as sglang or vllm can land. Reusing it
    for the probe's call-site attribution is a different question and the answer is
    no: torch is installed there too, so every collective attributes to a frame
    inside ``torch/distributed`` instead of to the framework helper that issued it.
    Torch is never a framework-rewrite target, and a root that matches it displaces
    the sites that are.

    Args:
        path: A candidate source root.

    Returns:
        True when the path is the package root itself; a specific package directory
        *inside* one (``dist-packages/sglang/``) returns False and is kept.
    """
    return PurePosixPath(path.rstrip("/")).name in _INTERPRETER_PACKAGE_DIRS


def build_probe_env(
    *,
    probe_dir: Path | str,
    source_roots: "list[str] | tuple[str, ...]",
    deep: bool = False,
) -> dict[str, str]:
    """Build the environment the benchmark process needs to run the probe.

    Args:
        probe_dir: Directory the per-rank reports are written into. Interpreter
            package roots are dropped; see :func:`_is_package_root`.
        source_roots: Framework source roots used to attribute call sites. With
            none supplied the probe still runs but attributes sites to whatever
            frame was innermost, which is rarely actionable.
        deep: Request the tier-2 hook.

    Returns:
        Environment variables to layer onto the benchmark process.         ``PYTHONPATH``
        is deliberately absent: it has to be *prepended* to whatever the
        materialized config already carries, which is the caller's job.
    """
    env: dict[str, str] = {
        "HYPERLOOM_HOST_PROBE": "1",
        "HYPERLOOM_HOST_PROBE_DIR": str(probe_dir),
    }
    roots = [r for r in (str(r).strip() for r in (source_roots or [])) if r and not _is_package_root(r)]
    if roots:
        env["HYPERLOOM_HOST_PROBE_ROOTS"] = os.pathsep.join(roots)
    if deep:
        env["HYPERLOOM_HOST_PROBE_DEEP"] = "1"
    return env


def promote_evidence_path(shared_state: Any, result: dict[str, Any] | None) -> str:
    """Lift an executor result's evidence path onto SharedState, if it carries one.

    Both the standalone ``profile`` action and the composite ``roofline`` action
    (which runs profile internally) can produce the document, so both have to
    promote it. Leaving that to one of them is how a live session ended up with 29
    measured candidates on disk and a specialist reporting that no host-side
    evidence was available: the roofline path never looked for it, and the prompt
    renderer reads SharedState, not the filesystem.

    Args:
        shared_state: The SharedState to update.
        result: An executor result that may carry ``framework_rewrite_evidence``.

    Returns:
        The promoted path, or ``""`` when the result carries none — in which case
        any path already on record is left alone, because a leg that produced no
        document is not evidence that the previous one was wrong.
    """
    path = str((result or {}).get("framework_rewrite_evidence") or "").strip()
    if not path:
        return ""
    shared_state.last_framework_rewrite_evidence = path
    return path


def _round(value: float, digits: int = 4) -> float:
    """Round ``value`` defensively.

    Args:
        value: Number to round.
        digits: Decimal places.

    Returns:
        The rounded value, or ``0.0`` when ``value`` is not a real number.
    """
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def read_probe_reports(probe_dir: Path | str) -> list[dict[str, Any]]:
    """Read every per-rank host-probe report in ``probe_dir``.

    Args:
        probe_dir: Directory the probe wrote its per-rank JSON into.

    Returns:
        The parsed reports, ordered by rank. Unreadable or non-conforming files
        are skipped with a warning: partial rank coverage still yields usable
        evidence, and a truncated file from a killed rank must not lose the rest.
    """
    root = Path(probe_dir)
    if not root.is_dir():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(root.glob(PROBE_FILE_GLOB)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("framework rewrite evidence: cannot read %s: %s", path, exc)
            continue
        if not isinstance(payload, dict) or not payload.get("schema", "").startswith("hyperloom.host_probe/"):
            log.warning("framework rewrite evidence: %s is not a host-probe report", path)
            continue
        reports.append(payload)
    reports.sort(key=lambda row: int(row.get("rank") or 0))
    return reports


def _merge_host_calls(reports: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Merge per-rank host-API rows into one table keyed by ``(api, site)``.

    Counts and wall time are averaged across the ranks that reported the site
    rather than summed, so a number stays comparable to one run's cost no matter
    how many ranks the workload used.

    Args:
        reports: Parsed per-rank host-probe reports.

    Returns:
        Mapping of ``(api, site)`` to merged statistics.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for report in reports:
        for row in report.get("host_calls") or []:
            if not isinstance(row, dict):
                continue
            key = (str(row.get("api") or ""), str(row.get("site") or ""))
            if not key[0]:
                continue
            entry = merged.setdefault(
                key,
                {
                    "api": key[0],
                    "site": key[1],
                    "ranks": 0,
                    "count": 0,
                    "wall_s": 0.0,
                    "bytes": 0,
                    "shape_sigs": set(),
                    "callers": set(),
                    "first_s": None,
                    "last_s": None,
                },
            )
            entry["ranks"] += 1
            entry["count"] += int(row.get("count") or 0)
            entry["wall_s"] += float(row.get("wall_s") or 0.0)
            entry["bytes"] += int(row.get("bytes") or 0)
            entry["shape_sigs"].update(str(s) for s in (row.get("shape_sigs") or []))
            entry["callers"].update(str(s) for s in (row.get("callers") or []))
            first = row.get("first_s")
            last = row.get("last_s")
            if isinstance(first, (int, float)) and first >= 0:
                entry["first_s"] = first if entry["first_s"] is None else min(entry["first_s"], first)
            if isinstance(last, (int, float)) and last >= 0:
                entry["last_s"] = last if entry["last_s"] is None else max(entry["last_s"], last)
    for entry in merged.values():
        ranks = max(1, int(entry["ranks"]))
        entry["count_per_rank"] = entry["count"] // ranks
        entry["wall_s_per_rank"] = _round(entry["wall_s"] / ranks, 6)
    return merged


def _hot_loop_start(*tables: dict[Any, dict[str, Any]]) -> float | None:
    """Return the timestamp at which the hot loop began, or None when undecidable.

    The hottest site by per-rank call count is taken as the anchor: with a loop
    product of blocks x steps x chunks it cannot be anywhere but the innermost loop.
    Its first call is therefore the moment steady-state work started, and anything
    that had already finished by then was preparation. See
    :data:`HOT_LOOP_DOMINANCE_RATIO` for why this replaced two timeline-position
    anchors that real data disproved.

    Both probe tiers timestamp against the same base, and the hot loop is a property
    of the process rather than of one table, so the anchor is derived across all of
    them — a tier-2 function can easily run more often than any wrapped host API.

    Args:
        *tables: Merged tables from :func:`_merge_host_calls` and
            :func:`_merge_framework_calls`.

    Returns:
        The hottest site's ``first_s``, or ``None`` when no report carried timestamps
        or no site dominates by call count (a flat distribution has no hot loop to
        anchor on, and guessing one would mislabel real findings).
    """
    timed = [
        entry
        for table in tables
        for entry in table.values()
        if isinstance(entry.get("first_s"), (int, float)) and float(entry["first_s"]) >= 0
    ]
    if not timed:
        return None
    counts = sorted((int(entry.get("count_per_rank") or 0) for entry in timed), reverse=True)
    if not counts or counts[0] <= 0:
        return None
    median = counts[len(counts) // 2] or 1
    if counts[0] / median < HOT_LOOP_DOMINANCE_RATIO:
        return None
    hottest = max(timed, key=lambda entry: int(entry.get("count_per_rank") or 0))
    return float(hottest["first_s"])


def _stops_before_hot_loop(entry: dict[str, Any], hot_loop_start: float | None) -> bool:
    """Return whether a site had already stopped being called when the hot loop began.

    Args:
        entry: A merged host-call or framework-call entry.
        hot_loop_start: Output of :func:`_hot_loop_start`.

    Returns:
        True only on positive evidence. A missing anchor or a report without
        timestamps yields False, so a site is never called set-up on absent data.
    """
    last = entry.get("last_s")
    if hot_loop_start is None or not isinstance(last, (int, float)) or float(last) < 0:
        return False
    return float(last) < hot_loop_start


def _merge_framework_calls(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge per-rank framework-function rows into one table keyed by function.

    Repeat rates are averaged over the reporting ranks; each rank samples the
    same code under the same workload, so an average is the right summary and a
    sum would be meaningless.

    Args:
        reports: Parsed per-rank host-probe reports.

    Returns:
        Mapping of ``file:line:name`` to merged statistics.
    """
    merged: dict[str, dict[str, Any]] = {}
    for report in reports:
        for row in report.get("framework_calls") or []:
            if not isinstance(row, dict):
                continue
            key = str(row.get("function") or "")
            if not key:
                continue
            entry = merged.setdefault(
                key,
                {
                    "function": key,
                    "ranks": 0,
                    "count": 0,
                    "wall_s": 0.0,
                    "arg_samples": 0,
                    "strict_repeat_sum": 0.0,
                    "loose_repeat_sum": 0.0,
                    "first_s": None,
                    "last_s": None,
                },
            )
            entry["ranks"] += 1
            entry["count"] += int(row.get("count") or 0)
            entry["wall_s"] += float(row.get("wall_s") or 0.0)
            entry["arg_samples"] += int(row.get("arg_samples") or 0)
            entry["strict_repeat_sum"] += float(row.get("strict_repeat_rate") or 0.0)
            entry["loose_repeat_sum"] += float(row.get("loose_repeat_rate") or 0.0)
            first = row.get("first_s")
            last = row.get("last_s")
            if isinstance(first, (int, float)) and first >= 0:
                entry["first_s"] = first if entry["first_s"] is None else min(entry["first_s"], first)
            if isinstance(last, (int, float)) and last >= 0:
                entry["last_s"] = last if entry["last_s"] is None else max(entry["last_s"], last)
    for entry in merged.values():
        ranks = max(1, int(entry["ranks"]))
        entry["count_per_rank"] = entry["count"] // ranks
        entry["wall_s_per_rank"] = _round(entry["wall_s"] / ranks, 6)
        entry["strict_repeat_rate"] = _round(entry["strict_repeat_sum"] / ranks)
        entry["loose_repeat_rate"] = _round(entry["loose_repeat_sum"] / ranks)
    return merged


def _candidate(
    *,
    category: str,
    site: str,
    signal: str,
    count: int,
    wall_s: float,
    wall_pct: float,
    extra: dict[str, Any] | None = None,
    setup_phase: bool = False,
    last_call_s: float | None = None,
    hot_loop_start_s: float | None = None,
) -> dict[str, Any]:
    """Assemble one rewrite candidate row.

    Args:
        category: One of the ``CATEGORY_*`` ids.
        site: The framework source location the candidate applies to.
        signal: Human-readable statement of the measured evidence.
        count: Per-rank call count backing the candidate.
        wall_s: Per-rank wall seconds backing the candidate.
        wall_pct: ``wall_s`` as a percentage of the measured run.
        extra: Additional category-specific fields.
        setup_phase: Whether the site had stopped being called before the hot loop
            began.
        last_call_s: The site's last observed call timestamp, probe-relative.
        hot_loop_start_s: When the hot loop began; see :func:`_hot_loop_start`.

    Returns:
        The candidate row.
    """
    row: dict[str, Any] = {
        "category": category,
        "taxonomy": _CATEGORY_TAXONOMY.get(category, "?"),
        "site": site,
        "signal": signal,
        "count_per_rank": int(count),
        "wall_s_per_rank": _round(wall_s, 6),
        "wall_pct": _round(wall_pct, 2),
        "suggested_rewrite": _CATEGORY_RECIPE.get(category, ""),
    }
    if last_call_s is not None:
        row["last_call_s"] = _round(last_call_s, 3)
    if hot_loop_start_s is not None:
        row["hot_loop_start_s"] = _round(hot_loop_start_s, 3)
    if setup_phase:
        row["setup_phase"] = True
        row["signal"] = (
            f"{signal} NOTE: this site stopped being called at "
            f"{_round(last_call_s or 0.0, 1)}s, before the hot loop started at "
            f"{_round(hot_loop_start_s or 0.0, 1)}s, so it is not steady-state per-step "
            f"work: either one-time set-up (weight loading, model construction) or work "
            f"confined to an earlier phase. Rewriting it does not change steady-state "
            f"throughput."
        )
    if extra:
        row.update(extra)
    return row


def _host_call_candidates(
    merged: dict[tuple[str, str], dict[str, Any]],
    wall_seconds: float,
    hot_loop_start: float | None = None,
) -> list[dict[str, Any]]:
    """Classify merged host-API rows into rewrite candidates.

    Args:
        merged: Output of :func:`_merge_host_calls`.
        wall_seconds: Measured run wall seconds, used for the percentage column.
        hot_loop_start: When the hot loop began, derived across every merged table.
            Defaults to deriving it from ``merged`` alone.

    Returns:
        Unsorted candidate rows.
    """
    out: list[dict[str, Any]] = []
    denom = wall_seconds if wall_seconds > 0 else 0.0
    if hot_loop_start is None:
        hot_loop_start = _hot_loop_start(merged)
    for entry in merged.values():
        api = str(entry["api"])
        count = int(entry["count_per_rank"])
        if count < MIN_HOST_CALLS:
            continue
        wall = float(entry["wall_s_per_rank"])
        pct = (wall / denom * 100.0) if denom else 0.0
        setup_phase = _stops_before_hot_loop(entry, hot_loop_start)
        if api in _OBJECT_COLLECTIVE_APIS:
            out.append(
                _candidate(
                    category=CATEGORY_HOST_ROUND_TRIP,
                    site=entry["site"],
                    signal=(
                        f"{api} called {count} times per rank, {wall:.2f}s host wall "
                        f"({pct:.1f}% of the run). Object collectives pickle through "
                        f"the host, so every call is a round-trip."
                    ),
                    count=count,
                    wall_s=wall,
                    wall_pct=pct,
                    extra={"api": api, "callers": sorted(entry["callers"])},
                    setup_phase=setup_phase,
                    last_call_s=entry.get("last_s"),
                    hot_loop_start_s=hot_loop_start,
                )
            )
        elif api in _HOST_SYNC_APIS:
            out.append(
                _candidate(
                    category=CATEGORY_HOST_SYNC,
                    site=entry["site"],
                    signal=(
                        f"{api} called {count} times per rank, {wall:.2f}s host wall "
                        f"({pct:.1f}% of the run). Each call stalls the pipeline until "
                        f"the device catches up."
                    ),
                    count=count,
                    wall_s=wall,
                    wall_pct=pct,
                    extra={"api": api, "callers": sorted(entry["callers"])},
                    setup_phase=setup_phase,
                    last_call_s=entry.get("last_s"),
                    hot_loop_start_s=hot_loop_start,
                )
            )
        elif api in _H2D_APIS:
            mib = float(entry["bytes"]) / max(1, int(entry["ranks"])) / (1024.0 * 1024.0)
            out.append(
                _candidate(
                    category=CATEGORY_DEVICE_RESIDENT,
                    site=entry["site"],
                    signal=(
                        f"{api} performed {count} host-to-device copies per rank "
                        f"({mib:.1f} MiB, {wall:.2f}s, {pct:.1f}% of the run) from a "
                        f"CPU-resident source."
                    ),
                    count=count,
                    wall_s=wall,
                    wall_pct=pct,
                    extra={"api": api, "mib_per_rank": _round(mib, 2), "callers": sorted(entry["callers"])},
                    setup_phase=setup_phase,
                    last_call_s=entry.get("last_s"),
                    hot_loop_start_s=hot_loop_start,
                )
            )
    out.extend(_fusion_candidates(merged, wall_seconds))
    return out


def _fusion_candidates(
    merged: dict[tuple[str, str], dict[str, Any]],
    wall_seconds: float,
) -> list[dict[str, Any]]:
    """Find tensor collectives issued from several adjacent same-shape sites.

    A collective wrapped in a framework helper attributes to one line inside
    that helper however many times it is called, so the enclosing frames are
    what distinguish three adjacent same-shape collectives (fusable into one
    padded call) from a single collective in a loop (not fusable).

    Args:
        merged: Output of :func:`_merge_host_calls`.
        wall_seconds: Measured run wall seconds, used for the percentage column.

    Returns:
        Unsorted fusion candidate rows.
    """
    out: list[dict[str, Any]] = []
    denom = wall_seconds if wall_seconds > 0 else 0.0
    for entry in merged.values():
        api = str(entry["api"])
        if api not in _TENSOR_COLLECTIVE_APIS:
            continue
        callers = sorted(entry["callers"])
        shape_sigs = sorted(entry["shape_sigs"])
        # Same enclosing function, different lines: adjacent calls in one body.
        by_function: dict[str, set[str]] = {}
        for caller in callers:
            parts = caller.rsplit(":", 2)
            if len(parts) != 3:
                continue
            by_function.setdefault(f"{parts[0]}:{parts[2]}", set()).add(caller)
        fusable = {fn: lines for fn, lines in by_function.items() if len(lines) >= MIN_FUSION_SITES}
        if not fusable or len(shape_sigs) != 1:
            continue
        wall = float(entry["wall_s_per_rank"])
        pct = (wall / denom * 100.0) if denom else 0.0
        for function, lines in sorted(fusable.items()):
            out.append(
                _candidate(
                    category=CATEGORY_FUSE_COLLECTIVES,
                    site=function,
                    signal=(
                        f"{api} is issued from {len(lines)} separate lines in "
                        f"{function} with one identical payload shape "
                        f"({shape_sigs[0]}); {int(entry['count_per_rank'])} calls per "
                        f"rank total, {wall:.2f}s ({pct:.1f}% of the run)."
                    ),
                    count=int(entry["count_per_rank"]),
                    wall_s=wall,
                    wall_pct=pct,
                    extra={
                        "api": api,
                        "call_lines": sorted(lines),
                        "shape_signature": shape_sigs[0],
                    },
                )
            )
    return out


def _framework_call_candidates(
    merged: dict[str, dict[str, Any]],
    wall_seconds: float,
    hot_loop_start: float | None = None,
) -> list[dict[str, Any]]:
    """Classify merged framework-function rows into rewrite candidates.

    Args:
        merged: Output of :func:`_merge_framework_calls`.
        wall_seconds: Measured run wall seconds, used for the percentage column.
        hot_loop_start: When the hot loop began, derived across every merged table.
            Defaults to deriving it from ``merged`` alone.

    Returns:
        Unsorted candidate rows.
    """
    out: list[dict[str, Any]] = []
    denom = wall_seconds if wall_seconds > 0 else 0.0
    if hot_loop_start is None:
        hot_loop_start = _hot_loop_start(merged)
    for entry in merged.values():
        count = int(entry["count_per_rank"])
        if count < MIN_FRAMEWORK_CALLS or int(entry["arg_samples"]) <= 0:
            continue
        strict = float(entry["strict_repeat_rate"])
        loose = float(entry["loose_repeat_rate"])
        wall = float(entry["wall_s_per_rank"])
        pct = (wall / denom * 100.0) if denom else 0.0
        setup_phase = _stops_before_hot_loop(entry, hot_loop_start)
        if strict >= MIN_STRICT_REPEAT_RATE:
            out.append(
                _candidate(
                    category=CATEGORY_MEMOIZE,
                    site=entry["function"],
                    signal=(
                        f"called {count} times per rank ({wall:.2f}s, {pct:.1f}% of the "
                        f"run); {strict * 100:.0f}% of sampled calls repeated an "
                        f"argument identity already seen, so that share of the work is "
                        f"recomputation of a value the process already had. Only "
                        f"actionable if the callee is a PURE function of its arguments — "
                        f"the probe cannot tell purity, and a coarse-grained method (a "
                        f"module's forward, anything that mutates state or reads a "
                        f"global) will show the same repeat rate while being impossible "
                        f"to memoize. Check before rewriting."
                    ),
                    count=count,
                    wall_s=wall,
                    wall_pct=pct,
                    extra={
                        "strict_repeat_rate": strict,
                        "loose_repeat_rate": loose,
                        "arg_samples": int(entry["arg_samples"]),
                    },
                    setup_phase=setup_phase,
                    last_call_s=entry.get("last_s"),
                    hot_loop_start_s=hot_loop_start,
                )
            )
        elif loose >= MIN_LOOSE_REPEAT_RATE:
            pure_enabler = strict <= MAX_STRICT_REPEAT_FOR_PURE_ENABLER
            out.append(
                _candidate(
                    category=CATEGORY_HOIST,
                    site=entry["function"],
                    signal=(
                        f"called {count} times per rank ({wall:.2f}s, {pct:.1f}% of the "
                        f"run); {loose * 100:.0f}% of sampled calls repeated an argument "
                        f"shape/dtype/device signature but only {strict * 100:.0f}% "
                        f"received the same tensor object again. The arguments look like "
                        f"the same value rebuilt every iteration, so memoizing this alone "
                        f"would never hit. Confirm against the source: the probe does not "
                        f"read tensor contents, because doing so would inject the very "
                        f"host stalls it measures."
                    ),
                    count=count,
                    wall_s=wall,
                    wall_pct=pct,
                    extra={
                        "strict_repeat_rate": strict,
                        "loose_repeat_rate": loose,
                        "arg_samples": int(entry["arg_samples"]),
                        "enabler": pure_enabler,
                    },
                    setup_phase=setup_phase,
                    last_call_s=entry.get("last_s"),
                    hot_loop_start_s=hot_loop_start,
                )
            )
    return out


def build_evidence(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the merged rewrite-evidence document from per-rank reports.

    Args:
        reports: Parsed per-rank host-probe reports.

    Returns:
        The evidence document. When ``reports`` is empty the document still
        carries its schema and an explanatory note, so a consumer can tell "the
        probe found nothing" apart from "the probe never ran".
    """
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not reports:
        return {
            "schema": SCHEMA,
            "generated_at": generated_at,
            "ranks_merged": 0,
            "wall_seconds": 0.0,
            "roots": [],
            "candidates": [],
            "deep_probe_ran": False,
            "notes": [
                "no host-probe reports found; the probe was not enabled for this "
                "run, or every rank exited before writing its report"
            ],
        }

    wall_seconds = max((float(r.get("wall_seconds") or 0.0) for r in reports), default=0.0)
    roots: list[str] = []
    for report in reports:
        for root in report.get("roots") or []:
            if root not in roots:
                roots.append(str(root))

    host_merged = _merge_host_calls(reports)
    framework_merged = _merge_framework_calls(reports)

    hot_loop_start = _hot_loop_start(host_merged, framework_merged)
    candidates = _host_call_candidates(host_merged, wall_seconds, hot_loop_start)
    candidates.extend(_framework_call_candidates(framework_merged, wall_seconds, hot_loop_start))
    timestamped = any(
        isinstance(entry.get("first_s"), (int, float)) and float(entry["first_s"]) >= 0
        for table in (host_merged, framework_merged)
        for entry in table.values()
    )
    # Hot-path work first, then wall time, then call count. Set-up work is sorted
    # to the bottom rather than dropped: on a real run, loading a 34 GB checkpoint
    # took the top four slots on transferred bytes alone and pushed the genuine
    # per-step findings below them, which is exactly the attention a specialist
    # cannot afford to spend. The count tie-break keeps a site the probe could not
    # time (a collective whose cost lands in the next kernel launch) ranked by how
    # often it runs instead of dropping to the bottom.
    candidates.sort(
        key=lambda row: (
            0 if row.get("setup_phase") else 1,
            row["wall_s_per_rank"],
            row["count_per_rank"],
        ),
        reverse=True,
    )
    truncated = len(candidates) > MAX_CANDIDATES
    candidates = candidates[:MAX_CANDIDATES]
    for index, row in enumerate(candidates, start=1):
        row["rank"] = index

    deep_ran = any(bool(r.get("framework_calls")) for r in reports)
    notes: list[str] = []
    if not deep_ran:
        notes.append(
            "the deep probe did not run (HYPERLOOM_HOST_PROBE_DEEP unset), so "
            "memoization and loop-hoisting candidates are absent from this report; "
            "only host round-trips, host syncs, host-to-device copies and "
            "collective fusion were observable"
        )
    if any(r.get("roots_unset") for r in reports):
        notes.append(
            "at least one rank ran without HYPERLOOM_HOST_PROBE_ROOTS, so its call "
            "sites are attributed to whatever frame was innermost rather than to "
            "framework source"
        )
    if any((r.get("truncated") or {}).get("host_calls") for r in reports):
        notes.append("host-call site table hit its cap on at least one rank; raise HYPERLOOM_HOST_PROBE_MAX_SITES")
    if any((r.get("truncated") or {}).get("framework_calls") for r in reports):
        notes.append("framework-call table hit its cap on at least one rank; raise HYPERLOOM_HOST_PROBE_MAX_SITES")
    if truncated:
        notes.append(f"candidate list truncated to the {MAX_CANDIDATES} costliest")
    if timestamped and hot_loop_start is None:
        notes.append(
            "no dominant call site, so the hot loop's start could not be located and "
            "no candidate was marked as set-up work; judge per-step relevance from the "
            "call counts, which are reported per candidate"
        )
    notes.append(
        "vendor-kernel substitution and no-op glue removal are not host-observable; "
        "take those from the GPU kernel breakdown and from reading the source"
    )
    notes.append(
        "the host-sync candidates below are the ones that go through a Python "
        "method; a tensor passed where the C++ argument parser wants a scalar "
        "(torch.full((n,), t) with a 0-dim device tensor, and similar) converts "
        "inside ATen without calling any Python method, so an absent sync here is "
        "not proof of absence — check the per-step path for those by reading it"
    )
    for report in reports:
        for probe_note in report.get("notes") or []:
            text = f"rank {report.get('rank')}: {probe_note}"
            if text not in notes:
                notes.append(text)

    return {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "ranks_merged": len(reports),
        "wall_seconds": _round(wall_seconds, 3),
        "roots": roots,
        "candidates": candidates,
        "deep_probe_ran": deep_ran,
        "notes": notes,
    }


def aggregate_probe_dir(probe_dir: Path | str, out_path: Path | str) -> dict[str, Any]:
    """Read a probe directory, build the evidence document and write it.

    Args:
        probe_dir: Directory holding the per-rank host-probe reports.
        out_path: Destination for the merged evidence JSON.

    Returns:
        The evidence document, whether or not the write succeeded (the caller
        decides how loudly to complain about a failed write).
    """
    evidence = build_evidence(read_probe_reports(probe_dir))
    try:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        log.warning("framework rewrite evidence: cannot write %s: %s", out_path, exc)
    return evidence


def summarize_for_prompt(evidence: dict[str, Any], *, limit: int = 12) -> str:
    """Render the top candidates as a compact block for a specialist prompt.

    Args:
        evidence: An evidence document from :func:`build_evidence`.
        limit: Maximum candidates rendered.

    Returns:
        A plain-text block, or ``""`` when there is nothing worth rendering.
    """
    candidates = evidence.get("candidates") if isinstance(evidence, dict) else None
    if not isinstance(candidates, list) or not candidates:
        return ""
    lines = [
        "HOST-SIDE REWRITE EVIDENCE (measured, this workload, per rank).",
        "Each row is a measured inefficiency that does not appear in the GPU",
        "kernel breakdown. `taxonomy` names the rewrite pattern it instantiates.",
        "",
    ]
    for row in candidates[:limit]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"[{row.get('rank')}] {row.get('category')} (taxonomy {row.get('taxonomy')}) "
            f"at {row.get('site')}"
        )
        lines.append(f"    evidence: {row.get('signal')}")
        if row.get("enabler"):
            lines.append(
                "    NOTE: enabler — expect no standalone gain. Declare what it "
                "unlocks in `enables` so it is measured as part of that bundle."
            )
        recipe = str(row.get("suggested_rewrite") or "").strip()
        if recipe:
            lines.append(f"    shape of the fix: {recipe}")
        lines.append("")
    for note in evidence.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "CATEGORY_DEVICE_RESIDENT",
    "CATEGORY_FUSE_COLLECTIVES",
    "CATEGORY_HOIST",
    "CATEGORY_HOST_ROUND_TRIP",
    "CATEGORY_HOST_SYNC",
    "CATEGORY_MEMOIZE",
    "MAX_CANDIDATES",
    "PROBE_FILE_GLOB",
    "SCHEMA",
    "aggregate_probe_dir",
    "build_evidence",
    "read_probe_reports",
    "summarize_for_prompt",
]
