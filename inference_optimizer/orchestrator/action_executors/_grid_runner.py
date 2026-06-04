"""Shared helper for the ``explore`` executor's grid runs.

The job is essentially: take a base Magpie YAML + a list of
(name, extra_server_args, extra_envs) variants, run Magpie once per
variant, parse `benchmark_report.json`, return the winners.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ._robustness_pulse import pulse as _robustness_pulse
from ._subprocess_kill import OVERTIME_KILL_RETURNCODE, run_with_session_kill
from .benchmark_result import (
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Content-based variant fingerprint (cross-action dedup ledger key).
#
# Identity-by-name was too easy for an LLM-supplied grid to bypass: re-emitting
# an already-tested ``--block-size 128`` under a freshly invented name silently
# re-burned the wall clock. The fingerprint hashes the *content* that actually
# changes Magpie behavior (server args + env overrides) so any rename ends up
# at the same key in ``SharedState.explore_search.tested``.
#
# Normalization rules:
#   * args: ``shlex.split`` → sorted token tuple. Sorting is intentionally
#     aggressive — two flag strings differing only in token order produce the
#     same fingerprint. Real "later wins" overrides should be expressed as a
#     single flag with the final value, not a re-emit in a different order;
#     this is documented in the Orchestration IR-26 prompt.
#   * envs: ``(str(k), str(v))`` pairs sorted by key. ``str()`` matches the
#     ``_baseline_params_fingerprint`` convention so ``"1"`` and ``1`` collide
#     (Magpie ultimately sees the value as a shell-exported string anyway).
#   * 16-char SHA-1 prefix: collision-resistant enough for a per-session dedup
#     ledger while staying compact in ``state.json`` and prompt summaries.
# ---------------------------------------------------------------------------
def variant_fingerprint(
    extra_server_args: str | None,
    extra_envs: dict[str, Any] | None,
) -> str:
    """Stable content fingerprint for a (extra_server_args, extra_envs) pair.

    See module-level rationale. Name and note are intentionally NOT part of
    the input — two variants with identical content but different names
    (e.g. ``A`` and ``A_v2``) must collapse to the same fingerprint.
    """
    args_text = str(extra_server_args or "")
    try:
        args_tokens = sorted(shlex.split(args_text))
    except ValueError:
        # Unbalanced quotes / shell-parse failure: fall back to a stable
        # whitespace split so we still produce *some* fingerprint instead
        # of crashing the grid pre-flight. Callers can still distinguish
        # different bad strings; identical bad strings still collide.
        args_tokens = sorted(args_text.split())
    env_pairs = sorted(
        (str(k), str(v)) for k, v in (extra_envs or {}).items()
    )
    payload = json.dumps(
        [args_tokens, [list(p) for p in env_pairs]],
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_magpie_python() -> str:
    """Resolve the Python interpreter for Magpie subprocesses.

    Order: $MAGPIE_PYTHON env > first `python3` on PATH that can
    ``import Magpie`` > /opt/venv/bin/python (if it exists).
    """
    env_val = os.environ.get("MAGPIE_PYTHON", "").strip()
    if env_val:
        return env_val

    def _can_import_magpie(py: str) -> bool:
        try:
            proc = run_with_session_kill(
                [py, "-c", "import Magpie"],
                capture_output=True, timeout=10,
            )
            return getattr(proc, "returncode", 1) == 0
        except Exception:
            return False

    candidate = shutil.which("python3")
    if candidate and _can_import_magpie(candidate):
        return candidate

    fallback = Path("/opt/venv/bin/python")
    if fallback.exists():
        return str(fallback)

    return candidate or "/opt/venv/bin/python"


def _resolve_session_dir() -> Path:
    """Resolve the active session_dir for executors that need an output root.

    Reads :func:`inference_optimizer.paths.session_dir`; this honors
    ``$USER_DATA_PATH`` and otherwise returns ``/workspace/hyperloom``.
    Used by executor-class fallback paths when ``ctx.extra["workspace"]``
    was not pre-mkdir'd by SubAgentRunner.
    """
    from ...paths import session_dir as _sd
    return _sd()


_MAGPIE_CWD_DEFAULT = "/tmp"
_VARIANT_TIMEOUT_SEC_DEFAULT = 7800  # 130 min; matches BASELINE_DEFAULT_TIMEOUT_SEC for Qwen3-32B TP=1 CONC=64 ISL/OSL=1024 NUM_PROMPTS=320 workload


# ---------------------------------------------------------------------------
# User-declared variant skip list
#
# Operators (or the brain agent via the prompt's Environment block) can
# pre-prune the search grid by declaring SKIP_VARIANTS. The value is a
# comma/whitespace-separated list of variant patterns; each pattern is
# matched against ``GridVariant.name`` either exactly or as a fnmatch glob
# (``*`` / ``?`` / ``[abc]`` supported). Empty patterns are ignored.
#
# Examples
# --------
#   SKIP_VARIANTS=attn_aiter                 # exact name
#   SKIP_VARIANTS=attn_aiter,sched_dfs       # two exact names
#   SKIP_VARIANTS=attn_*,vllm_aiter_fp8bmm   # glob + exact mixed
#
# Resolution order (most-specific wins):
#   params["skip_variants"]  >  $SKIP_VARIANTS  >  ""
#
# The helper is intentionally *only* a name-based filter; no model/TP
# predicates here. Model-aware static rules live in each executor's own
# ``_filter_incompatible_variants`` (kept as a safety net) and will
# eventually migrate to the KB.
import fnmatch as _fnmatch  # noqa: E402  (kept near callers for grep-ability)


def resolve_skip_spec(params: dict | None) -> str:
    """Resolve the active skip spec from task params + process env.

    ``params["skip_variants"]`` may be a list[str] or a single str; both are
    flattened to comma-joined form before pattern parsing.
    """
    val = ""
    if params and "skip_variants" in params:
        raw = params.get("skip_variants")
        if isinstance(raw, (list, tuple)):
            val = ",".join(str(x) for x in raw if x is not None)
        elif raw is not None:
            val = str(raw)
    if not val.strip():
        val = os.environ.get("SKIP_VARIANTS", "")
    return (val or "").strip()


def _parse_skip_spec(spec: str) -> list[str]:
    """Split ``spec`` on commas and whitespace; drop empties."""
    if not spec:
        return []
    out: list[str] = []
    for token in spec.replace("\n", ",").split(","):
        for sub in token.split():
            t = sub.strip()
            if t:
                out.append(t)
    return out


# Compiled here so callers don't re-compile per grid iteration.
# Matches both ``--cuda-graph-max-bs 64`` (space-separated) and
# ``--cuda_graph_max_bs=64`` (underscore + equals) so glob-style or
# pep8-style flag strings both parse. Captures the integer value.
_RE_CUDA_GRAPH_MAX_BS = re.compile(
    r"--cuda[-_]graph[-_]max[-_]bs[= ]+(\d+)"
)


def apply_multi_node_invalid_variants(
    grid: list["GridVariant"],
) -> tuple[list["GridVariant"], list[dict]]:
    """Drop variants known to underperform in multi-node mode.

    Currently enforces one rule, which has been confirmed empirically
    on the GLM-5 TP16 multi-node setup:

      - ``--cuda-graph-max-bs N`` where ``N < $CONC`` is auto-skipped.
        Justification: the cuda graph cache can hold only ``N`` distinct
        batch sizes; when the bench runs with concurrency > N, every
        cross-node decode tick misses the graph cache and falls back to
        eager mode, costing ~50% throughput. Three runs at
        ``max_bs`` ∈ {8, 16, 32} with CONC=64 produced ~245 tok/s vs the
        baseline 549 tok/s — i.e. 35-40 min per variant of certain
        regression. Single-node has the same theoretical issue but the
        per-variant cost is small enough (~10 min) that empirical
        confirmation is still useful, so the filter is multi-node-only.

    Returns ``(kept, dropped)`` matching ``apply_user_skip_list``'s
    shape. Single-node short-circuits to ``(list(grid), [])`` so the
    call site stays branch-free; the per-variant ``log.info`` in callers
    only fires for actual multi-node drops.
    """
    from ._multi_node_env import is_multi_node
    if not is_multi_node():
        return list(grid), []
    try:
        conc = int(os.environ.get("CONC", "64") or 64)
    except ValueError:
        conc = 64
    if conc <= 0:
        return list(grid), []
    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        m = _RE_CUDA_GRAPH_MAX_BS.search(v.extra_server_args or "")
        if m and int(m.group(1)) < conc:
            dropped.append({
                "name": v.name,
                "source": "multi_node_invalid",
                "reason": (
                    f"cuda_graph_max_bs={m.group(1)} < CONC={conc} "
                    "(multi-node graph-cache miss → known regression)"
                ),
            })
        else:
            kept.append(v)
    return kept, dropped


# Multi-node likely-winner priority for grid reordering. Tags are matched
# against ``GridVariant.name`` and ``GridVariant.note`` as case-insensitive
# substrings; earlier tags win. Rationale per tag:
#
# Params (`_MN_PARAMS_PRIORITY`):
#   * cuda_graph_max_bs — matches CONC is the empirical single best
#     server-param lever in multi-node TP runs (24ldn implied +0.93%
#     vs baseline; A1 already pre-filters bs<CONC, leaving only the
#     viable cap as the leading candidate).
#   * mem_fraction — bumping mem-fraction-static unlocks KV-cache
#     headroom for long-context workloads; cheap to test, sometimes
#     big.
#   * max_num_seqs — marathon KB validated +84% on Kimi-K2.5; tier-1
#     KV-cache class param.
#   * decode_steps — continuous decode steps; modest leverage.
#   * schedule / nccl — historically marginal or negative.
#
# Backends (`_MN_BACKENDS_PRIORITY`):
#   * aiter — covers attn_aiter / decode_aiter / moe_aiter, the largest
#     historical multi-node wins (+10-30%).
#   * tier3_fusion — enable_fused_moe / enable_mixed_chunk.
#   * tier2_schedule — lpm / dfs / overlap policies.
#   * tier5_comm — custom_ar (small leverage, expensive to test).
#
# Names not matching any tag sort to the end in original order.
_MN_PARAMS_PRIORITY: tuple[str, ...] = (
    "cuda_graph_max_bs",
    "mem_fraction",
    "max_num_seqs",
    "decode_steps",
    "schedule",
    "nccl",
)
_MN_BACKENDS_PRIORITY: tuple[str, ...] = (
    "aiter",
    "tier3_fusion",
    "tier2_schedule",
    "tier5_comm",
)


def reorder_grid_for_multi_node(
    grid: list["GridVariant"],
    *,
    priority_tags: tuple[str, ...],
) -> list["GridVariant"]:
    """Reorder grid so likely-winners run first in multi-node mode.

    Single-node short-circuits to ``list(grid)`` (preserves the original
    DEFAULT_*_GRID order bit-for-bit). Multi-node sorts each variant
    into a bucket by the first ``priority_tag`` that appears as a
    case-insensitive substring of ``variant.name`` or ``variant.note``.
    Variants matching no tag land at the end. Sort is stable so ties
    preserve original grid order.

    Why this matters for multi-node only: each variant costs ~35-40 min
    (cmd_restart_server + bench + cleanup), so a 5-6 hr grid easily
    hits the run's ``--max-hours`` cap before the likely winners get a
    chance. Reordering surfaces empirically-strong candidates in the
    first 1-3 rounds, leaving the long-tail variants to optional later
    rounds.
    """
    from ._multi_node_env import is_multi_node
    if not is_multi_node():
        return list(grid)

    def _priority(v: GridVariant) -> int:
        haystack = f"{v.name} {v.note or ''}".lower()
        for i, tag in enumerate(priority_tags):
            if tag.lower() in haystack:
                return i
        return len(priority_tags)

    return sorted(grid, key=_priority)


# ---------------------------------------------------------------------------
# Single-node + compatibility filters (companion to A1's multi-node filter).
#
# These two helpers protect single-node and incompatible-model paths from
# wasted sglang restarts on variants whose flags either (a) only make sense
# in multi-node, or (b) require a model class / sglang version that the
# current run does not have. Each fires BEFORE the variant is dispatched
# to a real benchmark, so a 5-10 min restart per filtered variant is saved.
#
# Both are conservative: probe failures (e.g. ``sglang --help`` not
# importable in the sandbox) fall through to "no filtering" so the
# downstream grid_runner's ``status="failed"`` + rejected-ledger path
# still handles the bad variant gracefully. Net effect: in the best
# case we save the time, in the worst case we waste one restart per
# bad variant (same as the no-filter baseline).
# ---------------------------------------------------------------------------


def apply_single_node_invalid_variants(
    grid: list["GridVariant"],
) -> tuple[list["GridVariant"], list[dict]]:
    """Drop variants whose ``note`` is ``multi_node_only_*`` when single-node.

    Companion to :func:`apply_multi_node_invalid_variants`. Variants the
    grid library has tagged with ``note="multi_node_only_..."`` (e.g.
    DeepEP MoE, ep_moe — flags that NCCL-cross-node-distribute MoE
    expert shards) are silently dropped in single-node mode where they
    would either reject the flag or no-op silently. Multi-node path
    returns ``(list(grid), [])`` so the multi-node grid is preserved
    bit-for-bit.

    The convention ``note="multi_node_only_*"`` is owned by the grid
    definitions in the ``explore`` executor; we never invent the
    classification here.
    """
    from ._multi_node_env import is_multi_node
    if is_multi_node():
        return list(grid), []
    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        note_l = (v.note or "").lower()
        if note_l.startswith("multi_node_only"):
            dropped.append({
                "name": v.name,
                "source": "single_node_invalid",
                "reason": (
                    f"variant note={v.note!r} is multi-node-only "
                    "(silently dropped in single-node path)"
                ),
            })
        else:
            kept.append(v)
    return kept, dropped


# Multi-node-hot sglang flags that depend on model class (MLA / MoE) or
# sglang version. Each entry maps a substring of ``extra_server_args`` to
# a compatibility predicate. Keep this list small and well-documented;
# anything more dynamic should live in the action_registry's
# ``applicable_when`` schema instead.
_COMPATIBILITY_FLAG_RULES: tuple[tuple[str, str], ...] = (
    ("--enable-flashinfer-mla", "mla"),
    ("--enable-deepep-moe",      "moe"),
    ("--enable-ep-moe",          "moe"),
)


# Per-framework cache for ``_probe_server_help_text`` — populated on
# first call per framework so we avoid spawning a subprocess per-variant.
# Cleared by ``importlib.reload`` during tests. Empty results are NOT
# cached so a transient failure (e.g. mocked subprocess raises once)
# re-probes on the next call. The single-key ``_SGLANG_HELP_CACHE``
# this replaces is preserved as a back-compat alias below; callers that
# pre-date the rename keep working through ``_probe_sglang_help_text``.
_HELP_TEXT_CACHE: dict[str, str] = {}

# Per-framework subprocess command for ``--help`` text extraction. Each
# command must be a single-shot ``python3 -c <inline>`` invocation so
# the probe's 10-second timeout covers the import cost. Failure paths
# (importerror / argparse exit / etc.) are captured by the broad
# ``except Exception`` in ``_probe_server_help_text``.
_HELP_PROBE_COMMANDS: dict[str, tuple[str, ...]] = {
    "sglang": (
        "python3", "-c",
        "from sglang.launch_server import parser; parser.print_help()",
    ),
    "vllm": (
        "python3", "-c",
        "from vllm.entrypoints.openai.api_server import make_arg_parser; "
        "make_arg_parser(None).print_help()",
    ),
    # atom branch: the audited atom version exposes EngineArgs.add_cli_args
    # on ``atom.model_engine.arg_utils`` (mirrors vLLM's EngineArgs). Build
    # a throwaway ArgumentParser, let atom populate it, and print the help
    # surface for substring matching against grid-variant flag literals.
    "atom": (
        "python3", "-c",
        "import argparse; from atom.model_engine.arg_utils import EngineArgs; "
        "p = argparse.ArgumentParser(); EngineArgs.add_cli_args(p); "
        "p.print_help()",
    ),
}


def _probe_server_help_text(framework: str) -> str:
    """Best-effort fetch of ``<framework> --help`` text for grid-variant
    flag validation.

    Supported frameworks: ``sglang``, ``vllm``, ``atom``. Unknown values
    return ``""`` (defer to graceful runtime failure). The cache is
    keyed by framework so a multi-framework test box doesn't leak the
    first-probed framework's output into the second's slot.

    Returns ``""`` on ANY failure (subprocess timeout, framework not
    importable in the current Python, sandbox without the framework
    installed, test-time subprocess mocks that mis-handle this probe's
    argv shape, ValueError from a too-strict mock side_effect, etc.).
    Callers MUST treat empty as "I don't know what this framework
    supports" and fall through to NOT filtering. Empty results are NOT
    cached so a transient mock-side failure does not poison the cache.

    The broad ``except Exception`` is deliberate: this probe is purely
    a perf optimisation (saves a wasted 10-min server restart per
    incompatible variant). It must NEVER crash the optimizer or fail
    a unit test that mocks ``subprocess.run`` for unrelated reasons.
    """
    fw = (framework or "").strip().lower()
    if fw in _HELP_TEXT_CACHE:
        return _HELP_TEXT_CACHE[fw]
    cmd = _HELP_PROBE_COMMANDS.get(fw)
    if cmd is None:
        return ""
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True, text=True, timeout=10,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
    except Exception:  # noqa: BLE001 — best-effort, see docstring
        out = ""
    if out:
        _HELP_TEXT_CACHE[fw] = out
    return out


def _probe_sglang_help_text() -> str:
    """Back-compat shim — defer to the framework-keyed probe.

    Pre-dates the multi-framework rename; kept so in-process tests that
    monkey-patch this exact name still work. New call sites should use
    ``_probe_server_help_text("sglang")`` directly.
    """
    return _probe_server_help_text("sglang")


def _detect_model_class(model_path: str) -> tuple[bool, bool]:
    """Heuristic detect of (is_mla_model, is_moe_model) from model path.

    Uses lowercased substring match on the model path. This is a cheap
    O(N) check intended to filter out an obviously-wrong variant before
    spending 10 min on a doomed sglang restart. False negatives (model
    we don't recognise) defer to graceful runtime failure; false
    positives (we mis-classify a model as MLA/MoE) cost one restart
    same as if we hadn't filtered.

    Known MLA models: DeepSeek (V2/V3/R1), GLM-5, Kimi-K2 — all share
    MLA-style multi-head latent attention. Known MoE models: anything
    in the MLA set + Qwen3-MoE.
    """
    p = model_path.lower()
    mla_keys = ("glm-5", "glm5", "deepseek", "kimi-k2", "kimi_k2", "kimi")
    moe_keys = (
        "glm-5", "glm5", "deepseek-v2", "deepseek-v3", "deepseek-r1",
        "kimi", "qwen3-moe", "qwen3_moe", "mixtral",
    )
    is_mla = any(k in p for k in mla_keys)
    is_moe = any(k in p for k in moe_keys)
    return is_mla, is_moe


def apply_compatibility_filter(
    grid: list["GridVariant"],
) -> tuple[list["GridVariant"], list[dict]]:
    """Skip variants known to be incompatible with current model/sglang.

    Two filter dimensions, each conservative on probe failure:

    1. **Model class** — variants requiring MLA attention (e.g.
       ``--enable-flashinfer-mla``) or expert-parallel MoE (e.g.
       ``--enable-deepep-moe``) are dropped when ``$MODEL_PATH`` lacks
       the corresponding model-family keyword. If ``$MODEL_PATH`` is
       unset, both predicates are assumed True (let the variant try).
    2. **sglang version** — variants whose flag literal does NOT appear
       in ``sglang launch_server --help`` output are dropped. If the
       help text can't be fetched (sglang not importable in sandbox),
       both predicates are assumed True (defer to graceful failure).

    Returns the same ``(kept, dropped)`` shape as
    ``apply_user_skip_list`` so callers can merge dropped entries
    uniformly.
    """
    model_path = os.environ.get("MODEL_PATH", "")
    if model_path:
        is_mla, is_moe = _detect_model_class(model_path)
    else:
        # No MODEL_PATH set -> can't detect -> assume compatible.
        is_mla, is_moe = True, True

    # Pick the live framework's --help text. Default to sglang so
    # existing test fixtures that don't pass ``framework=`` (and pre-
    # atom call sites) keep their old behaviour. atom / vllm flow in
    # through callers that thread the rendered ``benchmark.framework``
    # value down here.
    fw = (os.environ.get("FRAMEWORK", "") or "sglang").strip().lower()
    help_text = _probe_server_help_text(fw)
    help_available = bool(help_text)

    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        args = v.extra_server_args or ""
        skip_reason: str | None = None
        for flag, required_class in _COMPATIBILITY_FLAG_RULES:
            if flag not in args:
                continue
            # Model-class predicate
            class_ok = (
                (required_class == "mla" and is_mla)
                or (required_class == "moe" and is_moe)
            )
            if not class_ok:
                skip_reason = (
                    f"{flag} requires {required_class.upper()} model; "
                    f"MODEL_PATH={model_path!r} not recognised as "
                    f"{required_class.upper()}-class"
                )
                break
            # Framework flag-support predicate (only when help is
            # readable). Reason mentions the active framework so log
            # readers can tell which `--help` rejected the variant.
            if help_available and flag not in help_text:
                skip_reason = (
                    f"{flag} not present in `{fw} --help` output; "
                    f"current {fw} version likely too old"
                )
                break
        if skip_reason:
            dropped.append({
                "name": v.name,
                "source": "compatibility_filter",
                "reason": skip_reason,
            })
        else:
            kept.append(v)
    return kept, dropped


def apply_user_skip_list(
    grid: list["GridVariant"],
    *,
    skip_spec: str,
) -> tuple[list["GridVariant"], list[dict]]:
    """Drop variants whose name matches any pattern in ``skip_spec``.

    Returns ``(kept, dropped)`` where each dropped entry is
    ``{"name", "reason", "source"}`` with source=``"user_skip"`` so
    callers can distinguish user-driven skips from model/kernel
    incompatibility skips when both layers run.
    """
    patterns = _parse_skip_spec(skip_spec)
    if not patterns:
        return list(grid), []

    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        matched_pat: str | None = None
        for pat in patterns:
            # Exact name first (cheaper, more common), then fnmatch for
            # globs. fnmatch also accepts plain names so the second branch
            # alone would suffice, but keeping the fast-path makes logs
            # explicit ("matched 'attn_aiter'" vs "matched 'attn_*'").
            if pat == v.name or _fnmatch.fnmatchcase(v.name, pat):
                matched_pat = pat
                break
        if matched_pat is None:
            kept.append(v)
            continue
        dropped.append({
            "name": v.name,
            "source": "user_skip",
            "reason": f"matched SKIP_VARIANTS pattern '{matched_pat}'",
        })
    return kept, dropped


@dataclass(init=False)
class GridVariant:
    """One row of the grid we're going to test."""

    name: str                                    # human-readable label
    extra_server_args: str = ""                  # appended via EXTRA_{SGLANG,VLLM,ATOM}_ARGS env
    extra_envs: dict[str, str] = field(default_factory=dict)
    note: str = ""                                # optional reason / category

    def __init__(
        self,
        name: str,
        extra_server_args: str = "",
        extra_envs: dict[str, str] | None = None,
        note: str = "",
        *,
        extra_sglang_args: str | None = None,
    ) -> None:
        # Back-compat keyword alias for the historical
        # ``extra_sglang_args`` kwarg name. Operators / tests /
        # third-party callers may still construct
        # ``GridVariant(extra_sglang_args="x")``; route that into the
        # canonical attribute with a single DeprecationWarning so the
        # callsite shows up in audit logs.
        if extra_sglang_args is not None:
            import warnings as _warnings
            _warnings.warn(
                "GridVariant(extra_sglang_args=...) is a deprecation "
                "alias for GridVariant(extra_server_args=...) and will "
                "be removed in the next Hyperloom release.",
                DeprecationWarning,
                stacklevel=2,
            )
            if not extra_server_args:
                extra_server_args = extra_sglang_args
        self.name = name
        self.extra_server_args = extra_server_args
        self.extra_envs = dict(extra_envs) if extra_envs is not None else {}
        self.note = note

    @property
    def fingerprint(self) -> str:
        """Content fingerprint used as dedup-ledger key. See module doc."""
        return variant_fingerprint(self.extra_server_args, self.extra_envs)


def coerce_extra_envs(value: Any) -> dict[str, str]:
    """Normalize Orchestration-supplied ``extra_envs`` to ``dict[str,str]``.

    The Orchestration LLM tends to emit ``extra_envs`` in three shapes
    even though only the dict form is contractually correct:

    1. ``{"FOO": "1", "BAR": "2"}`` — canonical.
    2. ``"FOO=1 BAR=2"`` or ``"FOO=1\nBAR=2"`` — shell-style string; the
       LLM cribs this from `export FOO=1 BAR=2` examples in prompts.
    3. ``["FOO=1", "BAR=2"]`` — list of ``KEY=VAL`` tokens; sometimes
       emitted alongside ``synergy_groups`` lists.

    Previously the grid-construction sites accepted only shape #1 and
    silently propagated #2/#3 into :class:`GridVariant.extra_envs`, where
    :func:`_run_magpie` and :func:`variant_fingerprint` call ``.items()``
    and crash with ``AttributeError("'str' object has no attribute
    'items'")`` — taking the entire ``backends`` / ``params`` round
    down. This helper keeps the contract narrow at the boundary so the
    downstream pipeline only ever sees a clean dict.

    Unknown shapes (anything that isn't dict/str/list/None) are coerced
    to an empty dict; the action ledger records the round but no envs
    are exported. Caller logging surfaces the variant name so the LLM
    can self-correct on the next round.
    """
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if k is not None}
    if isinstance(value, str):
        out: dict[str, str] = {}
        # Accept newline, semicolon, or whitespace separation; values
        # never contain ``=`` in practice but we split on the first ``=``
        # only to preserve URL-style assignments like ``HF_ENDPOINT=https://...``.
        tokens = re.split(r"[\s;]+", value.strip())
        for tok in tokens:
            if not tok:
                continue
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            k = k.strip()
            if not k:
                continue
            out[k] = v.strip()
        return out
    if isinstance(value, (list, tuple)):
        out_l: dict[str, str] = {}
        for item in value:
            if isinstance(item, dict):
                # ``[{"FOO": "1"}, {"BAR": "2"}]`` — merge in order so
                # later entries win, mirroring shell ``export`` semantics.
                for k, v in item.items():
                    if k is None:
                        continue
                    out_l[str(k)] = str(v)
                continue
            if not isinstance(item, str) or "=" not in item:
                continue
            k, v = item.split("=", 1)
            k = k.strip()
            if not k:
                continue
            out_l[k] = v.strip()
        return out_l
    return {}


@dataclass
class VariantResult:
    """One bench run's parsed result."""

    name: str
    extra_server_args: str
    extra_envs: dict[str, str]
    status: str
    output_throughput: float | None = None
    request_throughput: float | None = None
    total_token_throughput: float | None = None
    completed_requests: int | None = None
    duration_seconds: float | None = None
    ttft_mean_ms: float | None = None
    e2el_mean_ms: float | None = None
    tpot_mean_ms: float | None = None
    workspace: str | None = None
    report_path: str | None = None
    raw_result_path: str | None = None
    reported_success: bool | None = None
    returncode: int | None = None
    nonfatal_warnings: list[str] = field(default_factory=list)
    error: str | None = None
    # Short tag for failure classification — matches the label used by
    # ``_write_variant_abort_marker`` (e.g. ``mn_server_restart_failed``,
    # ``magpie_timeout``, ``yaml_build_error``, ``no_benchmark_workspace``,
    # ``magpie_nonzero_invalid_measurement``, ``benchmark_report_missing``,
    # ``benchmark_report_invalid_metric``). Empty string for succeeded
    # variants. Threaded into ``coordinator._summarize_failed_variants``
    # so the LLM critic prompt sees ``failed_variants[*].error_class``
    # instead of a generic ``None``.
    error_class: str = ""
    note: str = ""
    # Fix-E (Q3 — Q3c): wall-clock seconds the Magpie subprocess
    # actually consumed. Populated on success AND on the
    # ``killed_overtime`` path so the ExploreExecutor can record
    # ``runtime_sec`` + ``wall_clock_ratio_vs_baseline`` against the
    # variant ledger without re-measuring.
    runtime_sec: float | None = None
    # Fix-E: True iff this variant was reaped by the
    # ``baseline_runtime_sec * explore_overtime_kill_ratio`` soft
    # deadline (vs a regular crash / hard timeout / success). Caller
    # is expected to demote this to a synthetic outcome
    # ``KILLED_OVERTIME`` (no tput, no fingerprint promotion).
    killed_overtime: bool = False

    @property
    def fingerprint(self) -> str:
        """Same fingerprint scheme as :class:`GridVariant`."""
        return variant_fingerprint(self.extra_server_args, self.extra_envs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":               self.name,
            "extra_server_args":  self.extra_server_args,
            "extra_envs":         self.extra_envs,
            "fingerprint":        self.fingerprint,
            "status":             self.status,
            "output_throughput":  self.output_throughput,
            "request_throughput": self.request_throughput,
            "total_token_throughput": self.total_token_throughput,
            "completed_requests": self.completed_requests,
            "duration_seconds":   self.duration_seconds,
            "ttft_mean_ms":       self.ttft_mean_ms,
            "e2el_mean_ms":       self.e2el_mean_ms,
            "tpot_mean_ms":       self.tpot_mean_ms,
            "workspace":          self.workspace,
            "report_path":        self.report_path,
            "raw_result_path":    self.raw_result_path,
            "reported_success":   self.reported_success,
            "returncode":         self.returncode,
            "nonfatal_warnings":  self.nonfatal_warnings,
            "error":              self.error,
            "error_class":        self.error_class,
            "note":               self.note,
            "runtime_sec":        self.runtime_sec,
            "killed_overtime":    self.killed_overtime,
        }


# ---------------------------------------------------------------------------
# Shared sanitization for Orchestration-supplied overrides. Magpie picks the
# benchmark script via ``cfg["benchmark"]["benchmark_script"]`` (a bare file
# name; Magpie prepends its own scripts dir) and writes ``inferencex_result.
# json`` into ``$RESULT_DIR``. Both knobs are surfaced as ``task.params``
# fields so Orchestration can route around scripts that hardcode
# ``--result-dir /workspace/`` (see SKILL.md "Magpie leak-path salvage"
# and the failure-recovery section). Both must be sanitized before they
# touch a YAML or an env var because they originate from an LLM proposal:
# we reject anything that contains path separators or shell metacharacters.
# The helpers raise ``ValueError`` so callers can surface ``error_class=
# bad_param`` (Coordinator promotes that to a ``policy_denied`` observation
# instead of running an unsafe subprocess).
_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.sh$")
_RESULT_DIR_FORBID_RE = re.compile(r"[\s\"'`$;&|<>(){}\[\]\\*?!]")


def sanitize_script_name(value: Any) -> str | None:
    """Return ``value`` if it's a safe Magpie benchmark script file name.

    Magpie prepends its own ``scripts/`` directory to the value, so the
    value MUST be a bare ``*.sh`` file name (no slashes / no ``..``).
    Empty / ``None`` returns ``None`` (caller should treat as "no
    override"). Raises ``ValueError`` for anything that looks like a
    shell injection attempt — Orchestration can then propose a corrected
    value on the next tick.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not _SCRIPT_NAME_RE.match(text):
        raise ValueError(
            f"benchmark_script={text!r} rejected: must be a bare *.sh "
            "file name (no path separators, no shell metacharacters)"
        )
    return text


def sanitize_result_dir(value: Any) -> str | None:
    """Return ``value`` if it's a safe absolute (or workspace-relative) dir.

    Magpie passes the value through to ``$RESULT_DIR``, which lands in a
    shell ``cd`` / ``mkdir`` inside the benchmark script. We reject any
    character class that would let an LLM-supplied override escape into
    a different shell word. Empty / ``None`` returns ``None``.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if _RESULT_DIR_FORBID_RE.search(text):
        raise ValueError(
            f"result_dir={text!r} rejected: contains whitespace or shell "
            "metacharacters; pass an absolute or workspace-relative path"
        )
    return text


def server_args_env_name(framework: str | None) -> str:
    """Return the Magpie env var used to append backend server args."""
    name = str(framework or "").strip().lower()
    # atom check first: "atom" is not a substring of vllm/sglang, but keep
    # ordering explicit so future framework names with overlapping substrings
    # cannot accidentally match the wrong branch.
    if "atom" in name:
        return "EXTRA_ATOM_ARGS"
    if "vllm" in name:
        return "EXTRA_VLLM_ARGS"
    return "EXTRA_SGLANG_ARGS"


def merge_server_args(*parts: str | None) -> str:
    """Merge server arg strings preserving left-to-right override semantics.

    vLLM/SGLang command lines are assembled by shell-appending
    ``EXTRA_{VLLM,SGLANG}_ARGS`` after the default server args. Some flags are
    intentionally repeated so later variants can override base args (e.g. a
    model-specific ``--block-size 1`` plus a grid candidate ``--block-size
    256``). Therefore this helper only removes empty chunks; it does not try to
    de-duplicate option names.
    """
    return " ".join(str(p).strip() for p in parts if str(p or "").strip())


def apply_runtime_benchmark_overrides(
    bench: dict[str, Any],
    *,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
) -> dict[str, Any]:
    """Apply runtime env/CLI overrides to a Magpie benchmark YAML.

    This is the single shared path used by baseline/profile and grid
    executors. Historically only ``baseline.py`` applied these overrides, so
    backends/params/sweep silently fell back to shipped YAML defaults like
    ``TP=1`` and ``ROCR_VISIBLE_DEVICES="1"``. Large models (DeepSeek-R1-0528)
    then OOM-failed even though the launch environment had ``TP=8``.

    ``benchmark_script`` (when non-empty) lets the caller force-select a
    specific Magpie script (e.g. a model-specific ``dsr1_fp8_mi300x.sh``)
    that the operator deliberately wants benchmarked. The value MUST
    already be sanitized via :func:`sanitize_script_name` — callers at
    the executor boundary do that so any ``ValueError`` surfaces as
    ``error_class=bad_param`` instead of an unsafe subprocess invocation.
    The override is written AFTER the ``gpu_type``-derived generic script
    below so the operator-supplied pick wins over Hyperloom's default.
    """
    if model_path:
        bench["model"] = str(model_path)

    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision

    if gpu_type:
        bench["runner_type"] = str(gpu_type)
        # Magpie priority: explicit benchmark_script > InferenceX native
        # script > runner_type-derived generic script. Force-pin the
        # generic ``{framework}_{gpu_type}.sh`` so Magpie's resolver hits
        # priority 1 and never falls through to InferenceX native
        # scripts (e.g. ``dsr1_fp8_mi300x.sh``) that hardcode
        # ``--result-dir /workspace/`` and ignore ``EXTRA_*_ARGS``. See
        # ``design/magpie-generic-script-and-user-data-path.md``.
        framework = str(bench.get("framework") or "").lower()
        if framework:
            bench["benchmark_script"] = f"{framework}_{gpu_type}.sh"
        else:
            bench.pop("benchmark_script", None)

    if benchmark_script:
        bench["benchmark_script"] = str(benchmark_script)

    envs = bench.setdefault("envs", {})
    for env_key in ("ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC"):
        val = os.environ.get(env_key, "").strip()
        if not val:
            continue
        # TP yaml-explicit wins: a stale state.tp re-exported on resume
        # must not silently downgrade a YAML-pinned TP (e.g. baseline
        # config TP=2 → sglang TP=1 because state.tp defaulted to 1).
        if env_key == "TP":
            yaml_tp = envs.get("TP")
            if yaml_tp not in (None, 0, "", "0"):
                continue
        envs[env_key] = int(val)

    explicit_rocr = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if explicit_rocr:
        envs["ROCR_VISIBLE_DEVICES"] = explicit_rocr
    else:
        tp_val = int(envs.get("TP", 1) or 1)
        existing_rocr = str(envs.get("ROCR_VISIBLE_DEVICES", "")).strip()
        existing_count = (
            len([x for x in existing_rocr.split(",") if x.strip()])
            if existing_rocr else 0
        )
        if tp_val > 1 and existing_count < tp_val:
            envs["ROCR_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp_val))

    return envs


# ---------------------------------------------------------------------------
def _build_variant_yaml(
    base_yaml_path: Path,
    base_extra_args: str,
    variant: GridVariant,
    *,
    output_subdir: Path,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
) -> Path:
    """Materialize a per-variant Magpie YAML on disk.

    Magpie's sglang_mi300x.sh honors ``EXTRA_SGLANG_ARGS`` from envs to
    append flags after the auto-generated server args, so we just inject
    the variant's flags there.

    ``model_path`` (when non-empty) overrides ``benchmark.model``; the
    shipped configs all have a legacy hardcoded Qwen-Qwen3-8B path that
    would otherwise win over the user's runtime selection.

    ``gpu_type`` (when non-empty) injects ``benchmark.runner_type`` and
    force-pins ``benchmark.benchmark_script`` to the generic
    ``{framework}_{gpu_type}.sh`` so Magpie's resolver does NOT fall
    through to an InferenceX native script (which hardcodes
    ``--result-dir /workspace/`` and ignores ``EXTRA_*_ARGS``).

    ``benchmark_script`` (when non-empty, must already be sanitized via
    :func:`sanitize_script_name`) force-pins the Magpie script per
    variant when an operator deliberately wants a specific (often
    model-specific) script benchmarked. Applied AFTER the
    ``gpu_type``-derived generic script so the operator pick wins.
    """
    with base_yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bench = cfg.setdefault("benchmark", {})
    envs = apply_runtime_benchmark_overrides(
        bench, model_path=model_path, gpu_type=gpu_type,
        benchmark_script=benchmark_script,
    )
    extra_args_env = server_args_env_name(bench.get("framework"))

    combined = merge_server_args(
        str(envs.get(extra_args_env, "")),
        base_extra_args,
        variant.extra_server_args,
    )
    if combined:
        envs[extra_args_env] = combined
    for k, v in variant.extra_envs.items():
        envs[str(k)] = str(v)

    output_subdir.mkdir(parents=True, exist_ok=True)
    out_path = output_subdir / "config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def _parse_report(workspace: Path) -> dict[str, Any] | None:
    report = workspace / "benchmark_report.json"
    if not report.exists():
        return None
    try:
        with report.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _kill_stale_servers() -> None:
    """Deep-clean any lingering inference server processes + shared memory.

    Magpie's server_cleanup.sh only kills the process group leader and waits,
    but vLLM::Worker / EngineCore children often escape the pgrp (Ray-spawned
    or multiprocessing.spawn). Without this pre-clean, the next vLLM startup
    hangs for 5 minutes on zmq socket / shared mem conflicts:
      "Did not receive response from front-end process within 5 minutes"

    We call this BEFORE every Magpie invocation so each grid variant starts
    on a pristine server state.

    NOTE: uses /proc scan instead of `subprocess.run(["pgrep",...])` to avoid
    conflicting with test mocks that patch subprocess.run for Magpie calls.

    Multi-node short-circuit: in --nodes>=2 mode the inference servers run
    inside the RayJob pods, NOT in this sandbox. Scanning sandbox /proc
    finds nothing matching, and clearing sandbox /dev/shm/vllm* would only
    remove unrelated state. Skip the whole sweep + the 2s settle sleep —
    server lifecycle there is owned by `multi_node restart-server`.
    """
    from ._multi_node_env import is_multi_node
    if is_multi_node():
        return

    import signal
    import glob
    import time

    _KILL_PATTERNS = ("VLLM::Worker", "VLLM::EngineCore", "vllm.entrypoints",
                      "vllm serve", "sglang.srt", "sglang.launch_server",
                      # atom server entrypoint (analogous to vllm.entrypoints).
                      "atom.entrypoints", "atom.entrypoints.openai_server")

    # atom spawns its ModelRunner workers via ``multiprocessing.spawn`` — their
    # cmdline is the generic ``spawn_main ... --multiprocessing-fork`` so they
    # cannot be matched by _KILL_PATTERNS. On server teardown they routinely
    # orphan to init (ppid=1) yet keep their full HIP/VRAM reservation (~87 %
    # of each MI3xx GPU for an 8B+ model), which OOM-kills the *next* atom
    # server (baseline ok, then roofline/profile/explore all fail to allocate
    # KV cache). We identify these survivors by the atom install (and aiter
    # JIT cache) mmap'd into their address space — a signature the optimizer's
    # own Ray / multiprocessing children never carry.
    _FORK_MARKERS = (b"--multiprocessing-fork", b"spawn_main")
    _ATOM_MAP_SIGNATURES = ("/ATOM/atom/", "/aiter/jit/", "/aiter-test/aiter/")

    my_pid = os.getpid()
    try:
        my_pgid = os.getpgrp()
    except OSError:
        my_pgid = -1

    def _is_orphaned_atom_worker(pid: int, cmdline: bytes) -> bool:
        if not any(m in cmdline for m in _FORK_MARKERS):
            return False
        # Never touch a worker that belongs to *our* process group.
        try:
            if my_pgid != -1 and os.getpgid(pid) == my_pgid:
                return False
        except (OSError, ProcessLookupError):
            return False
        try:
            with open(f"/proc/{pid}/maps", "r", errors="replace") as fh:
                maps = fh.read()
        except (OSError, PermissionError):
            return False
        return any(sig in maps for sig in _ATOM_MAP_SIGNATURES)

    killed_atom = False
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
        except (OSError, PermissionError):
            continue
        text = cmdline.replace(b"\0", b" ").decode("utf-8", "replace")
        is_atom_server = "atom.entrypoints" in text
        if any(pat in text for pat in _KILL_PATTERNS) or _is_orphaned_atom_worker(pid, cmdline):
            killed_atom = killed_atom or is_atom_server or b"--multiprocessing-fork" in cmdline
            # Kill the whole process group when we can — atom servers fan out
            # ModelRunner children that must die with the leader.
            try:
                pgid = os.getpgid(pid)
                if pgid not in (my_pgid, 0):
                    os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # Clear /dev/shm vllm/nccl/cuda/torch segments that prevent re-binding.
    for pattern in ("/dev/shm/vllm*", "/dev/shm/nccl*", "/dev/shm/cuda*",
                    "/dev/shm/torch*", "/dev/shm/atom*"):
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass

    # Brief pause for KFD (ROCm kernel driver) async VRAM release. Atom workers
    # hold tens-to-hundreds of GB, whose async teardown lags well past 2s, so
    # give the driver longer to actually reclaim before the next server boots.
    time.sleep(8 if killed_atom else 2)


def _run_magpie(
    *,
    magpie_python: str,
    config_path: Path,
    output_dir: Path,
    timeout_sec: int,
    cwd: str,
    result_dir: str | None = None,
    soft_deadline_sec: float | None = None,
) -> tuple[int, str, str]:
    """Blocking subprocess wrapper. Returns (rc, stdout, stderr).

    ``result_dir`` (when non-empty, must already be sanitized via
    :func:`sanitize_result_dir`) overrides ``$RESULT_DIR`` for this
    invocation. The env var is ALWAYS set (defaults to ``output_dir``
    when caller doesn't override) so even a Magpie script that respects
    ``$RESULT_DIR`` writes ``inferencex_result.json`` into the per-task
    workspace rather than ``/workspace/``.

    ``soft_deadline_sec`` is the Fix-E per-variant overtime cap. When
    set, ``run_with_session_kill`` reaps the tree once the deadline
    elapses and returns a sentinel ``returncode = OVERTIME_KILL_RETURNCODE``
    instead of raising ``TimeoutExpired``. The caller distinguishes
    "overtime kill" (returncode sentinel) from "hard timeout"
    (TimeoutExpired) and from "crash" (any other nonzero).
    """
    # Pre-clean: kill lingering server processes + clear shared memory so the
    # next vLLM/SGLang startup doesn't collide with stale resources.
    # Skip in test environments to avoid 2s sleep per variant.
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        _kill_stale_servers()

    env = os.environ.copy()
    env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
    magpie_dir = os.environ.get("MAGPIE_DIR", "")
    if magpie_dir:
        env["PYTHONPATH"] = f"{magpie_dir}:{env.get('PYTHONPATH', '')}"

    # Multi-node mode: tell Magpie to skip its local-server launch and
    # point benchmark_serving at the head pod's ClusterIP. Returns {} in
    # single-node mode so the env stays exactly as before (no-op guard).
    from ._multi_node_env import magpie_remote_env
    env.update(magpie_remote_env())

    # #210 (Deval, comment 8): pin Magpie's InferenceX-resolution to
    # ``$INFERENCEX_PATH`` so Magpie loads the SAME InferenceX checkout
    # that Hyperloom's ``_inferencex_patcher`` has patched. Without
    # this, Magpie's ``_resolve_default_inferencex_dir`` falls through
    # to ``./InferenceX`` next to its repo or
    # ``$XDG_CACHE_HOME/magpie/InferenceX``, either of which may be a
    # separate, unpatched checkout — the symptom reported in #210
    # comments 4 + 6. ``MAGPIE_INFERENCEX_PATH`` is the highest-
    # precedence resolution rung in Magpie itself
    # (``Magpie/modes/benchmark/inferencex.py:43``), so this is the
    # documented contract for tying the two checkouts together.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if inferencex_path:
        env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
    # Always-on RESULT_DIR default: covers Magpie scripts that respect
    # the env var (and would otherwise fall back to a hardcoded path).
    # Scripts that ignore RESULT_DIR (e.g. ``dsr1_fp8_mi300x.sh`` with its
    # hardcoded ``--result-dir /workspace/``) still leak; the
    # ``extract_benchmark_measurement`` salvage path picks those up.
    env["RESULT_DIR"] = result_dir or str(output_dir)
    # Magpie's ``InferenceX/benchmarks/single_node/*.sh`` wrappers default
    # ``SERVER_LOG=/workspace/server.log`` and the GPU monitor's
    # ``GPU_METRICS_CSV=/workspace/gpu_metrics.csv``. Both honor env-var
    # overrides — pin them per-task so server stdout/stderr and per-second
    # GPU telemetry land alongside ``benchmark_report.json`` instead of
    # leaking outside the session. Always overwrite (not ``setdefault``)
    # so a stale value inherited from the parent shell can't redirect a
    # variant's logs into a previous run's slot.
    # ``harvest_leaked_artifacts`` still runs as defense-in-depth for any
    # wrapper that ignores these vars (the older
    # ``inferencex_result.json`` leak path also stays covered).
    env["SERVER_LOG"] = str(output_dir / "server.log")
    env["GPU_METRICS_CSV"] = str(output_dir / "gpu_metrics.csv")
    cmd = [
        magpie_python, "-m", "Magpie", "-v", "benchmark",
        "--benchmark-config", str(config_path),
        "--output-dir", str(output_dir),
        "--run-mode", "local",
    ]
    # ``run_with_session_kill`` (imported at module level so tests can
    # patch it as ``_grid_runner.run_with_session_kill``) is the
    # ``subprocess.run``-compatible wrapper that launches Magpie in its
    # own POSIX session and tears down the whole descendant tree on
    # every exit path (bugs.md §B — leaked vLLM / SGLang servers across
    # grid variants were what later sourced half-truncated benchmark
    # scripts in bugs.md §C #1). See ``_subprocess_kill.py``.
    proc = run_with_session_kill(
        cmd, env=env, cwd=cwd, timeout=timeout_sec,
        soft_deadline_sec=soft_deadline_sec,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


# ---------------------------------------------------------------------------
async def run_grid(
    *,
    base_yaml_path: Path,
    base_extra_args: str,
    grid: list[GridVariant],
    output_root: Path,
    magpie_python: str | None = None,
    cwd: str = _MAGPIE_CWD_DEFAULT,
    variant_timeout_sec: int = _VARIANT_TIMEOUT_SEC_DEFAULT,
    keep_going_on_failure: bool = True,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
    result_dir: str | None = None,
    soft_deadline_sec: float | None = None,
) -> list[VariantResult]:
    """Execute every variant in ``grid`` once, in order.

    Returns the per-variant :class:`VariantResult` list (all attempts,
    including failed ones). Caller decides which variants are "winners".

    Synchronous subprocess call wrapped in ``asyncio.to_thread`` so the
    Coordinator reactor isn't blocked.

    ``model_path`` and ``gpu_type`` are forwarded to every variant's YAML
    render; pass the values resolved by the executor (task.params or
    $MODEL_PATH / $GPU_TYPE) so each Magpie invocation benchmarks the
    user's actual model on the user's actual GPU rather than the YAML's
    legacy default.

    ``benchmark_script`` / ``result_dir`` (both must already be sanitized
    via :func:`sanitize_script_name` / :func:`sanitize_result_dir`) let
    the executor route around a model-default script that hardcodes
    ``--result-dir /workspace/`` — see SKILL.md "Magpie leak-path
    salvage". ``benchmark_script`` rewrites the variant YAML so Magpie
    runs the operator-picked script; ``result_dir`` is forwarded as
    ``$RESULT_DIR`` so scripts that respect the env var write into the
    variant slot. The salvage path is still wired in (mtime-gated per
    variant below) for scripts that ignore both knobs.

    ``soft_deadline_sec`` (Fix E — per-variant overtime kill): when
    set, every variant's Magpie subprocess is reaped once its
    wall-clock exceeds this many seconds; the resulting
    :class:`VariantResult` has ``status='failed'``,
    ``killed_overtime=True`` and ``runtime_sec`` populated. Caller
    (ExploreExecutor) demotes those to the synthetic
    ``KILLED_OVERTIME`` outcome instead of running them through the
    normal KEEP / REVERT / FAILED ladder. None / 0 = disabled (legacy
    behaviour, only ``variant_timeout_sec`` is enforced).
    """
    if not magpie_python:
        magpie_python = _resolve_magpie_python()
    results: list[VariantResult] = []
    # Variant-boundary robustness pulse — runs a bounded deterministic
    # robustness tick after every variant (success OR failure) so that a
    # mid-grid GPU leak, SGLang crash, or ROCm error spike surfaces
    # between variants instead of waiting for the whole grid (often
    # 30+ minutes) to finish. Best-effort, ≤ ``_PULSE_TIMEOUT_SEC``;
    # see ``_robustness_pulse.py`` for the contract.
    async def _pulse_after_variant(idx: int) -> None:
        try:
            await _robustness_pulse(tick_index=idx)
        except Exception as exc:  # noqa: BLE001
            log.debug("robustness pulse swallowed: %r", exc)

    for i, variant in enumerate(grid):
        slot = output_root / f"variant_{i:02d}_{_safe(variant.name)}"
        try:
            cfg_path = _build_variant_yaml(
                base_yaml_path, base_extra_args, variant, output_subdir=slot,
                model_path=model_path,
                gpu_type=gpu_type,
                benchmark_script=benchmark_script,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: yaml_build_error: %r",
                i + 1, len(grid), variant.name, exc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="yaml_build_error",
                error_summary=repr(exc),
                extra_args=variant.extra_server_args,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed", error=f"yaml_build_error: {exc!r}",
                error_class="yaml_build_error",
                note=variant.note,
            ))
            await _pulse_after_variant(i)
            if not keep_going_on_failure:
                break
            continue

        from ._multi_node_env import log_mn_banner
        log_mn_banner(
            "grid_runner", log,
            variant=f"{i+1}/{len(grid)}:{variant.name}",
        )
        log.info(
            "grid_runner: variant %d/%d name=%s args=%s",
            i + 1, len(grid), variant.name, variant.extra_server_args,
        )

        # Multi-node only: restart sglang/vllm with this variant's
        # server-side flags so each grid row runs against a fresh server
        # (parity with single-node Magpie's PHASE=all). No-op in
        # single-node mode — the helper short-circuits when nodes<2.
        from ._multi_node_server_lifecycle import (
            ServerRestartFailed,
            restart_server_for_round,
        )
        try:
            # PD knobs auto-resolved by the helper from $PD_* env. The
            # grid runner doesn't sweep PD ratio yet, so PD config
            # stays constant across variants within one run.
            await restart_server_for_round(
                extra_server_args=merge_server_args(
                    base_extra_args, variant.extra_server_args,
                ),
                model_path=model_path,
                ep=int(os.environ.get("EP") or 0) or None,
            )
        except ServerRestartFailed as exc:
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: "
                "mn_server_restart_failed: %s",
                i + 1, len(grid), variant.name, exc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="mn_server_restart_failed",
                error_summary=str(exc),
                extra_args=variant.extra_server_args,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                error=f"mn_server_restart_failed: {exc}",
                error_class="mn_server_restart_failed",
                note=variant.note,
            ))
            if not keep_going_on_failure:
                break
            continue

        # Snapshot wall-clock immediately before launch so the salvage
        # path can mtime-gate documented Magpie leak destinations
        # per-variant. Without this gate a stale
        # ``/workspace/inferencex_result.json`` from a prior run (or
        # from an earlier variant in this same grid) silently
        # masquerades as the current variant's result.
        variant_started_unix = time.time()
        try:
            rc, stdout, stderr = await asyncio.to_thread(
                _run_magpie,
                magpie_python=magpie_python,
                config_path=cfg_path,
                output_dir=slot,
                timeout_sec=variant_timeout_sec,
                cwd=cwd,
                result_dir=result_dir,
                soft_deadline_sec=soft_deadline_sec,
            )
        except subprocess.TimeoutExpired as exc:
            # Harvest pre-timeout leaks (``server.log`` / GPU metrics /
            # partial profile relay) so the NFS clone of the variant
            # slot captures what the wrapper managed to write before
            # the timer fired — usually the smoking gun.
            to_candidates = sorted(slot.glob("benchmark_*"))
            to_destination = to_candidates[-1] if to_candidates else slot
            to_harvested = harvest_leaked_artifacts(
                to_destination,
                subprocess_started_unix=variant_started_unix,
            )
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: "
                "magpie timeout (timeout_sec=%d): %s",
                i + 1, len(grid), variant.name, variant_timeout_sec, exc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="magpie_timeout",
                error_summary=str(exc),
                extra_args=variant.extra_server_args,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed", error=f"timeout: {exc}",
                error_class="magpie_timeout",
                note=variant.note,
                runtime_sec=round(
                    max(0.0, time.time() - variant_started_unix), 2,
                ),
                nonfatal_warnings=[
                    f"harvested_leaked_artifact:{src}"
                    for src, _ in to_harvested
                ],
            ))
            await _pulse_after_variant(i)
            if not keep_going_on_failure:
                break
            continue

        # Soft overtime gate fired. Record a synthetic
        # ``killed_overtime=True`` VariantResult with no tput / report
        # so the ExploreExecutor can demote this variant to the
        # ``KILLED_OVERTIME`` ledger outcome (no fingerprint promotion,
        # no stack advance). We still run harvest_leaked_artifacts so
        # any server.log / GPU metrics the wrapper managed to write
        # before being reaped land alongside ``variant_NN_<name>/`` for
        # post-mortem.
        if rc == OVERTIME_KILL_RETURNCODE:
            variant_runtime_sec = round(
                max(0.0, time.time() - variant_started_unix), 2,
            )
            ok_candidates = sorted(slot.glob("benchmark_*"))
            ok_destination = ok_candidates[-1] if ok_candidates else slot
            ok_harvested = harvest_leaked_artifacts(
                ok_destination,
                subprocess_started_unix=variant_started_unix,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                returncode=rc,
                killed_overtime=True,
                runtime_sec=variant_runtime_sec,
                error=(
                    f"killed_overtime: wall-clock {variant_runtime_sec:.1f}s "
                    f"exceeded soft_deadline_sec={float(soft_deadline_sec or 0.0):.1f}s"
                ),
                note=variant.note,
                nonfatal_warnings=[
                    f"harvested_leaked_artifact:{src}"
                    for src, _ in ok_harvested
                ],
            ))
            log.info(
                "_grid_runner: variant %s killed_overtime "
                "(runtime=%.1fs deadline=%.1fs)",
                variant.name, variant_runtime_sec,
                float(soft_deadline_sec or 0.0),
            )
            await _pulse_after_variant(i)
            if not keep_going_on_failure:
                break
            continue

        # Locate workspace inside slot.
        candidates = sorted(slot.glob("benchmark_*"))
        # Always-on artifact harvest (parity with BaselineExecutor —
        # see ``harvest_leaked_artifacts``). Without this each variant
        # in a backends / params / sweep grid leaks its own
        # ``/workspace/server.log`` + ``gpu_metrics.csv`` + profile
        # relay, which makes the NFS clone of
        # ``<session>/runs/<action>/<task_id>/<variant>/`` empty of
        # wrapper diagnostics — exactly the per-variant evidence
        # Robustness needs to RCA a flaky variant.
        harvest_destination = candidates[-1] if candidates else slot
        harvested = harvest_leaked_artifacts(
            harvest_destination,
            subprocess_started_unix=variant_started_unix,
        )
        if harvested:
            log.info(
                "_grid_runner: variant=%s harvested %d leaked artifact(s): %s",
                variant.name,
                len(harvested),
                ", ".join(src.name for src, _ in harvested),
            )
        if not candidates:
            harvest_tags = [f"harvested_leaked_artifact:{src}" for src, _ in harvested]
            no_ws_error_summary = (
                (stderr or stdout)[-2000:]
                if rc != 0 else "no benchmark_* workspace produced"
            )
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: "
                "no_benchmark_workspace (rc=%s)",
                i + 1, len(grid), variant.name, rc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="no_benchmark_workspace",
                error_summary=no_ws_error_summary,
                extra_args=variant.extra_server_args,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                returncode=rc,
                error=no_ws_error_summary,
                error_class="no_benchmark_workspace",
                nonfatal_warnings=harvest_tags,
                note=variant.note,
            ))
            await _pulse_after_variant(i)
            if rc != 0 and not keep_going_on_failure:
                break
            continue
        workspace = candidates[-1]
        report = _parse_report(workspace)
        report_path = workspace / "benchmark_report.json"
        measurement = extract_benchmark_measurement(
            report,
            workspace=workspace,
            subprocess_started_unix=variant_started_unix,
        )
        warnings = list(measurement.pop("nonfatal_warnings", []) or [])
        if rc != 0:
            warnings.append("magpie_nonzero_after_valid_measurement")
        for leak_src, _ in harvested:
            warnings.append(f"harvested_leaked_artifact:{leak_src}")

        if not measurement.get("valid_measurement"):
            if rc != 0:
                error = (stderr or stdout)[-2000:]
                invalid_class = "magpie_nonzero_invalid_measurement"
            elif not report:
                error = "benchmark_report missing"
                invalid_class = "benchmark_report_missing"
            else:
                error = "benchmark_report missing valid throughput/completed requests"
                invalid_class = "benchmark_report_invalid_metric"
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: %s (rc=%s): %s",
                i + 1, len(grid), variant.name, invalid_class, rc, error[:200],
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class=invalid_class,
                error_summary=error,
                extra_args=variant.extra_server_args,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                workspace=str(workspace),
                report_path=str(report_path) if report_path.exists() else None,
                raw_result_path=measurement.get("raw_result_path"),
                reported_success=measurement.get("reported_success"),
                returncode=rc,
                nonfatal_warnings=warnings,
                error=error,
                error_class=invalid_class,
                note=variant.note,
            ))
            await _pulse_after_variant(i)
            if rc != 0 and not keep_going_on_failure:
                break
            continue

        results.append(VariantResult(
            name=variant.name, extra_server_args=variant.extra_server_args,
            extra_envs=dict(variant.extra_envs),
            status="succeeded",
            output_throughput=measurement.get("output_throughput"),
            request_throughput=measurement.get("request_throughput"),
            total_token_throughput=measurement.get("total_token_throughput"),
            completed_requests=measurement.get("completed_requests"),
            duration_seconds=measurement.get("duration_seconds"),
            ttft_mean_ms=measurement.get("ttft_mean_ms"),
            e2el_mean_ms=measurement.get("e2el_mean_ms"),
            tpot_mean_ms=measurement.get("tpot_mean_ms"),
            workspace=str(workspace),
            report_path=str(report_path) if report_path.exists() else None,
            raw_result_path=measurement.get("raw_result_path"),
            reported_success=measurement.get("reported_success"),
            returncode=rc,
            nonfatal_warnings=warnings,
            error=(stderr or stdout)[-2000:] if rc != 0 else None,
            note=variant.note,
            runtime_sec=round(
                max(0.0, time.time() - variant_started_unix), 2,
            ),
        ))
        log.info(
            "grid_runner: variant %s tput=%.1f tok/s",
            variant.name, results[-1].output_throughput or 0.0,
        )
        await _pulse_after_variant(i)
    return results


SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT = 1.0
MULTI_NODE_DEFAULT_KEEP_THRESHOLD_PCT = 2.0


def pick_winners(
    results: list[VariantResult],
    baseline_tput: float,
    *,
    keep_threshold_pct: float | None = None,
) -> list[VariantResult]:
    """Filter variants whose throughput beats ``baseline_tput`` by
    ``keep_threshold_pct`` percent (marathon §params: > 1% = KEEP).

    Resolution order for ``keep_threshold_pct``:

    1. Explicit caller value (any float, including 0.5/1.0/3.0) wins.
       This preserves legacy single-node behaviour bit-for-bit (callers
       like the params executor still pass their own 0.5% default).
    2. ``None`` (i.e. caller did not pass a value) falls back to:
       * **multi-node**: ``MULTI_NODE_DEFAULT_KEEP_THRESHOLD_PCT`` (2.0%).
         Multi-node noise floor is empirically ~1-2% (jitter from
         cross-node RDMA + Ray scheduling + GPU clock drift), so the
         1.0% single-node default produced false positives that wasted
         a ~40-min ``validate_stack`` round each.
       * **single-node**: ``SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT``
         (1.0%) — identical to the pre-multi-node-aware default.

    Single-node call sites are bit-for-bit equivalent: ``is_multi_node()``
    short-circuits to False without touching state, so the cutoff math
    is unchanged.
    """
    if keep_threshold_pct is None:
        from ._multi_node_env import is_multi_node
        keep_threshold_pct = (
            MULTI_NODE_DEFAULT_KEEP_THRESHOLD_PCT
            if is_multi_node()
            else SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT
        )
    cutoff = baseline_tput * (1.0 + keep_threshold_pct / 100.0)
    return [
        r for r in results
        if r.status == "succeeded"
        and isinstance(r.output_throughput, (int, float))
        and r.output_throughput > cutoff
    ]


def _safe(name: str) -> str:
    """Filesystem-safe slug for variant directory names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]


def _write_variant_abort_marker(
    slot: Path,
    *,
    variant_name: str,
    error_class: str,
    error_summary: str,
    extra_args: str = "",
) -> None:
    """Write ``abort_reason.json`` into the variant slot directory.

    Why this exists: when a variant aborts before benchmark_report.json
    is produced (ServerRestartFailed / yaml_build_error / magpie timeout
    / no benchmark_* workspace / invalid measurement), the slot dir is
    left with only ``config.yaml`` and a session reader cannot tell
    "tested-but-failed" from "untested / skipped". The final-report
    renderer and any later post-mortem tool then under-report grid
    coverage.

    Drop a small JSON marker so:

    * final-report / breakdown can count failed-but-tested variants;
    * a session reader inspecting ``runs/<action>/<task_id>/<variant>/``
      sees an explicit reason even after the main process log was
      rotated or truncated;
    * the marker pairs with the ``log.warning`` line emitted next to
      each catch site for grep-from-log triage.

    Failure to write the marker is non-fatal — log and continue so a
    full-disk / permissions issue can't escalate a single-variant
    abort into a whole-grid abort.
    """
    try:
        slot.mkdir(parents=True, exist_ok=True)
        marker = {
            "variant": variant_name,
            "error_class": error_class,
            "error": (error_summary or "")[:2000],
            "extra_args": extra_args,
            "aborted_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
            ),
        }
        (slot / "abort_reason.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning(
            "_grid_runner: failed to write abort_reason.json at %s: %s",
            slot, exc,
        )


__all__ = [
    "GridVariant",
    "MULTI_NODE_DEFAULT_KEEP_THRESHOLD_PCT",
    "SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT",
    "VariantResult",
    "apply_runtime_benchmark_overrides",
    "pick_winners",
    "reorder_grid_for_multi_node",
    "run_grid",
    "sanitize_result_dir",
    "sanitize_script_name",
    "server_args_env_name",
    "variant_fingerprint",
]
