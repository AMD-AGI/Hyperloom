# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Record a forge-loop run's best solution in the KB Store.

Called after each durable best and during graceful finalization. One run's
result becomes one record under the kernel five-tuple
(``kernel:<op>:<framework>:<framework_version>:<backend>:<gpu>``):

  * the record carries the metrics, the LLM-distilled strategy/recipe/lessons,
    and the implementation signature a later run gates reuse on;
  * the cumulative diff travels beside it as a ``solution.patch`` artifact, so a
    reader can rank candidates before deciding to pull a patch;
  * the store's champion pointer follows the best speedup recorded so far.

Identity is deterministic and never LLM-inferred. Only the free-text experience
(strategy / recipe / lessons) and the coarse ``category`` bucket come from a
single best-effort LLM call.

Write policy: only record a run that beat its own baseline (speedup > 1.0) and
produced a diff. Losing to a previously recorded run is not a reason to discard
the evidence, so the record is still written; only the champion pointer is
withheld.

Everything here is best-effort: if the store is unavailable, or the LLM
summarization fails, the run is simply not mirrored - it never raises into the
forge-loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from kernelforge.knowledge.implementation_identity import (
    canonical_owner_framework,
    hash_implementation_identity,
    implementation_signature,
)

log = logging.getLogger(__name__)

_CATEGORIES = {"gemm", "attention", "moe", "communication", "others"}
_UNKNOWN = "unknown"

# Bound the LLM inputs so a huge trajectory / source file can't blow the prompt.
_MAX_DIGEST_CHARS = 8000
_MAX_SOURCE_CHARS = 6000
_LLM_TIMEOUT_SEC = 150

# Frameworks whose kernels are detected from package-relative paths.
_FRAMEWORKS = ("aiter", "sglang", "vllm")

# Explicit "no framework" values for --framework: a standalone kernel file that
# belongs to no framework package. Treated identically to an undetected path.
_NO_FRAMEWORK_SENTINELS = {"standalone", "none", "unknown"}

_C_LIKE_LANGS = {"hip", "cuda", "cpp", "c"}


# --------------------------------------------------------------------------- #
# slug / value normalization
# --------------------------------------------------------------------------- #
def resolve_operation(kernel_source: str, kernel_path: str, target_functions: list[str] | None = None) -> str:
    """Return the operation identity (the entry function name, not the file name).

    Uses ``derive_kernel_names`` (parses ``@triton.jit`` / ``@*.kernel`` defs and
    HIP/CUDA ``__global__`` entries) on the anchor source, preferring a
    compute-kernel name over a host launcher/wrapper.

    Repository tasks whose anchor file is only a host wrapper (the real
    ``@triton.jit`` kernels live in OTHER files it imports) declare no GPU kernel
    in the anchor, so ``derive_kernel_names`` finds nothing. In that case fall
    back to ``target_functions`` before the last-resort file stem.

    The fallback selection is ORDER-INDEPENDENT: a producer with hand-declared
    ``--target-functions`` and a consumer deriving target functions from the
    source set may hand the same set of names in a different order.
    Picking ``target_functions[0]`` would then diverge and split the slug, so the
    candidate set is de-duplicated and sorted, preferring a compute kernel over a
    launcher/wrapper, before the first is chosen. Single-file tasks are
    unaffected: the anchor derive succeeds and ``target_functions`` is never
    consulted.
    """

    def _pick(names: list[str]) -> str | None:
        preferred = [n for n in names if not n.lower().startswith(("launch", "main", "wrapper", "run_"))]
        if preferred:
            return preferred[0]
        return names[0] if names else None

    try:
        from kernelforge.mcp_server.tools.pmc import derive_kernel_names

        # Anchor source order is stable for the same file, so keep it (the first
        # compute kernel is usually the primary one, helpers come later).
        picked = _pick(derive_kernel_names(kernel_source or ""))
        if picked:
            return picked
    except Exception as exc:  # noqa: BLE001 - best-effort; fall back below
        log.debug("resolve_operation: derive_kernel_names failed: %r", exc)
    # Fallback: order-independent (sorted, de-duplicated) so producer/consumer
    # converge even when their target-function lists are ordered differently.
    cand = sorted({fn.strip() for fn in (target_functions or []) if fn and fn.strip()})
    picked = _pick(cand)
    if picked:
        return picked
    return Path(kernel_path).stem


def detect_backend_language(kernel_backend: str) -> str:
    """Derive the implementation language exclusively from the selected kernel_backend."""
    lang = str(kernel_backend or "").split("-", 1)[0].strip().lower()
    return lang or _UNKNOWN


def detect_framework(kernel_path: str, framework_override: str = "") -> str:
    """Detect the owning framework.

    ``framework_override`` (an explicit ``--framework`` passed by the caller) is
    AUTHORITATIVE when given: relying on scanning ``Path(kernel_path).parts`` for
    a framework directory name is fragile across producer/consumer workspaces
    (e.g. a flattened scratch copy drops the ``vllm/`` directory), which would
    split the slug. A "no framework" sentinel (``standalone``/``none``/
    ``unknown``) explicitly means a standalone file.

    Without an override, fall back to the deterministic path scan. A standalone
    file with no known framework directory yields ``unknown``.
    """
    raw_fw = (framework_override or "").strip().lower()
    fw = canonical_owner_framework(raw_fw)
    if raw_fw:
        if raw_fw in _NO_FRAMEWORK_SENTINELS:
            return _UNKNOWN
        return fw
    parts = {canonical_owner_framework(p) for p in Path(kernel_path).parts}
    for fwname in _FRAMEWORKS:
        if fwname in parts:
            return fwname
    return _UNKNOWN


def _read_text_safe(
    path: str,
    source_contents: dict[str, str] | None = None,
) -> str:
    if source_contents is not None:
        candidates = (str(path), str(Path(path).resolve()))
        for candidate in candidates:
            if candidate in source_contents:
                return source_contents[candidate]
    try:
        return Path(path).read_text(errors="replace")
    except Exception:  # noqa: BLE001 - best-effort
        return ""


def find_defining_source(
    op: str,
    anchor_path: str,
    anchor_source: str,
    source_files: list[str] | None,
    *,
    source_contents: dict[str, str] | None = None,
) -> str:
    """Return the source text that DEFINES ``op`` (for signature/dtype parsing).

    For a repository task the operation may live in a file OTHER than the anchor
    (e.g. the anchor is a host wrapper), so scan the whole source set. Prefers the
    anchor when it defines ``op``. Falls back to the anchor source when nothing
    matches. Single-file tasks pass no extra ``source_files`` and simply reuse the
    anchor source.
    """
    if not op:
        return anchor_source or ""
    def_re = re.compile(r"\bdef\s+" + re.escape(op) + r"\b")
    glob_re = re.compile(r"__global__[^\n]*\b" + re.escape(op) + r"\b")
    if def_re.search(anchor_source or "") or glob_re.search(anchor_source or ""):
        return anchor_source or ""
    for f in source_files or []:
        txt = _read_text_safe(f, source_contents)
        if txt and (def_re.search(txt) or glob_re.search(txt)):
            return txt
    return anchor_source or ""


def find_defining_path(
    op: str,
    anchor_path: str,
    anchor_source: str,
    source_files: list[str] | None,
    *,
    source_contents: dict[str, str] | None = None,
) -> str:
    """Return the PATH of the file that DEFINES ``op`` (for framework detection).

    The framework identity must follow the file where the compute kernel is
    actually DEFINED, not the anchor that merely calls it. A common cross-package
    case: the ``--kernel`` anchor is a vLLM/SGLang entry/dispatch file, but the
    real ``@triton.jit`` / ``__global__`` kernel lives in aiter, listed in
    ``source_files``. Keying the framework off the anchor path would then yield
    ``vllm`` on one side and ``aiter`` on another and split the slug. Mirrors
    ``find_defining_source`` but returns the path; falls back to the anchor path
    when the anchor defines ``op`` or nothing matches.
    """
    if not op:
        return anchor_path
    def_re = re.compile(r"\bdef\s+" + re.escape(op) + r"\b")
    glob_re = re.compile(r"__global__[^\n]*\b" + re.escape(op) + r"\b")
    if def_re.search(anchor_source or "") or glob_re.search(anchor_source or ""):
        return anchor_path
    for f in source_files or []:
        txt = _read_text_safe(f, source_contents)
        if txt and (def_re.search(txt) or glob_re.search(txt)):
            return f
    return anchor_path


def infer_source_owner_framework(
    *,
    kernel_path: str,
    kernel_source: str,
    target_functions: list[str] | None = None,
    source_files: list[str] | None = None,
    framework_override: str = "",
    source_contents: dict[str, str] | None = None,
    concrete_operation: str = "",
) -> str:
    """Resolve the canonical framework that owns the concrete operation."""
    concrete_op = concrete_operation or resolve_operation(kernel_source, kernel_path, target_functions=target_functions)
    defining_path = find_defining_path(
        concrete_op,
        kernel_path,
        kernel_source,
        source_files,
        source_contents=source_contents,
    )
    return detect_framework(
        defining_path,
        framework_override=framework_override,
    )


# --------------------------------------------------------------------------- #
# deterministic signature -> input dtypes
# --------------------------------------------------------------------------- #
def _balanced_parens(source: str, open_idx: int) -> tuple[str, int]:
    """Return (inner, close_idx) for the parens opened at ``open_idx``."""
    depth = 0
    for i in range(open_idx, len(source)):
        c = source[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return source[open_idx + 1 : i], i
    return "", -1


def _signature_params(source: str, func: str) -> str | None:
    """Return the raw parameter-list string of ``func``'s definition, or None.

    Handles a Python ``def`` and a C/C++ definition (a ``func(...) {`` whose
    parens are followed by a body), skipping call sites.
    """
    m = re.search(r"\bdef\s+" + re.escape(func) + r"\s*\(", source)
    if m:
        inner, _ = _balanced_parens(source, m.end() - 1)
        return inner
    for m in re.finditer(r"\b" + re.escape(func) + r"\s*\(", source):
        inner, close = _balanced_parens(source, m.end() - 1)
        if close < 0:
            continue
        # A definition has a body after the (optional trailing return type).
        if re.match(r"\s*(?:->[^\{;]*)?\{", source[close + 1 : close + 60]):
            return inner
    return None


def _split_top_level(params: str) -> list[str]:
    """Split a parameter list on top-level commas (respecting brackets)."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for c in params:
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth = max(0, depth - 1)
        if c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _parse_param_py(p: str) -> tuple[str, str]:
    """Parse a Python parameter ``name: type = default`` -> (name, type)."""
    p = p.strip()
    if p in ("self", "cls") or p.startswith("*") or p == "/":
        return "", ""
    typ = ""
    if ":" in p:
        name_part, ann = p.split(":", 1)
        typ = ann.split("=")[0].strip()
    else:
        name_part = p.split("=")[0]
    return name_part.strip(), typ


def _parse_param_c(p: str) -> tuple[str, str]:
    """Parse a C/C++ parameter ``const float* a`` -> (name, type)."""
    p = p.split("=")[0].strip()
    if not p or p == "void":
        return "", ""
    # Peel trailing array subscripts, then the last identifier is the parameter
    # name and whatever precedes it is the type. Each subscript is matched
    # unambiguously as ``[<optional spaces><optional digits><optional spaces>]``
    # via a single leading ``\s*`` plus a ``(?:\d+\s*)?`` group (the ``\d+`` gates
    # entry), so there is no two-adjacent-``\s*`` split — this stays linear and
    # cannot backtrack catastrophically on hostile "a[ ][ ]..." input.
    body = p
    arr = ""
    m_arr = re.search(r"((?:\[\s*(?:\d+\s*)?\])+)\s*$", body)
    if m_arr:
        arr = m_arr.group(1)
        body = body[: m_arr.start()].rstrip()
    m_name = re.search(r"([A-Za-z_]\w*)\s*$", body)
    if not m_name:
        return "", ""
    typ = body[: m_name.start()].strip()
    name = m_name.group(1).strip()
    # Pointer/reference markers may attach to the name side ("float *a"); fold
    # them back into the type so the recorded dtype is faithful.
    for mark in ("*", "&"):
        n = p.count(mark)
        if n and mark not in typ:
            typ = (typ + " " + mark * n).strip()
    return name, (typ + arr).strip()


def _strip_param_comments(params: str, is_c: bool) -> str:
    """Remove comments from a raw parameter-list string.

    Signatures commonly carry per-line comments (e.g. AITER annotates each tensor
    param with a shape comment); their commas/newlines would otherwise corrupt
    top-level splitting and pollute the parsed parameter names.
    """
    if is_c:
        params = re.sub(r"/\*.*?\*/", "", params, flags=re.DOTALL)
        return "\n".join(re.sub(r"//.*$", "", ln) for ln in params.splitlines())
    return "\n".join(re.sub(r"#.*$", "", ln) for ln in params.splitlines())


def extract_input_dtypes(kernel_source: str, func: str, lang: str) -> dict[str, str]:
    """Parse ``func``'s signature into ``{param_name: declared_type}``.

    Deterministic best-effort: records the types literally declared in the entry
    signature (C types for HIP/CUDA, annotations for Python DSLs). Parameters
    with no declared type map to ``unknown``. Returns ``{}`` when the signature
    can't be located.
    """
    if not kernel_source or not func:
        return {}
    params = _signature_params(kernel_source, func)
    if params is None:
        return {}
    is_c = lang in _C_LIKE_LANGS
    params = _strip_param_comments(params, is_c)
    out: dict[str, str] = {}
    for p in _split_top_level(params):
        name, typ = _parse_param_c(p) if is_c else _parse_param_py(p)
        if name:
            out[name] = typ or _UNKNOWN
    return out


# --------------------------------------------------------------------------- #
# LLM summarization (strategy / recipe / lessons / category only)
# --------------------------------------------------------------------------- #
_SUMMARY_SYSTEM = (
    "You analyze the trajectory of an autonomous GPU-kernel optimization run and "
    "extract a compact, structured summary. You never write code - you only "
    "report what the winning change did and classify the operator. Answer with a "
    "single JSON object and nothing else."
)


def _summary_prompt(op: str, digest: str, kernel_source: str) -> str:
    """Build the user prompt asking for a strict-JSON structured summary."""
    digest = (digest or "")[:_MAX_DIGEST_CHARS]
    kernel_source = (kernel_source or "")[:_MAX_SOURCE_CHARS]
    return f"""\
Operator under optimization: {op}

## Kernel source (target of the run, possibly truncated)
{kernel_source or "(unavailable)"}

## Optimization trajectory (attempts, diffs, per-iteration lessons)
{digest or "(no trajectory recorded)"}

## Your task
Return ONE JSON object with EXACTLY these keys (use "" when you cannot determine
a value; never invent):
{{
  "category": "one of: GEMM, attention, MOE, communication, others",
  "strategy": "one sentence: the direction of the WINNING change",
  "recipe": "concrete, reproducible steps of the winning change",
  "lessons": "distilled: what worked / what failed / pitfalls for this operator"
}}
Output ONLY the JSON object.
"""


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply (tolerates code fences)."""
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end > start else ""
    if not candidate:
        return {}
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except (ValueError, TypeError):
        return {}


async def _query_llm(config, workspace: str, prompt: str, usage=None) -> str:
    """Run one no-edit query through the globally configured agent backend."""
    from kernelforge.agent_backends.base import (
        AgentRunSpec,
        AgentToolPolicy,
    )
    from kernelforge.agent_backends.registry import (
        create_registered_backend,
    )

    backend = create_registered_backend(config.agent_runtime())

    result = await backend.run(
        AgentRunSpec(
            system_prompt=_SUMMARY_SYSTEM,
            user_prompt=prompt,
            cwd=workspace,
            writable=False,
            timeout_sec=_LLM_TIMEOUT_SEC,
            reasoning_effort="high",
            tool_policy=AgentToolPolicy(
                read=False,
                search=False,
                write=False,
                shell=False,
                max_turns=1,
            ),
            protected_globs=["*"],
        ),
        usage=usage,
    )
    return result.text.strip()


def _normalize_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce the LLM's JSON into the fields we store, with safe fallbacks."""

    def _str(key: str) -> str:
        val = raw.get(key)
        return val.strip() if isinstance(val, str) and val.strip() else ""

    category = _str("category").lower()
    if category not in _CATEGORIES:
        category = "others"
    return {
        "category": category,
        "strategy": _str("strategy"),
        "recipe": _str("recipe"),
        "lessons": _str("lessons"),
    }


def summarize_run(config, workspace: str, op: str, digest: str, kernel_source: str, usage=None) -> dict[str, Any]:
    """Summarize a run with one LLM call; returns normalized fields (never raises).

    On any backend failure, timeout, or unparsable reply every field degrades to
    its empty / ``others`` default so the caller can still write a page.
    """
    defaults = _normalize_summary({})
    prompt = _summary_prompt(op, digest, kernel_source)
    try:
        reply = asyncio.run(
            asyncio.wait_for(
                _query_llm(config, workspace, prompt, usage=usage),
                timeout=_LLM_TIMEOUT_SEC,
            )
        )
    except Exception as exc:  # noqa: BLE001 - best-effort; keep defaults
        log.warning("experience summarize LLM failed: %r", exc)
        return defaults
    parsed = _normalize_summary(_extract_json(reply))
    log.info("experience summarize: category=%s strategy=%r", parsed["category"], parsed["strategy"][:60])
    return parsed


# --------------------------------------------------------------------------- #
# page rendering
# --------------------------------------------------------------------------- #
def _changed_files_from_diff(diff: str) -> list[str]:
    """Extract the list of changed file paths from a unified/git diff."""
    files: list[str] = []
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)", diff or "", re.MULTILINE):
        path = m.group(2)
        if path not in files:
            files.append(path)
    return files


def _measurement_line(metric: dict[str, Any]) -> str:
    """State the speedup with the two timings it was computed from."""
    speedup = metric.get("speedup")
    best = metric.get("wall_ms")
    baseline = metric.get("baseline_wall_ms")
    if not isinstance(speedup, (int, float)):
        return "- Speedup: not recorded\n"
    line = f"- Speedup: {float(speedup):.6g}x"
    if isinstance(best, (int, float)) and isinstance(baseline, (int, float)):
        line += f" ({float(best):.4g} ms vs {float(baseline):.4g} ms)"
    return line + "\n"


def _experience_markdown(
    *,
    canonical_id: str,
    knowledge: dict[str, Any],
    patch_name: str,
) -> str:
    """Render one recorded run for a reader.

    The record's own fields are what a later run compares and ranks; this is
    what a person or an agent reads when deciding whether a candidate is worth
    replaying. The diff is not inlined: it sits beside this file under its own
    name, and copying it here would store the same bytes twice.
    """
    metric = knowledge.get("metric") if isinstance(knowledge.get("metric"), dict) else {}
    kernel = knowledge.get("task_id") or canonical_id
    snr = metric.get("snr_db")
    changed = knowledge.get("changed_files") or []
    sources = knowledge.get("source_files") or []
    dtypes = knowledge.get("dtypes") or []

    head = [f"# {canonical_id}\n\n", f"- Task: `{kernel}`\n"]
    head.append(_measurement_line(metric))
    if isinstance(snr, (int, float)):
        head.append(f"- Correctness: SNR {float(snr):.1f} dB\n")
    if metric.get("gpu_arch"):
        head.append(f"- Compiled for: {metric['gpu_arch']}\n")
    if changed:
        head.append(f"- Changed files: {', '.join(str(p) for p in changed)}\n")
    head.append(f"- Patch: `{patch_name}`\n")

    body = []
    for title, key in (
        ("Strategy", "strategy"),
        ("Recipe", "recipe"),
        ("Lessons", "lessons"),
    ):
        body.append(f"\n## {title}\n\n{str(knowledge.get(key) or '(not recorded)')}\n")

    body.append("\n## Implementation\n\n")
    body.append(f"- Signature: `{knowledge.get('implementation_signature') or ''}`\n")
    if sources:
        body.append(f"- Source files: {', '.join(str(p) for p in sources)}\n")
    if dtypes:
        body.append(f"- Input dtypes: {', '.join(str(d) for d in dtypes)}\n")
    return "".join(head + body)


def write_run_experience(
    *,
    config,
    workspace: str,
    kernel_path: str,
    kernel_source: str,
    kernel_backend: str,
    gpu_target: str,
    experiment_id: str,
    baseline_wall_ms: float | None,
    best_wall_ms: float | None,
    mean_case_speedup: float | None = None,
    cumulative_diff: str,
    digest: str,
    snr_db: float | None = None,
    source_files: list[str] | None = None,
    target_functions: list[str] | None = None,
    operator_name: str = "",
    implementation_signature_override: str = "",
    implementation_identity_override: dict[str, Any] | None = None,
    framework: str = "",
    summary_override: dict[str, Any] | None = None,
    reused_speedup: float | None = None,
    usage=None,
) -> dict[str, Any]:
    """Mirror one run's best solution into the experience store. Never raises.

    Logical identity and the implementation signature are derived
    deterministically from the caller's operator, editable sources, concrete
    target symbols, framework, and backend. Only experience prose/category
    may come from an LLM. Returns a small
    status dict for logging: ``{"written": bool, "reason": str, ...}``.

    The caller persists that reason, and the store client this write opens
    authenticates with a bearer token, so a failure's text is redacted and
    bounded before it is returned or logged.

    ``summary_override`` supplies a pre-built ``{category, strategy, recipe,
    lessons}`` dict INSTEAD of the (expensive, ~150s) LLM summarization. The
    incremental-publish path (called after every new best, inside the running
    loop) passes a cheap heuristic summary so it neither stalls the loop nor
    nests an event loop; the final graceful write passes None to get the precise
    LLM summary, which overwrites the same solution page in place.
    """
    try:
        return _write_run_experience_impl(
            config=config,
            workspace=workspace,
            kernel_path=kernel_path,
            kernel_source=kernel_source,
            kernel_backend=kernel_backend,
            gpu_target=gpu_target,
            experiment_id=experiment_id,
            baseline_wall_ms=baseline_wall_ms,
            best_wall_ms=best_wall_ms,
            mean_case_speedup=mean_case_speedup,
            cumulative_diff=cumulative_diff,
            digest=digest,
            snr_db=snr_db,
            source_files=source_files,
            target_functions=target_functions,
            operator_name=operator_name,
            implementation_signature_override=implementation_signature_override,
            implementation_identity_override=implementation_identity_override,
            framework=framework,
            summary_override=summary_override,
            reused_speedup=reused_speedup,
            usage=usage,
        )
    except Exception as exc:  # noqa: BLE001 - a KB write must never break the loop
        # Imported here rather than at module scope: the reader imports this
        # module for detect_framework, so a top-level import would close a cycle.
        from kernelforge.knowledge.experience_reader import sanitize_read_error
        from kernelforge.rewrite_by_flydsl.agent_kb import kb_store_secrets

        reason = sanitize_read_error(exc, secrets=kb_store_secrets(config))
        log.warning("experience write failed (skipped): %s", reason)
        return {"written": False, "reason": reason}


def _write_run_experience_impl(
    *,
    config,
    workspace,
    kernel_path,
    kernel_source,
    kernel_backend,
    gpu_target,
    experiment_id,
    baseline_wall_ms,
    best_wall_ms,
    mean_case_speedup,
    cumulative_diff,
    digest,
    snr_db,
    source_files=None,
    target_functions=None,
    operator_name="",
    implementation_signature_override="",
    implementation_identity_override=None,
    framework="",
    summary_override=None,
    reused_speedup=None,
    usage=None,
) -> dict[str, Any]:
    # The hardware model addresses the record; without it the run would file its
    # experience under a GPU-less address that no read ever resolves to, so a
    # silent write is worse than no write at all.
    gpu_type = str(getattr(config, "gpu_type", "") or "").strip()
    if not gpu_type:
        return {"written": False, "reason": "missing_gpu_type"}

    if not isinstance(mean_case_speedup, (int, float)):
        return {"written": False, "reason": "missing_mean_case_speedup"}
    this_speedup = float(mean_case_speedup)
    if not math.isfinite(this_speedup) or this_speedup <= 0.0:
        return {"written": False, "reason": "invalid_mean_case_speedup"}
    if this_speedup <= 1.0:
        return {"written": False, "reason": "no_improvement"}
    # A warm-started run begins already holding a recorded solution. Recording it
    # again under this run's id would not be a new solution, just a second copy
    # of the one it started from, and enough copies crowd the ranking a later
    # warm start reads. The patch itself stays: it is still this run's result.
    if (
        isinstance(reused_speedup, (int, float))
        and math.isfinite(float(reused_speedup))
        and this_speedup <= float(reused_speedup)
    ):
        return {"written": False, "reason": "no_improvement_over_reuse"}
    if not (cumulative_diff or "").strip():
        return {"written": False, "reason": "empty_diff"}

    # Deterministic identity (never LLM-inferred, so the address is stable).
    # Read and write share one resolver so a warm start cannot look somewhere
    # a prior write never reached.
    from kernelforge.knowledge.loop_identity import (
        EXPERIENCE_ARTIFACT,
        PATCH_ARTIFACT,
        resolve_loop_identity,
    )

    identity, concrete_op, framework = resolve_loop_identity(
        kernel_path=kernel_path,
        kernel_source=kernel_source,
        kernel_backend=kernel_backend,
        gpu_type=gpu_type,
        target_functions=target_functions,
        source_files=source_files,
        framework=framework,
        operator_name=operator_name,
        producer=getattr(config, "producer", ""),
    )
    op = identity.kernel_name
    backend_lang = identity.backend
    op_source = find_defining_source(concrete_op, kernel_path, kernel_source, source_files)
    dtypes = extract_input_dtypes(op_source, concrete_op, backend_lang)
    if implementation_signature_override and implementation_identity_override:
        impl_signature = str(implementation_signature_override)
        impl_identity = dict(implementation_identity_override)
        if hash_implementation_identity(impl_identity) != impl_signature:
            raise ValueError("implementation identity override does not match its signature")
    elif implementation_signature_override or implementation_identity_override:
        raise ValueError("implementation signature and identity overrides must be supplied together")
    else:
        # Compatibility for direct/non-campaign callers. Forge campaigns always
        # supply the immutable pristine contract captured before warm-start.
        impl_signature, impl_identity = implementation_signature(
            workspace=workspace,
            kernel_path=kernel_path,
            source_files=source_files,
            framework=framework,
        )
    log.info(
        "experience identity: op=%s concrete=%s framework=%s backend=%s implementation=%s",
        op,
        concrete_op,
        framework,
        backend_lang,
        impl_signature[:12],
    )

    # The KB Store is addressed by the recipe identity, so the facade is opened
    # on it rather than on a composed slug. An unconfigured store yields an
    # inactive facade instead of raising, which is the cold-start outcome the
    # loop already handles.
    #
    # Imported here rather than at module scope: the facade's identity module
    # imports this one, so a top-level import would close a cycle.
    from kernelforge.rewrite_by_flydsl.agent_kb import KernelRecipeKB

    kb = KernelRecipeKB.open_identity(identity, config)
    if not kb.active:
        log.info("experience write skipped: %s", kb.reason or "not_configured")
        return {"written": False, "reason": kb.reason or "not_configured"}

    # Experience prose + category. Use the caller-supplied cheap summary when
    # given (incremental publish); otherwise pay for the LLM summary (final
    # graceful write). ``_normalize_summary`` guarantees all fields are present.
    if summary_override is not None:
        summary = _normalize_summary(summary_override)
    else:
        summary = summarize_run(
            config=config,
            workspace=workspace,
            op=op,
            digest=digest,
            kernel_source=kernel_source,
            usage=usage,
        )

    metric = {
        "wall_ms": best_wall_ms,
        "baseline_wall_ms": baseline_wall_ms,
        "speedup": round(this_speedup, 4),
        "snr_db": snr_db,
        "gpu_arch": gpu_target,
    }
    changed_files = _changed_files_from_diff(cumulative_diff)

    # Everything a later run needs to judge and reuse this solution, minus the
    # diff: that travels as an artifact so a reader can rank candidates without
    # pulling a patch it may not want.
    knowledge = {
        "task_id": experiment_id,
        "category": summary["category"],
        "strategy": summary["strategy"],
        "recipe": summary["recipe"],
        "lessons": summary["lessons"],
        "metric": metric,
        "changed_files": changed_files,
        "dtypes": dtypes,
        "source_files": list(impl_identity["source_paths"]),
        "implementation_signature": impl_signature,
        "implementation_identity": impl_identity,
    }

    with tempfile.TemporaryDirectory(prefix="forge-loop-kb-") as staging:
        patch_path = Path(staging) / PATCH_ARTIFACT
        # Bytes, not text: writing through a text handle would translate the
        # newlines a patch has to reproduce exactly.
        patch_path.write_bytes(cumulative_diff.encode("utf-8"))
        experience_path = Path(staging) / EXPERIENCE_ARTIFACT
        experience_path.write_text(
            _experience_markdown(
                canonical_id=kb.canonical_id,
                knowledge=knowledge,
                patch_name=PATCH_ARTIFACT,
            ),
            encoding="utf-8",
        )
        # The store names a record after its own content, so an LLM-written
        # summary of a solution this run already recorded is filed as a second
        # record: same patch, same speedup, richer prose. Suppressing that pair
        # needs a way to name the record being revised, which the store does not
        # expose, so the duplicate stands until the write strategy is settled.
        outcome = kb.write_candidate(
            knowledge,
            files={
                PATCH_ARTIFACT: patch_path,
                EXPERIENCE_ARTIFACT: experience_path,
            },
            speedup=this_speedup,
        )

    if not outcome.get("written"):
        return {"written": False, "reason": outcome.get("reason") or "write_failed"}

    solution = str(outcome.get("solution") or "")
    log.info(
        "experience written: %s (speedup %.3f, champion=%s)",
        solution,
        this_speedup,
        outcome.get("champion"),
    )
    return {
        "written": True,
        "kernel": kb.canonical_id,
        "solution": solution,
        "session_id": outcome.get("session_id", ""),
        "champion": bool(outcome.get("champion")),
        "speedup": this_speedup,
    }
