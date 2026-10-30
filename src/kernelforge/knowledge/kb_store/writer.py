# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Record a forge-loop run's best solution in the KB Store."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.unified_diff import touched_paths

from kernelforge.knowledge.kb_store.errors import (
    kb_store_secrets,
    sanitize_read_error,
)
from kernelforge.knowledge.kb_store.identity.implementation import (
    hash_implementation_identity,
    implementation_signature,
)
from kernelforge.knowledge.kb_store.identity.run import (
    EXPERIENCE_ARTIFACT,
    PATCH_ARTIFACT,
    LoopKernel,
    resolve_loop_identity,
)
from kernelforge.knowledge.kb_store.identity.source_resolution import (
    find_defining_source,
)

log = logging.getLogger(__name__)

_CATEGORIES = {"gemm", "attention", "moe", "communication", "others"}
_UNKNOWN = "unknown"

# Bound the LLM inputs so a huge trajectory / source file can't blow the prompt.
_MAX_DIGEST_CHARS = 8000
_MAX_SOURCE_CHARS = 6000
_LLM_TIMEOUT_SEC = 150

_C_LIKE_LANGS = {"hip", "cuda", "cpp", "c"}


# --------------------------------------------------------------------------- # deterministic signature -> input
# dtypes --------------------------------------------------------------------------- #
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
    """Return the raw parameter-list string of ``func``'s definition, or None."""
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
    # Peel trailing array subscripts, then the last identifier is the parameter name and whatever precedes it is the
    # type.
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
    # Pointer/reference markers may attach to the name side ("float *a"); fold them back into the type so the recorded
    # dtype is faithful.
    for mark in ("*", "&"):
        n = p.count(mark)
        if n and mark not in typ:
            typ = (typ + " " + mark * n).strip()
    return name, (typ + arr).strip()


def _strip_param_comments(params: str, is_c: bool) -> str:
    """Remove comments from a raw parameter-list string."""
    if is_c:
        params = re.sub(r"/\*.*?\*/", "", params, flags=re.DOTALL)
        return "\n".join(re.sub(r"//.*$", "", ln) for ln in params.splitlines())
    return "\n".join(re.sub(r"#.*$", "", ln) for ln in params.splitlines())


def extract_input_dtypes(kernel_source: str, func: str, lang: str) -> dict[str, str]:
    """Parse ``func``'s signature into ``{param_name: declared_type}``."""
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


# --------------------------------------------------------------------------- # LLM summarization (strategy / recipe /
# lessons / category only) --------------------------------------------------------------------------- #
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
    from kernelforge.agent_backends.runtime import agent_runtime

    backend = create_registered_backend(agent_runtime(config))

    result = await backend.run(
        AgentRunSpec(
            role="experience summary",
            system_prompt=_SUMMARY_SYSTEM,
            user_prompt=prompt,
            cwd=workspace,
            writable=False,
            timeout_sec=_LLM_TIMEOUT_SEC,
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


def summarize_run(config, workspace: str, prompt: str, usage=None) -> dict[str, Any]:
    """Summarize a run with one LLM call on ``prompt``; returns normalized fields."""
    reply = asyncio.run(
        asyncio.wait_for(
            _query_llm(config, workspace, prompt, usage=usage),
            timeout=_LLM_TIMEOUT_SEC,
        )
    )
    parsed = _normalize_summary(_extract_json(reply))
    log.info("experience summarize: category=%s strategy=%r", parsed["category"], parsed["strategy"][:60])
    return parsed


# --------------------------------------------------------------------------- # page rendering
# --------------------------------------------------------------------------- #
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
    """Render one recorded run for a reader."""
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


@dataclass(frozen=True)
class RunOutcome:
    """One run's best measured solution, its cumulative diff, and the implementation it was measured against."""

    experiment_id: str
    gpu_target: str
    baseline_wall_ms: float | None
    best_wall_ms: float | None
    cumulative_diff: str
    digest: str
    mean_case_speedup: float | None = None
    snr_db: float | None = None
    reused_speedup: float | None = None
    implementation_signature: str = ""
    implementation_identity: dict[str, Any] | None = None


def write_run_experience(
    *,
    config,
    kernel: LoopKernel,
    outcome: RunOutcome,
    summary_override: dict[str, Any] | None = None,
    usage=None,
) -> dict[str, Any]:
    """Mirror one run's best solution into the experience store. Never raises."""
    try:
        return _write_run_experience_impl(
            config=config,
            kernel=kernel,
            outcome=outcome,
            summary_override=summary_override,
            usage=usage,
        )
    except Exception as exc:  # noqa: BLE001 - isolation boundary: one experience publication
        reason = sanitize_read_error(exc, secrets=kb_store_secrets(config))
        # Sanitized reason and stack frames only: the raw message may carry the KB store credentials.
        log.warning(
            "experience write failed (skipped): %s (%s)\n%s",
            reason,
            type(exc).__name__,
            "".join(traceback.format_tb(exc.__traceback__)),
        )
        return {"written": False, "reason": reason}


def _write_run_experience_impl(
    *, config, kernel: LoopKernel, outcome: RunOutcome, summary_override=None, usage=None
) -> dict[str, Any]:
    # The hardware model addresses the record; without it the run would file its experience under a GPU-less address
    # that no read ever resolves to, so a silent write is worse than no write at all.
    gpu_type = config.gpu_type.strip()
    if not gpu_type:
        return {"written": False, "reason": "missing_gpu_type"}

    if not isinstance(outcome.mean_case_speedup, (int, float)):
        return {"written": False, "reason": "missing_mean_case_speedup"}
    this_speedup = float(outcome.mean_case_speedup)
    if not math.isfinite(this_speedup) or this_speedup <= 0.0:
        return {"written": False, "reason": "invalid_mean_case_speedup"}
    # Losing to the source baseline is deliberately not a refusal: an operator whose best port is still slower than
    # what ships is the one that most needs its progress carried forward, and withholding it made every later run read
    # the same losing seed. Only the champion pointer stays gated on speedup.
    #
    # A warm-started run begins already holding a recorded solution, so recording it again under this run's id would
    # file a second copy of the one it started from rather than progress.
    reused = outcome.reused_speedup
    if isinstance(reused, (int, float)) and math.isfinite(float(reused)) and this_speedup <= float(reused):
        return {"written": False, "reason": "no_improvement_over_reuse"}
    if not (outcome.cumulative_diff or "").strip():
        return {"written": False, "reason": "empty_diff"}

    # Deterministic identity (never LLM-inferred, so the address is stable).
    identity, concrete_op, framework = resolve_loop_identity(kernel, gpu_type=gpu_type, producer=config.producer)
    op_source = find_defining_source(concrete_op, kernel.kernel_path, kernel.kernel_source, kernel.source_files)
    dtypes = extract_input_dtypes(op_source, concrete_op, identity.backend)
    impl_signature, impl_identity = _implementation_identity(kernel, outcome, framework)
    log.info(
        "experience identity: op=%s concrete=%s framework=%s backend=%s implementation=%s",
        identity.kernel_name,
        concrete_op,
        framework,
        identity.backend,
        impl_signature[:12],
    )

    # The KB Store is addressed by the recipe identity, so the facade is opened on it rather than on a composed slug.
    from kernelforge.knowledge.kb_store.recipe import KernelRecipeKB

    kb = KernelRecipeKB.open_identity(identity, config)
    if not kb.active:
        log.info("experience write skipped: %s", kb.reason or "not_configured")
        return {"written": False, "reason": kb.reason or "not_configured"}

    if summary_override is not None:
        summary = _normalize_summary(summary_override)
    else:
        summary = summarize_run(
            config=config,
            workspace=kernel.workspace,
            prompt=_summary_prompt(identity.kernel_name, outcome.digest, kernel.kernel_source),
            usage=usage,
        )

    # Everything a later run needs to judge and reuse this solution, minus the diff: that travels as an artifact so a
    # reader can rank candidates without pulling a patch it may not want.
    knowledge = {
        "task_id": outcome.experiment_id,
        "category": summary["category"],
        "strategy": summary["strategy"],
        "recipe": summary["recipe"],
        "lessons": summary["lessons"],
        "metric": {
            "wall_ms": outcome.best_wall_ms,
            "baseline_wall_ms": outcome.baseline_wall_ms,
            "speedup": round(this_speedup, 4),
            "snr_db": outcome.snr_db,
            "gpu_arch": outcome.gpu_target,
        },
        "changed_files": touched_paths(outcome.cumulative_diff),
        "dtypes": dtypes,
        "source_files": list(impl_identity["source_paths"]),
        "implementation_signature": impl_signature,
        "implementation_identity": impl_identity,
    }
    return _file_experience(kb, knowledge, outcome.cumulative_diff, this_speedup)


def _implementation_identity(kernel: LoopKernel, outcome: RunOutcome, framework: str) -> tuple[str, dict]:
    """The implementation the run was measured against: the campaign's own record of it, or one computed here."""
    signature, identity = outcome.implementation_signature, outcome.implementation_identity
    if signature and identity:
        signature, identity = str(signature), dict(identity)
        if hash_implementation_identity(identity) != signature:
            raise ValueError("implementation identity override does not match its signature")
        return signature, identity
    if signature or identity:
        raise ValueError("implementation signature and identity overrides must be supplied together")
    # Compatibility for direct/non-campaign callers.
    return implementation_signature(
        workspace=kernel.workspace,
        kernel_path=kernel.kernel_path,
        source_files=kernel.source_files,
        framework=framework,
    )


def _file_experience(kb, knowledge: dict[str, Any], cumulative_diff: str, speedup: float) -> dict[str, Any]:
    """Stage the run's patch and experience page, and file them in the store as one candidate."""
    with tempfile.TemporaryDirectory(prefix="forge-loop-kb-") as staging:
        patch_path = Path(staging) / PATCH_ARTIFACT
        # Bytes, not text: writing through a text handle would translate the newlines a patch has to reproduce
        # exactly.
        patch_path.write_bytes(cumulative_diff.encode("utf-8"))
        experience_path = Path(staging) / EXPERIENCE_ARTIFACT
        experience_path.write_text(
            _experience_markdown(canonical_id=kb.canonical_id, knowledge=knowledge, patch_name=PATCH_ARTIFACT),
            encoding="utf-8",
        )
        # The store names a record after its own content, so an LLM-written summary of a solution this run already
        # recorded is filed as a second record: same patch, same speedup, richer prose.
        written = kb.write_candidate(
            knowledge, files={PATCH_ARTIFACT: patch_path, EXPERIENCE_ARTIFACT: experience_path}, speedup=speedup
        )

    if not written.get("written"):
        return {"written": False, "reason": written.get("reason") or "write_failed"}

    solution = str(written.get("solution") or "")
    log.info("experience written: %s (speedup %.3f, champion=%s)", solution, speedup, written.get("champion"))
    return {
        "written": True,
        "kernel": kb.canonical_id,
        "solution": solution,
        "session_id": written.get("session_id", ""),
        "champion": bool(written.get("champion")),
        "speedup": speedup,
    }
