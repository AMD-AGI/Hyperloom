# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse the discovery model's answer into ranked fusion recipes."""

from __future__ import annotations

import contextlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from hyperloom_common.coerce import to_float

from .diagnose import categories_in_text
from .discovery_prompt import DiscoveryTarget, declared_ops, declared_traits
from .locate import (
    _compile_pass_recipe,
    _read_source,
    _unclaimable_note,
    covered_by_vllm_compile_pass,
    out_of_scope_terms,
    rank_recipes,
    vllm_compile_pass_state,
)
from .models import Recipe
from .vllm_passes import resolve_target_runtime

log = logging.getLogger("forge_fusion")


def _json_object(fragment: str) -> dict[str, Any] | None:
    """``fragment`` parsed as JSON when it is an object, else None."""
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        obj = json.loads(fragment)
        if isinstance(obj, dict):
            return obj
    return None


def _salvage_objects(text: str) -> list[dict[str, Any]]:
    """Recover every complete top-level ``{...}`` object from (possibly truncated) text, ignoring braces inside strings.

    The object a cut left half-written comes last, once repaired, unless it repeats one already recovered.
    """
    out: list[dict[str, Any]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    commas: list[int] = []
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
                commas = []
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            obj = _json_object(text[start : i + 1]) if depth == 0 else None
            if obj is not None:
                out.append(obj)
        elif ch == "," and depth == 1:
            commas.append(i - start)
    repaired = _repair_truncated_object(text[start:], depth, in_str, commas) if depth else None
    if repaired is not None and repaired not in out:
        out.append(repaired)
    return out


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Pull JSON fusion proposals out of model text."""
    if not text:
        return []
    fences = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidates = list(fences)
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : i + 1])
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            return parsed
    # Fallback: salvage complete objects from a truncated/unclosed array, then the object the cut left half-written --
    # with three quarters of responses arriving truncated, that last object is often the only one there is.
    salvaged = _salvage_objects(text)
    if not salvaged:
        log.warning(
            "discovery: no JSON proposals parsed from %d chars of model text; "
            "this is a parse failure, NOT a no_opportunity result",
            len(text),
        )
    return salvaged


# A repaired object has to carry enough of the fusion description to act on.
_REPAIRED_REQUIRED_ANY = ("op_chain", "fusion_math")


def _repair_truncated_object(fragment: str, depth: int, in_str: bool, commas: list[int]) -> dict[str, Any] | None:
    """Recover the proposal that a cut-off response left half-written.

    ``fragment`` runs from the object's opening brace to the cut, with ``depth`` braces and maybe a string still
    open; ``commas`` are the offsets of its top-level separators, where a field before the cut ends complete.
    """
    attempts = [fragment + ('"' if in_str else "") + "}" * depth]
    attempts.extend(fragment[:mark] + "}" for mark in reversed(commas))
    for attempt in attempts:
        try:
            obj = json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or not str(obj.get("name") or "").strip():
            continue
        if any(str(obj.get(key) or "").strip() for key in _REPAIRED_REQUIRED_ANY):
            return obj
    return None


def _norm_env_flag(flag: str, model_type: str) -> str:
    """Normalize/model-prefix a proposed env flag (e.g. FUSED_QK -> ZAYA_FUSED_QK)."""
    f = re.sub(r"\s+", "_", (flag or "FUSED").strip()).upper()
    prefix = f"{model_type.upper()}_" if model_type else ""
    if prefix and not f.startswith(prefix):
        f = prefix + f
    return f


def _field_text(item: dict[str, Any], key: str, default: str = "") -> str:
    """A proposal field as text, ``default`` when the model left it empty or null."""
    return str(item.get(key) or default)


def _trace_lacks(categories: list[str], category_shares: dict[str, float] | None) -> bool:
    """Whether the trace measured category shares but none of ``categories``."""
    if not category_shares or not categories:
        return False
    return {category for category, share in category_shares.items() if float(share) > 0.0}.isdisjoint(categories)


def _as_sequence(value: Any) -> list[Any]:
    """A proposal field that may arrive as one string or a list of them."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def resolve_repo_file(value: Any, repo_root: str) -> str:
    """Resolve a repo-scope proposal's claimed path to a real file, or ``""``.

    Repo scope removes the offered-file list that
    :func:`resolve_proposed_source_file` matches against, so existing on disk and
    living inside the repository is the whole check -- and it is a HARD one.
    Falling back to the model file the way single-file discovery does would
    reintroduce the silent retarget that repo scope exists to remove: the caller
    would get a recipe pointing at a file the model never proposed.
    """
    claimed = str(value or "").strip()
    if not claimed:
        return ""
    root: Optional[Path] = None
    if repo_root:
        with contextlib.suppress(OSError):
            root = Path(repo_root).expanduser().resolve()
    candidates = [Path(claimed).expanduser()]
    if root is not None and not candidates[0].is_absolute():
        candidates.insert(0, root / candidates[0])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if root is not None and not resolved.is_relative_to(root):
            log.warning("discovery: ignoring %s, which is outside the framework repo %s", resolved, root)
            continue
        return str(resolved)
    return ""


def resolve_proposed_source_file(value: Any, in_scope_files: Sequence[str], default: str) -> str:
    """Map a proposal's claimed call-site file onto one the run actually offered.

    The model is asked to copy a path back verbatim, and mostly does; a trailing
    component still identifies the file unambiguously when it does not. Anything
    that matches nothing falls back to the primary file rather than inventing a
    target the loop cannot track.
    """
    claimed = str(value or "").strip()
    if not claimed or not in_scope_files:
        return default
    for path in in_scope_files:
        if claimed == path:
            return path
    claimed_parts = Path(claimed).parts
    for path in in_scope_files:
        parts = Path(path).parts
        if claimed_parts and parts[-len(claimed_parts) :] == claimed_parts:
            return path
    for path in in_scope_files:
        if Path(path).name == Path(claimed).name:
            return path
    log.warning(
        "discovery: proposed source_file %r matches no in-scope file; using %s",
        claimed[:120],
        Path(default).name or default,
    )
    return default


def _proposal_edit_files(item: dict[str, Any], target: DiscoveryTarget, name: str) -> list[str]:
    """The files a proposal wires itself into, its call site first; ``[]`` drops the proposal.

    Single-file discovery maps the claimed file onto the one source file the prompt embedded. Repo scope embedded
    nothing, so each proposal names its own files, and one whose call site does not resolve is dropped rather than
    retargeted onto the target's source file.
    """
    if not target.repo_scope:
        scope_files = [target.source_file] if target.source_file else []
        return [resolve_proposed_source_file(item.get("source_file"), scope_files, target.source_file)]
    call_site = resolve_repo_file(item.get("source_file"), target.repo_root)
    if not call_site:
        log.warning(
            "discovery: dropping %s -- its source_file %r is not a file in %s",
            name,
            str(item.get("source_file") or "")[:200],
            target.repo_root or "the framework repo",
        )
        return []
    files = [call_site]
    for value in _as_sequence(item.get("additional_files")):
        resolved = resolve_repo_file(value, target.repo_root)
        if not resolved:
            log.warning(
                "discovery: %s listed an additional file that does not resolve (%r); ignoring it",
                name,
                str(value)[:200],
            )
        elif resolved not in files:
            files.append(resolved)
    return files


def _scope_gate_drops(name: str, files: list[str], terms: list[str], explicit_target: bool) -> bool:
    """Whether the scope gate drops a proposal declaring ``terms`` that none of its edit ``files`` perform.

    A fusion is wired by replacing a call site in the files the proposal claims, so one claiming ops those files do
    not perform is unwireable no matter how good the kernel is. The files are the ones the model was shown or named,
    never a file the chain was not claimed to be in. The gate recognizes nine coarse terms, so it can reject a
    wireable chain; an operator-named target is kept with a warning, because discarding what was explicitly asked
    for hides that.
    """
    outside = out_of_scope_terms("\n".join(_read_source(path) for path in files), terms)
    if outside:
        log.log(
            logging.WARNING if explicit_target else logging.INFO,
            "discovery: %s %s (%s not performed in %s -- the fusion crosses "
            "a module boundary and has no wireable call site there)",
            "keeping operator-named" if explicit_target else "dropping",
            name,
            ",".join(outside),
            _file_names(files),
        )
    return bool(outside) and not explicit_target


def _file_names(paths: Sequence[str]) -> str:
    """``paths`` by file name, for a log line."""
    return ", ".join(Path(p).name or p for p in paths)


def parse_discovered_recipes(
    text: str,
    target: DiscoveryTarget,
    *,
    category_shares: dict[str, float] | None = None,
    explicit_target: bool = False,
) -> list[Recipe]:
    """Convert the LLM's JSON proposals into ranked :class:`Recipe` objects.

    ``explicit_target``: an operator named the target, so the scope gate keeps, with a warning, what it would drop.
    """
    runtime = resolve_target_runtime(target.install.framework, framework_root=target.install.framework_root)
    out: list[Recipe] = []
    for i, item in enumerate(_extract_json_array(text)):
        name = _field_text(item, "name", f"discovered_{i + 1}").strip()
        op_chain = _field_text(item, "op_chain")
        fusion_math = _field_text(item, "fusion_math", op_chain)
        # The fusion-DEFINING fields (name / op-chain / math) -- NOT the free-prose rationale or grep anchors, which
        # can mention an op in passing and would attach a category the fusion does not actually involve.
        defining_text = " ".join([name, op_chain, fusion_math])
        # What this fusion IS, as opposed to how this run described it.
        declared, traits = declared_ops(item), declared_traits(item)
        # The op categories: from the declaration when there is one, else from the prose.
        categories = categories_in_text(" ".join(declared or [defining_text]))
        # Hallucination gate FIRST: a proposal whose ops the trace never measured has nothing to remove, and that is
        # true whether we would author it or claim a framework pass for it.
        if _trace_lacks(categories, category_shares):
            log.info("discovery: dropping %s (proposed ops %s absent from the trace)", name, ",".join(categories))
            continue
        edit_files = _proposal_edit_files(item, target, name)
        if not edit_files or _scope_gate_drops(name, edit_files, [*declared, *traits], explicit_target):
            continue
        # Compile-pass gate: never author a chain vLLM fuses at compile time. It keys on precision and variant words
        # (quant, fp8, mla, kvcache) that no op category expresses, so it needs more than ``declared``.
        pass_name = covered_by_vllm_compile_pass(
            matched_categories=[],
            text=" ".join([*declared, *(traits or [defining_text])]),
            framework=target.install.framework,
        )
        state = vllm_compile_pass_state(pass_name, probe=target.install.pass_probe, runtime=runtime)
        if state is not None and state.enabled is True:
            log.info("discovery: dropping %s, vLLM compile pass %s already fuses it", name, state.flag or pass_name)
            continue  # the framework really does fuse this: authoring is a no-op
        # Absent / undecidable / pinned off: the framework is NOT fusing this for us, so keep the proposal as
        # authoring work and record why.
        pass_note = _unclaimable_note(state)
        if pass_note:
            log.info("compile pass not claimed for %s: %s", name, pass_note)
        existing_operator = _field_text(item, "existing_operator").strip()
        candidate_kind = _field_text(item, "candidate_kind").strip().lower()
        # ``integration`` is only meaningful with a named operator: the authoring prompt injects its "benchmark the
        # existing operator first" block only when both fields are set.
        if candidate_kind not in {"new_fusion", "replacement"}:
            candidate_kind = "integration" if existing_operator else "new_fusion"
        recipe = Recipe(
            pattern_id=f"llm:{name}",
            description=_field_text(item, "rationale", op_chain or name)[:300],
            env_flag=_norm_env_flag(_field_text(item, "env_flag", "FUSED"), target.model_type),
            source_file=edit_files[0],
            extra_files=edit_files[1:],
            source_hints=[str(a) for a in _as_sequence(item.get("source_anchors") or [])],
            fusion_math=fusion_math,
            eager_reference_hint=_field_text(item, "eager_reference"),
            shapes=target.shapes,
            matched_categories=categories,
            trigger_share=to_float(item.get("priority"), max(0.1, 1.0 - 0.1 * i)),  # preserve LLM order when absent
            source_confirmed=True,  # the LLM read the real source to propose it
            candidate_kind=candidate_kind,
            existing_operator=existing_operator,
            compile_pass_note=pass_note,
        )
        if state is not None and state.claimable:
            recipe = _compile_pass_recipe(recipe, state, "chain", recipe.description[:200])
        elif edit_files != [target.source_file]:
            log.info("discovery: %s wires into %s", name, _file_names(edit_files))
        out.append(recipe)
    return rank_recipes(sorted(out, key=lambda r: r.trigger_share, reverse=True))
