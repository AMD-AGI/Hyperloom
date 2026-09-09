# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 2 (deterministic half): assemble concrete recipes from matched patterns."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .calibration import DEFAULT_MIN_PREDICTED_GAIN, predict_cuda_graph_on_gain
from .models import Diagnosis, FusionPattern, Recipe
from .patterns import match_patterns
from .shapes import load_model_config, resolve_decode_shapes
from .vllm_passes import PassState, TargetRuntime, probe_pass_states, resolve_target_runtime

PassProbe = Callable[[str], PassState]

log = logging.getLogger("kernelforge.fusion.locate")


def resolve_framework_source_file(
    model_path: str,
    framework: str,
    *,
    framework_root: str = "",
    model_type: str = "",
) -> tuple[str, str]:
    """Best-effort path to the framework's model implementation file."""
    fw = (framework or "").strip().lower()
    if fw not in ("sglang", "vllm", "vllm-aiter"):
        return "", f"unsupported framework {fw!r}"

    config = load_model_config(model_path)
    mt = (model_type or str(config.get("model_type") or "")).strip()
    search_dirs = _model_search_dirs(fw, framework_root)

    if fw in _VLLM_FRAMEWORKS:
        registered = _vllm_registered_source(model_path)
        # The registry answers for whichever vLLM is importable in THIS process, which need not be the tree the
        # operator pinned.
        if registered and not _within_root(registered, framework_root):
            log.info(
                "vllm registry names %s, outside --framework-root %s; searching the pinned tree instead",
                registered,
                framework_root,
            )
            registered = ""
        if registered:
            log.info("source resolved to %s (vllm registry)", registered)
            return registered, "vllm registry"

    candidates: list[tuple[int, str]] = []
    for arch in _architecture_names(config):
        candidates.extend(_files_defining(arch, search_dirs))
    legacy = _legacy_source_file(mt, fw, framework_root) if mt else ""
    if legacy:
        candidates.append((_dir_rank(legacy, search_dirs), legacy))

    best = _best_implementation(candidates)
    if best:
        if best != legacy:
            how = "architecture search"
        elif fw in _VLLM_FRAMEWORKS:
            # Named apart so the manifest shows the registry was asked and missed, which is the case worth chasing:
            # vLLM knows and we did not hear it.
            how = "path convention (registry missed)"
        else:
            how = "path convention"
        log.info(
            "source resolved to %s (%s, %d candidate(s))",
            best,
            how,
            len({path for _rank, path in candidates}),
        )
        return best, how
    return "", "unresolved" if mt else "no model_type"


def _vllm_registered_source(model_path: str) -> str:
    """Source file of the class vLLM actually registers for this model, or \"\"."""
    archs = load_model_config(model_path).get("architectures") or []
    if not archs:
        return ""
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        models = ModelRegistry.models
    except (ImportError, AttributeError) as exc:
        log.warning("vllm registry unavailable (%s); using the path convention", exc)
        return ""
    for arch in archs:
        entry = models.get(arch)
        if not entry:
            continue
        try:
            # A model registered out of tree carries its class; an in-tree one names the module to import it from.
            cls = (
                entry.model_cls
                if hasattr(entry, "model_cls")
                else getattr(importlib.import_module(entry.module_name), entry.class_name)
            )
            source = inspect.getsourcefile(cls)
        except (ImportError, AttributeError, TypeError) as exc:
            log.warning(
                "vllm registers %s as %s, which did not resolve: %s: %s",
                arch,
                getattr(entry, "module_name", entry),
                type(exc).__name__,
                exc,
            )
            continue
        if source and Path(source).is_file():
            return source
    return ""


def _architecture_names(config: dict[str, Any]) -> list[str]:
    """Architectures worth searching for, text tower first."""
    names: list[str] = []
    for source in (config.get("text_config") or {}, config):
        for arch in source.get("architectures") or []:
            arch = str(arch).strip()
            if arch and arch not in names:
                names.append(arch)
    return names


def _legacy_source_file(model_type: str, framework: str, framework_root: str) -> str:
    """The historical ``<models dir>/<model_type>.py`` guess."""
    if framework == "sglang":
        rels = ("python/sglang/srt/models", "sglang/srt/models", "srt/models")
        return _first_source_file(model_type, framework_root, rels, pkg="sglang", pkg_models=("srt", "models"))
    rels = ("vllm/model_executor/models", "model_executor/models")
    return _first_source_file(model_type, framework_root, rels, pkg="vllm", pkg_models=("model_executor", "models"))


# Implementations live in the in-tree models package and, for newer families, an out-of-tree plugin package that sits
# beside it.
_MODEL_DIR_RELS = {
    "sglang": (("python", "sglang", "srt", "models"), ("sglang", "srt", "models"), ("srt", "models")),
    "vllm": (("vllm", "model_executor", "models"), ("model_executor", "models"), ("vllm", "models"), ("models",)),
}
_PKG_DIR_RELS = {
    "sglang": (("srt", "models"),),
    "vllm": (("model_executor", "models"), ("models",)),
}

# Configuration and dispatch helpers, never the model itself.
_NON_IMPLEMENTATION_FILES = frozenset({"config.py", "registry.py", "interfaces.py"})

# A file without any of these defines a wrapper, not a decoder.
_DECODER_MARKER = re.compile(r"^class\s+\w*(?:DecoderLayer|Attention|MLP)\b", re.MULTILINE)


def _model_search_dirs(framework: str, framework_root: str) -> list[Path]:
    """Directories that may hold model implementations, nearest first."""
    pkg = "sglang" if framework == "sglang" else "vllm"
    dirs: list[Path] = []
    if framework_root:
        for rel in _MODEL_DIR_RELS[pkg]:
            cand = Path(framework_root).joinpath(*rel)
            if cand.is_dir():
                dirs.append(cand)
    pkg_dir = _package_dir(pkg)
    if pkg_dir:
        for rel in _PKG_DIR_RELS[pkg]:
            cand = Path(pkg_dir).joinpath(*rel)
            if cand.is_dir():
                dirs.append(cand)
    seen: set[str] = set()
    unique: list[Path] = []
    for directory in dirs:
        key = str(directory.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    return unique


def _vendor_priority() -> tuple[str, ...]:
    """Accelerator sub-package preference for per-vendor model forks."""
    override = os.environ.get("FORGE_FUSION_VENDOR", "").strip().lower()
    if override:
        return (override,)
    try:
        import torch

        if getattr(torch.version, "hip", None):
            return ("amd",)
        if getattr(torch.version, "cuda", None):
            return ("nvidia",)
    except Exception:
        pass
    if Path("/opt/rocm").exists():
        return ("amd",)
    return ()


def _files_defining(arch: str, search_dirs: list[Path]) -> list[tuple[int, str]]:
    """Every file that defines ``class <arch>``, tagged with its search-dir rank."""
    if not arch:
        return []
    pattern = re.compile(r"^class\s+" + re.escape(arch) + r"\s*[(:]", re.MULTILINE)
    found: list[tuple[int, str]] = []
    for rank, directory in enumerate(search_dirs):
        for path in sorted(directory.rglob("*.py")):
            if path.name in _NON_IMPLEMENTATION_FILES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if pattern.search(text):
                found.append((rank, str(path)))
    return found


def _within_root(path_str: str, framework_root: str) -> bool:
    """Whether ``path_str`` sits under an explicitly pinned framework root."""
    if not framework_root:
        return True
    try:
        return Path(path_str).resolve().is_relative_to(Path(framework_root).resolve())
    except OSError:
        return False


def _dir_rank(path_str: str, search_dirs: list[Path]) -> int:
    """Rank of the search dir a resolved path came from; last when unknown."""
    resolved = Path(path_str).resolve()
    for rank, directory in enumerate(search_dirs):
        try:
            resolved.relative_to(directory.resolve())
        except ValueError:
            continue
        return rank
    return len(search_dirs)


def _best_implementation(candidates: list[tuple[int, str]]) -> str:
    """Pick the candidate most likely to hold fusible decode code."""
    if not candidates:
        return ""
    nearest = min(rank for rank, _ in candidates)
    vendors = set(_vendor_priority())
    best_key = None
    best_path = ""
    for path_str in dict.fromkeys(path for rank, path in candidates if rank == nearest):
        path = Path(path_str)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        has_decoder = 1 if _DECODER_MARKER.search(text) else 0
        vendor_match = 1 if (vendors and {part.lower() for part in path.parts} & vendors) else 0
        key = (has_decoder, vendor_match, len(text))
        if best_key is None or key > best_key:
            best_key, best_path = key, path_str
    return best_path


def _first_source_file(
    model_type: str,
    framework_root: str,
    root_rels: tuple[str, ...],
    *,
    pkg: str,
    pkg_models: tuple[str, ...],
) -> str:
    """Return the first existing ``<...>/<model_type>.py``, or \"\"."""
    fname = f"{model_type}.py"
    if framework_root:
        for rel in root_rels:
            cand = Path(framework_root).joinpath(*rel.split("/")) / fname
            if cand.is_file():
                return str(cand)
    pkg_dir = _package_dir(pkg)
    if pkg_dir:
        cand = Path(pkg_dir).joinpath(*pkg_models) / fname
        if cand.is_file():
            return str(cand)
    return ""


def _package_dir(pkg: str) -> str:
    """Directory of an installed package (``.../sglang`` or ``.../vllm``), or \"\"."""
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError, ModuleNotFoundError):
        return ""
    if spec is None or not spec.origin:
        return ""
    return str(Path(spec.origin).resolve().parent)


def _read_source(source_file: str) -> str:
    """Read a resolved model source file; "" when missing/unreadable."""
    if not source_file:
        return ""
    try:
        return Path(source_file).read_text(encoding="utf-8")
    except OSError:
        return ""


# A fusion is delivered by REPLACING one call site in the framework source the author was shown.
_SCOPE_MARKERS: tuple[tuple[str, str], ...] = (
    # The failure this table was written for: a decode fusion that folds in the KV-cache write.
    ("kvcache", r"kv_cache|key_cache|value_cache|slot_mapping|reshape_and_cache|kvcache"),
    ("rope", r"rotary|\brope\b"),
    ("rmsnorm", r"rms_?norm"),
    ("layernorm", r"layer_?norm"),
    ("activation", r"silu|gelu|\brelu\b|sigmoid|act_fn|activation"),
    ("conv", r"\bconv"),
    ("sample", r"sample|argmax|multinomial"),
    ("mla", r"\bmla\b|kv_lora|q_lora"),
    ("moe", r"\bmoe\b|expert"),
)


def out_of_scope_terms(source_text: str, terms: Sequence[str]) -> list[str]:
    """Declared terms the shown source file never performs."""
    if not source_text:
        return []
    lowered = source_text.lower()
    markers = dict(_SCOPE_MARKERS)
    return [
        term
        for term in dict.fromkeys(str(t).strip().lower() for t in terms if str(t).strip())
        if term in markers and not re.search(markers[term], lowered)
    ]


def _source_confirms(pattern: FusionPattern, source_text: str) -> bool:
    """Whether any of the pattern's source hints appear in the model source."""
    return any(h and h in source_text for h in pattern.source_hints)


def _already_fused(pattern: FusionPattern, source_text: str) -> bool:
    """Whether the model source already implements this fusion (no-op recipe)."""
    return any(re.search(m, source_text) for m in pattern.fused_markers)


# vLLM compile-time fusion passes (see torch.compile fusion config).
_VLLM_COMPILE_PASSES: tuple[tuple[str, str, frozenset[str], tuple[tuple[str, ...], ...]], ...] = (
    (
        "fuse_rope_kvcache_cat_mla",
        "fuse_rope_kvcache_cat_mla",
        frozenset(),
        (("mla",), ("rope", "rotary"), ("cat", "concat", "kvcache", "kv_cache", "kv cache")),
    ),
    (
        "fuse_mla_dual_rms_norm",
        "fuse_mla_dual_rms_norm",
        frozenset(),
        (("mla",), ("dual",), ("rms", "rmsnorm", "norm")),
    ),
    (
        "fuse_rope_kvcache",
        "fuse_rope_kvcache",
        frozenset(),
        (("rope", "rotary"), ("kvcache", "kv_cache", "kv cache", "kv-cache")),
    ),
    (
        "qk_norm_rope",
        "enable_qk_norm_rope_fusion",
        frozenset({"rmsnorm", "rope"}),
        (("q_norm", "k_norm", "qk_norm", "qk norm", "qk"), ("rope", "rotary")),
    ),
    ("fuse_attn_quant", "fuse_attn_quant", frozenset(), (("attn", "attention"), ("quant", "fp8", "scaled_mm"))),
    (
        "fuse_act_quant",
        "fuse_act_quant",
        frozenset(),
        (("silu", "gelu", "swiglu", "activation", "act"), ("quant", "fp8")),
    ),
    ("fuse_norm_quant", "fuse_norm_quant", frozenset(), (("rmsnorm", "rms", "layernorm", "norm"), ("quant", "fp8"))),
)

# Compile passes belong to vLLM's torch.compile pipeline; sglang does not run them, so the gate only applies to vllm
# targets.
_VLLM_FRAMEWORKS = frozenset({"vllm", "vllm-aiter"})


def covered_by_vllm_compile_pass(*, matched_categories: list[str], text: str, framework: str) -> str:
    """Name of the vLLM compile pass that implements this fusion, or ``\"\"``."""
    if (framework or "").strip().lower() not in _VLLM_FRAMEWORKS:
        return ""
    cats = {str(c).strip().lower() for c in (matched_categories or [])}
    blob = (text or "").lower()
    for pass_name, _flag, req_cats, kw_groups in _VLLM_COMPILE_PASSES:
        cat_ok = (not req_cats) or (not cats) or req_cats.issubset(cats)
        if not cat_ok:
            continue
        if all(any(k in blob for k in group) for group in kw_groups):
            return pass_name
    return ""


def vllm_pass_config_flag(pass_name: str) -> str:
    """``PassConfig`` field that switches this compile pass on, or ``""``."""
    for name, flag, _cats, _kw in _VLLM_COMPILE_PASSES:
        if name == pass_name:
            return flag
    return ""


def _all_pass_config_flags() -> tuple[str, ...]:
    """Every ``PassConfig`` flag in the table, de-duplicated and order-stable."""
    return tuple(dict.fromkeys(flag for _name, flag, _cats, _kw in _VLLM_COMPILE_PASSES if flag))


def vllm_compile_pass_state(
    pass_name: str,
    *,
    probe: Optional[PassProbe] = None,
    runtime: Optional[TargetRuntime] = None,
) -> Optional[PassState]:
    """Full state of the vLLM compile pass behind ``pass_name`` (``None`` if unmapped)."""
    flag = vllm_pass_config_flag(pass_name)
    if not flag:
        return None
    if probe is not None:
        return probe(flag)
    rt = runtime or TargetRuntime()
    if rt.error:
        # Target install not pinned: refuse to judge rather than probe whichever vLLM happens to be importable here
        # and then edit it.
        return PassState(flag=flag, error=rt.error)
    # Read the WHOLE table in one probe: the cost is importing vLLM, so asking per flag would re-pay it for every
    # matched pattern.
    return probe_pass_states(
        _all_pass_config_flags(),
        python=rt.python,
        require_root=rt.require_root,
    ).get(flag)


def _unclaimable_note(state: PassState) -> str:
    """Why a matched compile pass was not claimed, for the manifest."""
    if not state.present:
        return (
            f"vLLM compile pass `{state.flag}` does not exist in this install "
            f"(nothing to enable): authoring still applies"
        )
    if state.error:
        return (
            f"state of vLLM compile pass `{state.flag}` is UNDECIDABLE "
            f"({state.error[:160]}): not claimed, authoring still applies"
        )
    if state.enabled is None:
        return (
            f"vLLM resolves `{state.flag}` from the full engine config "
            f"(source={state.source}), so it cannot be decided here: "
            f"not claimed, authoring still applies"
        )
    # Disabled, but a level pins it: flipping the class default would not take.
    return (
        f"vLLM compile pass `{state.flag}` is off but pinned by the default "
        f"optimization level (source={state.source}), so flipping the "
        f"PassConfig default would have no effect: not claimed"
    )


def rank_recipes(recipes: list[Recipe]) -> list[Recipe]:
    """Order candidates so the cheapest, most certain win is attempted first."""
    return sorted(recipes, key=lambda r: 0 if r.candidate_kind == "compile_pass" else 1)


def _compile_pass_recipe(
    pat: FusionPattern,
    state: PassState,
    *,
    shapes: dict[str, Any],
    matched_categories: list[str],
    trigger_share: float,
    predicted_gain: float,
    mem_share: Optional[float],
    source_confirmed: Optional[bool],
) -> Recipe:
    """Recipe that claims the framework's own disabled fusion pass."""
    return Recipe(
        pattern_id=f"compile_pass:{state.flag}",
        description=(
            f"vLLM implements this fusion as compile pass `{state.flag}`, but it is "
            f"DISABLED in this install: enable the native pass instead of authoring a "
            f"kernel ({pat.description})"
        ),
        env_flag="",
        source_file=state.config_file,
        source_hints=[state.flag],
        fusion_math=pat.fusion_math,
        eager_reference_hint="",
        shapes=shapes,
        matched_categories=matched_categories,
        trigger_share=trigger_share,
        rocm_native=pat.rocm_native,
        source_confirmed=source_confirmed,
        already_satisfied=False,
        predicted_gain=predicted_gain,
        mem_share=float(mem_share or 0.0),
        candidate_kind="compile_pass",
        compile_pass_flag=state.flag,
    )


def build_recipes(
    diagnosis: Diagnosis,
    *,
    model_path: str,
    framework: str,
    framework_root: str = "",
    decode_batch: int = 16,
    min_predicted_gain: float = DEFAULT_MIN_PREDICTED_GAIN,
    include_unconfirmed: bool = False,
    pass_probe: Optional[PassProbe] = None,
) -> list[Recipe]:
    """Instantiate localized recipes from a diagnosis (deterministic skeleton)."""
    matched = match_patterns(diagnosis, framework)
    if not matched:
        return []
    # Pin ONE install for every compile-pass question in this run (probe now, edit and serve later), and make an
    # explicit --framework-root a precondition.
    runtime = resolve_target_runtime(framework, framework_root=framework_root)
    shapes = resolve_decode_shapes(model_path, decode_batch=decode_batch)
    model_type = str(shapes.get("model_type") or "")
    source_file, source_resolution_note = resolve_framework_source_file(
        model_path,
        framework,
        framework_root=framework_root,
        model_type=model_type,
    )
    source_text = _read_source(source_file)
    have_source = bool(source_text)
    # Model-prefix the env flag so it is unambiguous per model and matches the framework's convention (e.g. lfm2 ->
    # LFM2_FUSED_RESIDUAL, zaya -> ZAYA_FUSED_QK).
    prefix = f"{model_type.upper()}_" if model_type else ""

    bytes_share = diagnosis.category_bytes_share or {}
    recipes: list[Recipe] = []
    for pat, trigger_share in matched:
        confirmed = _source_confirms(pat, source_text) if have_source else None
        already = _already_fused(pat, source_text) if have_source else False
        # MEASURED memory-traffic share of this pattern's op chain (the slice of HBM traffic fusion would collapse).
        mem_share = sum(bytes_share.get(c, 0.0) for c in pat.trigger_categories) if bytes_share else None
        # Per-pattern predicted cg-ON gain, grounded in the measured memory channel when available (else the
        # launch-share discount).
        predicted_gain = predict_cuda_graph_on_gain(trigger_share, decode_batch=decode_batch, mem_share=mem_share)
        # Compile-pass gate: a fusion vLLM performs at compile time is a no-op to author -- but only while that pass
        # is switched ON, which is read from the install, not assumed.
        compile_pass = covered_by_vllm_compile_pass(
            matched_categories=list(pat.trigger_categories),
            # Fusion-defining fields only (id / math / env flag); exclude the prose description + grep hints so an
            # incidental mention of a native-fused op does not wrongly mark the pattern already-fused.
            text=" ".join((pat.id, pat.fusion_math, pat.env_flag)),
            framework=framework,
        )
        state = vllm_compile_pass_state(compile_pass, probe=pass_probe, runtime=runtime) if compile_pass else None
        # ONLY an enabled pass makes the candidate a no-op.
        covered = state is not None and state.enabled is True
        pass_note = "" if state is None or state.claimable or covered else _unclaimable_note(state)
        if pass_note:
            log.info("compile pass not claimed for %s: %s", pat.id, pass_note)
        already = already or covered
        if have_source and not include_unconfirmed and confirmed is False:
            # Drop wrong-model patterns once we can read the source.
            continue
        if already and not include_unconfirmed:
            # Drop no-op patterns (source already fuses OR an ENABLED vLLM pass covers).
            continue
        if state is not None and state.claimable:
            recipes.append(
                _compile_pass_recipe(
                    pat,
                    state,
                    shapes=shapes,
                    trigger_share=trigger_share,
                    predicted_gain=predicted_gain,
                    mem_share=mem_share,
                    source_confirmed=confirmed,
                    matched_categories=sorted(
                        c for c in pat.trigger_categories if diagnosis.category_shares.get(c, 0.0) > 0
                    ),
                )
            )
            continue
        recipes.append(
            Recipe(
                pattern_id=pat.id,
                description=pat.description,
                env_flag=f"{prefix}{pat.env_flag}",
                source_file=source_file,
                source_hints=list(pat.source_hints),
                fusion_math=pat.fusion_math,
                eager_reference_hint=pat.eager_reference_hint,
                shapes=shapes,
                matched_categories=sorted(
                    c for c in pat.trigger_categories if diagnosis.category_shares.get(c, 0.0) > 0
                ),
                trigger_share=trigger_share,
                rocm_native=pat.rocm_native,
                source_confirmed=confirmed,
                already_satisfied=already,
                predicted_gain=predicted_gain,
                mem_share=float(mem_share or 0.0),
                compile_pass_note=pass_note,
                source_resolution_note=source_resolution_note,
            )
        )
    return rank_recipes(recipes)
