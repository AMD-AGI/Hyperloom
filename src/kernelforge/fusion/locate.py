# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 2 (deterministic half): assemble concrete recipes from matched patterns.

Given a diagnosis + framework + model, this:

1. resolves the model's source file in the framework tree (sglang/vllm),
2. derives representative decode shapes from the model config,
3. instantiates each triggered :class:`FusionPattern` into a concrete
   :class:`Recipe`.

The LLM-driven half (confirming the exact call sites + adapting the recipe to the
real source) happens in the author stage (Phase 3); here we produce the localized
skeleton the author works from. No per-model literals live here.
"""

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
    """Best-effort path to the framework's model implementation file.

    vLLM's registry names the class it will actually construct, so it answers
    outright whenever vLLM can be imported. The other two mechanisms are pooled
    rather than tried in turn, because neither dominates: searching for the
    architecture the config names is the only one that covers sglang, but the
    class it finds can sit in a wrapper with nothing fusible, while the
    historical ``<models dir>/<model_type>.py`` guess lands on the decoder
    without ever naming it. gemma-4 is both at once -- the architecture resolves
    to ``gemma4_mm.py`` and the convention to ``gemma4.py`` -- so the candidates
    are ranked together and the decoder marker decides.

    Args:
        model_path: Model directory (used to read ``model_type`` if not given).
        framework: ``sglang`` / ``vllm`` / ``vllm-aiter``.
        framework_root: Explicit framework source root; when empty, the installed
            package location is auto-detected.
        model_type: Model type; when empty, read from the model config.

    Returns:
        ``(path, how)``: the resolved source-file path (``""`` when it cannot be
        located) and which mechanism produced it, recorded on the recipe so the
        manifest shows whether the registry answered or was fallen back from.
    """
    fw = (framework or "").strip().lower()
    if fw not in ("sglang", "vllm", "vllm-aiter"):
        return "", f"unsupported framework {fw!r}"

    config = load_model_config(model_path)
    mt = (model_type or str(config.get("model_type") or "")).strip()
    search_dirs = _model_search_dirs(fw, framework_root)

    if fw in _VLLM_FRAMEWORKS:
        registered = _vllm_registered_source(model_path)
        # The registry answers for whichever vLLM is importable in THIS process,
        # which need not be the tree the operator pinned. Locality wins for the
        # same reason it does in _best_implementation: the author stage patches
        # the pinned tree, so a file outside it is not the one being optimized.
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
            # Named apart so the manifest shows the registry was asked and missed,
            # which is the case worth chasing: vLLM knows and we did not hear it.
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
    """Source file of the class vLLM actually registers for this model, or "".

    The ``model_executor/models/<model_type>.py`` convention misses newer models,
    which live in a ``vllm/models/<name>/`` package whose ``__init__`` re-exports a
    per-platform implementation. Neither that package nor its ``__init__`` holds
    the forward pass, so ask the registry which class is used and follow it to the
    file that defines it -- on ROCm that is the ``amd/`` variant, which is the one
    worth fusing.

    ``ModelRegistry.models`` is read rather than ``_VLLM_MODELS``: vLLM has
    already applied its own prefix rule to build it, so ``module_name`` is a
    full path and ``class_name`` is the implementation, which for 93 of the 364
    architectures shipped here is not the architecture name.
    """
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
            # A model registered out of tree carries its class; an in-tree one
            # names the module to import it from.
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
    """Architectures worth searching for, text tower first.

    A multimodal checkpoint names its wrapper at the top level; decode-time
    fusion lives in the text decoder, so the nested ``text_config`` entry is the
    more useful lead when both are present.
    """
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


# Implementations live in the in-tree models package and, for newer families, an
# out-of-tree plugin package that sits beside it.
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

# A file without any of these defines a wrapper, not a decoder. ``gemma4_mm.py``
# holds the multimodal processor and nothing fusible; the decoder it delegates to
# lives in ``gemma4.py``.
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
    """Accelerator sub-package preference for per-vendor model forks.

    ``deepseek_v4`` and ``minimax_m3`` ship ``amd/``, ``nvidia/`` and ``xpu/``
    copies of the same class. Handing the author stage the fork the engine never
    executes yields fusions that cannot land, so the running platform decides.
    """
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
    """Every file that defines ``class <arch>``, tagged with its search-dir rank.

    The trailing ``[(:]`` stops ``DeepseekV4ForCausalLMConfig`` from matching
    ``DeepseekV4ForCausalLM``.
    """
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
    """Whether ``path_str`` sits under an explicitly pinned framework root.

    An unset root pins nothing, so every path qualifies.
    """
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
    """Pick the candidate most likely to hold fusible decode code.

    Locality comes first: an explicit ``--framework-root`` names the tree the
    author stage will patch, so a file from an unrelated installed copy must
    never outrank it however good it looks.
    """
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
    """Return the first existing ``<...>/<model_type>.py``, or "".

    Tries, in order: each ``root_rels`` under an explicit ``framework_root``, then
    the installed package's own models dir (``<pkg_dir>/<pkg_models...>``). Using
    the package dir directly is layout-agnostic: it works for both an editable
    checkout (``<root>/python/sglang/...``) and a site-packages install
    (``<site-packages>/sglang/...``), because the package dir is ``.../sglang``
    (or ``.../vllm``) in both.
    """
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
    """Directory of an installed package (``.../sglang`` or ``.../vllm``), or "".

    Layout-agnostic: returns the package's own directory regardless of whether it
    is an editable checkout or a site-packages install.
    """
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


# A fusion is delivered by REPLACING one call site in the framework source the
# author was shown. Every input the fused kernel needs therefore has to be in
# scope at that call site, and every op it computes has to be one the shown file
# actually performs -- otherwise the kernel cannot be wired at all, or wiring it
# would double-execute work the framework still does elsewhere.
#
# The map is deliberately partial. A term earns an entry only when it has an
# unambiguous source-level spelling; ``add``, ``mul``, ``copy`` and ``reduce``
# are absent because the shapes they take in real source are too varied to
# distinguish "absent" from "spelled differently", and a scope gate that fires
# on a spelling is worse than none. Unlisted terms are not judged.
_SCOPE_MARKERS: tuple[tuple[str, str], ...] = (
    # The failure this table was written for: a decode fusion that folds in the
    # KV-cache write. In vLLM v1 that write happens inside the attention
    # backend, several frames below the model file -- ``key_cache`` /
    # ``slot_mapping`` are simply not names the model's forward can reach.
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
    """Declared terms the shown source file never performs.

    A non-empty result means the proposal crosses a module boundary: it claims
    to compute something that is not in the file whose call site the author will
    replace. Such a fusion is unwireable by construction -- the author writes
    the kernel, cannot find a call site that has the inputs, and delivers the
    module with no wiring edit. That is exactly the shape
    :func:`kernelforge.fusion.validate.fused_symbol_invocation_evidence` catches
    at the far end of the pipeline, after a full authoring campaign has been
    spent on it; this catches it before the campaign starts.

    Fails OPEN in every uncertain case: an unreadable source is not judged, and
    neither is a term with no entry in :data:`_SCOPE_MARKERS`.
    """
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


# vLLM compile-time fusion passes (see torch.compile fusion config). Matching one
# of these means vLLM CAN fuse the chain natively -- it does NOT mean it does:
# most of these flags default to off, so the pass has to be probed
# (:mod:`vllm_passes`) before a candidate can be called already-satisfied.
# The source-marker check (``_already_fused``) only greps the eager model source
# and misses these compile passes -- this closes that gap.
#
# Each entry is (pass_name, config_flag, required_categories, keyword_groups): a
# candidate is covered when its matched categories include ``required_categories``
# (or its categories are unknown, e.g. LLM-discovered recipes) AND every keyword
# group has at least one keyword present in the candidate's op-chain / description
# text. ``config_flag`` is the ``PassConfig`` field that switches the pass on; it
# is the ONLY prior knowledge kept here (a stable name mapping). Whether the pass
# is on is version- and platform-dependent and is always read from the target
# install, never hardcoded.
# ``quant``-bearing passes REQUIRE a quant keyword so plain norm/act/rope fusions
# (which vLLM does NOT fuse without quant) are never dropped. Ordered specific
# (mla / cat) first so the reported pass is the most precise.
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

# Compile passes belong to vLLM's torch.compile pipeline; sglang does not run
# them, so the gate only applies to vllm targets.
_VLLM_FRAMEWORKS = frozenset({"vllm", "vllm-aiter"})


def covered_by_vllm_compile_pass(*, matched_categories: list[str], text: str, framework: str) -> str:
    """Name of the vLLM compile pass that implements this fusion, or ``""``.

    Reused by BOTH the pattern route (``build_recipes``) and the discovery route
    (``discover.parse_discovered_recipes``). A match only says vLLM CAN fuse the
    chain at compile time; :func:`missed_vllm_compile_pass` decides whether it
    actually does.
    """
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
    """Full state of the vLLM compile pass behind ``pass_name`` (``None`` if unmapped).

    Callers need all four outcomes, not a boolean: only ``enabled`` means the
    candidate is genuinely already satisfied. Collapsing ``absent`` (this vLLM has
    no such flag, so there is no framework implementation to reuse) or
    ``undecidable`` into "satisfied" would delete a candidate that should still be
    authored.
    """
    flag = vllm_pass_config_flag(pass_name)
    if not flag:
        return None
    if probe is not None:
        return probe(flag)
    rt = runtime or TargetRuntime()
    if rt.error:
        # Target install not pinned: refuse to judge rather than probe whichever
        # vLLM happens to be importable here and then edit it.
        return PassState(flag=flag, error=rt.error)
    # Read the WHOLE table in one probe: the cost is importing vLLM, so asking
    # per flag would re-pay it for every matched pattern.
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
    """Order candidates so the cheapest, most certain win is attempted first.

    Only the top recipe is acted on, so ordering by trigger share alone spends an
    LLM authoring loop (plus compile / parity / CUDA-graph risk) on a large slice
    while leaving a free one unclaimed. A ``compile_pass`` is a one-line
    deterministic flip that hands the work to the framework's own vendor-tuned
    kernel, so it goes first; ties and every other kind keep their existing
    share-descending order (stable sort).
    """
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
    """Recipe that claims the framework's own disabled fusion pass.

    Nothing is authored: the edit target is vLLM's pass config (hence
    ``source_file``), there is no env gate because the flip itself enables the
    fusion, and the resulting kernel is the framework's, not ours.
    """
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
    """Instantiate localized recipes from a diagnosis (deterministic skeleton).

    When the model source file is resolvable, each candidate pattern is confirmed
    against it: a pattern whose source hints do NOT appear is dropped (wrong-model
    red herring), and a pattern whose fusion is ALREADY implemented (a fused_*
    marker present) is dropped as already-satisfied (no-op recipe). When the source
    cannot be resolved, patterns are kept with ``source_confirmed=None``.

    Each recipe also gets a PER-PATTERN predicted cuda-graph-ON gain derived from
    its own ``trigger_share`` (the slice that pattern actually addresses), and is
    dropped when that is below ``min_predicted_gain``. This is tighter than the
    aggregate diagnose gate: a model can clear the diagnosis on total launch-bound
    share while any single pattern only addresses a sub-threshold slice.

    A pattern vLLM implements as a compile pass is dropped ONLY when that pass is
    actually enabled in the target install. When it exists but is switched off, the
    fusion is being missed, and the recipe becomes a ``compile_pass`` candidate:
    enable the framework's own pass instead of authoring a duplicate kernel.

    Args:
        include_unconfirmed: keep source-unconfirmed / already-fused / low-gain
            recipes (annotated) instead of dropping them; useful for diagnostics.
        pass_probe: reads a vLLM compile pass's resolved state (injectable for
            tests); defaults to probing the installed vLLM.

    Returns an empty list when the diagnosis is not a fusion candidate, nothing
    triggers, or every candidate is filtered out.
    """
    matched = match_patterns(diagnosis, framework)
    if not matched:
        return []
    # Pin ONE install for every compile-pass question in this run (probe now, edit
    # and serve later), and make an explicit --framework-root a precondition.
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
    # Model-prefix the env flag so it is unambiguous per model and matches the
    # framework's convention (e.g. lfm2 -> LFM2_FUSED_RESIDUAL, zaya -> ZAYA_FUSED_QK).
    prefix = f"{model_type.upper()}_" if model_type else ""

    bytes_share = diagnosis.category_bytes_share or {}
    recipes: list[Recipe] = []
    for pat, trigger_share in matched:
        confirmed = _source_confirms(pat, source_text) if have_source else None
        already = _already_fused(pat, source_text) if have_source else False
        # MEASURED memory-traffic share of this pattern's op chain (the slice of
        # HBM traffic fusion would collapse). None when the trace carried no shapes
        # -> predict falls back to the launch-share discount.
        mem_share = sum(bytes_share.get(c, 0.0) for c in pat.trigger_categories) if bytes_share else None
        # Per-pattern predicted cg-ON gain, grounded in the measured memory channel
        # when available (else the launch-share discount). Annotated for ranking /
        # author context; not a hard drop (the diagnose gate vetoes non-candidates).
        predicted_gain = predict_cuda_graph_on_gain(trigger_share, decode_batch=decode_batch, mem_share=mem_share)
        # Compile-pass gate: a fusion vLLM performs at compile time is a no-op to
        # author -- but only while that pass is switched ON, which is read from the
        # install, not assumed.
        compile_pass = covered_by_vllm_compile_pass(
            matched_categories=list(pat.trigger_categories),
            # Fusion-defining fields only (id / math / env flag); exclude the
            # prose description + grep hints so an incidental mention of a
            # native-fused op does not wrongly mark the pattern already-fused.
            text=" ".join((pat.id, pat.fusion_math, pat.env_flag)),
            framework=framework,
        )
        state = vllm_compile_pass_state(compile_pass, probe=pass_probe, runtime=runtime) if compile_pass else None
        # ONLY an enabled pass makes the candidate a no-op. Absent / undecidable /
        # level-pinned-off all mean the framework is not fusing this for us, so the
        # candidate survives as normal authoring work (annotated with why).
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
