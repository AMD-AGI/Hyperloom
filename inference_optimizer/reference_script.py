# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Reference launch-recipe parsing, rendering, and discovery.

A *reference recipe* is an InferenceX ``benchmarks/single_node/*.sh`` launch
script. The optimizer lifts only its **static, fully-resolved** server flags and
a whitelist of ``export`` lines and uses them as the lowest-priority base for
the baseline server args (EXPLORE can still override). The shell is never
executed — anything dynamic (``$VARS``: TP/CONC/ISL/OSL/model/port) is skipped,
because the optimizer's normal env seeding already owns those.

Three public entry points:

* :func:`parse_reference_script` — lift ``(server_args, envs, model)`` from a
  recipe (local path or http(s) URL). Fail-soft: an unreachable / missing
  source returns an empty recipe, never raises.
* :func:`render_reference_script` — the inverse: emit a recipe text from the
  current best ``server_args`` / ``envs`` (used to write the read-only
  ``current_setting.sh`` artifact). Round-trips with the parser.
* :func:`discover_reference_script` — when no recipe was supplied, find a
  matching one in the InferenceX checkout by filename. Tiered + fail-soft.
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# Exports we are willing to carry from a recipe. Deliberately an explicit list,
# NOT a ``VLLM_*`` glob: a glob would drag in path-valued vars like
# VLLM_CACHE_ROOT that won't exist in our sandbox. Add names here as recipes
# prove they need them.
_ENV_WHITELIST = frozenset({
    "VLLM_USE_BREAKABLE_CUDAGRAPH",
    "VLLM_USE_TRITON_FLASH_ATTN",
    "VLLM_FP8_PADDING",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MHA",
    "VLLM_ROCM_USE_AITER_MOE",
    "SGLANG_USE_AITER",
    "SGLANG_MOE_PADDING",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
})

# Flags that never belong in the lifted base: the optimizer's env seeding owns
# the workload + I/O, so drop these even when fully resolved.
_DROP_FLAGS = frozenset({
    "--port",
    "--host",
    "--served-model-name",
    "--result-dir",
    "--result-filename",
})
# Flags dropped by prefix (result-*, served-model-* variants, log redirection).
_DROP_PREFIXES = ("--result-", "--served-model")


@dataclass(frozen=True)
class ReferenceRecipe:
    """Static facts lifted from a reference launch recipe."""

    server_args: str = ""
    envs: dict[str, str] = field(default_factory=dict)
    model: str | None = None


def _read_source(source: str) -> str | None:
    """Return the recipe text, or None on any failure (fail-soft)."""
    s = str(source or "").strip()
    if not s:
        return None
    if s.startswith(("http://", "https://")):
        try:
            from .baseline_comparison.inferencex_client import _fetch_raw
            return _fetch_raw(s).decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 — fail-soft, never abort launch
            log.warning("reference-script: could not fetch %r: %s", s, exc)
            return None
    try:
        return Path(s).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 — fail-soft
        log.warning("reference-script: could not read %r: %s", s, exc)
        return None


def _entrypoint_markers(framework: str) -> tuple[str, ...]:
    fw = str(framework or "").strip().lower()
    if "atom" in fw:
        return ("atom.entrypoints",)
    if "vllm" in fw:
        return ("vllm serve",)
    return ("sglang.launch_server",)


def _join_continuations(text: str) -> list[str]:
    """Collapse backslash line-continuations into single logical lines."""
    logical: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
        else:
            buf += line
            logical.append(buf)
            buf = ""
    if buf:
        logical.append(buf)
    return logical


def _find_entrypoint_line(text: str, framework: str) -> str | None:
    markers = _entrypoint_markers(framework)
    for line in _join_continuations(text):
        if any(m in line for m in markers):
            return line
    return None


def _strip_redirection(tokens: list[str]) -> list[str]:
    """Drop shell redirection / backgrounding tail (``> log 2>&1 &``)."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("&", ";"):
            break
        if t.startswith(">") or t.startswith("<") or t.startswith("2>") or "2>&1" in t:
            # redirection target may be the next token
            if t in (">", "<", "2>") and i + 1 < len(tokens):
                i += 2
                continue
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def _has_var(s: str) -> bool:
    return "$" in s


def _is_flag(tok: str) -> bool:
    return tok.startswith("--")


def _flag_name(tok: str) -> str:
    return tok.split("=", 1)[0]


def _should_drop_flag(name: str) -> bool:
    if name in _DROP_FLAGS:
        return True
    return any(name.startswith(p) for p in _DROP_PREFIXES)


def parse_reference_script(source: str, *, framework: str) -> ReferenceRecipe:
    """Lift ``(server_args, envs, model)`` from a reference recipe.

    Fail-soft: a missing/unreachable source or a recipe with no recognizable
    entrypoint returns an empty :class:`ReferenceRecipe` (never raises), so a
    bad ``--reference-script`` can never block a launch.
    """
    text = _read_source(source)
    if text is None:
        return ReferenceRecipe()

    envs = _extract_envs(text)
    line = _find_entrypoint_line(text, framework)
    if not line:
        log.warning(
            "reference-script: no %s entrypoint found in %r; "
            "carrying exports only",
            _entrypoint_markers(framework), source,
        )
        return ReferenceRecipe(server_args="", envs=envs, model=None)

    try:
        tokens = shlex.split(line)
    except ValueError:
        log.warning("reference-script: could not shell-parse entrypoint line")
        return ReferenceRecipe(server_args="", envs=envs, model=None)

    server_args, model = _extract_server_args(tokens, framework)
    return ReferenceRecipe(server_args=server_args, envs=envs, model=model)


def _extract_envs(text: str) -> dict[str, str]:
    """Pull whitelisted ``export KEY=VALUE`` lines whose value has no ``$``."""
    envs: dict[str, str] = {}
    pat = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)\s*$")
    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if key not in _ENV_WHITELIST:
            continue
        if _has_var(val):
            continue
        # strip surrounding quotes if present
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        envs[key] = val
    return envs


def _extract_server_args(
    tokens: list[str], framework: str,
) -> tuple[str, str | None]:
    """Walk entrypoint tokens as (flag, value) pairs; keep static flags only.

    Returns ``(server_args, model_basename_or_None)``. A flag whose value
    contains ``$`` is dropped together with its value (no orphan flags). The
    positional model and drop-listed flags are removed from ``server_args`` but
    the model is captured for caller-side model-gating.
    """
    tokens = _strip_redirection(tokens)
    # Skip the entrypoint prefix itself.
    fw = str(framework or "").strip().lower()
    start = 0
    if "vllm" in fw and not ("atom" in fw):
        # ``vllm serve <model> ...`` → entrypoint is the first two tokens.
        for i, t in enumerate(tokens):
            if t == "serve":
                start = i + 1
                break
    else:
        # ``python3 -m sglang.launch_server ...`` / atom: entrypoint is the
        # ``-m module`` run; flags follow.
        for i, t in enumerate(tokens):
            if t == "-m" and i + 1 < len(tokens):
                start = i + 2
                break

    model: str | None = None
    kept: list[str] = []
    i = start
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not _is_flag(tok):
            # positional (the model for ``vllm serve $MODEL``); capture, drop.
            if model is None and tok not in ("serve",):
                model = None if _has_var(tok) else Path(tok).name
            i += 1
            continue
        name = _flag_name(tok)
        # capture model from --model / --model-path even though we drop it.
        is_model_flag = name in ("--model", "--model-path")
        if "=" in tok:  # --flag=value (self-contained)
            value = tok.split("=", 1)[1]
            if is_model_flag:
                if not _has_var(value):
                    model = Path(value).name
                i += 1
                continue
            if _has_var(value) or _should_drop_flag(name):
                i += 1
                continue
            kept.append(tok)
            i += 1
            continue
        # ``--flag value`` or bare ``--flag``
        has_value = i + 1 < n and not tokens[i + 1].startswith("-")
        if has_value:
            value = tokens[i + 1]
            if is_model_flag:
                if not _has_var(value):
                    model = Path(value).name
                i += 2
                continue
            if _has_var(value) or _should_drop_flag(name):
                i += 2  # drop BOTH flag and its value (no orphan flag)
                continue
            kept.append(tok)
            kept.append(value)
            i += 2
            continue
        # bare store-true flag
        if _should_drop_flag(name):
            i += 1
            continue
        kept.append(tok)
        i += 1

    return " ".join(kept), model


def render_reference_script(
    *,
    framework: str,
    server_args: str,
    envs: dict[str, str] | None = None,
    model: str | None = None,
) -> str:
    """Render a recipe text from the current best args/envs.

    Output is a valid recipe that :func:`parse_reference_script` round-trips:
    whitelisted ``export`` lines + a single entrypoint line. Read-only artifact
    (``current_setting.sh``) — nothing parses it back to restore state.
    """
    fw = str(framework or "sglang").strip().lower()
    lines = ["#!/usr/bin/env bash", "# Auto-generated by hyperloom — current best launch recipe."]
    if model:
        lines.append(f"# model: {model}")
    for k, v in (envs or {}).items():
        if k in _ENV_WHITELIST:
            lines.append(f"export {k}={v}")
    args = str(server_args or "").strip()
    if "atom" in fw:
        entry = f"python3 -m atom.entrypoints.openai_server {args}".rstrip()
    elif "vllm" in fw:
        entry = f"vllm serve $MODEL {args}".rstrip()
    else:
        entry = f"python3 -m sglang.launch_server --model-path=$MODEL {args}".rstrip()
    lines.append(entry)
    return "\n".join(lines) + "\n"


# ── discovery ──────────────────────────────────────────────────────────────

# Filename pattern: {model}_{precision}_{gpu}[_{suffix}...].sh
_FNAME_RE = re.compile(r"^([^_]+)_([^_]+)_([^_]+?)((?:_[^_]+)*)\.sh$")
_KNOWN_FW_SUFFIXES = ("atom", "sglang", "vllm")


def _normalize_model(s: str) -> str:
    """Lowercase, drop ``.``/``-``, take the leading alnum run."""
    base = Path(str(s or "")).name.lower()
    base = base.replace(".", "").replace("-", "").replace("_", "")
    m = re.match(r"^[a-z0-9]+", base)
    return m.group(0) if m else ""


def _trailing_digits(s: str) -> str:
    m = re.search(r"(\d+)$", s)
    return m.group(1) if m else ""


def _model_tier(file_seg: str, run_model: str) -> str:
    """Tier a filename model segment against the run's model.

    ``exact`` — normalized equal. ``fuzzy`` — one is a substring of the other
    AND their trailing version digits agree (so ``minimaxm2`` vs ``minimaxm3``
    demotes to ``none``). Otherwise ``none``.
    """
    a = _normalize_model(file_seg)
    b = _normalize_model(run_model)
    if not a or not b:
        return "none"
    if a == b:
        return "exact"
    if a in b or b in a:
        da, db = _trailing_digits(a), _trailing_digits(b)
        if da and db and da != db:
            return "none"
        return "fuzzy"
    return "none"


def models_compatible(reference_model: str, run_model: str) -> bool:
    """Whether a reference recipe's model is safe to apply to ``run_model``.

    Single source of truth for the model-gate (shared by discovery and the
    baseline executor) so the two never drift. Empty ``reference_model`` is
    treated as ungated (returns True). Uses the same normalized, version-aware
    tiering as discovery: an ``exact`` or ``fuzzy`` tier is compatible; a
    trailing-version mismatch (``minimaxm2`` vs ``minimaxm3``) is not.
    """
    ref = str(reference_model or "").strip()
    if not ref:
        return True
    return _model_tier(ref, run_model) in ("exact", "fuzzy")


def _parse_filename(name: str) -> tuple[str, str, str, set[str]] | None:
    """Return (model_seg, precision, gpu, suffix_set) or None."""
    m = _FNAME_RE.match(name)
    if not m:
        return None
    model_seg, precision, gpu, rest = m.groups()
    suffixes = {s for s in rest.split("_") if s}
    return model_seg, precision.lower(), gpu.lower(), suffixes


def discover_reference_script(
    inferencex_path: str,
    *,
    model_path: str,
    precision: str,
    gpu_type: str,
    framework: str,
) -> tuple[str | None, str]:
    """Find a matching single-node recipe in the InferenceX checkout.

    Returns ``(path_or_None, tier)`` where tier is ``exact`` / ``fuzzy`` /
    ``none``. Only ``exact`` should be auto-applied; ``fuzzy`` is a candidate
    to surface to the operator. Total fail-soft: any error → ``(None, "none")``.
    """
    try:
        root = Path(str(inferencex_path or "").strip())
        if not root or not root.is_dir():
            return (None, "none")
        want_prec = str(precision or "").strip().lower()
        want_gpu = str(gpu_type or "").strip().lower()
        fw = str(framework or "").strip().lower()
        want_fw = next((f for f in _KNOWN_FW_SUFFIXES if f in fw), "")

        search_dirs = [
            root / "benchmarks" / "single_node",
            root / "benchmarks" / "single_node" / "fixed_seq_len",
        ]
        candidates: list[Path] = []
        for d in search_dirs:
            if d.is_dir():
                candidates.extend(sorted(d.glob("*.sh")))

        best_fuzzy: str | None = None
        for path in candidates:
            parsed = _parse_filename(path.name)
            if not parsed:
                continue
            model_seg, prec, gpu, suffixes = parsed
            if want_prec and prec != want_prec:
                continue
            if want_gpu and gpu != want_gpu:
                continue
            # framework gate: a script tagged with another framework is out;
            # an untagged script is treated as the default (sglang/vllm) match.
            other_fw = (suffixes & set(_KNOWN_FW_SUFFIXES)) - ({want_fw} if want_fw else set())
            if other_fw:
                continue
            tier = _model_tier(model_seg, model_path)
            if tier == "exact":
                return (str(path), "exact")
            if tier == "fuzzy" and best_fuzzy is None:
                best_fuzzy = str(path)
        if best_fuzzy:
            return (best_fuzzy, "fuzzy")
        return (None, "none")
    except Exception as exc:  # noqa: BLE001 — total fail-soft
        log.warning("reference-script discovery failed: %s", exc)
        return (None, "none")


__all__ = [
    "ReferenceRecipe",
    "parse_reference_script",
    "render_reference_script",
    "discover_reference_script",
    "models_compatible",
]
