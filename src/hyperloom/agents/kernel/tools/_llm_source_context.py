###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Runtime context handed to the source-resolution model tiers.

Which file implements a kernel is a question about the *running* configuration,
not just about file names. The same MoE operator dispatches to different
implementations under ``--moe-runner-backend triton`` and ``aiter``; a model
shown only forty lines of each candidate cannot tell those apart. This module
assembles the four things that actually disambiguate a code path:

* the model's own config (architecture, quantization, expert layout)
* the serving flags and environment that select a backend
* the launcher call stack the trace tier already recovered
* the framework roots in play

Three properties keep this from becoming a liability:

* **Nothing secret leaves.** Both the environment and the serving command line
  are admitted by explicit allowlists of path-selecting names, never by scanning
  for interesting keys. The raw command line is never forwarded: it is tokenised
  and only allowlisted flags survive. Every surviving value is then dropped if
  its name or its shape looks like a credential.
* **Nothing large leaves.** Each section is independently truncated, and the
  whole block is capped, so a big config or a deep stack cannot silently grow
  the request.
* **Nothing required.** Every section degrades to omitted. A missing config, an
  unreadable trace or an absent root list narrows the context rather than
  failing the tier.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
from pathlib import Path
from typing import Any

#: Config keys that change which kernel implementation runs. The full config is
#: far larger and mostly irrelevant (tokenizer settings, generation defaults).
_CONFIG_KEYS = (
    "architectures",
    "model_type",
    "torch_dtype",
    "quantization_config",
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "num_experts_per_tok",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_hidden_layers",
)

#: Quantization selectors that can change the implementation family. Arbitrary
#: extension keys are denied because model configs are not a trusted secret
#: boundary and frequently carry vendor-specific metadata.
_QUANTIZATION_CONFIG_KEYS = (
    "activation_dtype",
    "activation_scheme",
    "bits",
    "compute_dtype",
    "fmt",
    "format",
    "group_size",
    "kv_cache_scheme",
    "quant_algo",
    "quant_method",
    "weight_block_size",
    "weight_dtype",
)

#: Environment names that select a code path. Allowlisted by name rather than
#: filtered by pattern: a denylist would leak whatever nobody thought to add.
_ENV_ALLOWLIST = (
    "SGLANG_USE_AITER",
    "SGLANG_MOE_RUNNER_BACKEND",
    "SGLANG_ENABLE_TORCH_COMPILE",
    "SGLANG_ATTENTION_BACKEND",
    "SGLANG_DISABLE_CUDA_GRAPH",
    "VLLM_USE_TRITON_FLASH_ATTN",
    "VLLM_ATTENTION_BACKEND",
    "HIP_VISIBLE_DEVICES",
    "TP",
    "TP_SIZE",
    "WORLD_SIZE",
)

#: Serving flags that select which kernel implementation runs. Allowlisted for
#: the same reason as the environment names: the serving command line also
#: carries credentials (``--api-key``), file-system layout and user data, none
#: of which help decide a source path.
_SERVER_ARG_ALLOWLIST = (
    "--attention-backend",
    "--block-size",
    "--decode-attention-backend",
    "--disable-cuda-graph",
    "--dp",
    "--dp-size",
    "--dtype",
    "--enable-aiter-allreduce-fusion",
    "--enable-dp-attention",
    "--enable-ep-moe",
    "--enable-torch-compile",
    "--ep",
    "--ep-size",
    "--kv-cache-dtype",
    "--moe-runner-backend",
    "--page-size",
    "--prefill-attention-backend",
    "--quantization",
    "--speculative-algorithm",
    "--tp",
    "--tp-size",
)

#: Belt and braces over the allowlist: never emit a value whose name looks like
#: a credential, even if it were added above by mistake.
_SECRET_NAME_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CRED|AUTH", re.IGNORECASE)

#: The name check cannot see a credential passed as the value of an innocuous
#: flag, so common credential prefixes and HTTP authorization forms are denied.
_SECRET_VALUE_RE = re.compile(
    r"^(?:sk-|ak-|pk-|hf_|ghp_|gho_|github_pat_|xox[baprs]-)"
    r"|^(?:authorization\s*:|basic(?:\s|$)|bearer(?:\s|$))",
    re.IGNORECASE,
)

_MAX_ARG_VALUE_CHARS = 120
_MAX_CONFIG_LIST_ITEMS = 16
_MAX_STACK_FRAMES = 8
_MAX_ROOTS = 12
_MAX_CONTEXT_CHARS = 6000

#: Selector values are deliberately narrower than general command-line text.
#: Each comma-separated component is short and uses only backend/dtype syntax.
_SELECTOR_VALUE_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,38}"
    r"(?:,[A-Za-z0-9][A-Za-z0-9_.-]{0,38})*"
)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_URL_VALUE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://|[?@]")
_JWT_VALUE_RE = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+={0,2}")
_OPAQUE_VALUE_RE = re.compile(r"[A-Za-z0-9+/=_-]{40,}")
_NON_FINITE_VALUE_RE = re.compile(r"(?:nan|[+-]?inf(?:inity)?)", re.IGNORECASE)


def _redact(name: str, value: Any) -> str | None:
    """Return a safe selector value for ``name``, or None when it must not leave."""
    if _SECRET_NAME_RE.search(name):
        return None
    raw = str(value)
    if _CONTROL_CHAR_RE.search(raw):
        return None
    text = raw.strip()
    if (
        not text
        or len(text) > _MAX_ARG_VALUE_CHARS
        or _SECRET_VALUE_RE.search(text)
        or _URL_VALUE_RE.search(text)
        or _JWT_VALUE_RE.fullmatch(text)
        or _OPAQUE_VALUE_RE.fullmatch(text)
        or _NON_FINITE_VALUE_RE.fullmatch(text)
        or not _SELECTOR_VALUE_RE.fullmatch(text)
    ):
        return None
    return text[:_MAX_ARG_VALUE_CHARS]


def _clean_config_value(name: str, value: Any) -> Any | None:
    """Bound and redact one allowlisted model-config value."""
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _redact(name, value)
    if isinstance(value, list):
        cleaned = []
        for item in value[:_MAX_CONFIG_LIST_ITEMS]:
            scalar = _clean_config_value(name, item)
            if scalar is not None and not isinstance(scalar, (dict, list)):
                cleaned.append(scalar)
        return cleaned or None
    return None


def _clean_quantization_config(value: Any) -> dict[str, Any] | None:
    """Keep only bounded, non-secret quantization selectors."""
    if not isinstance(value, dict):
        return None
    cleaned: dict[str, Any] = {}
    for key in _QUANTIZATION_CONFIG_KEYS:
        if key not in value:
            continue
        selected = _clean_config_value(key, value[key])
        if selected is not None:
            cleaned[key] = selected
    return cleaned or None


def server_arg_flags(server_args: str | None) -> dict[str, Any]:
    """Reduce a serving command line to its allowlisted backend selectors.

    The raw string is never emitted. It is tokenised, and only flags named in
    :data:`_SERVER_ARG_ALLOWLIST` survive -- with their values still subject to
    :func:`_redact`, since an allowlisted flag can carry a credential-shaped
    value. A denied flag consumes its value too, so the value cannot reappear
    later as a bare token.

    Args:
        server_args: The serving command line, or None.

    Returns:
        Flag name (without leading dashes) to value; ``True`` for bare flags.
    """
    if not server_args:
        return {}
    if _CONTROL_CHAR_RE.search(str(server_args)):
        return {}
    try:
        tokens = shlex.split(str(server_args))
    except ValueError:
        # An unbalanced quote means we cannot tell flags from values; emitting
        # a partial parse of an unparseable line risks leaking a fragment.
        return {}
    out: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.startswith("--"):
            continue
        flag, sep, inline = token.partition("=")
        has_value = bool(sep)
        value: Any = inline
        if not has_value and index < len(tokens) and not tokens[index].startswith("--"):
            value = tokens[index]
            has_value = True
            index += 1
        if flag not in _SERVER_ARG_ALLOWLIST:
            continue
        if not has_value:
            out[flag.lstrip("-")] = True
            continue
        emitted = _redact(flag, value)
        if emitted is not None:
            out[flag.lstrip("-")] = emitted
    return out


def model_config_summary(model_path: str | Path | None) -> dict[str, Any]:
    """Pull the path-selecting fields out of a model's config.json."""
    if not model_path:
        return {}
    cfg = Path(model_path) / "config.json"
    try:
        raw = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _CONFIG_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if key == "quantization_config":
            value = _clean_quantization_config(value)
        else:
            value = _clean_config_value(key, value)
        if value is not None:
            out[key] = value
    return out


def runtime_flags(
    server_args: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Collect the serving flags and allowlisted environment that pick a path."""
    out: dict[str, Any] = {}
    flags = server_arg_flags(server_args)
    if flags:
        out["server_args"] = flags
    source = env if env is not None else os.environ
    picked: dict[str, str] = {}
    for name in _ENV_ALLOWLIST:
        if name not in source:
            continue
        value = _redact(name, source[name])
        if value is not None:
            picked[name] = value
    if picked:
        out["env"] = picked
    return out


def launcher_stack(entry: dict[str, Any]) -> list[str]:
    """The call-site frames the trace tier recovered for one candidate."""
    frames = entry.get("launcher_frames")
    if isinstance(frames, list) and frames:
        return [str(f) for f in frames[:_MAX_STACK_FRAMES] if str(f).strip()]
    # The single resolved frame is still a call site worth showing.
    src = str(entry.get("source_file") or "").strip()
    line = entry.get("source_line")
    func = str(entry.get("source_function") or "").strip()
    if src and line and func:
        return [f"{src}({line}): {func}"]
    return []


def build_context_block(
    *,
    model_path: str | Path | None = None,
    server_args: str | None = None,
    framework_roots: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    framework: str | None = None,
    precision: str | None = None,
) -> str:
    """Render the shared runtime context, or "" when nothing is available.

    Returned as text rather than a dict because both tiers embed it in a prompt,
    and a single rendering keeps them from drifting apart.
    """
    sections: list[str] = []

    selection = {
        name: cleaned
        for name, value in (("framework", framework), ("precision", precision))
        if value is not None and (cleaned := _redact(name, value)) is not None
    }
    if selection:
        sections.append("Runtime selection:\n" + json.dumps(selection, indent=2))

    cfg = model_config_summary(model_path)
    if cfg:
        sections.append("Model config (selected fields):\n" + json.dumps(cfg, indent=2))

    flags = runtime_flags(server_args, env)
    if flags:
        sections.append("Serving configuration:\n" + json.dumps(flags, indent=2))

    roots = [str(r) for r in framework_roots if str(r).strip()][:_MAX_ROOTS]
    if roots:
        sections.append("Framework roots in play:\n" + "\n".join(f"- {r}" for r in roots))

    if not sections:
        return ""
    block = (
        "Runtime context -- use this to decide which implementation a kernel "
        "actually reaches. A backend flag or a quantization format often "
        "distinguishes two same-named candidates.\n\n" + "\n\n".join(sections)
    )
    return block[:_MAX_CONTEXT_CHARS]
