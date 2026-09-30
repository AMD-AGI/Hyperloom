# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The InferenceX server script a recipe boots through, and the edited copy that carries levers into it."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_text
from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json

log = logging.getLogger(__name__)

# ``export NAME=value`` with no ``${NAME:-...}`` guard: the recipe's own value
# wins over anything the caller exported under that name.
_UNGUARDED_EXPORT_RE = re.compile(r"^[^\S\n]*export\s+([A-Za-z_][A-Za-z0-9_]*)=(?!\"?\$\{?\1[:-])", re.MULTILINE)

# Spelled out rather than imported from ``agentx.deploy``: this module is on the
# default benchmark path, which is pinned not to import the agentx package.
_AGENTX_CLIENT_SCRIPT = "aiperf_client.sh"

_SPLICE_RE = re.compile(r'^"?\$\{([A-Za-z_][A-Za-z0-9_]*)\[@\]\}"?$')
_COPY_SUFFIX_RE = re.compile(r"\.hl-[0-9a-f]{12}(?=\.sh$)")

# The recipe simulates acceptance for its own draft, so what is drafted and how
# acceptance is simulated stay pinned; how fast the draft runs stays open.
_PINNED_DRAFT_FLAGS = frozenset(
    {
        "speculative-num-steps",
        "speculative-num-draft-tokens",
        "speculative-eagle-topk",
        "speculative-algorithm",
        "speculative-draft-model-path",
        "speculative-dspark-block-size",
        "num-speculative-tokens",
        "method",
        "draft-model",
    }
)
_PINNED_DRAFT_PREFIXES = ("spec-decode-acceptance", "speculative-config")
_PINNED_SPEC_CONFIG_KEYS = frozenset(
    {
        "method",
        "model",
        "num_speculative_tokens",
        "rejection_sample_method",
        "synthetic_acceptance_length",
        "draft_sample_method",
    }
)
_ACCEPTANCE_ENV_PREFIX = "SGLANG_SIMULATE_ACC_"

_SPEC_CONFIG = "speculative-config"
_SPEC_CONFIG_VAR = "HYPERLOOM_SPECULATIVE_CONFIG"
_JSON_MERGE = "import json,sys; c=json.loads(sys.argv[1]); c.update(json.loads(sys.argv[2])); print(json.dumps(c))"


class RecipeLeverUnavailableError(ValueError):
    """Raised when a lever or run mode cannot be expressed on the recipe that boots the server."""


def resolve_launch_server_script(bench: Mapping[str, Any]) -> str:
    """Path of the script that boots the server, or ``""`` when unresolvable.

    ``benchmark_script`` names a server launcher on every non-AgentX run. The
    AgentX switch pins the aiperf client there instead, so for that one recipe
    shape the launcher is the builtin the client delegates to. Resolution
    mirrors ``aiperf_client.sh``: the same ``AGENTX_SERVER_SCRIPT`` override,
    the same ``{framework}_{gpu}.sh`` fallback, the same ``<checkout>/benchmarks/``
    directory and no recursive search. Recipe-recorded values beat the ambient
    env because the recipe is the record of what actually ran.
    """
    envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    script = Path(str(bench.get("benchmark_script") or "").strip()).name
    if not script:
        return ""

    if script == _AGENTX_CLIENT_SCRIPT:
        script = str(envs.get("AGENTX_SERVER_SCRIPT") or os.environ.get("AGENTX_SERVER_SCRIPT") or "").strip()
        if not script:
            framework = str(bench.get("framework") or envs.get("FRAMEWORK") or "").strip().lower()
            if not framework:
                return ""
            gpu = (
                str(
                    envs.get("GPU_TYPE")
                    or envs.get("RUNNER_TYPE")
                    or bench.get("runner_type")
                    or os.environ.get("GPU_TYPE")
                    or os.environ.get("RUNNER_TYPE")
                    or "mi300x"
                )
                .strip()
                .lower()
            )
            script = f"{framework}_{gpu}.sh"

    for root in (
        str(bench.get("inferencex_path") or "").strip(),
        os.environ.get("INFERENCEX_PATH", "").strip(),
    ):
        if not root:
            continue
        benchmarks = Path(root) / "benchmarks"
        candidate = benchmarks / script
        # The builtin sources benchmark_lib.sh from its own directory and dies
        # without it, so a half-populated checkout resolves to nothing.
        if candidate.is_file() and (benchmarks / "benchmark_lib.sh").is_file():
            return str(candidate)
    return ""


def recipe_owns_argv(bench: Mapping[str, Any]) -> bool:
    """Whether the AgentX client boots an agentic recipe, which hardcodes its own argv and env.

    Same rule as ``aiperf_client.sh``: the ``AGENTX_SERVER_SCRIPT`` path relative
    to ``benchmarks/`` sits under an ``agentic/`` directory.
    """
    if Path(str(bench.get("benchmark_script") or "")).name != _AGENTX_CLIENT_SCRIPT:
        return False
    envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    script = str(envs.get("AGENTX_SERVER_SCRIPT") or os.environ.get("AGENTX_SERVER_SCRIPT") or "").strip()
    return "agentic" in Path(script).parent.parts and bool(resolve_launch_server_script(bench))


def launcher_overwritten_envs(bench: Mapping[str, Any]) -> frozenset[str]:
    """Env names the launcher re-exports unconditionally.

    Empty for an agentic recipe: its rendered copy re-exports every env lever
    after the recipe's own exports.
    """
    if recipe_owns_argv(bench):
        return frozenset()
    path = resolve_launch_server_script(bench)
    if not path:
        return frozenset()
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        log.warning("recipe: could not read the server script %s; assuming it constrains nothing", path)
        return frozenset()
    return frozenset(m.group(1) for m in _UNGUARDED_EXPORT_RE.finditer(text))


def _official(script: str) -> str:
    """The recipe a rendered copy was derived from (``x.hl-<digest>.sh`` -> ``x.sh``)."""
    return _COPY_SUFFIX_RE.sub("", script)


def _flag_key(token: str) -> str:
    """Identity of a ``--flag`` / ``--flag=value`` token; vLLM treats ``_`` and ``-`` alike."""
    return token.split("=", 1)[0].lstrip("-").lower().replace("_", "-")


def _words(line: str) -> list[str]:
    """Shell words of one array line, quotes kept, comments dropped."""
    lexer = shlex.shlex(line, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def _array_span(lines: list[str], name: str) -> tuple[int, int]:
    """Line indexes of ``NAME=(`` and its closing ``)`` for the one multi-line server array."""
    opens = [i for i, line in enumerate(lines) if re.fullmatch(rf"\s*{name}=\(\s*", line)]
    if len(opens) != 1:
        raise RecipeLeverUnavailableError(f"the recipe has {len(opens)} multi-line {name}=( arrays; expected one")
    start = opens[0]
    for end in range(start + 1, len(lines)):
        if lines[end].strip() == ")":
            return start, end
    raise RecipeLeverUnavailableError(f"the recipe's {name} array is never closed")


def _array_flags(lines: list[str], name: str) -> set[str]:
    """Flag keys assigned to ``name`` anywhere in the recipe, single- or multi-line."""
    flags: set[str] = set()
    for i, line in enumerate(lines):
        match = re.match(rf"\s*{name}\+?=\((.*)$", line)
        if not match:
            continue
        body = [match.group(1)]
        if not match.group(1).rstrip().endswith(")"):
            for tail in lines[i + 1 :]:
                if tail.strip() == ")":
                    break
                body.append(tail)
        flags.update(
            _flag_key(w) for part in body for w in _words(part.rstrip().removesuffix(")")) if w.startswith("--")
        )
    return flags


def _flag_groups(tokens: Sequence[str]) -> list[list[str]]:
    """Split argv tokens into ``[flag, *values]`` groups."""
    groups: list[list[str]] = []
    for token in tokens:
        if token.startswith("-") or not groups:
            groups.append([token])
        else:
            groups[-1].append(token)
    return groups


def _spec_config_overlay(group: Sequence[str]) -> dict[str, Any]:
    """The JSON object a ``--speculative-config`` lever merges into the recipe's own."""
    try:
        overlay = json.loads(group[1]) if len(group) == 2 else None
    except json.JSONDecodeError:
        overlay = None
    if not isinstance(overlay, dict):
        raise RecipeLeverUnavailableError(f"{' '.join(group)!r} is not one --speculative-config JSON object")
    return overlay


def _render(
    text: str,
    framework: str,
    tokens: Sequence[str],
    remove_args: Sequence[str],
    env_levers: Mapping[str, str | None],
) -> str:
    """Apply set/remove edits to the server array and export env levers ahead of its launch."""
    name = f"{framework.upper()}_CMD"
    lines = text.splitlines(keepends=True)
    start, end = _array_span(lines, name)
    groups = _flag_groups(tokens)
    overlays = [_spec_config_overlay(g) for g in groups if _flag_key(g[0]) == _SPEC_CONFIG]
    groups = [g for g in groups if _flag_key(g[0]) != _SPEC_CONFIG]
    drop = {_flag_key(g[0]) for g in groups} | {_flag_key(r.split()[0]) for r in remove_args if r.strip()}
    pinned = sorted(k for k in drop if k in _PINNED_DRAFT_FLAGS or k.startswith(_PINNED_DRAFT_PREFIXES))
    pinned += sorted(k for overlay in overlays for k in overlay if k in _PINNED_SPEC_CONFIG_KEYS)
    pinned += sorted(k for k in env_levers if k.startswith(_ACCEPTANCE_ENV_PREFIX))
    if pinned:
        raise RecipeLeverUnavailableError(f"{pinned} pin the recipe's draft and its simulated acceptance")

    words = [(i, w) for i in range(start + 1, end) for w in _words(lines[i])]
    edited = drop | ({_SPEC_CONFIG} if overlays else set())
    for _, word in words:
        splice = _SPLICE_RE.match(word)
        if splice and (clash := edited & _array_flags(lines, splice.group(1))):
            raise RecipeLeverUnavailableError(f"{sorted(clash)} are set inside {splice.group(1)}, not in {name}")

    kept: dict[int, list[str]] = {i: [] for i in range(start + 1, end)}
    value_pending = False
    spec_value_next = False
    spec_value: str | None = None
    for i, word in words:
        if value_pending:
            value_pending = False
            if not word.startswith("-") and not _SPLICE_RE.match(word):
                continue
        if spec_value_next:
            spec_value_next = False
            spec_value, word = word, f'"${_SPEC_CONFIG_VAR}"'
        elif word.startswith("--") and _flag_key(word) in drop:
            value_pending = "=" not in word
            continue
        elif overlays and word == f"--{_SPEC_CONFIG}":
            spec_value_next = True
        kept[i].append(word)
    if overlays and spec_value is None:
        raise RecipeLeverUnavailableError(f"{name} has no --{_SPEC_CONFIG} <value> to merge the lever into")

    indent = re.match(r"\s*", lines[start + 1]).group(0)
    body: list[str] = []
    for i in range(start + 1, end):
        if kept[i] == _words(lines[i]):
            body.append(lines[i])
        elif kept[i]:
            body.append(indent + " ".join(kept[i]) + "\n")
    body.extend(indent + " ".join(shlex.quote(t) for t in group) + "\n" for group in groups)
    lines[start + 1 : end] = body
    end = start + 1 + len(body)

    if env_levers:
        expansion = f"${{{name}[@]}}"
        launch = next((i for i in range(end + 1, len(lines)) if expansion in lines[i]), None)
        if launch is None:
            raise RecipeLeverUnavailableError(f"the recipe never expands {name} after defining it")
        lead = re.match(r"\s*", lines[launch]).group(0)
        block = [
            f"{lead}unset {key}\n" if value is None else f"{lead}export {key}={shlex.quote(value)}\n"
            for key, value in env_levers.items()
        ]
        lines[launch:launch] = block

    # Merged at launch, so the recipe's own value (often built from shell variables) keeps its pinned keys.
    lead = re.match(r"\s*", lines[start]).group(0)
    merges: list[str] = []
    for overlay in overlays:
        patch = shlex.quote(json.dumps(overlay, separators=(",", ":")))
        merges.append(f'{lead}{_SPEC_CONFIG_VAR}="$(python3 -c {shlex.quote(_JSON_MERGE)} {spec_value} {patch})"\n')
        spec_value = f'"${_SPEC_CONFIG_VAR}"'
    lines[start:start] = merges
    return "".join(lines)


def apply_recipe_levers(
    bench: Mapping[str, Any],
    *,
    inherited_script: str,
    server_args: str,
    remove_args: Sequence[str],
    env_levers: Mapping[str, str | None],
) -> str:
    """Render the levers into a copy of the agentic recipe; return its ``AGENTX_SERVER_SCRIPT`` value.

    The base is ``inherited_script`` when it derives from the session's recipe,
    otherwise the session's recipe itself (``""`` under replace mode). The copy
    sits beside the recipe as ``<stem>.hl-<digest>.sh``; with nothing to apply
    the base is returned unchanged.

    Args:
        bench: Benchmark mapping whose ``AGENTX_SERVER_SCRIPT`` names the session's recipe.
        inherited_script: ``AGENTX_SERVER_SCRIPT`` the loaded config declared.
        server_args: Declared lever string; each flag replaces the recipe's own,
            except a ``--speculative-config`` object, which merges into it.
        remove_args: Flags deleted from the recipe's server array.
        env_levers: Env name to value, or ``None`` to unset, applied in order.

    Raises:
        RecipeLeverUnavailableError: When the recipe's shape cannot carry an edit.
    """
    envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    current = str(envs.get("AGENTX_SERVER_SCRIPT") or os.environ.get("AGENTX_SERVER_SCRIPT") or "").strip()
    base = inherited_script if inherited_script and _official(inherited_script) == _official(current) else current
    parsed = tokenize_server_args_preserving_json(server_args)
    if parsed is None:
        raise RecipeLeverUnavailableError(f"{server_args!r} does not split into argv tokens")
    tokens = parsed[1]
    if not tokens and not remove_args and not env_levers:
        return base
    benchmarks = Path(resolve_launch_server_script(bench)).parents[len(Path(current).parts) - 1]
    text = (benchmarks / base).read_text(encoding="utf-8")
    rendered = _render(text, str(bench.get("framework") or ""), tokens, remove_args, env_levers)
    official = Path(_official(base))
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
    copy = official.with_name(f"{official.stem}.hl-{digest}.sh")
    if not (benchmarks / copy).exists():
        atomic_write_text(benchmarks / copy, rendered)
    return str(copy)


__all__ = [
    "RecipeLeverUnavailableError",
    "apply_recipe_levers",
    "launcher_overwritten_envs",
    "recipe_owns_argv",
    "resolve_launch_server_script",
]
