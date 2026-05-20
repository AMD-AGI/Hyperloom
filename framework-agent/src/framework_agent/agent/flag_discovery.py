"""DiscoveredFlag normalisation + dedup + ranking.

Output schema consumed by IO's ``SharedState.discovered_flags`` (see
hyperloom-framework-agent-design.md §6.6). Read by params action next
round so the bandit grid auto-expands with newly surfaced flags.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal


# 5 provenance markers: libcst-based variants get the strict name;
# grep-fallback variants carry the ``_grep`` suffix so downstream KB
# write can downgrade ``confidence`` (design §9.3 R3 mitigation).
DiscoverVia = Literal[
    "argparse", "dataclass", "pydantic",
    "argparse_grep", "dataclass_grep",
]


@dataclass(frozen=True)
class DiscoveredFlag:
    """One flag-shaped tunable surfaced from framework source.

    Frozen so call sites can use it as a dict key / dedup token.
    """

    flag_name: str       # "--max-model-len" (cli) or "max_num_seqs" (config)
    module: str          # dotted module path, e.g. "vllm.engine.arg_utils"
    source_path: str     # absolute path to the .py file
    line: int            # 1-based line number of the definition
    via: DiscoverVia
    type_hint: str       # "int" / "bool" / "Optional[str]" / "str" fallback
    default_repr: str    # repr of default; "_MISSING_" if no default
    help_text: str       # argparse help= text or class name slice (<=160 chars)
    surface: Literal["cli", "config"]
    # Optional framework label so IO can route the flag to the right
    # config block (e.g. EXTRA_SGLANG_ARGS vs EXTRA_VLLM_ARGS).
    framework: str = ""

    def fingerprint(self) -> tuple[str, str, str, int]:
        """Stable identity tuple used by dedup / KB lookup."""
        return (self.flag_name, self.surface, self.source_path, self.line)


_VIA_RANK = {
    "argparse":       0,
    "pydantic":       1,
    "dataclass":      2,
    "argparse_grep":  10,
    "dataclass_grep": 11,
}


def dedup_and_rank(flags: Iterable[DiscoveredFlag]) -> list[DiscoveredFlag]:
    """Drop duplicates by fingerprint; sort by (surface, via, name).

    Same flag may legitimately appear in two surfaces (argparse CLI +
    dataclass field) -- both are kept since they expose different
    surfaces of the same tunable. Identical (flag_name, surface,
    source_path, line) tuples are deduped (visiting the same file
    twice e.g. via grep fallback after libcst partial parse).
    """
    seen: set[tuple[str, str, str, int]] = set()
    unique: list[DiscoveredFlag] = []
    for f in flags:
        fp = f.fingerprint()
        if fp in seen:
            continue
        seen.add(fp)
        unique.append(f)
    unique.sort(key=lambda f: (
        0 if f.surface == "cli" else 1,
        _VIA_RANK.get(f.via, 99),
        f.flag_name,
    ))
    return unique


def cli_flag_names(flags: Iterable[DiscoveredFlag]) -> list[str]:
    """Convenience: pick only the cli-surface flag names (with --) for
    downstream EXTRA_SGLANG_ARGS / EXTRA_VLLM_ARGS string assembly."""
    return [f.flag_name for f in flags if f.surface == "cli"]


def to_json_records(flags: Iterable[DiscoveredFlag]) -> list[dict]:
    """Serialise to JSON-friendly list-of-dicts for the envelope."""
    return [asdict(f) for f in flags]


def path_to_module(path: Path, root: Path) -> str:
    """Derive a dotted module path relative to a framework source root.

    Examples:
        path  = /sgl-workspace/vllm/vllm/engine/arg_utils.py
        root  = /sgl-workspace/vllm
        -> "vllm.engine.arg_utils"
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.stem
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts) if parts else path.stem


__all__ = [
    "DiscoverVia",
    "DiscoveredFlag",
    "cli_flag_names",
    "dedup_and_rank",
    "path_to_module",
    "to_json_records",
]
