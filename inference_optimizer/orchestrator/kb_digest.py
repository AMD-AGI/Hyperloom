"""Load marathon-style KB (entries.jsonl + kb_query.py) for Orchestration prompts.

Default KB root: ``/wekafs/xiaofei/Hyperloom/marathon/skills/kb`` (validated stack).
Override with env ``INFERENCE_OPTIMIZER_KB_ROOT`` pointing at a directory that
contains ``kb_query.py`` and ``entries.jsonl``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType

log = logging.getLogger(__name__)

ENV_KB_ROOT = "INFERENCE_OPTIMIZER_KB_ROOT"
DEFAULT_KB_ROOT = Path("/wekafs/xiaofei/Hyperloom/marathon/skills/kb")

_QUERY_PACKS: tuple[tuple[str, str], ...] = (
    ("torch.compile inductor triton native kernel", ""),
    ("sglang server parameter tuning backends", ""),
    ("kernel optimization HIP AITER", ""),
)


def _kb_root() -> Path:
    return Path(os.environ.get(ENV_KB_ROOT, str(DEFAULT_KB_ROOT))).expanduser()


def _load_kb_query_module(kb_root: Path) -> ModuleType | None:
    script = kb_root / "kb_query.py"
    if not script.is_file():
        log.warning("kb_digest: missing %s — KB hints disabled", script)
        return None
    spec = importlib.util.spec_from_file_location(
        f"io_kb_query_{hash(str(kb_root)) % 10000}", script,
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def format_kb_digest_for_orchestration(
    *,
    model_name: str,
    framework: str = "sglang",  # unused; KB rows often omit framework
    top_k_per_query: int = 4,
    max_lines: int = 14,
) -> str:
    """Return a compact markdown-free bullet list for the reactor prompt.

    We intentionally **do not** hard-filter by ``model_name``: the seed KB is
    small and many lessons transfer across models; strict model match often
    returns nothing. Relevance ranking still surfaces the best generic hits.
    """
    kb_root = _kb_root()
    mod = _load_kb_query_module(kb_root)
    if mod is None:
        return ""

    query_fn = getattr(mod, "query", None)
    if not callable(query_fn):
        return ""

    lines: list[str] = []
    seen: set[str] = set()
    _framework = ""  # avoid dropping entries missing framework field

    for query_text, _cat in _QUERY_PACKS:
        try:
            rows = query_fn(
                query_text=query_text,
                model="",
                framework=_framework,
                top_k=top_k_per_query,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("kb_digest query failed: %s", exc)
            continue
        for e in rows or []:
            eid = str(e.get("id", ""))
            if eid and eid in seen:
                continue
            if eid:
                seen.add(eid)
            cat = e.get("category", "?")
            action = (e.get("action") or "")[:120]
            lesson = (e.get("lesson") or e.get("context") or "")[:220]
            rel = e.get("_relevance")
            rel_s = f" rel={rel:.2f}" if isinstance(rel, (int, float)) else ""
            line = f"- [{cat}] {action}"
            if lesson:
                line += f" — {lesson}"
            line += rel_s
            lines.append(line)
            if len(lines) >= max_lines:
                break

    if not lines:
        return ""
    out_lines = []
    if model_name:
        out_lines.append(
            f"(session model: {model_name} — KB not model-filtered; ranked by relevance)",
        )
    out_lines.extend(lines)
    return "\n".join(out_lines[:max_lines])
