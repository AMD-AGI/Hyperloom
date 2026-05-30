"""Bulk-ingest local recipe snapshots into gbrain (the read-side cache).

main's ``recipe_kb`` writes recipes LOCAL-only; the gbrain read remote
(:class:`recipe_kb.gbrain_remote_client.GbrainRemoteRecipeClient`) serves
them back to a future session's warm-start. This module is the
"separately-scheduled bulk ingest" that lifts the authoritative local
store into gbrain so a remote read actually returns the champion config
instead of a bare anchor.

Policy (mirrors the gain-gate on the write side):

* Only recipes carrying a concrete ``best_config`` are ingested — a bare
  identity anchor with no config is not worth a remote round-trip.
* Idempotent: each recipe maps to a stable ``type: recipe`` page keyed by
  its 5-tuple canonical id, so re-running overwrites in place.

The emitted page is the same better-landing shape the gbrain read client
expects: ``type: recipe`` + ``tags: kind:/model:/gpu:/framework:`` +
flat ``attrs`` (model/hardware/framework/framework_version/precision +
best_config_args / best_config_envs / best_throughput).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Mapping

from .gbrain_remote_client import _GbrainMcp

log = logging.getLogger(__name__)

# A scalar safe to emit BARE (unquoted) in YAML: starts with a letter and
# is otherwise alnum/._- only. This keeps identifier-ish values (``recipe``,
# ``mi300x``, ``sglang``, ``unknown_version``) bare — gbrain's frontmatter
# parser expects that for keys like ``type`` — while anything that could be
# reinterpreted (digit-leading versions like ``0_5_11`` -> octal, tokens
# with ``:`` / spaces / YAML keywords) is JSON-quoted.
_SAFE_BAREWORD = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")
_YAML_KEYWORDS = frozenset({
    "true", "false", "null", "yes", "no", "on", "off", "none", "~",
})

_TAG_CLEAN = str.maketrans({" ": "-", "\t": "-", "/": "-"})


def _tag_value(value: Any) -> str:
    return str(value or "").strip().lower().translate(_TAG_CLEAN).strip("-") or "unknown"


def _scalar(value: Any) -> str:
    """Render a scalar as a YAML-safe token.

    Bare-word identifiers (letter-leading, alnum/._-) are emitted
    unquoted so gbrain's frontmatter parser accepts well-known keys like
    ``type``. Everything else is JSON double-quoted (valid YAML) so the
    parser never reinterprets a number-ish / bool-ish / underscore-
    separated token — e.g. ``0_5_11`` must not become octal ``329``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    s = str(value)
    if s and _SAFE_BAREWORD.match(s) and s.lower() not in _YAML_KEYWORDS:
        return s
    return json.dumps(s, ensure_ascii=False)


def _emit_yaml(obj: Mapping[str, Any], indent: int = 0) -> str:
    """Minimal recursive YAML emitter for scalars / nested dicts / scalar lists."""
    pad = "  " * indent
    lines: list[str] = []
    for key, val in obj.items():
        if isinstance(val, Mapping):
            if not val:
                lines.append(f"{pad}{key}: {{}}")
            else:
                lines.append(f"{pad}{key}:")
                lines.append(_emit_yaml(val, indent + 1))
        elif isinstance(val, (list, tuple)):
            if not val:
                lines.append(f"{pad}{key}: []")
            else:
                lines.append(f"{pad}{key}:")
                for item in val:
                    lines.append(f"{pad}- {_scalar(item)}")
        else:
            lines.append(f"{pad}{key}: {_scalar(val)}")
    return "\n".join(lines)


def _best_config_split(best_config: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Split a v2 ``best_config`` dict into (sglang_args, envs).

    Inverse of ``GbrainRemoteRecipeClient._best_config_from_attrs`` so a
    round-trip ingest->read preserves the champion config.
    """
    args = str(best_config.get("extra_sglang_args") or "").strip()
    envs = {
        str(k): str(v) for k, v in best_config.items() if k != "extra_sglang_args"
    }
    return args, envs


def recipe_to_page(recipe: Mapping[str, Any]) -> tuple[str, str] | None:
    """Map a v2 recipe dict to a (slug, content) gbrain better-landing page.

    Returns ``None`` when the recipe carries no concrete ``best_config``
    (a bare anchor is skipped — nothing useful to cache remotely).
    """
    best_config = recipe.get("best_config") if isinstance(recipe.get("best_config"), Mapping) else {}
    if not best_config:
        return None
    canonical = str(recipe.get("canonical_id") or "").strip()
    if not canonical:
        return None
    args, envs = _best_config_split(best_config)
    model = str(recipe.get("model") or "")
    hardware = str(recipe.get("hardware") or "")
    framework = str(recipe.get("framework") or "")
    attrs: dict[str, Any] = {
        "model": model,
        "hardware": hardware,
        "framework": framework,
        "framework_version": str(recipe.get("framework_version") or ""),
        "precision": str(recipe.get("precision") or ""),
        "best_config_args": args,
        "best_config_envs": envs,
        "best_throughput": float(recipe.get("best_throughput") or 0.0),
        "validated_gain_pct": float(recipe.get("validated_gain_pct") or 0.0),
    }
    tags = [
        "kind:recipe",
        f"model:{_tag_value(model)}",
        f"gpu:{_tag_value(hardware)}",
        f"framework:{_tag_value(framework)}",
    ]
    frontmatter: dict[str, Any] = {
        "type": "recipe",
        "tags": tags,
        "kind": "recipe",
        "canonical_id": canonical,
        "authority": str(recipe.get("authority") or "EXPERIENTIAL"),
        "confidence": float(recipe.get("confidence") or 0.85),
        "attrs": attrs,
    }
    # Stable slug from the 5-tuple canonical (colons -> path levels).
    slug = "recipe-snapshot/" + canonical.replace(":", "/")
    body_lines = [
        f"# Recipe {canonical}",
        "",
        f"- model: {model}",
        f"- hardware: {hardware}",
        f"- framework: {framework}",
        f"- best_throughput: {attrs['best_throughput']}",
        f"- validated_gain_pct: {attrs['validated_gain_pct']}",
        f"- best_config_args: {args}",
    ]
    content = "---\n" + _emit_yaml(frontmatter) + "\n---\n\n" + "\n".join(body_lines) + "\n"
    return slug, content


def ingest_local_to_gbrain(
    *,
    recipes: list[dict[str, Any]],
    mcp: _GbrainMcp | None,
    dry_run: bool,
) -> dict[str, int]:
    """Ingest a list of v2 recipe dicts into gbrain. Returns counters."""
    stats = {"total": len(recipes), "ingested": 0, "skipped_no_config": 0, "errors": 0}
    for recipe in recipes:
        page = recipe_to_page(recipe)
        if page is None:
            stats["skipped_no_config"] += 1
            continue
        slug, content = page
        if dry_run or mcp is None:
            stats["ingested"] += 1
            continue
        try:
            mcp.call("put_page", {"slug": slug, "content": content})
            stats["ingested"] += 1
        except Exception as exc:  # noqa: BLE001 - count, keep going
            stats["errors"] += 1
            log.warning("gbrain ingest put_page failed for %s: %r", slug, exc)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bulk-ingest local recipe snapshots into gbrain.")
    ap.add_argument("--local-kb-root", default=os.environ.get("HYPERLOOM_LOCAL_KB_ROOT", ""),
                    help="LocalRecipeStore root (default: $HYPERLOOM_LOCAL_KB_ROOT)")
    ap.add_argument("--gbrain-url", default=os.environ.get("GBRAIN_BASE_URL", ""))
    ap.add_argument("--token", default=os.environ.get("GBRAIN_TOKEN", ""))
    ap.add_argument("--limit", type=int, default=0, help="max recipes to scan (0=all)")
    ap.add_argument("--write", action="store_true", help="actually put pages (default dry-run)")
    args = ap.parse_args(argv)

    if not args.local_kb_root:
        print("requires --local-kb-root (or $HYPERLOOM_LOCAL_KB_ROOT)")
        return 2
    from pathlib import Path

    from .local_store import LocalRecipeStore

    store = LocalRecipeStore(root=Path(args.local_kb_root))
    recipes = store.list_recent(limit=args.limit or 100000)
    dry = not args.write
    mcp = None
    if not dry:
        if not args.gbrain_url or not args.token:
            print("--write requires GBRAIN_BASE_URL + GBRAIN_TOKEN")
            return 2
        from .. import recipe_snapshot_constants as C
        mcp = _GbrainMcp(args.gbrain_url, args.token, C.DEFAULT_HTTP_TIMEOUT_SEC)

    stats = ingest_local_to_gbrain(recipes=recipes, mcp=mcp, dry_run=dry)
    print(f"=== gbrain recipe ingest ({'DRY-RUN' if dry else 'WRITE'}) ===")
    for key, val in stats.items():
        print(f"  {key:18}: {val}")
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["recipe_to_page", "ingest_local_to_gbrain", "main"]
