# Copyright Advanced Micro Devices, Inc. All rights reserved.

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

from ..compat.payload_aliases import (
    CANONICAL_KEY,
    LEGACY_KEY,
    read_extra_server_args,
)
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


# best_config keys that are NOT environment variables (launch args under
# the canonical/legacy name + the nested env containers + current_best
# passthrough metadata copied by ``coordinator._build_recipe_payload``).
_NON_ENV_BEST_CONFIG_KEYS = frozenset({
    CANONICAL_KEY, LEGACY_KEY, "extra_envs", "envs", "args",
    "name", "tput", "accuracy",
})


def _best_config_split(best_config: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Split a ``best_config`` dict into (launch_args, envs).

    Handles BOTH canonical best_config shapes so an ingest->read
    round-trip preserves the champion config:

    * NESTED (authoritative local shape from
      ``coordinator._build_recipe_payload``): launch args under the
      canonical (or read-only legacy-alias) key and the env map nested
      under ``extra_envs`` / ``envs``. The nested dict MUST be unwrapped —
      treating ``extra_envs`` as a scalar env and ``str()``-ing it would
      serialize a Python ``dict`` repr into a single bogus env value and
      drop the real envs.
    * FLAT (legacy / direct-dict shape): launch args under the
      canonical/legacy key and each env as a sibling scalar key.

    Reads the args via the compat helper (canonical with read-only legacy
    fallback). For envs, a nested map wins; otherwise the remaining scalar
    sibling keys (minus non-env metadata) are taken as flat envs.
    """
    args = read_extra_server_args(dict(best_config)).strip()
    nested = best_config.get("extra_envs")
    if not isinstance(nested, Mapping):
        nested = best_config.get("envs")
    if isinstance(nested, Mapping):
        envs = {str(k): str(v) for k, v in nested.items()}
    else:
        envs = {
            str(k): str(v)
            for k, v in best_config.items()
            if k not in _NON_ENV_BEST_CONFIG_KEYS
            and not isinstance(v, (Mapping, list, tuple))
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
    # Negative-knowledge + provenance lists ride the recipe page so a
    # gbrain warm-start gets the same anti-priors the local row carries.
    # Without them ``cortex_t0`` reads empty ``pitfalls`` / ``lessons``
    # off the warm recipe and the next session loses its "avoid known-
    # dead knobs" priors. The minimal YAML emitter only handles scalar
    # lists, so structured list-of-dict fields are stored as JSON strings
    # (round-tripped by ``GbrainRemoteRecipeClient._json_list`` on read).
    for _field in ("what_worked", "what_failed", "remaining_gaps", "pitfalls", "lessons"):
        _value = recipe.get(_field)
        if _value:
            attrs[_field] = json.dumps(_value, ensure_ascii=False, default=str)
    # Stack fingerprint (aiter / rocm / framework versions) rides the page as
    # a nested dict so a gbrain warm-start can derive framework_version / rocm
    # / aiter without the local store. The reader already expects
    # attrs["stack_fingerprint"] as a dict (gbrain_remote_client._page_to_recipe);
    # without this the write side never emits it and the reader always sees {}.
    _stack_fp = recipe.get("stack_fingerprint")
    if isinstance(_stack_fp, Mapping) and _stack_fp:
        attrs["stack_fingerprint"] = {str(k): str(v) for k, v in _stack_fp.items()}
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


def mirror_recipe(recipe: Mapping[str, Any], mcp: _GbrainMcp | None) -> bool:
    """Best-effort mirror of ONE recipe dict into gbrain (read cache).

    Returns True when a page was written, False when skipped (no config /
    no mcp) or on a transport error. Never raises — the local write is
    authoritative and a gbrain hiccup must not affect it.
    """
    if mcp is None:
        return False
    page = recipe_to_page(recipe)
    if page is None:
        return False
    slug, content = page
    try:
        mcp.call("put_page", {"slug": slug, "content": content})
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort
        log.warning("gbrain mirror put_page failed for %s: %r", slug, exc)
        return False


def build_mirror_mcp_from_env() -> _GbrainMcp | None:
    """Build a write-side gbrain MCP client from env (background timeout)."""
    base_url = (os.environ.get("GBRAIN_BASE_URL", "") or "").strip()
    token = (os.environ.get("GBRAIN_TOKEN", "") or "").strip()
    if not base_url or not token:
        return None
    from .. import recipe_snapshot_constants as C
    return _GbrainMcp(base_url, token, C.DEFAULT_HTTP_TIMEOUT_SEC)


class GbrainMirroringRecipeKB:
    """Wrap a :class:`recipe_kb.RecipeKB` so a local ``put_recipe`` also
    mirrors the recipe into gbrain (the read cache), best-effort.

    Preserves the local-first contract: the wrapped dispatcher's local
    write is authoritative and runs first; the gbrain mirror is a
    post-write side-effect that never blocks or fails the local result.
    Only recipes with a concrete ``best_config`` are mirrored (a T0
    anchor has none -> skipped). Every other call (reads / append_attempt
    / ...) delegates to the wrapped dispatcher unchanged.
    """

    def __init__(self, inner: Any, mcp: _GbrainMcp | None) -> None:
        self._inner = inner
        self._mcp = mcp

    def put_recipe(self, **kwargs: Any) -> Any:
        result = self._inner.put_recipe(**kwargs)
        try:
            mirror_recipe(kwargs, self._mcp)
        except Exception as exc:  # noqa: BLE001 - never break the local write
            log.warning("gbrain mirror skipped: %r", exc)
        return result

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (get_recipe / search / append_attempt /
        # local / remote / ...) to the wrapped dispatcher.
        return getattr(self._inner, name)


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


__all__ = [
    "recipe_to_page",
    "ingest_local_to_gbrain",
    "mirror_recipe",
    "build_mirror_mcp_from_env",
    "GbrainMirroringRecipeKB",
    "main",
]
