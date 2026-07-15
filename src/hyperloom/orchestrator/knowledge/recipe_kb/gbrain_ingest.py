# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Mirror local recipe snapshots into gbrain (the read-side cache).

``recipe_kb`` writes recipes LOCAL-only; the gbrain read remote
(:class:`recipe_kb.gbrain_remote_client.GbrainRemoteRecipeClient`) serves them
back to a future session's warm-start. This module is the bulk ingest that
lifts the authoritative local store into gbrain.

Mirror gate (default permissive):

* Any recipe with a ``canonical_id`` is mirrored, including bare seed-only
  anchors (empty ``best_config``). Set ``RECIPE_KB_MIRROR_REQUIRE_SIGNAL=1``
  to restore the stricter gate (``best_config`` OR reusable prior signal).
* Idempotent: each recipe maps to a stable ``type: recipe`` page keyed by its
  canonical id, so re-running overwrites in place.

The emitted page shape: ``type: recipe`` + ``tags:
kind:/model:/gpu:/framework_name:`` + flat ``attrs``
(model/hardware/framework_name/framework_version/precision +
best_config_args / best_config_envs / best_throughput).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Mapping

from .gbrain_remote_client import _GbrainMcp

log = logging.getLogger(__name__)

# A scalar safe to emit bare (unquoted) in YAML: letter-leading, otherwise
# alnum/._- only. Anything reinterpretable (digit-leading versions, tokens with
# ``:`` / spaces / YAML keywords) is JSON-quoted instead.
_SAFE_BAREWORD = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")
_YAML_KEYWORDS = frozenset(
    {
        "true",
        "false",
        "null",
        "yes",
        "no",
        "on",
        "off",
        "none",
        "~",
    }
)

_TAG_CLEAN = str.maketrans({" ": "-", "\t": "-", "/": "-"})
_DEFAULT_RECIPE_SLUG_PREFIX = "hyperloom-recipe-kb"
_RECIPE_SLUG_PREFIX_ENV = "GBRAIN_RECIPE_SLUG_PREFIX"
_EXTRA_SERVER_ARGS_KEY = "extra_server_args"


def _tag_value(value: Any) -> str:
    """Normalize a value into a slug-style tag token.

    Lowercases the value and replaces whitespace and slashes with
    hyphens, trimming leading/trailing hyphens.

    Args:
        value: Arbitrary value to slugify.

    Returns:
        The tag slug, or ``"unknown"`` when the value is empty.
    """
    return str(value or "").strip().lower().translate(_TAG_CLEAN).strip("-") or "unknown"


def _recipe_slug_prefix() -> str:
    """Return the configured gbrain recipe page slug prefix.

    Returns:
        The normalized ``GBRAIN_RECIPE_SLUG_PREFIX`` value, or the default
        production recipe namespace when the env is unset.
    """
    raw = os.environ.get(_RECIPE_SLUG_PREFIX_ENV, "").strip().strip("/")
    return raw or _DEFAULT_RECIPE_SLUG_PREFIX


def _scalar(value: Any) -> str:
    """Render a scalar as a YAML-safe token.

    Bare-word identifiers (letter-leading, alnum/._-) are emitted unquoted;
    everything else is JSON double-quoted so the parser never reinterprets a
    number-ish / bool-ish / underscore-separated token.

    Args:
        value: The scalar value to render.

    Returns:
        A YAML-safe token: bare for safe identifiers, JSON-quoted otherwise.
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
    """Minimal recursive YAML emitter for scalars / nested dicts / scalar lists.

    Args:
        obj: The mapping to render as YAML.
        indent: Current indentation depth (two spaces per level).

    Returns:
        The rendered YAML text.
    """
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


# best_config keys that are NOT environment variables (launch args, nested env
# containers, and current_best passthrough metadata).
_NON_ENV_BEST_CONFIG_KEYS = frozenset(
    {
        _EXTRA_SERVER_ARGS_KEY,
        "extra_envs",
        "envs",
        "args",
        "name",
        "tput",
        "accuracy",
    }
)


def _coerce_server_args(value: Any) -> str:
    """Return canonical ``extra_server_args`` as a launch-arg string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value)


def _best_config_split(best_config: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Split a ``best_config`` dict into (launch_args, envs).

    Handles both canonical best_config shapes so an ingest->read round-trip
    preserves the champion config:

    * NESTED (authoritative local shape from
      ``coordinator._build_recipe_payload``): launch args under the
      canonical key and the env map nested under ``extra_envs``.
      The nested dict MUST be unwrapped —
      treating ``extra_envs`` as a scalar env and ``str()``-ing it would
      serialize a Python ``dict`` repr into a single bogus env value and
      drop the real envs.
    * FLAT (direct-dict shape): launch args under the canonical key and each
      env as a sibling scalar key.

    For envs, a nested map wins; otherwise the remaining scalar sibling keys
    (minus non-env metadata) are taken as flat envs.

    Args:
        best_config: The champion config dict in either nested or flat shape.

    Returns:
        A tuple of the launch-args string and the env-var dict.
    """
    args = _coerce_server_args(best_config.get(_EXTRA_SERVER_ARGS_KEY)).strip()
    nested = best_config.get("extra_envs")
    if isinstance(nested, Mapping):
        envs = {str(k): str(v) for k, v in nested.items()}
    else:
        envs = {
            str(k): str(v)
            for k, v in best_config.items()
            if k not in _NON_ENV_BEST_CONFIG_KEYS and not isinstance(v, (Mapping, list, tuple))
        }
    return args, envs


def _has_shareable_signal(recipe: Mapping[str, Any]) -> bool:
    """Return True when a seed-only recipe carries reusable prior signal.

    Args:
        recipe: The recipe dict to inspect.

    Returns:
        ``True`` when the recipe has a positive-throughput session, any
        negative-knowledge list, or architecture/model-class hints.
    """
    for s in recipe.get("sessions") or []:
        if not isinstance(s, Mapping):
            continue
        try:
            tput = float(s.get("throughput_after") or 0.0)
        except (TypeError, ValueError):
            tput = 0.0
        if tput > 0.0 or s.get("actions_taken"):
            return True
    for field in ("what_worked", "what_failed", "remaining_gaps", "pitfalls", "lessons"):
        if recipe.get(field):
            return True
    if recipe.get("architectures") or recipe.get("model_class"):
        return True
    return False


def recipe_to_page(recipe: Mapping[str, Any]) -> tuple[str, str] | None:
    """Map a v2 recipe dict to a (slug, content) gbrain better-landing page.

    Returns ``None`` only when the recipe has no ``canonical_id``. By default
    even pure seed-only anchors are mirrored; set
    ``RECIPE_KB_MIRROR_REQUIRE_SIGNAL=1`` for the stricter gate (best_config OR
    reusable prior).

    Args:
        recipe: The v2 recipe dict to convert.

    Returns:
        A ``(slug, content)`` page tuple, or ``None`` when the recipe lacks a
        canonical id (or fails the strict mirror gate).
    """
    best_config = recipe.get("best_config") if isinstance(recipe.get("best_config"), Mapping) else {}
    canonical = str(recipe.get("canonical_id") or "").strip()
    if not canonical:
        return None
    if str(os.environ.get("RECIPE_KB_MIRROR_REQUIRE_SIGNAL", "")).strip().lower() in ("1", "true", "yes"):
        if not best_config and not _has_shareable_signal(recipe):
            return None
    args, envs = _best_config_split(best_config)
    model = str(recipe.get("model") or "")
    hardware = str(recipe.get("hardware") or "")
    # Back-compat: rows predating the framework_name rename use ``framework``.
    framework_name = str(recipe.get("framework_name") or recipe.get("framework") or "")
    attrs: dict[str, Any] = {
        "model": model,
        "hardware": hardware,
        "framework_name": framework_name,
        "framework_version": str(recipe.get("framework_version") or ""),
        "precision": str(recipe.get("precision") or ""),
        "model_type": str(recipe.get("model_type") or ""),
        "architectures": list(recipe.get("architectures") or []),
        "best_config_args": args,
        "best_config_envs": envs,
        "best_throughput": float(recipe.get("best_throughput") or 0.0),
        "validated_gain_pct": float(recipe.get("validated_gain_pct") or 0.0),
    }
    # Negative-knowledge + provenance lists ride the recipe page so a gbrain
    # warm-start gets the same anti-priors the local row carries. The minimal
    # YAML emitter only handles scalar lists, so structured list-of-dict fields
    # are stored as JSON strings (decoded by ``_json_list`` on read).
    for _field in ("what_worked", "what_failed", "remaining_gaps", "pitfalls", "lessons", "prs_tested"):
        _value = recipe.get(_field)
        if _value:
            attrs[_field] = json.dumps(_value, ensure_ascii=False, default=str)
    # Stack fingerprint rides the page as a nested dict so a gbrain warm-start
    # can derive framework_version / rocm / aiter without the local store.
    _stack_fp = recipe.get("stack_fingerprint")
    if isinstance(_stack_fp, Mapping) and _stack_fp:
        attrs["stack_fingerprint"] = {str(k): str(v) for k, v in _stack_fp.items()}
    _model_type = str(recipe.get("model_type") or "")
    _architectures_raw = recipe.get("architectures") or []
    if isinstance(_architectures_raw, list):
        _arch_str = "+".join(sorted(str(a).strip().lower() for a in _architectures_raw if str(a or "").strip()))
    else:
        _arch_str = str(_architectures_raw).strip().lower()
    tags = [
        "kind:recipe",
        f"model:{_tag_value(model)}",
        f"gpu:{_tag_value(hardware)}",
        f"framework_name:{_tag_value(framework_name)}",
    ]
    if _model_type:
        tags.append(f"model_type:{_tag_value(_model_type)}")
    if _arch_str:
        tags.append(f"architectures:{_arch_str}")
    frontmatter: dict[str, Any] = {
        "type": "recipe",
        "tags": tags,
        "kind": "recipe",
        "canonical_id": canonical,
        "authority": str(recipe.get("authority") or "EXPERIENTIAL"),
        "confidence": float(recipe.get("confidence") or 0.85),
        "attrs": attrs,
    }
    # Stable slug from the canonical id (colons -> path levels).
    slug = _recipe_slug_prefix() + "/" + canonical.replace(":", "/")
    body_lines = [
        f"# Recipe {canonical}",
        "",
        f"- model: {model}",
        f"- hardware: {hardware}",
        f"- framework_name: {framework_name}",
        f"- best_throughput: {attrs['best_throughput']}",
        f"- validated_gain_pct: {attrs['validated_gain_pct']}",
        f"- best_config_args: {args}",
    ]
    content = "---\n" + _emit_yaml(frontmatter) + "\n---\n\n" + "\n".join(body_lines) + "\n"
    return slug, content


def mirror_recipe(recipe: Mapping[str, Any], mcp: _GbrainMcp | None) -> bool:
    """Best-effort mirror of ONE recipe dict into gbrain (read cache).

    Returns True when a page was written, False when skipped (no
    ``canonical_id``, strict-gate rejection, no mcp) or on a transport
    error. Never raises — the local write is authoritative and a gbrain
    hiccup must not affect it.

    Args:
        recipe: The recipe dict to mirror.
        mcp: The gbrain MCP client, or ``None`` to skip mirroring.

    Returns:
        ``True`` when a page was written, ``False`` when skipped or on error.
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
    """Build a write-side gbrain MCP client from env (background timeout).

    Returns:
        A configured :class:`_GbrainMcp`, or ``None`` when ``GBRAIN_BASE_URL``
        / ``GBRAIN_TOKEN`` are not set.
    """
    base_url = (os.environ.get("GBRAIN_BASE_URL", "") or "").strip()
    token = (os.environ.get("GBRAIN_TOKEN", "") or "").strip()
    if not base_url or not token:
        return None
    from hyperloom.inference_optimizer import recipe_snapshot_constants as C

    return _GbrainMcp(base_url, token, C.DEFAULT_HTTP_TIMEOUT_SEC)


class GbrainMirroringRecipeKB:
    """Wrap a :class:`recipe_kb.RecipeKB` so a local ``put_recipe`` also
    mirrors the recipe into gbrain (the read cache), best-effort.

    Preserves the local-first contract: the local write is authoritative and
    runs first; the gbrain mirror is a post-write side-effect that never blocks
    or fails the local result. Every other call delegates to the wrapped
    dispatcher unchanged.
    """

    def __init__(self, inner: Any, mcp: _GbrainMcp | None) -> None:
        """Wrap an inner dispatcher with a gbrain mirroring side-effect.

        Args:
            inner: The wrapped recipe dispatcher to delegate to.
            mcp: Optional gbrain MCP client used for mirroring writes.
        """
        self._inner = inner
        self._mcp = mcp

    def put_recipe(self, **kwargs: Any) -> Any:
        """Write a recipe locally, then best-effort mirror it to gbrain.

        The local write is authoritative; mirroring failures are logged
        and swallowed so they never block the local result.

        Args:
            **kwargs: Recipe fields forwarded to the inner dispatcher.

        Returns:
            The result of the wrapped ``put_recipe`` call.
        """
        result = self._inner.put_recipe(**kwargs)
        try:
            mirror_recipe(kwargs, self._mcp)
        except Exception as exc:  # noqa: BLE001 - never break the local write
            log.warning("gbrain mirror skipped: %r", exc)
        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to the wrapped dispatcher.

        Args:
            name: Attribute name to resolve on the inner dispatcher.

        Returns:
            The corresponding attribute from the wrapped dispatcher.
        """
        return getattr(self._inner, name)


__all__ = [
    "recipe_to_page",
    "mirror_recipe",
    "build_mirror_mcp_from_env",
    "GbrainMirroringRecipeKB",
]
