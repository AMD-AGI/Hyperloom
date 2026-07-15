# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Gbrain-backed read-only remote for the recipe-snapshot KB.

The sole read-side remote for the recipe KB. It exposes ``get_recipe``,
``search``, ``close``, and the ``enabled`` attribute so
:class:`recipe_kb.RecipeKB` can hold it in the ``remote`` slot. The corpus is
the gbrain page store (JSON-RPC MCP over HTTP). Honours the recipe_kb
local-first contract: writes go local-only through the dispatcher; gbrain is
consulted for READS only.

Schema adaptation (read-time): gbrain stores recipes under
``hyperloom-recipe-kb/`` by default (overridable via
``GBRAIN_RECIPE_SLUG_PREFIX``) as better-landing pages (``type: recipe`` +
``tags: model:/gpu:/framework_name:`` + flat ``attrs``). On read we re-derive
the 7-tuple identity from the page attrs, fall back to the ``unknown_*``
default slugs for missing dimensions, and project the page into the unified
nested KB-interface envelope. ``search`` lists the ``recipe`` type and filters
client-side, restricted to the configured slug prefix.

Every read returns the unified nested KB-interface envelope (``labels`` /
``body`` / ``metrics`` / ``findings`` / ``failures`` / ``gaps`` / ``lessons``
/ ``pitfalls``), so the dispatcher runs a single ``_v2_to_arbor`` translation.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from hyperloom.inference_optimizer import recipe_snapshot_constants as C
from .canonical_id import recipe_canonical_id
from .remote_client import RemoteRecipeClientError

log = logging.getLogger(__name__)

# Nested env-map key mirroring the warm-start consumers and the local recipe
# shape, so a round-tripped champion config is consumable verbatim.
_ENVS_KEY = "extra_envs"

# Listing the whole recipe type is a fallback path; prefer exact slug reads and
# cache broad scans so one warm-start tick does not issue thousands of MCP
# get_page calls.
_RECIPE_SCAN_CAP = 25000
_LIST_PAGE_SIZE = 100
_SCAN_CACHE_TTL_SEC = 60.0
_DEFAULT_RECIPE_SLUG_PREFIX = "hyperloom-recipe-kb"
_RECIPE_SLUG_PREFIX_ENV = "GBRAIN_RECIPE_SLUG_PREFIX"
_EXTRA_SERVER_ARGS_KEY = "extra_server_args"

# Wall-clock budget for one full ``_scan_recipes`` pass. The scan is
# best-effort: when the budget is exceeded it returns whatever it has gathered
# rather than blocking the foreground loop. Override via
# ``GBRAIN_RECIPE_SCAN_BUDGET_SEC``.
_RECIPE_SCAN_BUDGET_SEC = 20.0


class GbrainRemoteError(RemoteRecipeClientError):
    """Raised on an unrecoverable gbrain MCP interaction.

    Subclasses :class:`RemoteRecipeClientError` so the dispatcher's
    ``except RemoteRecipeClientError`` fall-through catches a gbrain transport /
    bad-envelope / JSON-RPC error and degrades to the local store.
    """

    def __init__(self, message: str, *, category: str = "transport", **kwargs: Any) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error message.
            category: Error category (defaults to ``transport``).
            **kwargs: Additional fields forwarded to the base exception.
        """
        super().__init__(message, category=category, **kwargs)


def _iter_sse_objects(raw: str):
    """Yield JSON objects decoded from an MCP HTTP response body.

    Handles the three framings the gbrain ``/mcp`` endpoint uses:

    * A plain JSON body (the whole payload is one object).
    * A single ``text/event-stream`` event whose ``data`` field is split
      across multiple ``data:`` lines (joined with ``\\n`` per the SSE spec,
      one optional leading space after the colon stripped).
    * Multiple SSE events in one response (e.g. a heartbeat before the result
      event). Each event is decoded independently so a non-result event can
      never corrupt the result parse.

    Malformed / incomplete events are skipped, so this is safe to call on a
    partially-read buffer.
    """
    import re

    text = raw.lstrip()
    if text.startswith("{") or text.startswith("["):
        try:
            yield json.loads(text)
        except json.JSONDecodeError:
            pass
        return
    for block in re.split(r"\r?\n\r?\n", raw):
        parts: list[str] = []
        for line in block.splitlines():
            if line.startswith("data:"):
                seg = line[5:]
                parts.append(seg[1:] if seg.startswith(" ") else seg)
        payload = "\n".join(parts).strip()
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


_BARE_RESULT_TOOLS = {
    "get_links",
    "get_backlinks",
    "traverse_graph",
}


def _select_mcp_response(raw: str, want_id: Any = None, *, allow_bare_result: bool = False) -> Any:
    """Pick the MCP response object from an MCP response body.

    Prefers the event whose ``id`` matches ``want_id`` (the request id); falls
    back to the first object carrying a ``result`` or ``error`` member. Some
    native gbrain graph tools return the tool result directly (for example a
    bare edge list) instead of wrapping it in a JSON-RPC envelope; those are
    accepted only when ``allow_bare_result`` is set.
    """
    fallback: Any = None
    bare: Any = None
    for obj in _iter_sse_objects(raw):
        if isinstance(obj, dict):
            if want_id is not None and obj.get("id") == want_id:
                return obj
            if fallback is None and ("result" in obj or "error" in obj):
                fallback = obj
            elif allow_bare_result and bare is None:
                bare = obj
        elif allow_bare_result and bare is None:
            bare = obj
    return fallback if fallback is not None else bare


class _GbrainMcp:
    """Minimal JSON-RPC-over-HTTP MCP client (list_pages / get_page)."""

    # Request id stamped on every envelope; used to select the matching
    # response event out of a multi-event SSE stream.
    _RPC_ID = "1"

    def __init__(self, base_url: str, token: str, timeout_sec: float) -> None:
        """Initialize the MCP client.

        Args:
            base_url: Base URL of the gbrain MCP endpoint.
            token: Bearer token for authorization.
            timeout_sec: Per-request timeout in seconds (floored at 0.5).
        """
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = max(0.5, float(timeout_sec))

    def _read_body(self, resp: Any, *, allow_bare_result: bool = False) -> str:
        """Read the MCP response body without blocking on a non-closing stream.

        gbrain's ``/mcp`` endpoint answers with ``text/event-stream`` framing.
        Some proxies keep the TCP connection open after the final line instead
        of sending EOF, so a plain read-to-EOF would block indefinitely. This
        reads incrementally under a hard wall-clock deadline and returns as soon
        as a complete SSE ``data:`` line is available. Non-SSE JSON bodies with
        a ``Content-Length`` are still read whole; the deadline only caps the
        pathological open-stream case.

        Args:
            resp: The open ``http.client.HTTPResponse`` from ``urlopen``.

        Returns:
            The decoded response body text.
        """
        deadline = time.monotonic() + max(0.5, float(self._timeout))
        chunks: list[bytes] = []
        buf = b""
        # Non-SSE JSON with a declared length: read it whole.
        ctype = ""
        clen = ""
        try:
            headers = resp.headers
            ctype = (headers.get("Content-Type") or "").lower()
            clen = headers.get("Content-Length") or ""
        except Exception:  # noqa: BLE001 — header access is best-effort
            ctype = ""
            clen = ""
        if "text/event-stream" not in ctype and clen:
            return resp.read().decode()
        readline = getattr(resp, "readline", None)
        if "text/event-stream" in ctype and callable(readline):
            # SSE is line-framed; fixed-size reads can block on an open stream
            # after the final short chunk and return a truncated JSON event.
            while True:
                if time.monotonic() >= deadline:
                    break
                try:
                    piece = readline()
                except (OSError, ValueError):
                    break
                if not piece:
                    break
                chunks.append(piece)
                buf += piece
                if (
                    _select_mcp_response(
                        buf.decode("utf-8", "replace"),
                        self._RPC_ID,
                        allow_bare_result=allow_bare_result,
                    )
                    is not None
                ):
                    break
            return b"".join(chunks).decode("utf-8", "replace")
        while True:
            if time.monotonic() >= deadline:
                break
            try:
                piece = resp.read(4096)
            except TypeError:
                # Simple stand-ins (and some response shims) expose a
                # zero-argument ``read()`` that returns the whole body.
                piece = resp.read()
            except (OSError, ValueError):
                break
            if not piece:
                break
            chunks.append(piece)
            buf += piece
            # Stop as soon as the buffer holds a complete JSON-RPC response
            # matching our request id (partial/heartbeat events keep reading;
            # a non-closing stream never hangs).
            if (
                _select_mcp_response(
                    buf.decode("utf-8", "replace"),
                    self._RPC_ID,
                    allow_bare_result=allow_bare_result,
                )
                is not None
            ):
                break
        return b"".join(chunks).decode("utf-8", "replace")

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke an MCP tool over JSON-RPC and return its decoded result.

        Args:
            tool: MCP tool name to call.
            arguments: Tool arguments.

        Returns:
            The decoded tool result: parsed JSON content when present, else the
            raw content text, else the result object.

        Raises:
            GbrainRemoteError: On transport failures, malformed envelopes, or
                JSON-RPC / tool-level errors.
        """
        envelope = {
            "jsonrpc": "2.0",
            "id": self._RPC_ID,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        req = urllib.request.Request(
            self._base + "/mcp",
            data=json.dumps(envelope).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {self._token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = self._read_body(resp, allow_bare_result=tool in _BARE_RESULT_TOOLS)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise GbrainRemoteError(f"gbrain {tool} transport error: {exc!r}") from exc
        # Select the JSON-RPC response event matching our request id.
        obj = _select_mcp_response(raw, self._RPC_ID, allow_bare_result=tool in _BARE_RESULT_TOOLS)
        if obj is None:
            prefix = raw[:300].replace("\n", "\\n").replace("\r", "\\r")
            raise GbrainRemoteError(
                f"gbrain {tool} bad envelope: no parseable JSON-RPC response in body; body_prefix={prefix!r}"
            )
        if not isinstance(obj, dict):
            return obj
        if tool in _BARE_RESULT_TOOLS and "result" not in obj and "error" not in obj:
            return obj
        # JSON-RPC transport-level error envelope; surfacing it prevents a
        # failed call from parsing as an empty success.
        if obj.get("error") is not None:
            raise GbrainRemoteError(f"gbrain {tool} JSON-RPC error: {obj.get('error')!r}")
        result = obj.get("result") or {}
        # MCP tool-level error (``tools/call`` result with ``isError``):
        # the tool ran but reported failure in-band.
        if isinstance(result, dict) and result.get("isError"):
            raise GbrainRemoteError(f"gbrain {tool} tool error: {result.get('content')!r}")
        content = result.get("content") or []
        if content and isinstance(content[0], dict) and content[0].get("text"):
            try:
                return json.loads(content[0]["text"])
            except json.JSONDecodeError:
                return content[0]["text"]
        return result


def _as_float(value: Any) -> float:
    """Coerce a value to float, defaulting to ``0.0`` on failure.

    Args:
        value: Value to convert.

    Returns:
        The parsed float, or ``0.0`` when conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _json_list(value: Any) -> list[Any]:
    """Decode a recipe-page list field stored as a JSON string.

    The mirror writer encodes structured list-of-dict fields as JSON strings
    because the minimal YAML emitter only renders scalar lists. Tolerates an
    already-decoded list and degrades to ``[]`` on absence or malformed content.

    Args:
        value: A list, a JSON-encoded list string, or anything else.

    Returns:
        The decoded list, or ``[]`` when the value is absent or malformed.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _best_config_from_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Project gbrain recipe attrs into the canonical ``best_config`` dict.

    gbrain recipe pages carry the champion config as a flat ``best_config_args``
    string (+ optional ``best_config_envs`` dict). We surface it in the shape
    the local recipe and warm-start consumers use: launch args under the
    canonical ``extra_server_args`` key and the env map NESTED under
    ``extra_envs`` (a flat env map would be invisible to the consumer's nested
    read).

    Args:
        attrs: The flat gbrain recipe page attrs mapping.

    Returns:
        The canonical ``best_config`` dict with launch args under the
        canonical key and the env map nested under ``extra_envs``.
    """
    out: dict[str, Any] = {}
    args = str(attrs.get("best_config_args") or "").strip()
    if args:
        out[_EXTRA_SERVER_ARGS_KEY] = args
    envs = attrs.get("best_config_envs")
    if isinstance(envs, Mapping) and envs:
        out[_ENVS_KEY] = {str(key): str(val) for key, val in envs.items()}
    return out


def _session_entries_from_attrs(attrs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build traceability session entries from session-sourced gbrain attrs.

    Args:
        attrs: The flat gbrain attrs mapping.

    Returns:
        A list of ``{"session_id": ...}`` entries, preserving the page's
        contributing session ids. Falls back to the canonical ``session_id``
        when ``session_ids`` is absent.
    """
    raw_ids = attrs.get("session_ids")
    ids: list[str] = []
    if isinstance(raw_ids, list):
        ids = [str(x).strip() for x in raw_ids if str(x or "").strip()]
    sid = str(attrs.get("session_id") or "").strip()
    if sid and sid not in ids:
        ids.insert(0, sid)
    return [{"session_id": sid_val} for sid_val in ids]


def _provenance_from_attrs(frontmatter: Mapping[str, Any], attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Merge page provenance with session-sourced traceability metadata."""
    provenance = dict(frontmatter.get("provenance") or {})
    provenance.setdefault("source", _recipe_slug_prefix())
    sid = str(attrs.get("session_id") or "").strip()
    if sid:
        provenance["session_id"] = sid
    raw_ids = attrs.get("session_ids")
    if isinstance(raw_ids, list):
        session_ids = [str(x).strip() for x in raw_ids if str(x or "").strip()]
        if session_ids:
            provenance["session_ids"] = session_ids
    return provenance


def _page_to_recipe(frontmatter: Mapping[str, Any]) -> dict[str, Any] | None:
    """Adapt a gbrain recipe page's frontmatter to the unified KB shape.

    Returns the nested KB-interface envelope so the dispatcher runs the single
    :func:`recipe_kb.dispatcher._v2_to_arbor` translation on it. Returns
    ``None`` when the page lacks the minimum identity (model / hardware) needed
    to build a canonical id.

    Args:
        frontmatter: The gbrain recipe page frontmatter mapping.

    Returns:
        The nested KB-interface recipe envelope, or ``None`` when the page
        lacks the model/hardware identity needed for a canonical id.
    """
    attrs = frontmatter.get("attrs") if isinstance(frontmatter.get("attrs"), Mapping) else {}
    model = str(attrs.get("model") or "").strip()
    hardware = str(attrs.get("hardware") or "").strip()
    if not model or not hardware:
        return None
    # Back-compat: pages predating the framework_name rename use ``framework``.
    framework_name = str(attrs.get("framework_name") or attrs.get("framework") or "").strip()
    framework_version = str(attrs.get("framework_version") or "").strip()
    precision = str(attrs.get("precision") or "").strip()
    model_type = str(attrs.get("model_type") or "").strip()
    architectures = attrs.get("architectures") or []
    canonical = recipe_canonical_id(
        model=model,
        hardware=hardware,
        framework_name=framework_name,
        model_type=model_type,
        architectures=architectures,
        framework_version=framework_version,
        precision=precision,
    )
    throughput = _as_float(attrs.get("best_throughput") or attrs.get("output_throughput"))
    validated_gain_pct = _as_float(attrs.get("validated_gain_pct"))
    return {
        C.F_CANONICAL_ID: canonical,
        C.F_VERSION: 1,
        "created_at": str(frontmatter.get("created_at") or ""),
        "updated_at": str(frontmatter.get("updated_at") or ""),
        # Slug-clean 7-tuple identity; ``_labels_match`` normalizes both sides
        # so the dispatcher's ``_v2_to_arbor`` reads identity from one place.
        "labels": C.canonical_labels(
            model=model,
            hardware=hardware,
            framework_name=framework_name,
            model_type=model_type,
            architectures=architectures,
            framework_version=framework_version,
            precision=precision,
        ),
        "body": {
            "best_config": _best_config_from_attrs(attrs),
            "best_throughput": throughput,
            "stack_fingerprint": dict(attrs.get("stack_fingerprint") or {}),
            "sessions": _session_entries_from_attrs(attrs),
            "last_profiled": "",
            "prs_tested": _json_list(attrs.get("prs_tested")),
        },
        "metrics": {
            "throughput": throughput,
            "validated_gain_pct": validated_gain_pct,
        },
        "findings": _json_list(attrs.get("what_worked")),
        "failures": _json_list(attrs.get("what_failed")),
        "gaps": _json_list(attrs.get("remaining_gaps")),
        "pitfalls": _json_list(attrs.get("pitfalls")),
        "lessons": _json_list(attrs.get("lessons")),
        C.F_AUTHORITY: str(frontmatter.get("authority") or C.AUTHORITY_EXPERIENTIAL),
        C.F_CONFIDENCE: _as_float(frontmatter.get("confidence")) or C.DEFAULT_CONFIDENCE,
        C.F_EVIDENCE_REFS: list(frontmatter.get("evidence_refs") or []),
        C.F_PROVENANCE: _provenance_from_attrs(frontmatter, attrs),
    }


def _labels_match(recipe: Mapping[str, Any], label_match: Mapping[str, Any]) -> bool:
    """True when every provided label matches the recipe's value.

    Scalar labels require equality; ``architectures`` uses *contains* semantics
    (the recipe's list must include all queried architectures). Both sides are
    run through ``canonical_labels`` so raw and slugged values converge.

    Args:
        recipe: The candidate recipe carrying slugged ``labels``.
        label_match: The constraining labels (raw or slugged); empty matches
            everything.

    Returns:
        ``True`` when every constrained label equals the recipe's slugged
        value.
    """
    if not label_match:
        return True
    recipe_labels = recipe.get("labels")
    if not isinstance(recipe_labels, Mapping):
        recipe_labels = {}
    want = C.canonical_labels(
        model=str(label_match.get(C.F_LABEL_MODEL, "") or recipe_labels.get(C.F_LABEL_MODEL, "")),
        hardware=str(label_match.get(C.F_LABEL_HARDWARE, "") or recipe_labels.get(C.F_LABEL_HARDWARE, "")),
        framework_name=str(label_match.get(C.F_LABEL_FRAMEWORK_NAME, "") or recipe_labels.get(C.F_LABEL_FRAMEWORK_NAME, "")),
        framework_version=str(
            label_match.get(C.F_LABEL_FRAMEWORK_VERSION, "") or recipe_labels.get(C.F_LABEL_FRAMEWORK_VERSION, "")
        ),
        precision=str(label_match.get(C.F_LABEL_PRECISION, "") or recipe_labels.get(C.F_LABEL_PRECISION, "")),
        model_type=str(label_match.get(C.F_LABEL_MODEL_TYPE, "") or recipe_labels.get(C.F_LABEL_MODEL_TYPE, "")),
        architectures=str(
            label_match.get(C.F_LABEL_ARCHITECTURES, "") or recipe_labels.get(C.F_LABEL_ARCHITECTURES, "")
        ),
    )
    # Only compare the dimensions the caller actually constrained.
    for key in label_match:
        if key == C.F_LABEL_ARCHITECTURES:
            # Contains: recipe architectures must include all queried slugs.
            recipe_arch = str(recipe_labels.get(key, "")).lower()
            query_arch = str(want.get(key, "")).lower()
            if query_arch and query_arch not in ("unknown_arch", ""):
                query_parts = set(query_arch.split("+"))
                recipe_parts = set(recipe_arch.split("+"))
                if not query_parts.issubset(recipe_parts):
                    return False
        elif key in recipe_labels and recipe_labels[key] != want.get(key):
            return False
    return True


def _recipe_slug_prefix() -> str:
    """Return the configured gbrain recipe page slug prefix."""
    raw = os.environ.get(_RECIPE_SLUG_PREFIX_ENV, "").strip().strip("/")
    return raw or _DEFAULT_RECIPE_SLUG_PREFIX


def _is_session_recipe_slug(slug: str) -> bool:
    """Return whether a gbrain slug belongs to the configured recipe KB."""
    return str(slug or "").startswith(_recipe_slug_prefix() + "/")


def _slug_for_canonical(canonical_id: str) -> str:
    """Build the configured gbrain slug for a canonical_id."""
    return _recipe_slug_prefix() + "/" + canonical_id.replace(":", "/")


class GbrainRemoteRecipeClient:
    """Read-only recipe-snapshot client backed by gbrain.

    The read-side ``remote`` for :class:`recipe_kb.RecipeKB`, constructed when
    ``GBRAIN_*`` is configured. Every read returns the unified nested
    KB-interface envelope so the dispatcher runs a single translation.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        enabled: bool = True,
        timeout_sec: float | None = None,
    ) -> None:
        """Initialize the gbrain-backed recipe client.

        Args:
            base_url: Base URL of the gbrain endpoint.
            token: Bearer token for authorization.
            enabled: Whether the client is active (also requires a URL/token).
            timeout_sec: Per-request timeout; defaults to the foreground HTTP
                budget when ``None``.
        """
        self.base_url = (base_url or "").strip()
        self.token = (token or "").strip()
        # Foreground-friendly default: never block the main loop longer than
        # the recipe_kb foreground budget.
        self.timeout_sec = float(timeout_sec) if timeout_sec is not None else C.FOREGROUND_HTTP_TIMEOUT_SEC
        self.enabled = bool(enabled and self.base_url and self.token)
        self._mcp = _GbrainMcp(self.base_url, self.token, self.timeout_sec) if self.enabled else None
        self._scan_cache: list[dict[str, Any]] | None = None
        self._scan_cache_ts = 0.0
        self._scan_cache_complete = False

    def close(self) -> None:
        """Release the underlying MCP client."""
        self._mcp = None

    def _scan_cache_ttl(self) -> float:
        """Return the recipe-scan cache TTL in seconds.

        Returns:
            The value from ``GBRAIN_RECIPE_SCAN_TTL_SEC`` when valid, otherwise
            the default TTL.
        """
        raw = os.environ.get("GBRAIN_RECIPE_SCAN_TTL_SEC", "").strip()
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                log.warning("invalid GBRAIN_RECIPE_SCAN_TTL_SEC=%r; using default", raw)
        return _SCAN_CACHE_TTL_SEC

    def _scan_budget(self) -> float:
        """Return the wall-clock budget (seconds) for one full recipe scan.

        ``0`` disables the budget (unbounded scan). Override via
        ``GBRAIN_RECIPE_SCAN_BUDGET_SEC``.

        Returns:
            The value from ``GBRAIN_RECIPE_SCAN_BUDGET_SEC`` when valid,
            otherwise the default budget.
        """
        raw = os.environ.get("GBRAIN_RECIPE_SCAN_BUDGET_SEC", "").strip()
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                log.warning("invalid GBRAIN_RECIPE_SCAN_BUDGET_SEC=%r; using default", raw)
        return _RECIPE_SCAN_BUDGET_SEC

    def _get_page_recipe(self, slug: str) -> dict[str, Any] | None:
        """Fetch one gbrain recipe page by slug and project it.

        Args:
            slug: The gbrain page slug to fetch.

        Returns:
            The projected recipe envelope, or ``None`` when disabled, missing,
            or lacking frontmatter.
        """
        if not self.enabled or self._mcp is None:
            return None
        page = self._mcp.call("get_page", {"slug": slug})
        fm = page.get("frontmatter") if isinstance(page, dict) else None
        if not isinstance(fm, Mapping):
            return None
        return _page_to_recipe(fm)

    def _search_recipe_candidates(self, label_match: Mapping[str, Any], *, limit: int) -> list[dict[str, Any]]:
        """Use gbrain page search to preselect recipe candidates by labels.

        Cheaper than a full corpus scan; results are still verified with
        ``_labels_match`` (the search query is only a prefilter).
        """
        if not self.enabled or self._mcp is None or not label_match:
            return []
        terms: list[str] = []
        for key in (
            C.F_LABEL_MODEL,
            C.F_LABEL_HARDWARE,
            C.F_LABEL_FRAMEWORK_NAME,
            C.F_LABEL_FRAMEWORK_VERSION,
            C.F_LABEL_PRECISION,
            C.F_LABEL_MODEL_TYPE,
            C.F_LABEL_ARCHITECTURES,
        ):
            value = str(label_match.get(key) or "").strip()
            if value and not value.startswith("unknown_") and value not in terms:
                terms.append(value)
        if not terms:
            return []
        try:
            raw = self._mcp.call(
                "search",
                {
                    "query": " ".join(terms),
                    "limit": min(max(int(limit or 1) * 10, _LIST_PAGE_SIZE), _RECIPE_SCAN_CAP),
                },
            )
        except GbrainRemoteError:
            return []
        hits: Any
        if isinstance(raw, dict):
            hits = raw.get("results") or raw.get("pages") or raw.get("hits") or []
        elif isinstance(raw, list):
            hits = raw
        else:
            hits = []
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            if not isinstance(hit, Mapping):
                continue
            slug = str(hit.get("slug") or hit.get("id") or "")
            if not slug or not _is_session_recipe_slug(slug) or slug in seen:
                continue
            seen.add(slug)
            recipe = self._get_page_recipe(slug)
            if recipe is not None and _labels_match(recipe, label_match):
                out.append(recipe)
                if limit and len(out) >= int(limit):
                    break
        out.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=True)
        return out

    def _scan_recipes(self, *, limit: int) -> list[dict[str, Any]]:
        """List recipe pages and project each to a Recipe dict.

        Uses forward pagination (``updated_asc`` + ``updated_after``) since the
        reverse cursor can terminate early when many pages share the same
        ``updated_at``. Dedups by slug across page boundaries and returns
        newest-first.

        Args:
            limit: Maximum number of recipes to return (capped at the internal
                scan cap).

        Returns:
            Newest-first projected recipe dicts, or ``[]`` when disabled.
        """
        if not self.enabled or self._mcp is None:
            return []
        now = time.monotonic()
        ttl = self._scan_cache_ttl()
        if self._scan_cache is not None and ttl > 0.0 and now - self._scan_cache_ts <= ttl:
            # Complete and non-empty partial scans are cached within the TTL to
            # avoid repeating the slow first-page scan; empty partials are not.
            if self._scan_cache_complete or self._scan_cache:
                return list(self._scan_cache[: int(limit) if limit and limit > 0 else len(self._scan_cache)])
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_slugs: set[str] = set()
        cap = min(int(limit) if limit and limit > 0 else _RECIPE_SCAN_CAP, _RECIPE_SCAN_CAP)
        pages = 0
        # Best-effort wall-clock budget: when exceeded, return what we have
        # rather than blocking the foreground loop. A budget hit is not cached
        # so a later tick can complete the scan.
        budget = self._scan_budget()
        deadline = time.monotonic() + budget if budget > 0.0 else None
        budget_hit = False
        while len(out) < cap:
            if deadline is not None and time.monotonic() >= deadline:
                budget_hit = True
                break
            params: dict[str, Any] = {
                "type": "recipe",
                "limit": _LIST_PAGE_SIZE,
                "sort": "updated_asc",
            }
            if cursor:
                params["updated_after"] = cursor
            batch = self._mcp.call("list_pages", params)
            if not isinstance(batch, list) or not batch:
                break
            new_slugs = 0
            for entry in batch:
                if deadline is not None and time.monotonic() >= deadline:
                    budget_hit = True
                    break
                slug = entry.get("slug") if isinstance(entry, dict) else None
                if not slug or not _is_session_recipe_slug(str(slug)) or slug in seen_slugs:
                    continue
                seen_slugs.add(slug)
                new_slugs += 1
                recipe = self._get_page_recipe(str(slug))
                if recipe is not None:
                    out.append(recipe)
                    if len(out) >= cap:
                        break
            pages += 1
            if budget_hit:
                break
            if len(batch) < _LIST_PAGE_SIZE:
                break
            last = batch[-1].get("updated_at") if isinstance(batch[-1], dict) else None
            if not last or (last == cursor and new_slugs == 0):
                break
            cursor = last
        if budget_hit:
            log.warning(
                "gbrain recipe scan exceeded %.1fs budget after %d page(s)/%d recipe(s); "
                "returning partial results (warm-start is best-effort)",
                budget,
                pages,
                len(out),
            )
        out.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=True)
        if budget_hit:
            if out:
                self._scan_cache = list(out)
                self._scan_cache_ts = time.monotonic()
                self._scan_cache_complete = False
        else:
            self._scan_cache = list(out)
            self._scan_cache_ts = time.monotonic()
            self._scan_cache_complete = True
        return out

    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the recipe for ``canonical_id`` (top label-match), or None.

        gbrain keeps no per-version archive, so ``version`` is accepted
        for interface parity but only ``version in (None, 1)`` can match.

        Args:
            canonical_id: The canonical recipe id to look up.
            version: Optional version; only ``None`` or ``1`` can match.

        Returns:
            The matching recipe envelope, or ``None`` on miss / disabled /
            non-matching version.

        Raises:
            ValueError: If ``canonical_id`` is empty.
        """
        if not self.enabled:
            return None
        if not canonical_id:
            raise ValueError("get_recipe requires a non-empty canonical_id")
        if version is not None and int(version) != 1:
            return None
        try:
            from .canonical_id import cid_to_path_components

            model, hardware, framework_name, model_type, architectures, framework_version, precision = (
                cid_to_path_components(canonical_id)
            )
        except Exception:  # noqa: BLE001 - malformed id -> remote miss
            return None
        label_match = {
            C.F_LABEL_MODEL: model,
            C.F_LABEL_HARDWARE: hardware,
            C.F_LABEL_FRAMEWORK_NAME: framework_name,
            C.F_LABEL_FRAMEWORK_VERSION: framework_version,
            C.F_LABEL_PRECISION: precision,
            C.F_LABEL_MODEL_TYPE: model_type,
            C.F_LABEL_ARCHITECTURES: architectures,
        }
        # Fast path: recipe slugs are the canonical id with ':' as path
        # separators, so exact reads skip the full corpus scan.
        direct = self._get_page_recipe(_slug_for_canonical(canonical_id))
        if direct is not None and direct.get(C.F_CANONICAL_ID) == canonical_id:
            return direct
        if framework_version != C.DEFAULT_FRAMEWORK_VERSION_SLUG:
            # Older sessions lacking framework_version were mirrored under
            # unknown_version; fall back there before a corpus scan.
            unknown_version_id = recipe_canonical_id(
                model=model,
                hardware=hardware,
                framework_name=framework_name,
                model_type=model_type,
                architectures=architectures,
                framework_version=C.DEFAULT_FRAMEWORK_VERSION_SLUG,
                precision=precision,
            )
            direct = self._get_page_recipe(_slug_for_canonical(unknown_version_id))
            relaxed_labels = {
                C.F_LABEL_MODEL: model,
                C.F_LABEL_HARDWARE: hardware,
                C.F_LABEL_FRAMEWORK_NAME: framework_name,
                C.F_LABEL_PRECISION: precision,
                C.F_LABEL_MODEL_TYPE: model_type,
                C.F_LABEL_ARCHITECTURES: architectures,
            }
            if direct is not None and _labels_match(direct, relaxed_labels):
                return direct
        rows = self.search(label_match=label_match, limit=1)
        return rows[0] if rows else None

    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        metric_filters: dict[str, Any] | None = None,
        updated_since: str | None = None,
        order_by: str = C.ORDER_BY_UPDATED_AT_DESC,
        limit: int = 50,
        prefer: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Client-side filtered recipe search over the gbrain corpus.

        ``prefer`` is accepted for signature parity; the dispatcher applies the
        rerank, so this adapter only honours the ``required`` (``label_match`` /
        metric) filter.

        Args:
            label_match: Identity labels to filter on (empty matches all).
            metric_filters: ``{metric: {min, max}}`` bounds to apply.
            updated_since: Keep only rows with ``updated_at`` at or after this.
            order_by: Ordering key; ASC variants reverse the default newest-
                first order.
            limit: Maximum number of rows to return.
            prefer: Accepted for signature parity; ignored here (rerank lives
                in the dispatcher).

        Returns:
            The filtered, ordered recipe rows, or ``[]`` when disabled.
        """
        del prefer  # client-side rerank lives in RecipeKB
        if not self.enabled:
            return []
        rows: list[dict[str, Any]] = []
        cache_satisfied = False
        if label_match and self._scan_cache is not None:
            ttl = self._scan_cache_ttl()
            cache_valid = ttl > 0.0 and time.monotonic() - self._scan_cache_ts <= ttl
            if cache_valid and (self._scan_cache_complete or self._scan_cache):
                rows = [r for r in self._scan_cache if _labels_match(r, label_match or {})]
                cache_satisfied = bool(rows) or self._scan_cache_complete
        if not cache_satisfied:
            rows = self._search_recipe_candidates(label_match or {}, limit=int(limit or 0)) if label_match else []
        if not cache_satisfied and (not rows or (limit and len(rows) < int(limit))):
            candidates = self._scan_recipes(limit=_RECIPE_SCAN_CAP)
            seen = {str(r.get(C.F_CANONICAL_ID) or "") for r in rows}
            rows.extend(
                r
                for r in candidates
                if str(r.get(C.F_CANONICAL_ID) or "") not in seen and _labels_match(r, label_match or {})
            )
        if updated_since:
            rows = [r for r in rows if str(r.get("updated_at") or "") >= str(updated_since)]
        if metric_filters:
            rows = [r for r in rows if _passes_metric_filters(r, metric_filters)]
        # _scan_recipes returns updated_desc; reverse for ASC asks.
        if order_by in (C.ORDER_BY_UPDATED_AT_ASC, C.ORDER_BY_CREATED_AT_ASC):
            rows = list(reversed(rows))
        return rows[: int(limit)] if limit and limit > 0 else rows

def _passes_metric_filters(recipe: Mapping[str, Any], metric_filters: Mapping[str, Any]) -> bool:
    """Apply ``{metric: {min,max}}`` filters against the recipe's metrics.

    Throughput lives under ``metrics.throughput`` / ``body.best_throughput``;
    both the ``throughput`` and ``best_throughput`` filter aliases are accepted.

    Args:
        recipe: The nested KB-interface recipe envelope.
        metric_filters: ``{metric: {min, max}}`` bounds to apply.

    Returns:
        ``True`` when the recipe satisfies every metric bound.
    """
    metrics = recipe.get("metrics") if isinstance(recipe.get("metrics"), Mapping) else {}
    body = recipe.get("body") if isinstance(recipe.get("body"), Mapping) else {}
    for metric, bounds in (metric_filters or {}).items():
        val = metrics.get(metric)
        if val is None and metric in ("best_throughput", "throughput"):
            val = body.get("best_throughput") or metrics.get("throughput")
        if val is None:
            return False
        fval = _as_float(val)
        if isinstance(bounds, Mapping):
            lo = bounds.get(C.F_METRIC_MIN)
            hi = bounds.get(C.F_METRIC_MAX)
            if lo is not None and fval < _as_float(lo):
                return False
            if hi is not None and fval > _as_float(hi):
                return False
    return True


def build_gbrain_remote_from_env() -> GbrainRemoteRecipeClient | None:
    """Construct a client from ``GBRAIN_BASE_URL`` / ``GBRAIN_TOKEN``.

    Returns ``None`` when the env is not configured so the caller can
    fall back to local-only or the cortex remote.

    Returns:
        A configured :class:`GbrainRemoteRecipeClient`, or ``None`` when the
        base URL / token env vars are not set.
    """
    base_url = (os.environ.get("GBRAIN_BASE_URL", "") or "").strip()
    token = (os.environ.get("GBRAIN_TOKEN", "") or "").strip()
    if not base_url or not token:
        return None
    timeout_env = os.environ.get("GBRAIN_HTTP_TIMEOUT_SEC")
    timeout_sec: float | None = None
    if timeout_env:
        try:
            timeout_sec = float(timeout_env)
        except ValueError:
            timeout_sec = None
    return GbrainRemoteRecipeClient(
        base_url=base_url,
        token=token,
        enabled=True,
        timeout_sec=timeout_sec,
    )


__all__ = [
    "GbrainRemoteRecipeClient",
    "GbrainRemoteError",
    "build_gbrain_remote_from_env",
]
