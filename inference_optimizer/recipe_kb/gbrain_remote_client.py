# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Gbrain-backed read-only remote for the recipe-snapshot KB.

A drop-in alternative to :class:`recipe_kb.RemoteRecipeClient`: it
exposes the identical read surface (``health`` / ``get_recipe`` /
``get_history`` / ``list_recent`` / ``search`` / ``list_attempts`` /
``list_session_attempts`` / ``session_summary`` / ``close`` plus the
``enabled`` attribute) so :class:`recipe_kb.RecipeKB` can hold it in
the ``remote`` slot without any dispatcher change. The corpus is the
gbrain page store (JSON-RPC MCP over HTTP) instead of the central
cortex kb-service.

Design fit: this honours the recipe_kb local-first contract verbatim —
writes still go local-only through the dispatcher; gbrain is consulted
for READS only.

Schema adaptation (read-time): gbrain stores recipes as better-landing
pages (``type: recipe`` + ``tags: model:/gpu:/framework:`` + flat
``attrs``). On read we re-derive the 5-tuple identity
(``inference:model:hardware:framework:framework_version:precision``)
from the page attrs, fall back to the ``unknown_*`` default slugs for
dimensions gbrain never recorded, and project the page into the unified
nested KB-interface envelope. The recipe corpus is small
(tens-to-hundreds of rows), so ``search`` lists the ``recipe`` type and
filters client-side, which sidesteps slug-scheme differences between
gbrain tags and the canonical id.

Wire shape returned by every read is the SAME nested KB-interface
envelope (``labels`` / ``body`` / ``metrics`` / ``findings`` /
``failures`` / ``gaps`` / ``lessons`` / ``pitfalls``) the cortex
kb-service emits, so the :class:`recipe_kb.RecipeKB` dispatcher runs a
single ``_v2_to_arbor`` translation regardless of which backend served
the row. The gbrain page store is the adapter's private storage detail.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Mapping

from .. import recipe_snapshot_constants as C
from ..compat.payload_aliases import CANONICAL_KEY
from .canonical_id import recipe_canonical_id
from .remote_client import RemoteRecipeClientError

log = logging.getLogger(__name__)

# Nested env-map key used across the warm-start consumers
# (``coordinator._maybe_enqueue_warm_replay`` / ``_inject_warm_recipe_history``)
# and the authoritative local recipe shape (``_build_recipe_payload``). The
# gbrain read surface mirrors this shape so a round-tripped champion config is
# consumable verbatim by warm-replay.
_ENVS_KEY = "extra_envs"

# Listing the whole recipe type is cheap (the corpus is small) but we
# still cap the scan so a runaway corpus can never block the main loop.
_RECIPE_SCAN_CAP = 500
_LIST_PAGE_SIZE = 100


class GbrainRemoteError(RemoteRecipeClientError):
    """Raised on an unrecoverable gbrain MCP interaction.

    Subclasses :class:`RemoteRecipeClientError` so the ``RecipeKB``
    dispatcher's existing ``except RemoteRecipeClientError`` fall-through
    catches a gbrain transport / bad-envelope / JSON-RPC error and
    degrades to the local store — without it, a gbrain network blip would
    bubble straight up the warm-start path and could lose the local
    recipe a fall-through would have surfaced.
    """

    def __init__(self, message: str, *, category: str = "transport", **kwargs: Any) -> None:
        super().__init__(message, category=category, **kwargs)


class _GbrainMcp:
    """Minimal JSON-RPC-over-HTTP MCP client (list_pages / get_page)."""

    def __init__(self, base_url: str, token: str, timeout_sec: float) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = max(0.5, float(timeout_sec))

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        envelope = {
            "jsonrpc": "2.0", "id": "1", "method": "tools/call",
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
                raw = resp.read().decode()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise GbrainRemoteError(f"gbrain {tool} transport error: {exc!r}") from exc
        try:
            if raw.lstrip().startswith("{"):
                obj = json.loads(raw)
            else:  # text/event-stream framing
                data_lines = [l[5:] for l in raw.splitlines() if l.startswith("data:")]
                obj = json.loads(data_lines[0]) if data_lines else {}
        except (json.JSONDecodeError, IndexError) as exc:
            raise GbrainRemoteError(f"gbrain {tool} bad envelope: {exc!r}") from exc
        if not isinstance(obj, dict):
            raise GbrainRemoteError(f"gbrain {tool} unexpected envelope type: {type(obj).__name__}")
        # JSON-RPC transport-level error envelope. Without surfacing this a
        # failed put_page / list_pages would parse as an empty success —
        # ingest/mirror would over-count "ingested" and health() would
        # falsely report healthy.
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
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _json_list(value: Any) -> list[Any]:
    """Decode a recipe-page list field stored as a JSON string.

    The ingest/mirror writer (``gbrain_ingest.recipe_to_page``) encodes
    structured list-of-dict fields (``what_failed`` / ``pitfalls`` /
    ``lessons`` / ...) as JSON strings because the minimal YAML emitter
    only renders scalar lists. Tolerates an already-decoded list (in case
    a future page stores them natively) and degrades to ``[]`` on absence
    or malformed content so a bad page never breaks warm-start.
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

    gbrain recipe pages carry the champion config as a flat
    ``best_config_args`` string (+ optional ``best_config_envs`` dict).
    We surface it in the SAME shape the authoritative local recipe
    (``coordinator._build_recipe_payload``) and the warm-start consumers
    (``_maybe_enqueue_warm_replay``) use: the launch args under the
    canonical ``extra_server_args`` key and the env map NESTED under
    ``extra_envs`` (not flattened as sibling keys). A flat env map would
    be invisible to the consumer's nested ``best_config["extra_envs"]``
    read and the high-confidence warm recipe would be skipped as
    ``best_config_empty``.
    """
    out: dict[str, Any] = {}
    args = str(attrs.get("best_config_args") or "").strip()
    if args:
        out[CANONICAL_KEY] = args
    envs = attrs.get("best_config_envs")
    if isinstance(envs, Mapping) and envs:
        out[_ENVS_KEY] = {str(key): str(val) for key, val in envs.items()}
    return out


def _page_to_recipe(frontmatter: Mapping[str, Any]) -> dict[str, Any] | None:
    """Adapt a gbrain recipe page's frontmatter to the unified KB shape.

    Returns the nested KB-interface envelope (``labels`` / ``body`` /
    ``metrics`` / ``findings`` / ``failures`` / ``gaps`` / ``lessons`` /
    ``pitfalls``) so the ``RecipeKB`` dispatcher runs the single
    :func:`recipe_kb.dispatcher._v2_to_arbor` translation on it — exactly
    like the cortex kb-service. gbrain's page store is the adapter's
    private storage detail; the dispatcher never sees the flat page shape.

    Returns ``None`` when the page lacks the minimum identity (model /
    hardware) needed to build a canonical id.
    """
    attrs = frontmatter.get("attrs") if isinstance(frontmatter.get("attrs"), Mapping) else {}
    model = str(attrs.get("model") or "").strip()
    hardware = str(attrs.get("hardware") or "").strip()
    if not model or not hardware:
        return None
    framework = str(attrs.get("framework") or "").strip()
    framework_version = str(attrs.get("framework_version") or "").strip()
    precision = str(attrs.get("precision") or "").strip()
    canonical = recipe_canonical_id(
        model=model, hardware=hardware, framework=framework,
        framework_version=framework_version, precision=precision,
    )
    throughput = _as_float(
        attrs.get("best_throughput") or attrs.get("output_throughput")
    )
    validated_gain_pct = _as_float(attrs.get("validated_gain_pct"))
    return {
        C.F_CANONICAL_ID: canonical,
        C.F_VERSION: 1,
        "created_at": str(frontmatter.get("created_at") or ""),
        "updated_at": str(frontmatter.get("updated_at") or ""),
        # Slug-clean 5-tuple identity. ``_labels_match`` runs both sides
        # through ``canonical_labels`` so the raw page model string and
        # the slugged label converge; we surface the canonical labels so
        # the dispatcher's ``_v2_to_arbor`` reads identity from one place.
        "labels": C.canonical_labels(
            model=model, hardware=hardware, framework=framework,
            framework_version=framework_version, precision=precision,
        ),
        "body": {
            "best_config": _best_config_from_attrs(attrs),
            "best_throughput": throughput,
            "stack_fingerprint": dict(attrs.get("stack_fingerprint") or {}),
            "sessions": [],
            "last_profiled": "",
            "prs_tested": [],
        },
        "metrics": {
            "throughput": throughput,
            # gbrain-only signal surfaced for consumers that want it.
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
        C.F_PROVENANCE: dict(frontmatter.get("provenance") or {}),
    }


def _labels_match(recipe: Mapping[str, Any], label_match: Mapping[str, Any]) -> bool:
    """True when every provided label equals the recipe's slugged value.

    The recipe carries already-slugged identity under ``labels`` (set by
    :func:`_page_to_recipe`); the caller's ``label_match`` may be a raw or
    slugged value, so we re-run it through ``canonical_labels`` to make
    the comparison slug-normalized (``Qwen/Qwen3`` vs ``qwen3`` converge).
    """
    if not label_match:
        return True
    recipe_labels = recipe.get("labels")
    if not isinstance(recipe_labels, Mapping):
        recipe_labels = {}
    want = C.canonical_labels(
        model=str(
            label_match.get(C.F_LABEL_MODEL, "")
            or recipe_labels.get(C.F_LABEL_MODEL, "")
        ),
        hardware=str(
            label_match.get(C.F_LABEL_HARDWARE, "")
            or recipe_labels.get(C.F_LABEL_HARDWARE, "")
        ),
        framework=str(
            label_match.get(C.F_LABEL_FRAMEWORK, "")
            or recipe_labels.get(C.F_LABEL_FRAMEWORK, "")
        ),
        framework_version=str(
            label_match.get(C.F_LABEL_FRAMEWORK_VERSION, "")
            or recipe_labels.get(C.F_LABEL_FRAMEWORK_VERSION, "")
        ),
        precision=str(
            label_match.get(C.F_LABEL_PRECISION, "")
            or recipe_labels.get(C.F_LABEL_PRECISION, "")
        ),
    )
    # Only compare the dimensions the caller actually constrained.
    for key in label_match:
        if key in recipe_labels and recipe_labels[key] != want.get(key):
            return False
    return True


class GbrainRemoteRecipeClient:
    """Read-only recipe-snapshot client backed by gbrain.

    Duck-types :class:`recipe_kb.RemoteRecipeClient`. Constructed by
    ``cli._build_recipe_kb_dispatcher`` when ``RECIPE_KB_REMOTE=gbrain``.
    Every read returns the unified nested KB-interface envelope (same
    shape as the cortex kb-service) so the ``RecipeKB`` dispatcher runs a
    single translation regardless of which backend served the row.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        enabled: bool = True,
        timeout_sec: float | None = None,
    ) -> None:
        self.base_url = (base_url or "").strip()
        self.token = (token or "").strip()
        # Foreground-friendly default: gbrain is a read side-channel, so
        # never block the main loop longer than the recipe_kb foreground
        # budget.
        self.timeout_sec = (
            float(timeout_sec) if timeout_sec is not None
            else C.FOREGROUND_HTTP_TIMEOUT_SEC
        )
        self.enabled = bool(enabled and self.base_url and self.token)
        self._mcp = _GbrainMcp(self.base_url, self.token, self.timeout_sec) if self.enabled else None

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:  # symmetry with RemoteRecipeClient
        self._mcp = None

    def health(self) -> bool:
        if not self.enabled or self._mcp is None:
            return False
        try:
            self._mcp.call("list_pages", {"type": "recipe", "limit": 1})
            return True
        except GbrainRemoteError as exc:
            log.info("gbrain-remote health probe failed: %s", exc)
            return False

    # -- internal scan -----------------------------------------------------
    def _scan_recipes(self, *, limit: int) -> list[dict[str, Any]]:
        """List recipe pages (newest first) and project each to a Recipe dict."""
        if not self.enabled or self._mcp is None:
            return []
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        cap = min(int(limit) if limit and limit > 0 else _RECIPE_SCAN_CAP, _RECIPE_SCAN_CAP)
        pages = 0
        while len(out) < cap and pages < (_RECIPE_SCAN_CAP // _LIST_PAGE_SIZE + 2):
            params: dict[str, Any] = {
                "type": "recipe", "limit": _LIST_PAGE_SIZE, "sort": "updated_desc",
            }
            if cursor:
                params["updated_before"] = cursor
            batch = self._mcp.call("list_pages", params)
            if not isinstance(batch, list) or not batch:
                break
            for entry in batch:
                slug = entry.get("slug") if isinstance(entry, dict) else None
                if not slug:
                    continue
                page = self._mcp.call("get_page", {"slug": slug})
                fm = page.get("frontmatter") if isinstance(page, dict) else None
                if not isinstance(fm, Mapping):
                    continue
                recipe = _page_to_recipe(fm)
                if recipe is not None:
                    out.append(recipe)
                    if len(out) >= cap:
                        break
            pages += 1
            if len(batch) < _LIST_PAGE_SIZE:
                break
            last = batch[-1].get("updated_at") if isinstance(batch[-1], dict) else None
            if not last or last == cursor:
                break
            cursor = last
        return out

    # -- read surface (mirrors RemoteRecipeClient) -------------------------
    def get_recipe(
        self, *, canonical_id: str, version: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the recipe for ``canonical_id`` (top label-match), or None.

        gbrain keeps no per-version archive, so ``version`` is accepted
        for interface parity but only ``version in (None, 1)`` can match.
        """
        if not self.enabled:
            return None
        if not canonical_id:
            raise ValueError("get_recipe requires a non-empty canonical_id")
        if version is not None and int(version) != 1:
            return None
        try:
            from .canonical_id import cid_to_path_components
            model, hardware, framework, framework_version, precision = cid_to_path_components(
                canonical_id
            )
        except Exception:  # noqa: BLE001 - malformed id -> remote miss
            return None
        label_match = {
            C.F_LABEL_MODEL: model, C.F_LABEL_HARDWARE: hardware,
            C.F_LABEL_FRAMEWORK: framework,
            C.F_LABEL_FRAMEWORK_VERSION: framework_version,
            C.F_LABEL_PRECISION: precision,
        }
        rows = self.search(label_match=label_match, limit=1)
        return rows[0] if rows else None

    def get_history(self, *, canonical_id: str, limit: int = 100) -> list[dict[str, Any]]:
        # gbrain does not retain a versioned recipe archive.
        return []

    def list_recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        return self._scan_recipes(limit=limit)

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

        ``prefer`` (workload-similarity hints) is accepted for the
        unified KB-interface signature; the dispatcher applies the
        client-side rerank over the normalized rows, so this adapter
        only honours the ``required`` (``label_match`` / metric) filter.
        """
        del prefer  # client-side rerank lives in RecipeKB
        if not self.enabled:
            return []
        candidates = self._scan_recipes(limit=_RECIPE_SCAN_CAP)
        rows = [r for r in candidates if _labels_match(r, label_match or {})]
        if updated_since:
            rows = [r for r in rows if str(r.get("updated_at") or "") >= str(updated_since)]
        if metric_filters:
            rows = [r for r in rows if _passes_metric_filters(r, metric_filters)]
        # _scan_recipes already returns updated_desc; honour the ASC asks.
        if order_by in (C.ORDER_BY_UPDATED_AT_ASC, C.ORDER_BY_CREATED_AT_ASC):
            rows = list(reversed(rows))
        return rows[: int(limit)] if limit and limit > 0 else rows

    def list_attempts(self, *, canonical_id: str, limit: int = 100) -> list[dict[str, Any]]:
        # Attempt rows live on local store under this design; gbrain
        # absence falls the dispatcher through to local.
        return []

    def list_session_attempts(self, *, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return []

    def session_summary(self, *, session_id: str) -> dict[str, Any] | None:
        return None


def _passes_metric_filters(recipe: Mapping[str, Any], metric_filters: Mapping[str, Any]) -> bool:
    """Apply ``{metric: {min,max}}`` filters against the recipe's metrics.

    The recipe is the nested KB-interface envelope, so throughput lives
    under ``metrics.throughput`` / ``body.best_throughput``. We accept
    both the ``throughput`` and the ``best_throughput`` filter aliases.
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
        base_url=base_url, token=token, enabled=True, timeout_sec=timeout_sec,
    )


__all__ = [
    "GbrainRemoteRecipeClient",
    "GbrainRemoteError",
    "build_gbrain_remote_from_env",
]
