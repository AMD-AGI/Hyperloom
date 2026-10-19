# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local-only dispatcher for the on-disk Recipe KB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .local_store import RecipeFilters


log = logging.getLogger(__name__)


def _prefer_score(row: dict[str, Any], prefer: dict[str, Any]) -> int:
    """Count matching workload hints for stable local search reranking."""

    score = 0
    for key, wanted in prefer.items():
        if wanted in (None, "", 0):
            continue
        actual = row.get(key)
        if isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            try:
                score += int(float(actual) == float(wanted))
            except (TypeError, ValueError):
                continue
        elif str(actual or "").strip().lower() == str(wanted).strip().lower():
            score += 1
    return score


@dataclass(frozen=True)
class _ReadRequest:
    """What a read asked for, as the audit event's request block records it."""

    canonical_id: str = ""
    label_match: dict[str, Any] | None = None
    prefer: dict[str, Any] | None = None


@dataclass
class RecipeKB:
    """Facade over one :class:`LocalRecipeStore`."""

    local: Any
    audit_hook: Any = None
    mode: str = "local"
    backend_name: str = "local-json"
    knowledge_config: Any = None

    def _backend_label(self) -> str:
        return self.backend_name or "local-json"

    def _emit_audit(self, event: dict[str, Any]) -> None:
        hook = self.audit_hook
        if not callable(hook):
            return
        try:
            hook(event)
        except Exception:  # caller-supplied hook; audit must not break the KB call
            log.debug("recipe_kb: audit_hook raised", exc_info=True)

    def _read_event(
        self,
        method: str,
        row: dict[str, Any] | None,
        request: _ReadRequest = _ReadRequest(),
        *,
        candidates: int = 0,
    ) -> dict[str, Any]:
        result = None
        if isinstance(row, dict):
            result = {
                "canonical_id": str(row.get("canonical_id") or ""),
                "exact": bool(request.canonical_id and str(row.get("canonical_id") or "") == request.canonical_id),
                "best_throughput": float(row.get("best_throughput") or 0.0),
                "best_config_nonempty": bool(row.get("best_config")),
            }
        return {
            "op": "read",
            "method": method,
            "mode": "local",
            "backend": self._backend_label(),
            "remote": "none",
            "resolution": "local",
            "hit": row is not None,
            "candidates": int(candidates),
            "request": {
                "canonical_id": request.canonical_id or None,
                "prefer_keys": sorted((request.prefer or {}).keys()),
                "label_match": request.label_match or None,
            },
            "result": result,
            "provenance": {
                "component": "recipe_kb",
                "backend": self._backend_label(),
            },
        }

    def put_recipe(self, **kwargs: Any) -> dict[str, Any]:
        """Write one local Recipe row and emit a best-effort audit event."""

        canonical_id = str(kwargs.get("canonical_id") or "")
        try:
            result = self.local.put_recipe(**kwargs)
        except Exception as exc:
            self._emit_audit(
                {
                    "op": "write",
                    "method": "put_recipe",
                    "mode": "local",
                    "backend": self._backend_label(),
                    "remote": "none",
                    "resolution": "local_error",
                    "success": False,
                    "hit": False,
                    "request": {"canonical_id": canonical_id or None},
                    "error": {"type": type(exc).__name__},
                    "provenance": {"component": "recipe_kb"},
                }
            )
            raise

        provenance = kwargs.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        details = provenance.get("details")
        details = details if isinstance(details, dict) else {}
        counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
        prior = result.get("prior_counts") if isinstance(result.get("prior_counts"), dict) else {}
        self._emit_audit(
            {
                "op": "write",
                "method": "put_recipe",
                "mode": "local",
                "backend": self._backend_label(),
                "remote": "none",
                "resolution": "local_write",
                "success": True,
                "hit": True,
                "generator": str(provenance.get("generator") or ""),
                "phase": str(details.get("phase") or ""),
                "request": {"canonical_id": canonical_id or None},
                "identity": {
                    key: str(kwargs.get(key) or "")
                    for key in (
                        "model",
                        "hardware",
                        "framework_name",
                        "framework_version",
                        "precision",
                    )
                },
                "result": {
                    "canonical_id": str(result.get("canonical_id") or canonical_id),
                    "version": int(result.get("version") or 0),
                    "created": bool(result.get("created")),
                    "best_throughput": float(kwargs.get("best_throughput") or 0.0),
                    "best_config_nonempty": bool(kwargs.get("best_config")),
                    "write_safety": {},
                },
                "counts": {key: int(value) for key, value in counts.items()},
                "delta": {key: int(value) - int(prior.get(key, 0) or 0) for key, value in counts.items()},
                "provenance": {
                    "component": "recipe_kb",
                    "generator": str(provenance.get("generator") or ""),
                },
            }
        )
        return result

    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
        prefer: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Read one exact local Recipe row."""

        row = self.local.get_recipe(canonical_id=canonical_id, version=version)
        self._emit_audit(self._read_event("get_recipe", row, _ReadRequest(canonical_id=canonical_id, prefer=prefer)))
        return row

    def get_authoritative_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Read the exact authoritative local row."""

        row = self.local.get_recipe(canonical_id=canonical_id, version=version)
        self._emit_audit(self._read_event("get_authoritative_recipe", row, _ReadRequest(canonical_id=canonical_id)))
        return row

    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        limit: int = 50,
        prefer: dict[str, Any] | None = None,
        filters: RecipeFilters = RecipeFilters(),
    ) -> list[dict[str, Any]]:
        """Search local Recipe rows and optionally rerank by workload hints."""

        rows = self.local.search(label_match=label_match, limit=limit, filters=filters)
        if prefer:
            rows = sorted(
                rows,
                key=lambda row: _prefer_score(row, prefer),
                reverse=True,
            )
        self._emit_audit(
            self._read_event(
                "search",
                rows[0] if rows else None,
                _ReadRequest(label_match=label_match, prefer=prefer),
                candidates=len(rows),
            )
        )
        return rows

    def close(self) -> None:
        """Close the local store when it exposes a transport."""

        close = getattr(self.local, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            log.exception("recipe_kb: local store close raised")


__all__ = ["RecipeKB"]
