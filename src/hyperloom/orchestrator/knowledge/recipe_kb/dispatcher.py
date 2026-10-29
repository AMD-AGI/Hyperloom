# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local-only dispatcher for the on-disk Recipe KB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from .local_store import RecipeUpsert


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ReadRequest:
    """What a read asked for, as the audit event's request block records it."""

    canonical_id: str = ""
    label_match: dict[str, Any] | None = None


@dataclass
class RecipeKB:
    """Facade over one :class:`LocalRecipeStore`."""

    mode: ClassVar[str] = "local"
    backend_name: ClassVar[str] = "local-json"

    local: Any
    audit_hook: Any = None
    knowledge_config: Any = None

    def _emit_audit(self, event: dict[str, Any]) -> None:
        hook = self.audit_hook
        if not callable(hook):
            return
        try:
            hook(event)
        except OSError:  # caller-supplied hook; audit must not break the KB call
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
            "backend": self.backend_name,
            "remote": "none",
            "resolution": "local",
            "hit": row is not None,
            "candidates": int(candidates),
            "request": {
                "canonical_id": request.canonical_id or None,
                "label_match": request.label_match or None,
            },
            "result": result,
            "provenance": {
                "component": "recipe_kb",
                "backend": self.backend_name,
            },
        }

    def put_recipe(self, upsert: RecipeUpsert) -> dict[str, Any]:
        """Write one local Recipe row and emit a best-effort audit event."""

        canonical_id = str(upsert.canonical_id or "")
        try:
            result = self.local.put_recipe(upsert)
        except Exception as exc:
            self._emit_audit(
                {
                    "op": "write",
                    "method": "put_recipe",
                    "mode": "local",
                    "backend": self.backend_name,
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

        provenance = upsert.provenance
        provenance = provenance if isinstance(provenance, dict) else {}
        details = provenance.get("details")
        details = details if isinstance(details, dict) else {}
        counts = result["counts"]
        prior = result["prior_counts"]
        self._emit_audit(
            {
                "op": "write",
                "method": "put_recipe",
                "mode": "local",
                "backend": self.backend_name,
                "remote": "none",
                "resolution": "local_write",
                "success": True,
                "hit": True,
                "generator": str(provenance.get("generator") or ""),
                "phase": str(details.get("phase") or ""),
                "request": {"canonical_id": canonical_id or None},
                "identity": {
                    "model": str(upsert.model or ""),
                    "hardware": str(upsert.hardware or ""),
                    "framework_name": str(upsert.framework_name or ""),
                    "framework_version": str(upsert.framework_version or ""),
                    "precision": str(upsert.precision or ""),
                },
                "result": {
                    "canonical_id": result["canonical_id"],
                    "version": result["version"],
                    "created": result["created"],
                    "best_throughput": float(upsert.best_throughput),
                    "best_config_nonempty": bool(upsert.best_config),
                    "write_safety": {},
                },
                "counts": {key: int(value) for key, value in counts.items()},
                "delta": {key: int(value) - prior[key] for key, value in counts.items()},
                "provenance": {
                    "component": "recipe_kb",
                    "generator": str(provenance.get("generator") or ""),
                },
            }
        )
        return result

    def get_recipe(self, *, canonical_id: str) -> dict[str, Any] | None:
        """Read one exact local Recipe row."""

        row = self.local.get_recipe(canonical_id=canonical_id)
        self._emit_audit(self._read_event("get_recipe", row, _ReadRequest(canonical_id=canonical_id)))
        return row

    def get_authoritative_recipe(self, *, canonical_id: str) -> dict[str, Any] | None:
        """Read the exact authoritative local row."""

        row = self.local.get_recipe(canonical_id=canonical_id)
        self._emit_audit(self._read_event("get_authoritative_recipe", row, _ReadRequest(canonical_id=canonical_id)))
        return row

    def search(self, *, label_match: dict[str, Any] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Search local Recipe rows, newest first."""

        rows = self.local.search(label_match=label_match, limit=limit)
        self._emit_audit(
            self._read_event(
                "search",
                rows[0] if rows else None,
                _ReadRequest(label_match=label_match),
                candidates=len(rows),
            )
        )
        return rows


__all__ = ["RecipeKB"]
