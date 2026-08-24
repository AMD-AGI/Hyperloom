# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local-only dispatcher for the on-disk Recipe KB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


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
        except Exception:  # noqa: BLE001 - audit must never break Recipe KB
            log.debug("recipe_kb: audit_hook raised", exc_info=True)

    def _read_event(
        self,
        *,
        method: str,
        row: dict[str, Any] | None,
        canonical_id: str = "",
        label_match: dict[str, Any] | None = None,
        prefer: dict[str, Any] | None = None,
        candidates: int = 0,
    ) -> dict[str, Any]:
        result = None
        if isinstance(row, dict):
            result = {
                "canonical_id": str(row.get("canonical_id") or ""),
                "exact": bool(canonical_id and str(row.get("canonical_id") or "") == canonical_id),
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
                "canonical_id": canonical_id or None,
                "prefer_keys": sorted((prefer or {}).keys()),
                "label_match": label_match or None,
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
        self._emit_audit(
            self._read_event(
                method="get_recipe",
                row=row,
                canonical_id=canonical_id,
                prefer=prefer,
            )
        )
        return row

    def get_authoritative_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Read the exact authoritative local row."""

        row = self.local.get_recipe(canonical_id=canonical_id, version=version)
        self._emit_audit(
            self._read_event(
                method="get_authoritative_recipe",
                row=row,
                canonical_id=canonical_id,
            )
        )
        return row

    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        metric_filters: dict[str, Any] | None = None,
        updated_since: str | None = None,
        order_by: str = "updated_at DESC",
        limit: int = 50,
        prefer: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search local Recipe rows and optionally rerank by workload hints."""

        rows = self.local.search(
            label_match=label_match,
            metric_filters=metric_filters,
            updated_since=updated_since,
            order_by=order_by,
            limit=limit,
        )
        if prefer:
            rows = sorted(
                rows,
                key=lambda row: _prefer_score(row, prefer),
                reverse=True,
            )
        self._emit_audit(
            self._read_event(
                method="search",
                row=rows[0] if rows else None,
                label_match=label_match,
                prefer=prefer,
                candidates=len(rows),
            )
        )
        return rows

    def append_attempt(self, **kwargs: Any) -> dict[str, Any]:
        """Append one local attempt row."""

        return self.local.append_attempt(**kwargs)

    def list_attempts(
        self,
        *,
        canonical_id: str,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List local attempts, optionally filtered by session id."""

        return self.local.list_attempts(
            canonical_id=canonical_id,
            session_id=session_id,
        )

    def close(self) -> None:
        """Close the local store when it exposes a transport."""

        close = getattr(self.local, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:  # noqa: BLE001 - lifecycle cleanup is best-effort
            log.exception("recipe_kb: local store close raised")


__all__ = ["RecipeKB"]
