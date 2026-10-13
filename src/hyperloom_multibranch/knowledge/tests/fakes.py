# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An in-memory Recipe KB Store with the calls the multi-branch write-back and warm start make."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FakeRecipeStore:
    """Holds one partition's champion; records every upload, knowledge write and promotion."""

    champion: dict[str, Any] | None = None
    uploads: dict[str, bytes] = field(default_factory=dict)
    knowledge: dict[str, Any] | None = None
    objective_schema: str = ""
    promoted: tuple[str, str, float] | None = None
    scopes: list[dict[str, Any] | None] = field(default_factory=list)

    def get_rollup(self, canonical_id: str, *, scope: dict[str, Any] | None = None) -> dict[str, Any] | None:
        self.scopes.append(scope)
        if self.champion is None:
            return None
        return {"champion": self.champion, "sessions": [{"session_id": self.champion["session_id"]}]}

    def put_dir(self, canonical_id: str, session_id: str, local_dir: str | Path, **_: Any) -> dict[str, str]:
        root = Path(local_dir)
        self.uploads = {
            path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
        }
        return {rel: f"kb://{canonical_id}/{session_id}/{rel}" for rel in self.uploads}

    def put_knowledge(
        self,
        canonical_id: str,
        knowledge: dict[str, Any],
        *,
        session_id: str = "",
        mode: str = "merge",
        scope: dict[str, Any] | None = None,
        objective_schema: str = "",
    ) -> dict[str, Any]:
        self.knowledge, self.objective_schema = knowledge, objective_schema
        return {"ok": True}

    def set_champion(
        self, canonical_id: str, session_id: str, *, metric: str, value: float, scope: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.promoted = (session_id, metric, value)
        return {"ok": True}
