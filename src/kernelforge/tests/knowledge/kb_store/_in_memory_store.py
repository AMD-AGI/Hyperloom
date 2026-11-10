# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An in-memory KB Store and the remote configuration that reaches it, shared by the KB store, rewrite and loop tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from kernelforge.config import Config
from kernelforge.config import KnowledgeConfig
from kernelforge.knowledge.kb_store import record_store


class InMemoryKBStore:
    """The subset of the KB Store surface the rewrite records use."""

    def __init__(self, *_args, **_kwargs):
        self.knowledge: dict[tuple[str, str], dict] = {}
        self.files: dict[tuple[str, str], dict[str, bytes]] = {}
        self.champions: dict[str, dict] = {}
        self.order: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []

    def get_rollup(self, canonical_id):
        sessions = [
            {"session_id": session_id, "updated_at": f"{index:04d}"}
            for index, (identity, session_id) in enumerate(self.order)
            if identity == canonical_id
        ]
        if not sessions and canonical_id not in self.champions:
            return None
        return {"sessions": sessions, "champion": self.champions.get(canonical_id, {})}

    def search_identities(
        self,
        *,
        scheme,
        match=None,
        offset=0,
        limit=50,
    ):
        names = (
            "producer",
            "kernel_name",
            "framework",
            "framework_version",
            "backend",
            "gpu",
        )
        found = []
        for canonical_id, _session_id in self.order:
            parts = canonical_id.split(":")
            if len(parts) != 7 or parts[0] != scheme:
                continue
            dimensions = dict(zip(names, parts[1:]))
            if any(dimensions.get(key) != value for key, value in (match or {}).items()):
                continue
            if canonical_id not in {item["canonical_id"] for item in found}:
                found.append(
                    {
                        "canonical_id": canonical_id,
                        "dimensions": dimensions,
                        "updated_at": f"{len(found):04d}",
                    }
                )
        page = found[offset : offset + limit]
        next_offset = offset + len(page)
        return {
            "items": page,
            "total": len(found),
            "next_offset": next_offset if next_offset < len(found) else None,
        }

    def get_top_sessions(
        self,
        canonical_id,
        *,
        metric="speedup",
        limit=3,
        offset=0,
    ):
        champion_id = str(self.champions.get(canonical_id, {}).get("session_id") or "")
        ranked = []
        for index, (identity, candidate_session_id) in enumerate(self.order):
            if identity != canonical_id:
                continue
            score = self.knowledge[(identity, candidate_session_id)].get(metric)
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                continue
            ranked.append(
                {
                    "session_id": candidate_session_id,
                    "score": float(score),
                    "updated_at": f"{index:04d}",
                    "is_champion": candidate_session_id == champion_id,
                }
            )
        ranked.sort(
            key=lambda item: (
                -item["score"],
                -int(item["updated_at"]),
                item["session_id"],
            )
        )
        return {"sessions": ranked[offset : offset + limit]}

    def get_session(self, canonical_id, session_id):
        knowledge = self.knowledge.get((canonical_id, session_id))
        return (
            None
            if knowledge is None
            else {
                "canonical_id": canonical_id,
                "session_id": session_id,
                "knowledge": knowledge,
            }
        )

    def list_session_files(self, canonical_id, session_id, *, kind=""):
        del kind
        return {
            "files": [
                {
                    "path": rel_path,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content),
                    "download_url": f"memory://{rel_path}",
                }
                for rel_path, content in self.files.get((canonical_id, session_id), {}).items()
            ]
        }

    def put_knowledge(self, canonical_id, knowledge, *, session_id="", mode="merge"):
        # Mirrors the SDK: "merge" shallow-merges over the stored section and "replace" overwrites it.
        if mode not in ("merge", "replace"):
            raise record_store.KBStoreError(f"mode must be 'merge' or 'replace', got {mode!r}")
        key = (canonical_id, session_id)
        if key not in self.knowledge:
            self.order.append(key)
        if mode == "merge":
            merged = dict(self.knowledge.get(key) or {})
            merged.update(knowledge)
            self.knowledge[key] = merged
        else:
            self.knowledge[key] = dict(knowledge)
        return {"session_id": session_id, "mode": mode}

    def put_file(self, canonical_id, session_id, rel_path, local_path, *, kind="other", meta=None):
        self.files.setdefault((canonical_id, session_id), {})[rel_path] = Path(local_path).read_bytes()
        return f"kb://{canonical_id}/{session_id}/{rel_path}"

    def download_session(self, canonical_id, session_id, destination, *, include_values=True):
        del include_values
        self.downloads.append((canonical_id, session_id))
        root = Path(destination) / "files"
        root.mkdir(parents=True, exist_ok=True)
        for rel_path, content in self.files.get((canonical_id, session_id), {}).items():
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def set_champion(self, canonical_id, session_id, *, metric="throughput", value=0.0):
        self.champions[canonical_id] = {
            "session_id": session_id,
            "metric": metric,
            "value": value,
        }
        return {}


def _remote_config(tmp_path):
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="remote",
        local_root=tmp_path / "knowledge",
        kb_store_url="http://in-memory",
        kb_store_token="token",
    )
    return Config.from_env(
        workspace=str(tmp_path),
        gpu_target="gfx950",
        gpu_type="mi355x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )


def _use_in_memory_kb_store(monkeypatch):
    store = InMemoryKBStore()
    monkeypatch.setattr(record_store, "KBStoreClient", lambda *a, **k: store)
    return store
