# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract guard for the vendored upstream KB Store SDK."""

from __future__ import annotations

import hashlib
from pathlib import Path

from kernelforge.knowledge.remote_exp import kb_store_client


def test_vendored_sdk_matches_upstream_git_blob() -> None:
    content = Path(kb_store_client.__file__).read_bytes()
    digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    assert digest == "8e60b4798a819dbdabdc4c57aa073466f1834c5e"


def test_identity_search_uses_the_kb_store_discovery_route(monkeypatch) -> None:
    client = kb_store_client.KBStoreClient("https://kb.example", "token")
    captured = {}

    def request(method, path, payload=None):
        captured.update(method=method, path=path, payload=payload)
        return {"items": []}

    monkeypatch.setattr(client, "_request", request)

    assert client.search_identities(
        scheme="kernel",
        match={"producer": "forge-loop", "backend": "triton"},
        offset=10,
        limit=20,
    ) == {"items": []}
    assert captured == {
        "method": "POST",
        "path": "/v1/kb/search",
        "payload": {
            "scheme": "kernel",
            "match": {"producer": "forge-loop", "backend": "triton"},
            "offset": 10,
            "limit": 20,
        },
    }
