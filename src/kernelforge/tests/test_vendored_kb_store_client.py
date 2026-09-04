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
    assert digest == "0ed8fff30f4d45826c198d21c367e933408efade"
