#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for apply_and_bench helpers (no serving / no GPU required).

Covers the patch-operation coverage gate (_diff_unsupported_ops) and the
measurement spread/significance helper (_spread).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"


def _load():
    spec = importlib.util.spec_from_file_location("apply_and_bench_under_test", _TOOLS_DIR / "apply_and_bench.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ab = _load()


# _diff_unsupported_ops: modify/add allowed; delete/rename/copy/mode/binary refused

_MODIFY = "diff --git a/x.cu b/x.cu\n--- a/x.cu\n+++ b/x.cu\n@@ -1 +1 @@\n-a\n+b\n"
_ADD = "diff --git a/n.cu b/n.cu\nnew file mode 100644\n--- /dev/null\n+++ b/n.cu\n@@ -0,0 +1 @@\n+z\n"


def test_unsupported_ops_allows_modify_and_add():
    assert ab._diff_unsupported_ops(_MODIFY) == []
    assert ab._diff_unsupported_ops(_ADD) == []
    # a modify + add multi-file diff is still all-clear
    assert ab._diff_unsupported_ops(_MODIFY + _ADD) == []


def test_unsupported_ops_flags_delete():
    d = "diff --git a/x.cu b/x.cu\ndeleted file mode 100644\n--- a/x.cu\n+++ /dev/null\n"
    assert ab._diff_unsupported_ops(d) == ["delete"]


def test_unsupported_ops_flags_rename_and_copy():
    assert ab._diff_unsupported_ops("diff --git a/x b/y\nrename from x\nrename to y\n") == ["rename"]
    assert ab._diff_unsupported_ops("diff --git a/x b/y\ncopy from x\ncopy to y\n") == ["copy"]


def test_unsupported_ops_flags_mode_and_binary():
    assert ab._diff_unsupported_ops("diff --git a/x b/x\nold mode 100644\nnew mode 100755\n") == ["mode-change"]
    assert ab._diff_unsupported_ops("diff --git a/x b/x\nGIT binary patch\n") == ["binary"]
    assert ab._diff_unsupported_ops("diff --git a/x b/x\nBinary files a/x and b/x differ\n") == ["binary"]


def test_unsupported_ops_dedup_and_sorted():
    mixed = (
        "diff --git a/x b/x\ndeleted file mode 100644\n"
        "diff --git a/y b/z\nrename from y\nrename to z\n"
        "diff --git a/y b/z\ndeleted file mode 100644\n"
    )
    assert ab._diff_unsupported_ops(mixed) == ["delete", "rename"]


# _spread: median + p25/p75 + stdev, None-safe


def test_spread_basic():
    s = ab._spread([10.0, 12.0, 11.0, 13.0, 9.0])
    assert s["n"] == 5
    assert s["median"] == 11.0
    assert s["p25"] is not None and s["p75"] is not None
    assert s["p25"] <= s["median"] <= s["p75"]
    assert s["stdev"] is not None and s["stdev"] > 0


def test_spread_edge_cases():
    assert ab._spread([])["median"] is None
    one = ab._spread([5.0])
    assert one["median"] == 5.0 and one["stdev"] == 0.0 and one["n"] == 1
