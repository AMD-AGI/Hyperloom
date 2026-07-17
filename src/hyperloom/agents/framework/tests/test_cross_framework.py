# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.cross_framework (cross-framework audit).

Hermetic: redirects the KB root via FRAMEWORK_AGENT_KB_DIR so the seeded
``cross_framework_map.jsonl`` and persisted ``semantic_audit.json`` never
touch a real workspace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hyperloom.agents.framework.audit as audit
import hyperloom.agents.framework.cross_framework as cf


_SGLANG_DIFF = """diff --git a/python/sglang/srt/mem_cache/radix_cache.py b/python/sglang/srt/mem_cache/radix_cache.py
--- a/python/sglang/srt/mem_cache/radix_cache.py
+++ b/python/sglang/srt/mem_cache/radix_cache.py
@@ -1,2 +1,4 @@
 class RadixCache:
+    def match_prefix(self, key):
+        return None
"""

_MAP_ROW = {
    "src_framework": "sglang",
    "dst_framework": "vllm",
    "feature": "radix_prefix_cache",
    "src_module": "python/sglang/srt/mem_cache/radix_cache.py",
    "dst_module": "vllm/core/block/prefix_caching_block.py",
    "confidence": "medium",
    "notes": "port radix prefix reuse",
}


@pytest.fixture
def kb_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KB resolution at a clean tmp_path for every test."""
    monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path))
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
    return tmp_path


def _seed_map(kb_root: Path, rows: list[dict]) -> Path:
    """Write cross_framework_map.jsonl under the active KB root."""
    d = kb_root / "framework_optimization"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "cross_framework_map.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _mk_target_module(root: Path, rel: str) -> Path:
    """Create an existing dst-framework module file under a target source root."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("class PrefixCachingBlock:\n    pass\n", encoding="utf-8")
    return p


# --- _paths_match -----------------------------------------------------------


def test_paths_match_exact_and_suffix() -> None:
    assert cf._paths_match("a/b/c.py", "a/b/c.py") is True
    assert cf._paths_match("python/sglang/srt/x.py", "sglang/srt/x.py") is True
    assert cf._paths_match("x/y.py", "z/y.py") is False


def test_paths_match_basename_only() -> None:
    assert cf._paths_match("foo.py", "foo.py") is True
    assert cf._paths_match("foo.py", "bar.py") is False


def test_paths_match_empty() -> None:
    assert cf._paths_match("", "a.py") is False
    assert cf._paths_match("a.py", "") is False


# --- load_cross_framework_map ----------------------------------------------


def test_load_map_missing_file_returns_empty(kb_root: Path) -> None:
    assert cf.load_cross_framework_map("sglang", "vllm") == []


def test_load_map_filters_by_pair(kb_root: Path) -> None:
    _seed_map(
        kb_root,
        [
            _MAP_ROW,
            {**_MAP_ROW, "src_framework": "vllm", "dst_framework": "sglang"},
        ],
    )
    out = cf.load_cross_framework_map("sglang", "vllm")
    assert len(out) == 1
    assert out[0]["feature"] == "radix_prefix_cache"


def test_load_map_case_insensitive(kb_root: Path) -> None:
    _seed_map(kb_root, [_MAP_ROW])
    assert len(cf.load_cross_framework_map("SGLang", "VLLM")) == 1


def test_load_map_tolerates_malformed_lines(kb_root: Path) -> None:
    d = kb_root / "framework_optimization"
    d.mkdir(parents=True, exist_ok=True)
    (d / "cross_framework_map.jsonl").write_text(
        json.dumps(_MAP_ROW) + "\n" + "{ not json\n" + "[1,2,3]\n",
        encoding="utf-8",
    )
    out = cf.load_cross_framework_map("sglang", "vllm")
    assert len(out) == 1


# --- run_cross_framework_audit ---------------------------------------------


def test_audit_no_patch_text(kb_root: Path) -> None:
    res = cf.run_cross_framework_audit(
        {"framework": "sglang", "target_framework": "vllm", "candidate": {"candidate_id": "c1"}}
    )
    assert res["layer"] == "cross_framework"
    assert res["semantic_status"] == "unknown"
    assert res["applicability"] == "needs_human_review"
    assert res["recommended_next_step"] == "author_via_specialist"


def test_audit_seed_missing(kb_root: Path) -> None:
    res = cf.run_cross_framework_audit(
        {
            "framework": "sglang",
            "target_framework": "vllm",
            "candidate": {"candidate_id": "c1"},
            "diff_text": _SGLANG_DIFF,
        }
    )
    assert res["semantic_status"] == "unknown"
    assert res["metrics"]["map_source"] == "missing_seed_file"


def test_audit_no_pair_match(kb_root: Path) -> None:
    _seed_map(kb_root, [{**_MAP_ROW, "src_framework": "vllm", "dst_framework": "sglang"}])
    res = cf.run_cross_framework_audit(
        {
            "framework": "sglang",
            "target_framework": "vllm",
            "candidate": {"candidate_id": "c1"},
            "diff_text": _SGLANG_DIFF,
        }
    )
    assert res["metrics"]["map_source"] == "no_pair_match"


def test_audit_mapped_target_present(kb_root: Path, tmp_path: Path) -> None:
    _seed_map(kb_root, [_MAP_ROW])
    target = tmp_path / "vllm_src"
    _mk_target_module(target, "vllm/core/block/prefix_caching_block.py")
    res = cf.run_cross_framework_audit(
        {
            "framework": "sglang",
            "target_framework": "vllm",
            "candidate": {"candidate_id": "c1"},
            "diff_text": _SGLANG_DIFF,
            "target_framework_source_roots": [str(target)],
        }
    )
    assert res["semantic_status"] == "partially_present"
    assert res["applicability"] == "needs_rewrite"
    assert res["metrics"]["mapped_files"] == 1
    assert res["metrics"]["dst_modules_present"] == 1
    assert res["metrics"]["roots_source"] == "explicit"
    assert res["recommended_next_step"] == "author_via_specialist"


def test_audit_mapped_target_absent_non_explicit_roots(kb_root: Path, tmp_path: Path) -> None:
    _seed_map(kb_root, [_MAP_ROW])
    empty_root = tmp_path / "empty_src"
    empty_root.mkdir()
    res = cf.run_cross_framework_audit(
        {
            "framework": "sglang",
            "target_framework": "vllm",
            "candidate": {"candidate_id": "c1"},
            "diff_text": _SGLANG_DIFF,
            "framework_source_roots": [str(empty_root)],
        }
    )
    assert res["semantic_status"] == "not_present"
    assert res["applicability"] == "needs_rewrite"
    assert res["metrics"]["dst_modules_present"] == 0
    assert res["metrics"]["roots_source"] == "fallback"
    assert any("roots" in str(r).lower() for r in res["risks"])


def test_audit_no_mapped_file(kb_root: Path, tmp_path: Path) -> None:
    _seed_map(kb_root, [_MAP_ROW])
    other_diff = _SGLANG_DIFF.replace("radix_cache.py", "some_unmapped_file.py")
    res = cf.run_cross_framework_audit(
        {
            "framework": "sglang",
            "target_framework": "vllm",
            "candidate": {"candidate_id": "c1"},
            "diff_text": other_diff,
            "target_framework_source_roots": [str(tmp_path)],
        }
    )
    assert res["semantic_status"] == "not_present"
    assert res["metrics"]["mapped_files"] == 0


def test_audit_use_llm_ignored_note(kb_root: Path, tmp_path: Path) -> None:
    _seed_map(kb_root, [_MAP_ROW])
    target = tmp_path / "vllm_src"
    _mk_target_module(target, "vllm/core/block/prefix_caching_block.py")
    res = cf.run_cross_framework_audit(
        {
            "framework": "sglang",
            "target_framework": "vllm",
            "candidate": {"candidate_id": "c1"},
            "diff_text": _SGLANG_DIFF,
            "target_framework_source_roots": [str(target)],
            "use_llm": True,
        }
    )
    assert any("llm" in str(r).lower() for r in res["risks"])


# --- run_phase_audit dispatch ----------------------------------------------


def test_phase_audit_dispatches_to_cross_framework(kb_root: Path, tmp_path: Path) -> None:
    _seed_map(kb_root, [_MAP_ROW])
    target = tmp_path / "vllm_src"
    _mk_target_module(target, "vllm/core/block/prefix_caching_block.py")
    work = tmp_path / "audit_work"
    res = audit.run_phase_audit(
        {
            "framework": "sglang",
            "target_framework": "vllm",
            "candidate": {"candidate_id": "c1"},
            "diff_text": _SGLANG_DIFF,
            "target_framework_source_roots": [str(target)],
            "work_dir": str(work),
        }
    )
    assert res["layer"] == "cross_framework"
    assert (work / "semantic_audit.json").is_file()


def test_phase_audit_same_framework_not_cross(kb_root: Path, tmp_path: Path) -> None:
    work = tmp_path / "audit_work2"
    res = audit.run_phase_audit(
        {
            "framework": "sglang",
            "target_framework": "sglang",
            "candidate": {"candidate_id": "c1"},
            "diff_text": _SGLANG_DIFF,
            "framework_source_roots": [str(tmp_path)],
            "work_dir": str(work),
        }
    )
    assert res.get("layer") != "cross_framework"


# --- H1: symbol-level landing ----------------------------------------------


def _mk_target_module_with_symbol(root: Path, rel: str) -> Path:
    """Create a dst module that already defines the src diff's added symbol."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("class PrefixCachingBlock:\n    def match_prefix(self, key):\n        return None\n", encoding="utf-8")
    return p


def test_h1_symbol_anchor_present(kb_root: Path, tmp_path: Path) -> None:
    _seed_map(kb_root, [_MAP_ROW])
    target = tmp_path / "vllm_src"
    _mk_target_module_with_symbol(target, "vllm/core/block/prefix_caching_block.py")
    res = cf.run_cross_framework_audit(
        {
            "framework": "sglang",
            "target_framework": "vllm",
            "candidate": {"candidate_id": "c1"},
            "diff_text": _SGLANG_DIFF,
            "target_framework_source_roots": [str(target)],
        }
    )
    hit = res["evidence"][0]
    assert hit["dst_symbol_present"] is True
    assert hit["dst_symbol"] == "match_prefix"
    assert res["metrics"]["dst_symbols_present"] == 1


def test_h1_symbol_anchor_absent_but_file_present(kb_root: Path, tmp_path: Path) -> None:
    _seed_map(kb_root, [_MAP_ROW])
    target = tmp_path / "vllm_src"
    _mk_target_module(target, "vllm/core/block/prefix_caching_block.py")
    res = cf.run_cross_framework_audit(
        {
            "framework": "sglang",
            "target_framework": "vllm",
            "candidate": {"candidate_id": "c1"},
            "diff_text": _SGLANG_DIFF,
            "target_framework_source_roots": [str(target)],
        }
    )
    hit = res["evidence"][0]
    assert hit["dst_present"] is True
    assert hit["dst_symbol_present"] is False
    assert hit["dst_symbol"] == ""
    assert res["metrics"]["dst_symbols_present"] == 0


def test_h1_symbol_anchor_raises_confidence(kb_root: Path, tmp_path: Path) -> None:
    # Symbol anchor raises confidence over file-only presence at the same coverage.
    two_file_diff = _SGLANG_DIFF + (
        "diff --git a/python/sglang/srt/managers/scheduler.py b/python/sglang/srt/managers/scheduler.py\n"
        "--- a/python/sglang/srt/managers/scheduler.py\n"
        "+++ b/python/sglang/srt/managers/scheduler.py\n"
        "@@ -1,1 +1,2 @@\n"
        " class Scheduler:\n"
        "+    def schedule(self):\n"
        "+        return None\n"
    )
    _seed_map(
        kb_root,
        [
            _MAP_ROW,
            {
                "src_framework": "sglang",
                "dst_framework": "vllm",
                "feature": "chunked_prefill_scheduler",
                "src_module": "python/sglang/srt/managers/scheduler.py",
                "dst_module": "vllm/core/absent_scheduler.py",
            },
        ],
    )
    req = {
        "framework": "sglang",
        "target_framework": "vllm",
        "candidate": {"candidate_id": "c1"},
        "diff_text": two_file_diff,
    }
    with_sym = tmp_path / "with_sym"
    _mk_target_module_with_symbol(with_sym, "vllm/core/block/prefix_caching_block.py")
    without_sym = tmp_path / "without_sym"
    _mk_target_module(without_sym, "vllm/core/block/prefix_caching_block.py")
    res_sym = cf.run_cross_framework_audit({**req, "target_framework_source_roots": [str(with_sym)]})
    res_nosym = cf.run_cross_framework_audit({**req, "target_framework_source_roots": [str(without_sym)]})
    assert res_sym["metrics"]["dst_modules_present"] == res_nosym["metrics"]["dst_modules_present"]
    assert res_sym["confidence"] > res_nosym["confidence"]
