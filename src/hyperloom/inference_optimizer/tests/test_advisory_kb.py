# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the human-editable, framework-partitioned advisory KB loader."""

from __future__ import annotations

from hyperloom.orchestrator.knowledge import advisory_kb as akb


# ---- routing: generic reaches all, framework folder reaches only that framework ----


def test_generic_hints_reach_all_frameworks():
    generic = akb.hints_from_markdown("")
    assert generic  # generic partition is non-empty
    # every generic hint is unrouted (framework == "")
    assert all(h["framework"] == "" for h in generic)


def test_vllm_run_gets_generic_plus_vllm():
    generic = akb.hints_from_markdown("")
    vllm = akb.hints_from_markdown("vllm")
    assert len(vllm) > len(generic)
    # a vLLM-only knob appears for vLLM, not in generic
    assert any("VLLM_ROCM_USE_AITER" in h["what"] for h in vllm)
    assert not any("VLLM_ROCM_USE_AITER" in h["what"] for h in generic)


def test_sglang_run_does_not_get_vllm_flags():
    sglang = akb.hints_from_markdown("sglang")
    assert not any("VLLM_ROCM_USE_AITER" in h["what"] for h in sglang)
    # sglang still gets the generic reasoning
    assert any("throughput" in (h.get("expected_impact") or "").lower() for h in sglang)


# ---- attributability: an entry without a source is dropped ----


def test_hints_require_source(tmp_path, monkeypatch):
    root = tmp_path / "advisory_kb"
    (root / "generic").mkdir(parents=True)
    (root / "generic" / "x.md").write_text(
        "# t\n\n## has source\n- source: paper\n\nbody\n\n## no source\n- impact: throughput\n\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HYPERLOOM_ADVISORY_KB_DIR", str(root))
    hints = akb.hints_from_markdown("")
    whats = " ".join(h["what"] for h in hints)
    assert "has source" in whats
    assert "no source" not in whats


# ---- checklist parsing + gpu/precision fields ----


def test_checklist_parses_applies_when_and_dirs():
    entries = akb.checklist_from_markdown("vllm")
    by_id = {e["id"]: e for e in entries}
    e = by_id["rocm.fp4.aiter_master_switch_gap"]
    assert e["applies_when"] == {"gpu": "rocm", "precision": "fp4"}
    assert e["source_dirs"]  # non-empty
    assert e["evidence"]  # source split into evidence tuple
    assert e["domain_hint"] == "kernel_switch_specialist"


def test_checklist_is_framework_routed():
    vllm = {e["id"] for e in akb.checklist_from_markdown("vllm")}
    sglang = {e["id"] for e in akb.checklist_from_markdown("sglang")}
    assert "rocm.fp4.aiter_master_switch_gap" in vllm
    assert "rocm.fp4.aiter_master_switch_gap" not in sglang


def test_env_override_of_kb_root(tmp_path, monkeypatch):
    root = tmp_path / "kb"
    (root / "vllm").mkdir(parents=True)
    (root / "vllm" / "levers.md").write_text(
        "# t\n\n## custom vllm knob\n- source: mytest\n- impact: throughput\n\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HYPERLOOM_ADVISORY_KB_DIR", str(root))
    hints = akb.hints_from_markdown("vllm")
    assert any("custom vllm knob" in h["what"] for h in hints)
