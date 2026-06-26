# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Step 2 — ``fa phase-audit`` static local-source judging.

Hermetic: no network / LLM. The opt-in LLM layer is exercised only via the
no-credentials skip path.
"""

from __future__ import annotations

import json
from pathlib import Path

import framework_agent.runtime.cli as cli
from framework_agent.audit import parse_unified_diff, run_phase_audit


def _make_root(tmp_path: Path, rel: str, content: str) -> Path:
    """Create ``<tmp>/vllm/<rel>`` with content; return the ``vllm`` root."""
    root = tmp_path / "vllm"
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return root


def _diff(*, added: list[str], context: list[str], path: str = "vllm/model_executor/layer.py") -> str:
    body = [
        f"diff --git a/{path} b/{path}",
        f"--- a/{path}",
        f"+++ b/{path}",
        "@@ -1,4 +1,6 @@",
    ]
    body.extend(f" {c}" for c in context)
    body.extend(f"+{a}" for a in added)
    return "\n".join(body) + "\n"


# parse_unified_diff
def test_parse_unified_diff_basic():
    patch = _diff(
        added=["    scale = compute(q)", "    return q * scale"],
        context=["import torch", "", "def op(q):"],
    )
    changes = parse_unified_diff(patch)
    assert len(changes) == 1
    c = changes[0]
    assert c.path == "vllm/model_executor/layer.py"
    assert len(c.added) == 2
    assert "import torch" in c.context


def test_parse_unified_diff_new_file():
    patch = (
        "diff --git a/vllm/x.py b/vllm/x.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/vllm/x.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def g():\n"
        "+    return 1\n"
    )
    changes = parse_unified_diff(patch)
    assert len(changes) == 1
    assert changes[0].is_new is True
    assert changes[0].path == "vllm/x.py"


# run_phase_audit — static verdict classes
def test_audit_already_equivalent(tmp_path: Path):
    content = "import torch\n\ndef scaled_op(q, k):\n    scale = compute_scale_factor(q)\n    return q * scale + k\n"
    root = _make_root(tmp_path, "model_executor/layer.py", content)
    patch = _diff(
        added=["    scale = compute_scale_factor(q)", "    return q * scale + k"],
        context=["import torch", "def scaled_op(q, k):"],
    )
    out = run_phase_audit(
        {
            "candidate": {"candidate_id": "ROCm/vllm#1", "pr_number": 1},
            "framework": "vllm",
            "framework_source_roots": [str(root)],
            "diff_text": patch,
            "work_dir": str(tmp_path / "wd"),
        }
    )
    assert out["semantic_status"] == "already_equivalent"
    assert out["applicability"] == "not_applicable"
    assert out["recommended_next_step"] == "skip"
    assert out["evidence"]  # evidence-gated
    assert (tmp_path / "wd" / "semantic_audit.json").exists()


def test_audit_not_present_direct_apply(tmp_path: Path):
    content = "import torch\n\ndef scaled_op(q, k):\n    return q + k\n"
    root = _make_root(tmp_path, "model_executor/layer.py", content)
    patch = _diff(
        added=["    q = apply_rotary_embedding(q)"],
        context=["import torch", "def scaled_op(q, k):", "    return q + k"],
    )
    out = run_phase_audit(
        {
            "candidate": {"candidate_id": "ROCm/vllm#2"},
            "framework_source_roots": [str(root)],
            "diff_text": patch,
            "work_dir": str(tmp_path / "wd"),
        }
    )
    assert out["semantic_status"] == "not_present"
    assert out["applicability"] == "direct_apply"
    assert out["recommended_next_step"] == "direct_framework_pr"


def test_audit_partially_present_needs_rewrite(tmp_path: Path):
    # One of the two added lines already present locally -> ~0.5 ratio.
    content = "import torch\n\ndef scaled_op(q, k):\n    scale = compute_scale_factor(q)\n    return q + k\n"
    root = _make_root(tmp_path, "model_executor/layer.py", content)
    patch = _diff(
        added=["    scale = compute_scale_factor(q)", "    bias = load_bias_vector(k)"],
        context=["def scaled_op(q, k):"],
    )
    out = run_phase_audit(
        {
            "candidate": {"candidate_id": "ROCm/vllm#3"},
            "framework_source_roots": [str(root)],
            "diff_text": patch,
            "work_dir": str(tmp_path / "wd"),
        }
    )
    assert out["semantic_status"] == "partially_present"
    assert out["applicability"] == "needs_rewrite"
    assert out["recommended_next_step"] == "author_via_specialist"


def test_audit_not_applicable_when_file_absent(tmp_path: Path):
    root = _make_root(tmp_path, "model_executor/layer.py", "x = 1\n")
    patch = _diff(
        added=["    y = 2"],
        context=["def other():"],
        path="vllm/does/not/exist.py",
    )
    out = run_phase_audit(
        {
            "candidate": {"candidate_id": "ROCm/vllm#4"},
            "framework_source_roots": [str(root)],
            "diff_text": patch,
            "work_dir": str(tmp_path / "wd"),
        }
    )
    assert out["semantic_status"] == "not_present"
    assert out["applicability"] == "not_applicable"
    assert out["recommended_next_step"] == "skip"


def test_audit_unknown_without_roots(tmp_path: Path):
    out = run_phase_audit(
        {
            "candidate": {"candidate_id": "c"},
            "framework_source_roots": [],
            "diff_text": _diff(added=["    y = 2"], context=["def f():"]),
            "work_dir": str(tmp_path / "wd"),
        }
    )
    assert out["semantic_status"] == "unknown"
    assert out["applicability"] == "needs_human_review"


def test_audit_unknown_without_patch(tmp_path: Path):
    root = _make_root(tmp_path, "model_executor/layer.py", "x = 1\n")
    out = run_phase_audit(
        {
            "candidate": {"candidate_id": "c"},
            "framework_source_roots": [str(root)],
            "work_dir": str(tmp_path / "wd"),
        }
    )
    assert out["semantic_status"] == "unknown"
    assert "no patch material" in " ".join(out["risks"]).lower()


def test_audit_patches_path_source(tmp_path: Path):
    content = "def scaled_op(q, k):\n    scale = compute_scale_factor(q)\n    return q * scale + k\n"
    root = _make_root(tmp_path, "model_executor/layer.py", content)
    patch_file = tmp_path / "pr.patches"
    patch_file.write_text(
        _diff(
            added=["    scale = compute_scale_factor(q)", "    return q * scale + k"],
            context=["def scaled_op(q, k):"],
        ),
        encoding="utf-8",
    )
    out = run_phase_audit(
        {
            "candidate": {"candidate_id": "c"},
            "framework_source_roots": [str(root)],
            "patches_path": str(patch_file),
            "work_dir": str(tmp_path / "wd"),
        }
    )
    assert out["metrics"]["patch_source"] == "patches_path"
    assert out["semantic_status"] == "already_equivalent"


# opt-in LLM layer: no creds -> static verdict kept, risk noted (hermetic)
def test_audit_use_llm_without_creds_keeps_static(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    content = "def scaled_op(q, k):\n    scale = compute_scale_factor(q)\n    return q * scale + k\n"
    root = _make_root(tmp_path, "model_executor/layer.py", content)
    out = run_phase_audit(
        {
            "candidate": {"candidate_id": "c"},
            "framework_source_roots": [str(root)],
            "diff_text": _diff(
                added=["    scale = compute_scale_factor(q)", "    return q * scale + k"],
                context=["def scaled_op(q, k):"],
            ),
            "work_dir": str(tmp_path / "wd"),
            "use_llm": True,
        }
    )
    assert out["layer"] == "static"
    assert any("missing" in r.lower() for r in out["risks"])


# CLI end-to-end
def test_cli_phase_audit_end_to_end(tmp_path: Path, capsys):
    content = "def scaled_op(q, k):\n    return q + k\n"
    root = _make_root(tmp_path, "model_executor/layer.py", content)
    req = {
        "candidate": {"candidate_id": "ROCm/vllm#9"},
        "framework": "vllm",
        "framework_source_roots": [str(root)],
        "diff_text": _diff(
            added=["    q = apply_rotary_embedding(q)"],
            context=["def scaled_op(q, k):", "    return q + k"],
        ),
        "work_dir": str(tmp_path / "wd"),
    }
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    rc = cli.main(["phase-audit", "--request", str(req_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_id"] == "ROCm/vllm#9"
    assert payload["semantic_status"] in ("not_present", "partially_present")
    assert payload["applicability"] in ("direct_apply", "needs_rewrite")


def test_cli_schema_lists_phase_audit(capsys):
    rc = cli.main(["schema"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "phase-audit" in payload["subcommands_available"]
    assert "semantic_status_values" in payload["phase_audit"]
