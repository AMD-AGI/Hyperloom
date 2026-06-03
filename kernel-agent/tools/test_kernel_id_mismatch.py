"""kernel_id mismatch handling at the kernel-agent boundary.

The Orchestration LLM occasionally supplies a ``kernel_id`` that does not
match any TraceLens candidate (e.g. it echoes an operator name like
``aiter.silu_and_mul`` or a hallucinated ``kn001`` instead of the real
``k001``). Previously ``find_candidate`` raised ``KeyError`` on the first
lookup, crashing the entire kernel-optimization subprocess before GEAK was
ever invoked.

Contract:

* ``find_candidate`` resolves by exact ``kernel_id`` and normalized
  ``kernel_id`` match (case-insensitive, ``kn``/``rn`` prefix folded to
  ``k``). It accepts a ``name`` only when that name uniquely identifies a
  routable candidate.
* When nothing matches it returns ``None`` instead of raising, so the caller
  can skip the kernel gracefully rather than aborting the whole run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_optimization as ko  # noqa: E402


ROUTABLE = {
    "kernel_id": "k002",
    "name": "sgl_kernel::silu_and_mul",
    "reusable_native_kernel": True,
    "source_file": "/sgl-kernel/elementwise.py",
}

CANDIDATES = [
    {"kernel_id": "k001", "name": "aten::mm"},
    ROUTABLE,
    {"kernel_id": "k010", "name": "aiter::rmsnorm", "source_file": "/aiter/rms.py"},
]

SKIPPED_CANDIDATES = [
    {
        "kernel_id": "k001",
        "name": "aten::mm",
        "reusable_native_kernel": False,
        "source_file": "",
        "skip_reason": "source file not resolved",
    },
    {
        "kernel_id": "k003",
        "name": "aten::mm",
        "reusable_native_kernel": False,
        "source_file": "",
        "skip_reason": "source file not resolved",
    },
]


def test_exact_id_match():
    assert ko.find_candidate(CANDIDATES, "k002")["name"] == "sgl_kernel::silu_and_mul"


def test_unique_routable_name_match():
    assert ko.find_candidate(CANDIDATES, "sgl_kernel::silu_and_mul")["kernel_id"] == "k002"


def test_operator_name_for_skipped_or_non_unique_candidates_returns_none():
    # Real TraceLens failures often list several skipped ``aten::mm`` rows.
    # The operator name is non-unique and non-routable, so it must not resolve
    # to k001 and accidentally send a skipped candidate into optimization.
    assert ko.find_candidate(SKIPPED_CANDIDATES, "aten::mm") is None


def test_normalized_prefix_match():
    # LLM hallucinated ``kn001`` / ``rn010``; fold the synthetic prefix back to
    # the real ``k`` numbering.
    assert ko.find_candidate(CANDIDATES, "kn001")["kernel_id"] == "k001"
    assert ko.find_candidate(CANDIDATES, "rn010")["kernel_id"] == "k010"


def test_unknown_id_returns_none_not_raise():
    # Pure hallucination that maps to nothing must NOT crash the subprocess.
    assert ko.find_candidate(CANDIDATES, "aiter.silu_and_mul") is None
    assert ko.find_candidate(CANDIDATES, "framework_sglang_silu_and_mul_m64") is None


@pytest.mark.parametrize("requested", ["kn001", "aten::mm"])
def test_main_skips_skipped_candidates_without_backend_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    requested: str,
) -> None:
    candidates_path = tmp_path / "kernel_candidates.json"
    candidates_path.write_text(
        json.dumps({"hot_kernels": [], "skipped_kernels": SKIPPED_CANDIDATES}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kernel_optimization.py",
            "--kernel-id", requested,
            "--session-id", "s1",
            "--workspace-path", str(tmp_path),
            "--candidates-path", str(candidates_path),
        ],
    )

    assert ko.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "skipped"
    if requested == "kn001":
        assert out["kernel_id"] == "k001"
        assert out["requested_kernel_id"] == requested
        assert out["decision"] == "REVERT"
        assert out["error_class"] == "missing_native_source"
        assert out["reason"] == "non_routable_candidate"
    else:
        assert out["kernel_id"] == requested
        assert out["error_class"] == "invalid_kernel_id"
        assert out["reason"] == "kernel_id_not_in_candidates"
