"""Regression: GEAK finalized-success status recognition in build_verification.

Root cause of the hands-off loop never closing: GEAK finalizes a multi-round
run as ``status="incremental_after_round_<N>"`` (and single-round as
``auto_finalized``/``finalized``), but the correctness-trust gate only accepted
``{complete, succeeded, ok}``. A GEAK best with a VERIFIED speedup (fused_moe
1.5257x, status=incremental_after_round_2) was therefore scored
``correctness=missing`` -> NEEDS_REVIEW and never reached integrate. The fix
trusts any non-failure finalized status, or a report carrying a verified
speedup >= 1.0, as a successful + correct GEAK run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_TOOL_PATH = Path(__file__).resolve().parent / "kernel_optimization.py"


@pytest.fixture(scope="module")
def ko() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_ko_status_under_test", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _report(tmp_path: Path, status: str, verified=1.5257) -> Path:
    p = tmp_path / "final_report.json"
    doc: dict = {"status": status}
    if verified is not None:
        doc["verified_speedup"] = verified
        doc["best_speedup_verified"] = verified
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _attempt(report: Path, speedup="1.5257") -> dict:
    return {
        "status": "completed",
        "backend": "geak",
        "attempt_id": "geak-test",
        "returncode": 0,
        "optimized_path": str(report.parent / "opt.py"),
        "backend_paths": {
            "geak_final_report": str(report),
            "geak_per_task_best_speedup": speedup,
        },
    }


def _args(src: Path) -> SimpleNamespace:
    return SimpleNamespace(
        correctness_passed=None, micro_speedup=None, source_file=str(src),
        kernel_repo="", repo="", accuracy_passed=None, e2e_gain_pct=None,
    )


def _patch_artifact(ko, monkeypatch, src: Path) -> None:
    src.write_text("def get_default_config():\n    return {}\n", encoding="utf-8")
    monkeypatch.setattr(ko, "_select_source_artifact", lambda *a, **k: (str(src), "source_file", ""))


@pytest.mark.parametrize(
    "status",
    ["incremental_after_round_2", "incremental_after_round_5", "complete",
     "completed", "auto_finalized", "finalized"],
)
def test_geak_finalized_status_is_trusted_and_keeps(ko, tmp_path, monkeypatch, status):
    src = tmp_path / "opt.py"
    _patch_artifact(ko, monkeypatch, src)
    monkeypatch.setenv("HYPERLOOM_TRUST_GEAK_CORRECTNESS", "1")
    v = ko.build_verification(_args(src), [_attempt(_report(tmp_path, status))], True)
    assert v["correctness_passed"] is True, v
    assert v["correctness_source"] == "geak_assumed_pass"
    assert v["micro_speedup"] >= 1.10
    # A verified >=1.10 micro + trusted correctness -> straight KEEP (the loop closes).
    assert ko.make_proposal(v, high_impact=True)["decision"] == "KEEP"


def test_geak_verified_speedup_trusted_even_with_unknown_status(ko, tmp_path, monkeypatch):
    src = tmp_path / "opt.py"
    _patch_artifact(ko, monkeypatch, src)
    monkeypatch.setenv("HYPERLOOM_TRUST_GEAK_CORRECTNESS", "1")
    # status the whitelist doesn't know, but a verified speedup is present.
    v = ko.build_verification(_args(src), [_attempt(_report(tmp_path, "brand_new_status", 1.4), "1.4")], True)
    assert v["correctness_passed"] is True


def test_geak_failed_status_without_verified_is_not_trusted(ko, tmp_path, monkeypatch):
    src = tmp_path / "opt.py"
    _patch_artifact(ko, monkeypatch, src)
    monkeypatch.setenv("HYPERLOOM_TRUST_GEAK_CORRECTNESS", "1")
    # status=failed AND no verified speedup -> must NOT be auto-trusted.
    v = ko.build_verification(_args(src), [_attempt(_report(tmp_path, "failed", verified=None), "1.4")], True)
    assert v["correctness_passed"] is False


def test_trust_disabled_restores_conservative_behaviour(ko, tmp_path, monkeypatch):
    src = tmp_path / "opt.py"
    _patch_artifact(ko, monkeypatch, src)
    monkeypatch.setenv("HYPERLOOM_TRUST_GEAK_CORRECTNESS", "0")
    v = ko.build_verification(_args(src), [_attempt(_report(tmp_path, "incremental_after_round_2"))], True)
    assert v["correctness_passed"] is False


# ---------------------------------------------------------------------------
# chaojhou review (point c): TERMINAL-ONLY ``incremental_after_round_`` match.
# A non-terminal ``incremental_round_<N>`` (a mid-run snapshot, NOT a finalized
# success) must NOT be accepted as correctness-verified; only the terminal
# finalized form ``incremental_after_round_<N>`` (and auto_finalized/finalized)
# is. These cases write a report with NO verified_speedup so ONLY the status
# gate can grant correctness, isolating the status-prefix behaviour.
# ---------------------------------------------------------------------------
def test_geak_non_terminal_incremental_round_is_not_trusted(ko, tmp_path, monkeypatch):
    src = tmp_path / "opt.py"
    _patch_artifact(ko, monkeypatch, src)
    monkeypatch.setenv("HYPERLOOM_TRUST_GEAK_CORRECTNESS", "1")
    # incremental_round_2 has NO ``_after_round_`` finalize and no verified
    # speedup -> must remain correctness=missing.
    v = ko.build_verification(
        _args(src),
        [_attempt(_report(tmp_path, "incremental_round_2", verified=None))],
        True,
    )
    assert v["correctness_passed"] is False, v


def test_geak_terminal_incremental_after_round_is_trusted_status_only(ko, tmp_path, monkeypatch):
    src = tmp_path / "opt.py"
    _patch_artifact(ko, monkeypatch, src)
    monkeypatch.setenv("HYPERLOOM_TRUST_GEAK_CORRECTNESS", "1")
    # incremental_after_round_3 is the TERMINAL finalized form -> trusted on
    # the status gate alone (no verified_speedup present).
    v = ko.build_verification(
        _args(src),
        [_attempt(_report(tmp_path, "incremental_after_round_3", verified=None))],
        True,
    )
    assert v["correctness_passed"] is True, v
    assert v["correctness_source"] == "geak_assumed_pass"
