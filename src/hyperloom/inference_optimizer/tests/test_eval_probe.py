# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy gate's reading of the lm-eval generation-pathology probe a round's eval left."""

from __future__ import annotations

import json
import os
from pathlib import Path


from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    EVAL_KIND_GENERATION_PATHOLOGY,
    EVAL_PROBE_FILENAME,
    eval_probe_summary,
    read_eval_probe,
)
from inference_suite.contracts import ProbeOutcome, RunLayout


def test_read_eval_probe_prefers_the_newest_sidecar(tmp_path):
    """``integrate_patch`` searches the grid slot, where sibling variants each own a sidecar, and attempt dirs are hash-named — so path order says nothing about which eval ran last."""
    older = tmp_path / "zzz_first" / EVAL_PROBE_FILENAME
    newer = tmp_path / "aaa_second" / EVAL_PROBE_FILENAME
    for path, hits in ((older, 11), (newer, 22)):
        path.parent.mkdir()
        path.write_text(json.dumps({"cap_hits": hits}), encoding="utf-8")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    record = read_eval_probe(tmp_path)

    assert record is not None
    assert record["cap_hits"] == 22


def test_read_eval_probe_is_none_without_a_sidecar(tmp_path):
    """No sidecar is the ordinary case: the model terminated its answers."""
    assert read_eval_probe(tmp_path) is None


def test_read_eval_probe_tolerates_corrupt_json(tmp_path):
    (tmp_path / EVAL_PROBE_FILENAME).write_text("{not json", encoding="utf-8")
    assert read_eval_probe(tmp_path) is None


def test_read_eval_probe_tolerates_an_unscannable_tree(tmp_path, monkeypatch):
    """Parallel xdist workers can delete sibling dirs while rglob is walking."""

    def _rglob_raises(self, pattern):
        raise FileNotFoundError(self)

    monkeypatch.setattr(Path, "rglob", _rglob_raises)
    assert read_eval_probe(tmp_path) is None


def _suite_probe(run_dir: Path, outcome: ProbeOutcome) -> None:
    probe = RunLayout(run_dir).eval_probe
    probe.parent.mkdir(parents=True)
    probe.write_text(json.dumps(outcome.to_document()), encoding="utf-8")


def test_read_eval_probe_ignores_a_suite_eval_whose_probe_did_not_trip(tmp_path):
    """inference_suite records its probe outcome on every eval, so the file alone is no trip."""
    _suite_probe(tmp_path / "suite_run_0", ProbeOutcome(tripped=False, samples_seen=1319, length_ratio_observed=0.006))

    assert read_eval_probe(tmp_path) is None


def test_read_eval_probe_reports_a_tripped_suite_probe_with_its_counts(tmp_path):
    _suite_probe(tmp_path / "suite_run_0", ProbeOutcome(tripped=True, samples_seen=128, length_ratio_observed=0.9))

    record = read_eval_probe(tmp_path)

    assert record is not None and record["kind"] == EVAL_KIND_GENERATION_PATHOLOGY
    summary = eval_probe_summary(record)
    assert "128 sampled responses" in summary and "90%" in summary


def test_eval_probe_summary_is_empty_without_a_probe():
    assert eval_probe_summary(None) == ""


def test_eval_probe_summary_names_the_kind_and_the_evidence():
    summary = eval_probe_summary({"observed_samples": 16, "cap_hits": 16, "max_completion_tokens_seen": 16384})
    assert EVAL_KIND_GENERATION_PATHOLOGY in summary
    assert "16/16" in summary
    assert "16384" in summary
