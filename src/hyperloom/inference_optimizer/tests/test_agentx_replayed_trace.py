# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK is told the trace an accepted AgentX round replayed, read from the aiperf export the round left.

GEAK keeps its own default for each ``workload_spec`` field left out, and its default corpus is the 256k-context cut:
a session on a 1M-context model replays the full corpus, so a handoff without the corpus searches another workload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.agentx.report import replayed_trace

_FULL = "semianalysis_cc_traces_weka_062126"
_SHAPE = {
    "input_sequence_length": {"avg": 155473.2, "p50": 108378.5},
    "output_sequence_length": {"avg": 785.9, "p50": 383.0},
}


def _export(run_dir: Path, **document: object) -> Path:
    artifacts = run_dir / "client" / "aiperf_artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "profile_export_aiperf.json").write_text(json.dumps(document), encoding="utf-8")
    return run_dir


def _dataset(loader: object = _FULL, entries: object = 393) -> dict:
    return {"metadata": {"dataset": {"loader": loader, "num_dataset_entries": entries}}}


def test_the_corpus_its_entry_count_and_mean_shape_come_from_the_rounds_export(tmp_path: Path) -> None:
    assert replayed_trace(_export(tmp_path, **_dataset(), **_SHAPE)) == {
        "corpus": _FULL,
        "canonical_corpus": _FULL,
        "num_entries": 393,
        "canonical_num_entries": 393,
        "observed_isl": 155473,
        "observed_osl": 785,
    }


def test_an_export_without_a_measured_shape_still_names_the_corpus(tmp_path: Path) -> None:
    assert replayed_trace(_export(tmp_path, **_dataset())) == {
        "corpus": _FULL,
        "canonical_corpus": _FULL,
        "num_entries": 393,
        "canonical_num_entries": 393,
    }


@pytest.mark.parametrize(
    "document",
    [{}, _dataset(loader=""), _dataset(entries=None), _dataset(entries=True), {"metadata": {}}],
    ids=["no metadata", "no loader", "no entry count", "boolean entry count", "no dataset"],
)
def test_an_export_that_names_no_corpus_declares_nothing(tmp_path: Path, document: dict) -> None:
    assert replayed_trace(_export(tmp_path, **document, **_SHAPE)) == {}


def test_a_round_without_an_export_declares_nothing(tmp_path: Path) -> None:
    assert replayed_trace(tmp_path) == {}
    assert replayed_trace(None) == {}
