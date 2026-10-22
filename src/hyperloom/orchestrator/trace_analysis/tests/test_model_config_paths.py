# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Candidate ``config.json`` locations for a model name, including the optional HF hub cache probe."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.trace_analysis.tracelens_analysis import _candidate_model_config_paths

_DIRECT = [Path("org/model/config.json")]


@pytest.fixture(autouse=True)
def _no_models_root(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_MODELS_ROOT", raising=False)


def _hub_probe_returning(monkeypatch, outcome):
    hub = pytest.importorskip("huggingface_hub")

    def probe(*, repo_id, filename):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(hub, "try_to_load_from_cache", probe)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("/cache/snapshots/abc/config.json", [*_DIRECT, Path("/cache/snapshots/abc/config.json")]),
        (ValueError("Repo id must be in the form 'repo_name' or 'namespace/repo_name'"), _DIRECT),
        (OSError("cache unreadable"), _DIRECT),
        (ImportError("hub client unavailable"), _DIRECT),
    ],
    ids=["cached", "invalid-repo-id", "oserror", "importerror"],
)
def test_a_hub_probe_adds_a_candidate_only_when_it_answers(monkeypatch, outcome, expected):
    _hub_probe_returning(monkeypatch, outcome)
    candidates = _candidate_model_config_paths("org/model")
    assert candidates == expected


def test_an_unexpected_hub_probe_failure_propagates(monkeypatch):
    _hub_probe_returning(monkeypatch, RuntimeError("hub client broken"))
    with pytest.raises(RuntimeError, match="hub client broken"):
        _candidate_model_config_paths("org/model")
