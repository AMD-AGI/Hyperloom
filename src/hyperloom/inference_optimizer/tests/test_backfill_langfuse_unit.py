# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exit status of the Langfuse backfill ingest."""

from __future__ import annotations

import sys
import types

from hyperloom.inference_optimizer.tools import backfill_langfuse


class _Observation:
    def start_observation(self, **_kwargs):
        return _Observation()

    def update_trace(self, **_kwargs):
        return None

    def end(self, **_kwargs):
        return None


class _RejectingClient(_Observation):
    def create_score(self, **_kwargs):
        raise RuntimeError("score rejected")

    def flush(self):
        return None


def test_a_failed_decision_score_fails_the_ingest(monkeypatch, capsys):
    client = _RejectingClient()
    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(get_client=lambda: client))
    monkeypatch.setattr(
        backfill_langfuse.lfmap,
        "decision_to_scores",
        lambda _row: [{"name": "decision", "value": 1.0, "metadata": {}}],
    )
    plan = {
        "trace_id_seed": "session",
        "created_at": "2026-01-01T00:00:00Z",
        "name": "session",
        "metadata": {},
        "session_id": "session",
        "phases": {},
        "recipe_audit": [],
        "decisions": [{}],
    }

    assert backfill_langfuse.ingest(plan) == 1
    assert capsys.readouterr().err == "ERROR: 1 decision score(s) failed to upload\n"
