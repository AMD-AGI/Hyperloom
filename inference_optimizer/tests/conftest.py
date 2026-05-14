"""Shared test helpers for the inference_optimizer tests package.

Currently exposes a single helper, :func:`seed_target_analysis_marker`,
that writes a structured ``target_baseline.json`` marker JSON at the
session directory's ``target_analysis/`` subdirectory.

Why this exists
---------------
:class:`Coordinator` now hard-gates ``target_analysis`` as TODO 0
*unconditionally*: when the marker JSON is missing it denies any
sequence action other than ``target_analysis`` and the
``_required_next_step`` text demands it (independent of the
``--compare-against-gpu`` flag — when unset, the executor still runs
and writes a ``reason='no_target_gpu_configured'`` marker).

Most existing tests don't care about the prep gate; they construct a
Coordinator and immediately exercise downstream behaviour (resume,
proposal flow, e2e mock adapters, kernel handlers, ...). For those
tests the cheapest fix is to write the marker JSON in the session
fixture so the gate is satisfied from tick 0.

Usage
-----

.. code-block:: python

    from .conftest import seed_target_analysis_marker

    @pytest.fixture
    def session_dir(tmp_path, monkeypatch):
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        sd = make_session_dir()
        seed_target_analysis_marker(sd)
        return sd

Tests that *want* to exercise the gate itself (e.g.
``test_required_step_gates`` / the dedicated executor tests) should
NOT call this helper; they construct their own session_dir fixture and
write the marker JSON only inside the specific assertions where the
gate-clearing transition is the subject under test.
"""

from __future__ import annotations

import json
from pathlib import Path


def seed_target_analysis_marker(session_dir: Path) -> Path:
    """Write a ``no_target_gpu_configured`` marker JSON at the session dir.

    Returns the path of the written file. Idempotent: if the file already
    exists it is overwritten with the same contents (cheap; the JSON is
    a few hundred bytes).
    """
    from inference_optimizer.session_paths import target_baseline_json
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "status": "skipped",
            "reason": "no_target_gpu_configured",
            "warning": "compare_against_gpu is empty",
        }),
        encoding="utf-8",
    )
    return path
