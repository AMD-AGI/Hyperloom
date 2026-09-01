"""Tests for format-independent PR reference exposure provenance."""

from __future__ import annotations

import json

from kernelforge.cli import _write_pr_provenance
from kernelforge.knowledge import pr_monitor_refs
from kernelforge.knowledge.pr_monitor_refs import refs_dir


SURFACED = ("ROCm/FlyDSL#959", "ROCm/FlyDSL#930")


def test_no_sidecar_when_no_references_were_surfaced(tmp_path):
    _write_pr_provenance(
        workspace_dir=str(tmp_path),
        surfaced=(),
        winning_iteration=3,
    )

    assert not (refs_dir(str(tmp_path)) / "provenance.json").exists()


def test_sidecar_records_exposure_without_parsing_free_form_lessons(tmp_path):
    _write_pr_provenance(
        workspace_dir=str(tmp_path),
        surfaced=SURFACED,
        winning_iteration=2,
        experiment_id="exp-1",
    )

    payload = json.loads((refs_dir(str(tmp_path)) / "provenance.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["winning_iteration"] == 2
    assert payload["experiment_id"] == "exp-1"
    assert payload["surfaced"] == list(SURFACED)
    assert set(payload) == {
        "schema_version",
        "winning_iteration",
        "experiment_id",
        "surfaced",
    }


def test_a_failed_sidecar_write_cannot_fail_a_finished_run(
    monkeypatch,
    tmp_path,
):
    def full_disk(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(pr_monitor_refs, "write_provenance", full_disk)

    _write_pr_provenance(
        workspace_dir=str(tmp_path),
        surfaced=SURFACED,
        winning_iteration=1,
    )
