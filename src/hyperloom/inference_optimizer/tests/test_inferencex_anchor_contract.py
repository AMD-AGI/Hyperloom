# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract between Hyperloom's patch anchors and the pinned InferenceX tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli.preflight import _INFERENCEX_REF_DEFAULT
from hyperloom.orchestrator.actions.executors._inferencex_anchor_contract import (
    MAGPIE_LIB_PATH,
    PROBE_TARGET_PATH,
    REFRESH_CMD,
    anchors_by_file,
    anchors_fingerprint,
    fetch_pinned_file,
    magpie_patch_applies,
)
from hyperloom.orchestrator.actions.executors._inferencex_patcher import _ANCHOR_CONTRACT, count_anchor_hits

CONTRACT_PATH = Path(__file__).parent / "fixtures" / "inferencex_anchor_contract.json"


def load_record() -> dict:
    """Return the checked-in contract record."""
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


# --- hermetic ---------------------------------------------------------------


def test_recorded_ref_matches_the_pin_the_code_installs():
    """Bumping INFERENCEX_REF is exactly when an anchor silently rots, so the bump cannot be allowed to pass without someone re-verifying."""
    record = load_record()

    assert record["ref"] == _INFERENCEX_REF_DEFAULT, (
        f"INFERENCEX_REF moved to {_INFERENCEX_REF_DEFAULT} but the anchor contract was last "
        f"verified against {record['ref']}. Re-verify and refresh it: {REFRESH_CMD}"
    )


def test_recorded_fingerprint_matches_the_current_anchors():
    """The counts below describe what the anchors matched as they were written; editing one invalidates the record just
    as a pin bump does.
    """
    record = load_record()

    assert record["anchors_fingerprint"] == anchors_fingerprint(), (
        f"the patch anchors changed since the contract was recorded. Re-verify and refresh it: {REFRESH_CMD}"
    )


def test_record_covers_every_anchor_in_the_contract():
    """A newly added patch must be verified against upstream too, not just inherit the previous record's silence."""
    record = load_record()

    recorded = {name for spec in record["files"].values() for name in spec["anchors"]}
    assert recorded == {name for name, *_ in _ANCHOR_CONTRACT}, f"refresh the contract: {REFRESH_CMD}"


def test_every_recorded_anchor_matched_exactly_one_site():
    """One site is the whole contract: zero means the patch is inert, and more than one means the file drifted into a shape the patcher never handled."""
    record = load_record()

    hits = {name: n for spec in record["files"].values() for name, n in spec["anchors"].items()}
    assert all(n == 1 for n in hits.values()), hits


def test_record_covers_the_magpie_patch():
    """The louder of the two patchers aimed at benchmark_lib.sh."""
    record = load_record()

    assert record.get("magpie_patch", {}).get("applies") is True, (
        f"the contract predates the magpie patch check, or the splice no longer applies. {REFRESH_CMD}"
    )
    assert record["magpie_patch"]["path"] == MAGPIE_LIB_PATH


# --- networked --------------------------------------------------------------


@pytest.mark.parametrize("rel_path", sorted(anchors_by_file()))
def test_pinned_upstream_still_matches_every_anchor(rel_path):
    """The layer that actually re-verifies."""
    record = load_record()
    text = fetch_pinned_file(rel_path, record["ref"])
    if text is None:
        pytest.skip(f"InferenceX@{record['ref'][:9]} unreachable (needs `gh` + repo access)")

    hits = {name: count_anchor_hits(text, anchor) for name, anchor in anchors_by_file()[rel_path]}
    assert hits == record["files"][rel_path]["anchors"], (
        f"upstream {rel_path} no longer matches the recorded anchors. Re-anchor the affected "
        f"patches in _inferencex_patcher.py, then refresh: {REFRESH_CMD}"
    )
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == record["files"][rel_path]["sha256"], (
        f"the anchors still match, but {rel_path} is not the file the contract recorded. Refresh it: {REFRESH_CMD}"
    )


def test_recorded_probe_target_is_the_path_the_patcher_appends_to():
    """The probe has no anchor to rot, but it does need this file to exist: if upstream moves it the patch degrades to a warning and the eval runs unbounded again -- the exact failure the probe was written to stop."""
    record = load_record()

    assert record["probe_target"]["path"] == PROBE_TARGET_PATH, (
        f"the probe target moved to {PROBE_TARGET_PATH}. Re-verify and refresh: {REFRESH_CMD}"
    )


def test_pinned_upstream_still_carries_the_probe_target():
    """Networked counterpart: confirm the file is really there at the pin."""
    record = load_record()
    text = fetch_pinned_file(PROBE_TARGET_PATH, record["ref"])
    if text is None:
        pytest.skip(f"InferenceX@{record['ref'][:9]} unreachable (needs `gh` + repo access)")

    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == record["probe_target"]["sha256"], (
        f"{PROBE_TARGET_PATH} changed upstream. The probe and the bounds are appended to it, so "
        f"re-read it before refreshing: {REFRESH_CMD}"
    )


def test_pinned_upstream_still_takes_the_magpie_splice():
    """Re-verify the splice against upstream, not just against the record."""
    record = load_record()
    text = fetch_pinned_file(MAGPIE_LIB_PATH, record["ref"])
    if text is None:
        pytest.skip(f"InferenceX@{record['ref'][:9]} unreachable (needs `gh` + repo access)")

    assert magpie_patch_applies(text), (
        f"{MAGPIE_LIB_PATH} at {record['ref'][:9]} no longer takes the run_lm_eval "
        f"--concurrent-requests splice; install.sh would die(). Re-anchor _magpie_patcher.py."
    )


# --- recording the contract (scripts/refresh_inferencex_anchor_contract.py) ----


@pytest.fixture
def upstream(monkeypatch):
    """Stand in for the pinned upstream tree: every file fetches, every anchor matches once."""
    import hyperloom.orchestrator.actions.executors._inferencex_anchor_contract as contract

    files: dict[str, str | None] = {path: f"text of {path}\n" for path in anchors_by_file()}
    files[PROBE_TARGET_PATH] = "probe target\n"
    files.setdefault(MAGPIE_LIB_PATH, "magpie lib\n")
    hits: dict[str, int] = {}
    monkeypatch.setattr(contract, "fetch_pinned_file", lambda rel_path, ref: files.get(rel_path))
    monkeypatch.setattr(contract, "count_anchor_hits", lambda text, anchor: hits.get(anchor, 1))
    monkeypatch.setattr(contract, "magpie_patch_applies", lambda text: files.get("applies", "yes") == "yes")
    return contract, files, hits


def test_build_record_records_each_file_the_probe_and_the_magpie_patch(upstream):
    contract, files, _hits = upstream
    record = contract.build_record("abc123")
    assert record["ref"] == "abc123"
    assert record["refresh_with"] == REFRESH_CMD
    assert record["anchors_fingerprint"] == anchors_fingerprint()
    assert set(record["files"]) == set(anchors_by_file())
    for rel_path, entry in record["files"].items():
        assert entry["sha256"] == hashlib.sha256(files[rel_path].encode("utf-8")).hexdigest()
        assert set(entry["anchors"].values()) == {1}
    assert record["probe_target"]["sha256"] == hashlib.sha256(b"probe target\n").hexdigest()
    assert record["magpie_patch"] == {"path": MAGPIE_LIB_PATH, "applies": True}


def test_build_record_refuses_an_unreachable_file(upstream):
    contract, files, _hits = upstream
    missing = sorted(anchors_by_file())[0]
    files[missing] = None
    with pytest.raises(RuntimeError, match=f"cannot fetch {missing} at abc123"):
        contract.build_record("abc123")


def test_build_record_refuses_an_anchor_that_does_not_match_exactly_once(upstream):
    contract, _files, hits = upstream
    name, _rel_parts, _sentinel, anchor = _ANCHOR_CONTRACT[0]
    hits[anchor] = 2
    with pytest.raises(RuntimeError, match=rf"expected each anchor to match exactly one site, got \{{'{name}': 2\}}"):
        contract.build_record("abc123")


def test_build_record_refuses_a_missing_probe_target(upstream):
    contract, files, _hits = upstream
    files[PROBE_TARGET_PATH] = None
    with pytest.raises(RuntimeError, match=f"cannot fetch {PROBE_TARGET_PATH} at abc123. The probe"):
        contract.build_record("abc123")


def test_build_record_refuses_a_magpie_patch_that_no_longer_applies(upstream):
    contract, files, _hits = upstream
    files["applies"] = "no"
    with pytest.raises(RuntimeError, match="the run_lm_eval --concurrent-requests splice no longer finds its site"):
        contract.build_record("abc123")


def test_fetch_pinned_file_returns_none_when_gh_fails_or_is_missing(monkeypatch):
    import subprocess

    import hyperloom.orchestrator.actions.executors._inferencex_anchor_contract as contract

    def run(stdout: bytes, returncode: int):
        return lambda *args, **kwargs: subprocess.CompletedProcess(args, returncode, stdout=stdout)

    monkeypatch.setattr(contract.subprocess, "run", run(b"body\n", 0))
    assert fetch_pinned_file("a/b.sh", "abc123") == "body\n"
    monkeypatch.setattr(contract.subprocess, "run", run(b"", 1))
    assert fetch_pinned_file("a/b.sh", "abc123") is None

    def missing(*args, **kwargs):
        raise OSError("gh not found")

    monkeypatch.setattr(contract.subprocess, "run", missing)
    assert fetch_pinned_file("a/b.sh", "abc123") is None


def test_magpie_patch_applies_needs_the_parser_site():
    assert magpie_patch_applies("#!/bin/bash\necho nothing to splice\n") is False
