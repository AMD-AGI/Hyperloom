# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ci/build_rolling_pool.py."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import build_rolling_pool as brp  # noqa: E402


def test_norm():
    assert brp._norm("  Org/Model  ") == "org/model"
    assert brp._norm(None) == ""


def test_candidate_keys_full_and_basename():
    keys = brp._candidate_keys("Org/Model-7B")
    assert brp.slugify("Org/Model-7B") in keys
    assert brp.slugify("Model-7B") in keys


def test_pulse_model_keys_paginates(monkeypatch):
    pages = [
        {"results": [{"model_name": "Org/A"}, {"model_name": "Org/B"}], "pagination": {"total": 3}},
        {"results": [{"model_name": "Org/C"}], "pagination": {"total": 3}},
    ]
    calls = {"i": 0}

    def fake_urlopen(url, timeout=0):
        idx = calls["i"]
        calls["i"] += 1
        payload = pages[idx] if idx < len(pages) else {"results": []}
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(brp.urllib.request, "urlopen", fake_urlopen)
    keys = brp._pulse_model_keys()
    assert brp.slugify("Org/A") in keys
    assert brp.slugify("Org/C") in keys


def test_pulse_model_keys_empty(monkeypatch):
    monkeypatch.setattr(
        brp.urllib.request, "urlopen", lambda url, timeout=0: io.BytesIO(json.dumps({"results": []}).encode())
    )
    assert brp._pulse_model_keys() == set()


def test_pulse_model_keys_retries_then_succeeds(monkeypatch):
    state = {"n": 0}

    def flaky(url, timeout=0):
        state["n"] += 1
        if state["n"] == 1:
            raise OSError("boom")
        return io.BytesIO(json.dumps({"results": [{"model_name": "m"}], "pagination": {"total": 1}}).encode())

    monkeypatch.setattr(brp.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(brp.time, "sleep", lambda *_a, **_k: None)
    keys = brp._pulse_model_keys()
    assert brp.slugify("m") in keys


def test_main_merges_dedups_and_writes(tmp_path: Path, monkeypatch, capsys):
    prod = tmp_path / "prod.json"
    newm = tmp_path / "new.json"
    manual_out = tmp_path / "manual.json"
    unrun_out = tmp_path / "unrun.json"

    prod.write_text(
        json.dumps(
            {
                "policy": {"excluded_exact_ids": ["bad/excluded"], "exclusion_keywords": ["gpt-oss"]},
                "candidates": [
                    {"repo_id": "org/keep", "params_b": 7, "downloads": 500},
                    {"repo_id": "org/gpt-oss-120b", "downloads": 999},  # keyword-excluded
                    {"repo_id": "bad/excluded", "downloads": 999},  # exact-excluded
                ],
            }
        ),
        encoding="utf-8",
    )
    newm.write_text(
        json.dumps(
            {
                "models": [
                    {"repo_id": "org/new1", "num_parameters": 7_000_000_000, "downloads": 1000},
                    {"repo_id": "org/keep", "downloads": 1},  # dup of production -> skipped
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(brp, "PROD", prod)
    monkeypatch.setattr(brp, "NEWM", newm)
    monkeypatch.setattr(brp, "MANUAL_OUT", manual_out)
    monkeypatch.setattr(brp, "UNRUN_OUT", unrun_out)
    monkeypatch.setattr(brp, "MANUAL_N", 1)
    # org/new1 already "run" via pulse -> excluded from unrun.
    monkeypatch.setattr(brp, "_pulse_model_keys", lambda: {brp.slugify("new1")})

    rc = brp.main()
    assert rc == 0

    merged = json.loads(prod.read_text(encoding="utf-8"))
    repos = {c["repo_id"] for c in merged["candidates"]}
    assert repos == {"org/keep", "org/new1"}  # excluded + dup gone

    manual = json.loads(manual_out.read_text(encoding="utf-8"))
    assert manual["count"] == 1  # MANUAL_N

    unrun = json.loads(unrun_out.read_text(encoding="utf-8"))
    unrun_repos = {c["repo_id"] for c in unrun["candidates"]}
    # new1 is run (pulse) AND org/new1 may be in manual top-1; keep is in manual top-1
    assert "org/new1" not in unrun_repos
