"""Tests for framework_agent.runtime.cli.main.

Hermetic - exercises argv-only paths. The ``candidates`` and ``explore``
subcommands are tested via stubs that replace
``sources.enumerate_candidates`` / ``explorer.explore`` so no network
or git is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hyperloom.agents.framework.runtime.cli as cli
from hyperloom.agents.framework.models import Candidate


# main() top-level -------------------------------------------------------


def test_main_schema_returns_zero(capsys) -> None:
    """fa schema must exit 0 and emit JSON to stdout."""
    rc = cli.main(["schema"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "schema" in payload["subcommands_available"]
    assert payload["promotion_policy"] == "manual_only"


def test_main_missing_subcommand_exits_nonzero(capsys) -> None:
    """No subcommand should trigger argparse usage error (SystemExit != 0)."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code != 0


def test_main_candidates_happy_path(monkeypatch, tmp_path: Path, capsys) -> None:
    """fa candidates loads request, calls enumerate_candidates, emits JSON."""
    req_payload = {
        "framework": "sglang",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": str(tmp_path / "w"),
        "baseline": {"throughput": 1.0},
        "candidate_refs": ["main"],
    }
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req_payload), encoding="utf-8")

    # Stub the dispatcher so the CLI never tries the network.
    import framework_agent.sources as src

    def fake_enum(r):
        return [Candidate(ref="main", repo=r.repo_url, source="explicit")]

    monkeypatch.setattr(src, "enumerate_candidates", fake_enum)

    rc = cli.main(["candidates", "--request", str(req_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["candidates"][0]["ref"] == "main"


def test_main_candidates_bad_request_exits_two(tmp_path: Path, capsys) -> None:
    """A missing JSON file should yield exit 2 and an ERROR line on stderr."""
    rc = cli.main(["candidates", "--request", str(tmp_path / "missing.json")])
    assert rc == 2
    assert "request file not found" in capsys.readouterr().err


def test_main_candidates_bad_json_exits_two(tmp_path: Path, capsys) -> None:
    """An invalid JSON request file should yield exit 2."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    rc = cli.main(["candidates", "--request", str(bad)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not valid JSON" in err


def test_main_explore_plan_happy_path(monkeypatch, tmp_path: Path, capsys) -> None:
    """fa explore (plan) calls explorer.explore(execute=False) and emits JSON."""
    req_payload = {
        "framework": "sglang",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": str(tmp_path / "w"),
        "baseline": {"throughput": 1.0},
        "candidate_refs": ["main"],
    }
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req_payload), encoding="utf-8")

    import framework_agent.explorer as ex

    captured = {}

    def fake_explore(req, *, execute=False):
        captured["execute"] = execute
        return {"mode": "plan" if not execute else "execute", "ok": True}

    monkeypatch.setattr(ex, "explore", fake_explore)
    rc = cli.main(["explore", "--request", str(req_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"mode": "plan", "ok": True}
    assert captured["execute"] is False


def test_main_explore_execute_flag_propagates(monkeypatch, tmp_path: Path) -> None:
    """--execute must reach explorer.explore as execute=True."""
    req_payload = {
        "framework": "sglang",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": str(tmp_path / "w"),
        "baseline": {"throughput": 1.0},
        "candidate_refs": ["main"],
    }
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req_payload), encoding="utf-8")

    import framework_agent.explorer as ex

    captured = {}

    def fake_explore(req, *, execute=False):
        captured["execute"] = execute
        return {"mode": "execute" if execute else "plan", "ok": True}

    monkeypatch.setattr(ex, "explore", fake_explore)
    rc = cli.main(["explore", "--request", str(req_path), "--execute"])
    assert rc == 0
    assert captured["execute"] is True
