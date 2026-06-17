# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the offline Langfuse backfill plan builder.

Focuses on the recipe-snapshot / gbrain audit folding (the live emitter's
``_flush_recipe_kb_audit`` counterpart): a finished session's
``runtime/recipe_snapshot/.audit.jsonl`` must surface in the backfill plan so
a recovery replay re-creates the KB read spans on the same trace.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.scripts import backfill_langfuse as bf
from inference_optimizer.session_paths import recipe_snapshot_audit_jsonl


def _seed(session_dir: Path) -> None:
    tdir = session_dir / "reports" / "trace"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "llm_calls.jsonl").write_text(
        json.dumps({
            "session_id": "SID", "component": "critic", "role": "critic",
            "model": "gpt-5.4", "ts": "2026-06-09T15:14:54Z",
            "input_tokens": 10, "output_tokens": 5,
        }) + "\n",
        encoding="utf-8",
    )
    (session_dir / "manifest.json").write_text(
        json.dumps({"session_id": "SID", "model_name": "M"}), encoding="utf-8",
    )


def test_build_plan_includes_recipe_audit(tmp_path):
    sd = tmp_path / "SID"
    _seed(sd)
    audit = recipe_snapshot_audit_jsonl(sd)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("\n".join(json.dumps(r) for r in [
        {"ts": "2026-06-09T15:14:54Z", "method": "get_recipe",
         "remote": "gbrain", "resolution": "remote", "hit": True},
        {"ts": "2026-06-09T15:14:55Z", "method": "search",
         "remote": "cortex", "resolution": "local", "hit": False},
    ]) + "\n", encoding="utf-8")

    plan = bf.build_plan(sd)
    assert plan["stats"]["recipe_audit"] == 2
    assert len(plan["recipe_audit"]) == 2
    assert plan["recipe_audit"][0]["method"] == "get_recipe"


def test_build_plan_recipe_audit_empty_when_absent(tmp_path):
    sd = tmp_path / "SID"
    _seed(sd)
    plan = bf.build_plan(sd)
    assert plan["stats"]["recipe_audit"] == 0
    assert plan["recipe_audit"] == []


def test_print_plan_reports_recipe_reads(tmp_path, capsys):
    sd = tmp_path / "SID"
    _seed(sd)
    audit = recipe_snapshot_audit_jsonl(sd)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        json.dumps({"ts": "2026-06-09T15:14:54Z", "method": "get_recipe",
                    "remote": "gbrain", "resolution": "remote", "hit": True}) + "\n",
        encoding="utf-8",
    )
    bf.print_plan(bf.build_plan(sd))
    out = capsys.readouterr().out
    assert "Recipe-snapshot reads: 1 audit row(s)" in out
