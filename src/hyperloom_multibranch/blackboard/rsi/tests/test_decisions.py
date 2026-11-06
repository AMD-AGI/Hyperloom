# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The judge's exchanges beside the archive: written once, without secrets or host paths, read back for R2."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom_multibranch.blackboard.rsi.archive import TreeArchive
from hyperloom_multibranch.blackboard.rsi.decisions import KEPT, archive_decisions, decision_calls

KEY = "sk-ant-api03-" + "abcdefghij" * 6


def test_the_judges_exchanges_go_beside_the_archive_once_without_secrets_or_host_paths(tmp_path: Path) -> None:
    transcript = tmp_path / "MB-judge.jsonl"
    lines = [
        {
            "at": 10.0,
            "prompt": "b1 stalled 7 tests; its log is /home/someone/run/MB-x/b1.log",
            "reply": "{}",
            "error": "",
        },
        {"at": 20.0, "prompt": f"the key {KEY} leaked into the view", "reply": '{"stop": []}', "error": "late"},
    ]
    transcript.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    archive = TreeArchive(tmp_path / "archive")

    kept = archive_decisions(transcript, archive, "sess-1")
    again = archive_decisions(transcript, archive, "sess-1")
    calls = decision_calls(archive, "sess-1")

    assert (kept, again, len(calls)) == (2, None, 2)
    assert set(calls[0]) <= set(KEPT) and calls[0]["at"] == 10.0 and calls[1]["reply"] == '{"stop": []}'
    assert "/home/someone" not in str(calls[0]["prompt"]) and KEY not in str(calls[1]["prompt"])
    assert decision_calls(archive, "another") == ()
