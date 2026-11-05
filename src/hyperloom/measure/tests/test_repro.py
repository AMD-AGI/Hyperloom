# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Running a repro case: its exit code is the verdict, anything else is no verdict."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hyperloom.measure.repro import (
    CASE_SCRIPT,
    LOG_FILE,
    REPRO_DIR,
    VERDICT_FILE,
    ReproError,
    find_case,
    read_verdict,
    run_case,
)


def _case(tmp_path: Path, body: str, *, attachment: str = "case") -> Path:
    script = tmp_path / "attach" / attachment / CASE_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(body, encoding="utf-8")
    return script


def _run(script: Path, tmp_path: Path, *, timeout: float = 30) -> str:
    return run_case(script, base_url="http://127.0.0.1:9", model="m", timeout_sec=timeout, output_dir=tmp_path / "out")


@pytest.mark.parametrize(("code", "verdict"), [(0, "not_reproduced"), (1, "reproduced")])
def test_the_exit_code_is_the_verdict_and_the_verdict_is_written_where_the_bundle_carries_it(tmp_path, code, verdict):
    script = _case(tmp_path, f"echo the case ran\nexit {code}\n")

    assert _run(script, tmp_path) == verdict

    out = tmp_path / "out" / REPRO_DIR
    assert json.loads((out / VERDICT_FILE).read_text()) == {"verdict": verdict, "exit_code": code}
    assert (out / LOG_FILE).read_text() == "the case ran\n"
    assert read_verdict(tmp_path / "out") == verdict


@pytest.mark.parametrize("code", [2, 3, 127, 255])
def test_any_other_exit_code_is_no_verdict_and_leaves_no_verdict_file(tmp_path, code):
    script = _case(tmp_path, f"echo broke\nexit {code}\n")

    with pytest.raises(ReproError, match=f"exited with {code}"):
        _run(script, tmp_path)

    assert not (tmp_path / "out" / REPRO_DIR / VERDICT_FILE).exists()
    assert (tmp_path / "out" / REPRO_DIR / LOG_FILE).read_text() == "broke\n", "the log of a failed case is kept"


def test_a_case_killed_by_a_signal_is_no_verdict(tmp_path):
    script = _case(tmp_path, "kill -9 $$\n")

    with pytest.raises(ReproError, match="exited with -9"):
        _run(script, tmp_path)


def test_a_case_that_outlives_its_timeout_is_killed_and_is_no_verdict(tmp_path):
    marker = tmp_path / "survived"
    script = _case(tmp_path, f"(sleep 3; touch {marker}) &\nsleep 30\n")
    started = time.monotonic()

    with pytest.raises(ReproError, match="did not finish within"):
        _run(script, tmp_path, timeout=1)

    assert time.monotonic() - started < 10
    time.sleep(3.5)
    assert not marker.exists(), "the whole process group of the case is killed, not only its shell"
    assert not (tmp_path / "out" / REPRO_DIR / VERDICT_FILE).exists()


def test_the_case_sees_the_server_and_the_model_and_nothing_that_could_be_a_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_API_TOKEN", "do-not-leak")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    script = _case(
        tmp_path, 'echo "$HL_BASE_URL|$HL_MODEL|${SOME_API_TOKEN:-unset}|${HTTPS_PROXY:-unset}|$PWD"\nexit 0\n'
    )

    _run(script, tmp_path)

    seen = (tmp_path / "out" / REPRO_DIR / LOG_FILE).read_text().strip()
    assert seen == f"http://127.0.0.1:9|m|unset|unset|{script.parent}"


def test_only_the_tail_of_a_long_log_is_kept(tmp_path):
    script = _case(tmp_path, "head -c 300000 /dev/zero | tr '\\0' a\necho THE-END\nexit 1\n")

    _run(script, tmp_path)

    log = (tmp_path / "out" / REPRO_DIR / LOG_FILE).read_bytes()
    assert len(log) <= 64 * 1024 and log.endswith(b"THE-END\n")


def test_the_case_is_found_in_the_one_attachment_that_holds_repro_sh(tmp_path):
    _case(tmp_path, "exit 0\n", attachment="b")
    (tmp_path / "attach" / "a").mkdir()
    (tmp_path / "attach" / "data").write_text("a file attachment")

    assert find_case(tmp_path / "attach", ["a", "b", "data"]) == tmp_path / "attach" / "b" / CASE_SCRIPT


def test_no_case_is_no_run(tmp_path):
    (tmp_path / "attach" / "a").mkdir(parents=True)

    with pytest.raises(ReproError, match="no attachment holds repro.sh"):
        find_case(tmp_path / "attach", ["a"])


def test_two_cases_are_ambiguous_not_a_pick(tmp_path):
    _case(tmp_path, "exit 0\n", attachment="a")
    _case(tmp_path, "exit 1\n", attachment="b")

    with pytest.raises(ReproError, match="more than one attachment"):
        find_case(tmp_path / "attach", ["a", "b"])


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (None, "no readable repro/verdict.json"),
        ("not json", "no readable repro/verdict.json"),
        ('["reproduced"]', "holds None"),
        ('{"verdict": "fixed"}', "holds 'fixed'"),
        ('{"verdict": ""}', "holds ''"),
    ],
)
def test_a_landed_bundle_without_a_real_verdict_does_not_read_as_one(tmp_path, content, match):
    if content is not None:
        (tmp_path / REPRO_DIR).mkdir()
        (tmp_path / REPRO_DIR / VERDICT_FILE).write_text(content)

    with pytest.raises(ReproError, match=match):
        read_verdict(tmp_path)


def test_a_case_that_floods_its_output_is_killed_and_is_no_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr("hyperloom.measure.repro._MAX_FILE_BYTES", 1024 * 1024)
    script = _case(tmp_path, "yes\nexit 1\n")

    with pytest.raises(ReproError, match="exited with -25"):  # SIGXFSZ
        _run(script, tmp_path)

    assert not (tmp_path / "out" / REPRO_DIR / VERDICT_FILE).exists()
