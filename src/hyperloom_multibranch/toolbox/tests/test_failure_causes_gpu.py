# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A failed test names the cause its server log states, on a real server (W90).

``hyperloom.measure.log_causes`` and ``boot_watch`` were proven on fixtures. Here each cause is made by a real SGLang
server on a real card, through the real toolbox: a flag the server refuses (``serve_flag``), an import that fails at
boot (``import_error``), a Python exception at boot (``python_exception``), an evaluation request the server answers
``400`` (``lm_eval_http_400``) and a boot that is not ready within its limit (``server_boot_timeout``, return code
-924). What was recorded is printed (``PROBE``): the result's failure fields, the log lines that state the cause, and
whether the cards came back.

Needs what ``test_toolbox_kinds_gpu`` needs; anywhere else the module is skipped.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from hyperloom.contracts.measurement import FailureClass, Purpose, TestKind, TestResult
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.measure.returncodes import SERVER_BOOT_TIMEOUT_RC
from hyperloom_multibranch.toolbox.tests.helpers import req
from hyperloom_multibranch.toolbox.tests.test_toolbox_kinds_gpu import (  # noqa: F401  (workload_environment is a fixture)
    CARDS,
    _probe,
    _run_root,
    _toolbox,
    pytestmark,
    workload_environment,
)

#: The lines of a log that state what went wrong, or that a boot got somewhere.
_CAUSE_LINES = re.compile(r"error:|Error|Exception|Traceback|fired up|400", re.IGNORECASE)
_LOG_LINES_SHOWN = 6
_CARDS_BACK_WITHIN_SEC = 60
SERVER_PROCESSES = ["pgrep", "-af", "sglang::|launch_server"]


@dataclass(frozen=True)
class Case:
    """One test to submit and the reason prefix its failure must carry (``None``: it must succeed)."""

    name: str
    request_args: dict
    delta: RecipeDelta
    reason_starts: str | None
    failure_class: FailureClass | None
    attempts: int = 1


def _delta(server_args: str = "", **envs: str) -> RecipeDelta:
    return RecipeDelta(config=ConfigDelta(server_args=f"--max-running-requests 64 {server_args}".strip(), envs=envs))


CASES = (
    Case("bad_flag", {}, _delta("--foo-bar 1"), "serve_flag: ", FailureClass.CANDIDATE),
    Case("bad_choice", {}, _delta("--attention-backend nonexistent"), "serve_flag: ", FailureClass.CANDIDATE),
    Case("import", {}, _delta("--load-format fastsafetensors"), "import_error", FailureClass.CANDIDATE),
    Case("exception", {}, _delta("--json-model-override-args notjson"), "python_exception: ", FailureClass.UNCLEAR, 2),
    Case(
        "eval_400",
        {"kind": TestKind.ACCURACY, "accuracy": True},
        _delta("--context-length 400"),
        "lm_eval_http_400: ",
        FailureClass.UNCLEAR,
        2,
    ),
)


def _logs_of(root: Path, request_id: str) -> dict[str, str]:
    """Every server and benchmark-stderr log an attempt of the test landed, by path below the test's directory."""
    base = root / "session" / "tests" / request_id
    paths = [*base.rglob("server*.log"), *base.rglob("benchmark_stderr.log")]
    return {str(p.relative_to(base)): p.read_text(errors="replace") for p in sorted(paths)}


def _cause_lines(text: str) -> list[str]:
    return [line.strip()[:300] for line in text.splitlines() if _CAUSE_LINES.search(line)][-_LOG_LINES_SHOWN:]


def _rc_files(root: Path, request_id: str, code: int) -> list[str]:
    """The landed files that carry ``code`` as a return code."""
    base = root / "session" / "tests" / request_id
    found = []
    for path in sorted(base.rglob("*.json")):
        if re.search(rf'"returncode":\s*{code}\b', path.read_text(errors="replace")):
            found.append(str(path.relative_to(base)))
    return found


async def _cards_back(toolbox) -> float | None:
    """Seconds until every card of the pool is free again, ``None`` when some stay leased."""
    began = time.monotonic()
    while time.monotonic() - began < _CARDS_BACK_WITHIN_SEC:
        if sorted(toolbox.provider.pool.free_cards()) == sorted(CARDS):
            return round(time.monotonic() - began, 1)
        await asyncio.sleep(1)
    return None


def _servers_left() -> list[str]:
    done = subprocess.run(SERVER_PROCESSES, capture_output=True, text=True, check=False)
    return [line[:160] for line in done.stdout.splitlines() if "pgrep" not in line]


def _facts(root: Path, case: str, request_id: str, result: TestResult) -> dict[str, object]:
    logs = _logs_of(root, request_id)
    facts = {
        "case": case,
        "validity": result.validity.value,
        "failure_class": result.failure_class.value if result.failure_class else None,
        "failure_reason": result.failure_reason,
        "attempts": [(a.failure_class.value if a.failure_class else None, a.failure_reason) for a in result.attempts],
        "wall_sec": None if result.finished_at is None else round(result.finished_at - (result.started_at or 0), 1),
        "log_cause_lines": {name: _cause_lines(text) for name, text in logs.items()},
        "log_tail": {name: text.splitlines()[-3:] for name, text in logs.items()},
        "rc_924_in": _rc_files(root, request_id, SERVER_BOOT_TIMEOUT_RC),
    }
    _probe("case", **facts)
    return facts


async def _run(root_name: str, name: str, requests: list, timeout_sec: int) -> tuple[Path, dict]:
    """Submit every request to one toolbox, wait for them all and return each one's facts and the pool's state."""
    root = _run_root(root_name)
    toolbox, *_ = _toolbox(root, name, timeout_sec=timeout_sec)
    results: dict[str, tuple[str, TestResult]] = {}
    try:
        await toolbox.start()
        ids = {label: await toolbox.submit_test(request) for label, request in requests}
        await asyncio.wait_for(toolbox.drain(), 3300)
        for label, request_id in ids.items():
            results[label] = (request_id, await toolbox.outcome(request_id))
        back = await _cards_back(toolbox)
    finally:
        await toolbox.close()
    facts = {label: _facts(root, label, rid, result) for label, (rid, result) in results.items()}
    _probe("pool", name=name, cards_back_after_sec=back, servers_left=_servers_left(), cards=sorted(CARDS))
    return root, {"facts": facts, "results": results, "cards_back": back}


def _check(case: Case, fact: dict[str, object]) -> list[str]:
    wrong = []
    reason, failure_class = fact["failure_reason"] or "", fact["failure_class"]
    if case.reason_starts is not None and not str(reason).startswith(case.reason_starts):
        wrong.append(f"{case.name}: reason {reason!r} does not start with {case.reason_starts!r}")
    if case.failure_class is not None and failure_class != case.failure_class.value:
        wrong.append(f"{case.name}: class {failure_class!r}, expected {case.failure_class.value!r}")
    if len(fact["attempts"]) != case.attempts:
        wrong.append(f"{case.name}: {len(fact['attempts'])} attempts, expected {case.attempts}")
    return wrong


async def test_a_failed_boot_names_the_cause_its_server_log_states() -> None:
    requests = [
        (case.name, req(f"b-{case.name}", purpose=Purpose.EXPLORE, change=case.delta, **case.request_args))
        for case in CASES
    ]
    _, done = await _run("w90-causes", "w90-causes", requests, timeout_sec=2400)

    wrong = [problem for case in CASES for problem in _check(case, done["facts"][case.name])]
    assert not wrong, wrong
    assert done["cards_back"] is not None, "every card is back in the pool"


async def test_a_boot_not_ready_within_its_limit_ends_at_the_boot_limit_and_frees_the_card() -> None:
    # The limit is half of what a round may take: a 40 s round gives a server 20 s, which no SGLang boot keeps.
    request = req("b-timeout", purpose=Purpose.EXPLORE, change=_delta())
    _, done = await _run("w90-boot-limit", "w90-boot-limit", [("boot_limit", request)], timeout_sec=40)

    fact = done["facts"]["boot_limit"]
    assert str(fact["failure_reason"]).startswith("server_boot_timeout: the server was not ready within 20s"), fact
    assert "; its log ends: " in str(fact["failure_reason"]), "the reason shows where the server's log stopped"
    assert fact["failure_class"] == FailureClass.CANDIDATE.value and len(fact["attempts"]) == 1, fact
    assert fact["rc_924_in"], "the return code -924 is in what the test landed"
    assert done["cards_back"] is not None, "the card is back in the pool"
