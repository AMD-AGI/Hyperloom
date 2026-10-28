# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.llm_prices import TokenCounts, price_of
from hyperloom_multibranch.contracts.blackboard.records import LlmUsage
from hyperloom_multibranch.contracts.blackboard.vocab import UsageSource
from hyperloom_multibranch.specialist.agent_onboard import ALLOWED_TOOLS, TOOLS, cli_onboard, onboard_argv
from hyperloom_multibranch.specialist.agent_onboarding import AttemptFailed
from hyperloom_multibranch.specialist.agent_session import AgentCommand

MODEL = "claude-sonnet-5-5"
SUB_MODEL = "claude-haiku-5-5"

# Stands in for `claude --print --output-format stream-json`: records how it was run, then streams as FAKE_MODE says.
FAKE_CLI = """#!{python}
import json, os, sys, time

record = {{"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": sys.stdin.read(), "pid": os.getpid()}}
with open(os.environ["FAKE_RECORD"], "w") as handle:
    json.dump(record, handle)
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "hang":
    import subprocess
    usage = {{"input_tokens": 10, "output_tokens": 40}}
    message = {{"id": "m0", "model": "{model}", "usage": usage, "content": []}}
    print(json.dumps({{"type": "assistant", "message": message, "session_id": "s-1"}}), flush=True)
    # A Claude CLI runs each shell command in a session of its own.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
    with open(os.environ["FAKE_RECORD"] + ".child", "w") as handle:
        handle.write(str(child.pid))
    time.sleep(600)
if mode == "crash":
    print("line one\\nauth failed: no key", file=sys.stderr)
    sys.exit(3)
print(json.dumps({{"type": "system", "subtype": "init", "session_id": "s-1"}}))
for number, (model, usage) in enumerate([
    ("{model}", {{"input_tokens": 10, "output_tokens": 40, "cache_read_input_tokens": 30,
                 "cache_creation_input_tokens": 20}}),
    ("{sub_model}", {{"input_tokens": 5, "output_tokens": 6}}),
]):
    message = {{"id": f"m{{number}}", "model": model, "usage": usage, "content": []}}
    print(json.dumps({{"type": "assistant", "message": message, "session_id": "s-1"}}))
error = mode == "error"
answer = "max turns reached" if error else '{{"source": "agent"}}'
print(json.dumps({{"type": "result", "is_error": error, "result": answer, "session_id": "s-1"}}))
"""


class _Usage:
    def __init__(self) -> None:
        self.recorded: list[LlmUsage] = []

    async def record_llm_usage(self, usage: LlmUsage) -> None:
        self.recorded.append(usage)


def _setup(tmp_path: Path, mode: str = "ok") -> tuple[AgentCommand, dict[str, str], Path, Path]:
    cli = tmp_path / "claude"
    cli.write_text(FAKE_CLI.format(python=sys.executable, model=MODEL, sub_model=SUB_MODEL), encoding="utf-8")
    cli.chmod(0o755)
    checkout = tmp_path / "agent" / "src"
    checkout.mkdir(parents=True)
    record = tmp_path / "record.json"
    env = {"PATH": os.environ.get("PATH", ""), "FAKE_MODE": mode, "FAKE_RECORD": str(record)}
    return AgentCommand("claude", str(cli), MODEL), env, checkout, record


async def test_an_answer_is_the_run_s_result_and_its_calls_are_the_branch_s_spend(tmp_path: Path) -> None:
    command, env, checkout, record = _setup(tmp_path)
    usage = _Usage()

    answer = await cli_onboard(command, env, "b07", usage)(checkout, "read the checkout")

    assert answer == '{"source": "agent"}'
    ran = json.loads(record.read_text(encoding="utf-8"))
    assert (ran["cwd"], ran["stdin"]) == (str(checkout), "read the checkout")
    main, sub = TokenCounts(10, 40, 30, 20), TokenCounts(5, 6)
    [recorded] = usage.recorded
    assert (recorded.branch, recorded.source, recorded.tokens, recorded.launch) == (
        "b07", UsageSource.SPECIALIST, main + sub, "s-1",
    )  # fmt: skip
    assert recorded.cost_usd == pytest.approx(price_of(MODEL).cost_usd(main) + price_of(SUB_MODEL).cost_usd(sub))
    assert (tmp_path / "agent" / "onboard-1.jsonl").exists()


def test_a_run_reads_and_nothing_else() -> None:
    argv = onboard_argv(AgentCommand("claude", "claude", MODEL))

    assert argv[:5] == ("claude", "--print", "--output-format", "stream-json", "--verbose")
    assert argv[argv.index("--model") + 1] == MODEL
    assert argv[argv.index("--tools") + 1] == ",".join(TOOLS)
    allowed = argv[argv.index("--allowedTools") + 1].split(",")
    assert allowed == list(ALLOWED_TOOLS) and not {"Edit", "Write", "Bash"} & set(allowed)
    # git log and git show write any file with --output.
    assert not [tool for tool in allowed if tool.startswith(("Bash(git log", "Bash(git show"))]
    assert "--strict-mcp-config" in argv and "--mcp-config" not in argv and "--restricted" in argv
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert "--model" not in onboard_argv(AgentCommand("claude", "claude"))


def test_codex_is_refused() -> None:
    with pytest.raises(ValueError, match="codex"):
        cli_onboard(AgentCommand("codex", "codex"), {}, "b07", _Usage())


async def test_a_run_that_exits_non_zero_fails_the_attempt_with_its_tail(tmp_path: Path) -> None:
    command, env, checkout, _ = _setup(tmp_path, "crash")
    usage = _Usage()

    with pytest.raises(AttemptFailed, match=r"exited 3: line one \| auth failed: no key"):
        await cli_onboard(command, env, "b07", usage)(checkout, "p")

    assert usage.recorded == []


async def test_a_run_that_ends_in_an_error_fails_the_attempt_and_its_spend_counts(tmp_path: Path) -> None:
    command, env, checkout, _ = _setup(tmp_path, "error")
    usage = _Usage()
    onboard = cli_onboard(command, env, "b07", usage)

    with pytest.raises(AttemptFailed, match="ended with an error: max turns reached"):
        await onboard(checkout, "p")
    with pytest.raises(AttemptFailed):
        await onboard(checkout, "p")

    assert len(usage.recorded) == 2
    assert (tmp_path / "agent" / "onboard-2.jsonl").exists()


async def test_a_run_past_its_time_is_killed_with_what_it_started(tmp_path: Path) -> None:
    command, env, checkout, record = _setup(tmp_path, "hang")
    child = Path(f"{record}.child")

    usage = _Usage()

    with pytest.raises(AttemptFailed, match="did not answer within 2 s"):
        await cli_onboard(command, env, "b07", usage, seconds=2.0)(checkout, "p")

    assert [recorded.tokens for recorded in usage.recorded] == [TokenCounts(10, 40)]
    assert child.exists()
    for pid in (json.loads(record.read_text(encoding="utf-8"))["pid"], int(child.read_text(encoding="utf-8"))):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
