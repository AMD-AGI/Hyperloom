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
from hyperloom_multibranch.specialist.agent_onboarding import AttemptFailed, GatewayDown, ModelRefused
from hyperloom_multibranch.specialist.agent_session import AgentCommand
from hyperloom_multibranch.specialist.dsh_cli import MODULE as DSH_CLI
from hyperloom_multibranch.specialist.dsh_cli import PACKAGES as APP_PACKAGES

MODEL = "claude-sonnet-5-5"
SUB_MODEL = "claude-haiku-5-5"
TOO_OLD = (
    "API Error: 400 {'error': 'Claude Code 2.1.273 does not support this model; version 2.1.280 or newer is required'}"
)
UNREACHABLE = "API Error: Unable to connect to API (ERR_PROXY_TUNNEL)"
API_ERRORS = {
    "api_too_old": (400, TOO_OLD),
    "api_overloaded": (529, "API Error: 529 overloaded"),
    "api_unreachable": (None, UNREACHABLE),
}

# Stands in for `claude --print --output-format stream-json`: records how it was run, then streams as FAKE_MODE says.
FAKE_CLI = """#!{python}
import json, os, sys, time

API_ERRORS = {api_errors!r}

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
if mode.startswith("retried"):
    # A CLI retries a model call that failed to connect; one that died meanwhile prints no result.
    print(json.dumps({{"type": "system", "subtype": "init", "session_id": "s-1"}}))
    for attempt in range(1, 4):
        print(json.dumps({{"type": "system", "subtype": "api_retry", "attempt": attempt, "max_retries": 10}}))
    if mode == "retried_then_answered":
        message = {{"id": "m0", "model": "{model}", "content": []}}
        print(json.dumps({{"type": "assistant", "message": message, "session_id": "s-1"}}))
    print("killed", file=sys.stderr)
    sys.exit(1)
if mode.startswith("api_"):
    # How the CLI streams a model call the gateway refused: a synthetic assistant turn and an error result, exit 1.
    status, text = API_ERRORS[mode]
    print(json.dumps({{"type": "system", "subtype": "init", "session_id": "s-1"}}))
    for line in range(30):
        print(json.dumps({{"type": "system", "subtype": "status", "line": line}}))
    message = {{"id": "m0", "model": "<synthetic>", "content": [{{"type": "text", "text": text}}]}}
    row = {{"type": "assistant", "message": message, "is_api_error_message": True, "api_error_status": status}}
    print(json.dumps(row))
    result = {{"type": "result", "is_error": True, "api_error_status": status, "result": text, "session_id": "s-1"}}
    print(json.dumps(result))
    sys.exit(1)
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


def _stand_in(tmp_path: Path, script: str, command: AgentCommand, mode: str):
    """The stand-in ``script`` as ``command``'s CLI, the environment that runs it in ``mode``, the checkout it reads
    and where it records how it was run."""
    cli = Path(command.executable)
    cli.write_text(script, encoding="utf-8")
    cli.chmod(0o755)
    checkout = tmp_path / "agent" / "src"
    checkout.mkdir(parents=True)
    record = tmp_path / "record.json"
    env = {"PATH": os.environ.get("PATH", ""), "FAKE_MODE": mode, "FAKE_RECORD": str(record)}
    return command, env, checkout, record


def _setup(tmp_path: Path, mode: str = "ok") -> tuple[AgentCommand, dict[str, str], Path, Path]:
    script = FAKE_CLI.format(python=sys.executable, model=MODEL, sub_model=SUB_MODEL, api_errors=API_ERRORS)
    return _stand_in(tmp_path, script, AgentCommand("claude", str(tmp_path / "claude"), MODEL), mode)


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


def test_a_deepseek_harness_run_is_its_bridge_s_one_shot_in_the_composition_that_only_reads() -> None:
    argv = onboard_argv(AgentCommand("deepseek", sys.executable, HARNESS_MODEL))

    assert argv == (sys.executable, "-m", DSH_CLI, "--print", "--read-only", "--model", HARNESS_MODEL)
    assert "--mcp-config" not in argv and "--system-prompt-file" not in argv


# Stands in for the harness's bridge run as `python -m ... --print --read-only`: records how it was run, the home it
# was given and that it could write there, then streams what the bridge streams for FAKE_MODE.
FAKE_BRIDGE = """#!{python}
import json, os, pathlib, sys

home = pathlib.Path(os.environ["DSH_HOME"])
(home / "sessions").mkdir(parents=True)
record = {{"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": sys.stdin.read(), "home": str(home)}}
record["imports"] = os.environ.get("PYTHONPATH")
with open(os.environ["FAKE_RECORD"], "w") as handle:
    json.dump(record, handle)
mode = os.environ.get("FAKE_MODE", "ok")
usage = {{"input_tokens": 10, "output_tokens": 40, "cache_read_input_tokens": 30, "cache_creation_input_tokens": 0}}
print(json.dumps({{"type": "system", "subtype": "init", "session_id": "s-1", "harness": "dsh"}}))
if mode == "ok":
    message = {{"id": "m0", "model": "{model}", "usage": usage, "content": []}}
    print(json.dumps({{"type": "assistant", "message": message, "session_id": "s-1"}}))
    result = {{"type": "result", "is_error": False, "result": '{{"source": "agent"}}', "usage": usage}}
    print(json.dumps({{**result, "session_id": "s-1"}}))
    sys.exit(0)
if mode == "unreachable":
    for attempt in range(1, 6):
        print(json.dumps({{"type": "system", "subtype": "api_retry", "attempt": attempt, "max_retries": 5}}))
    failed = {{"result": "TRANSPORT: DeepSeek Messages transport failed"}}
else:
    failed = {{"result": "AUTH: invalid key", "api_error_status": 401}}
print(json.dumps({{"type": "result", "is_error": True, "num_turns": 1, "session_id": "s-1", **failed}}))
sys.exit(1)
"""
HARNESS_MODEL = "deepseek-v4-flash"


def _harness(tmp_path: Path, mode: str = "ok") -> tuple[AgentCommand, dict[str, str], Path, Path]:
    script = FAKE_BRIDGE.format(python=sys.executable, model=HARNESS_MODEL)
    return _stand_in(tmp_path, script, AgentCommand("deepseek", str(tmp_path / "python"), HARNESS_MODEL), mode)


async def test_a_deepseek_harness_answers_from_a_home_of_the_run_s_own_that_goes_with_it(tmp_path: Path) -> None:
    command, env, checkout, record = _harness(tmp_path)
    usage = _Usage()

    answer = await cli_onboard(command, env, "b07", usage)(checkout, "read the checkout")

    assert answer == '{"source": "agent"}'
    ran = json.loads(record.read_text(encoding="utf-8"))
    assert (ran["cwd"], ran["stdin"]) == (str(checkout), "read the checkout")
    assert ran["argv"] == ["-m", DSH_CLI, "--print", "--read-only", "--model", HARNESS_MODEL]
    # The home is beside the log, never in the checkout, and nothing of the conversation is left once the run ended.
    assert ran["home"] == str(tmp_path / "agent" / "onboard-1.dsh") and not Path(ran["home"]).exists()
    # The bridge is a module of this app, which the environment the run was given does not say where to import from.
    assert ran["imports"] == str(APP_PACKAGES)
    [recorded] = usage.recorded
    assert (recorded.branch, recorded.tokens, recorded.launch) == ("b07", TokenCounts(10, 40, 30), "s-1")
    assert recorded.cost_usd == pytest.approx(price_of(HARNESS_MODEL).cost_usd(TokenCounts(10, 40, 30)))


async def test_a_deepseek_harness_that_could_not_reach_the_gateway_is_the_gateway_s_failure(tmp_path: Path) -> None:
    command, env, checkout, record = _harness(tmp_path, "unreachable")

    with pytest.raises(GatewayDown) as failed:
        await cli_onboard(command, env, "b07", _Usage())(checkout, "p")

    assert str(failed.value) == "the onboarding model's call failed: TRANSPORT: DeepSeek Messages transport failed"
    assert not Path(json.loads(record.read_text(encoding="utf-8"))["home"]).exists()


async def test_a_deepseek_harness_the_gateway_refused_ends_the_preflight_naming_the_refusal(tmp_path: Path) -> None:
    command, env, checkout, _ = _harness(tmp_path, "refused")

    with pytest.raises(ModelRefused) as refused:
        await cli_onboard(command, env, "b07", _Usage())(checkout, "p")

    assert str(refused.value) == "the onboarding model's call failed: AUTH: invalid key"


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
        assert _ended(pid)


def _ended(pid: int) -> bool:
    # Under a PID 1 that reaps nothing, a killed orphan stays a zombie, which `os.kill(pid, 0)` still finds.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return True
    return stat.rsplit(") ", 1)[1].startswith("Z")


async def test_a_model_call_the_gateway_refuses_ends_the_attempt_naming_the_api_error(tmp_path: Path) -> None:
    command, env, checkout, _ = _setup(tmp_path, "api_too_old")

    with pytest.raises(ModelRefused) as refused:
        await cli_onboard(command, env, "b07", _Usage())(checkout, "p")

    assert str(refused.value) == f"the onboarding model's call failed: {TOO_OLD}"


async def test_a_model_call_that_failed_for_a_moment_fails_only_its_attempt_naming_it(tmp_path: Path) -> None:
    command, env, checkout, _ = _setup(tmp_path, "api_overloaded")

    with pytest.raises(AttemptFailed) as failed:
        await cli_onboard(command, env, "b07", _Usage())(checkout, "p")

    assert isinstance(failed.value, GatewayDown)
    assert str(failed.value) == "the onboarding model's call failed: API Error: 529 overloaded"


async def test_a_model_call_that_could_not_reach_the_gateway_is_the_gateway_s_failure(tmp_path: Path) -> None:
    command, env, checkout, _ = _setup(tmp_path, "api_unreachable")

    with pytest.raises(GatewayDown) as failed:
        await cli_onboard(command, env, "b07", _Usage())(checkout, "p")

    assert str(failed.value) == f"the onboarding model's call failed: {UNREACHABLE}"


async def test_a_run_that_died_retrying_its_model_call_is_the_gateway_s_failure(tmp_path: Path) -> None:
    command, env, checkout, _ = _setup(tmp_path, "retried")

    with pytest.raises(GatewayDown, match="exited 1"):
        await cli_onboard(command, env, "b07", _Usage())(checkout, "p")


async def test_a_run_that_died_after_its_retried_call_was_answered_is_its_own_failure(tmp_path: Path) -> None:
    command, env, checkout, _ = _setup(tmp_path, "retried_then_answered")

    with pytest.raises(AttemptFailed, match="exited 1") as failed:
        await cli_onboard(command, env, "b07", _Usage())(checkout, "p")

    assert not isinstance(failed.value, GatewayDown)
