# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.llm_prices import TokenCounts, price_of
from hyperloom_multibranch.contracts.blackboard.vocab import UsageSource
from hyperloom_multibranch.specialist.agent_spend import AgentSpend, transcript_dir

MODEL = "claude-sonnet-5-5"
AGENT_MODEL = "claude-opus-5-5"


class _Usage:
    def __init__(self) -> None:
        self.recorded: list = []

    async def record_llm_usage(self, usage) -> None:
        self.recorded.append(usage)


def _message(message_id: str, session: str, output: int, model: str = AGENT_MODEL) -> str:
    usage = {"input_tokens": 2, "output_tokens": output, "cache_read_input_tokens": 100}
    message = {"id": message_id, "model": model, "usage": usage, "content": []}
    return json.dumps({"type": "assistant", "sessionId": session, "message": message}) + "\n"


def _transcript(config: Path, cwd: Path, session: str) -> Path:
    path = transcript_dir(config, cwd) / f"{session}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def test_the_agent_s_cli_calls_under_its_branch_are_the_branch_s_spend_each_counted_once(tmp_path: Path) -> None:
    config, branch_dir = tmp_path / "claude", tmp_path / "MB-branches" / "b01"
    usage = _Usage()
    spend = AgentSpend(config, (branch_dir,), tmp_path / "spend.json", "b01", MODEL, usage)
    agent = _transcript(config, branch_dir / "session" / "agent" / "src", "s-1")
    other = _transcript(config, tmp_path / "MB-branches" / "b010", "s-9")
    # A message streamed as several blocks repeats its usage on each line; a line still being written is not read.
    agent.write_text(_message("m1", "s-1", 40) + _message("m1", "s-1", 40) + _message("m2", "s-1", 7)[:30])
    other.write_text(_message("m9", "s-9", 1000))

    await spend.record()
    with agent.open("a") as handle:
        handle.write(_message("m2", "s-1", 7)[30:] + _message("m3", "s-1", 5, model=""))
    await AgentSpend(config, (branch_dir,), tmp_path / "spend.json", "b01", MODEL, usage).record()
    await spend.record()

    first, second = usage.recorded
    price = price_of(AGENT_MODEL)
    assert (first.branch, first.source, first.launch, first.tokens) == (
        "b01",
        UsageSource.SPECIALIST,
        "s-1",
        TokenCounts(2, 40, 100),
    )
    assert first.cost_usd == pytest.approx(price.cost_usd(TokenCounts(2, 40, 100)))
    assert second.tokens == TokenCounts(4, 12, 200)
    assert second.cost_usd == pytest.approx(
        price.cost_usd(TokenCounts(2, 7, 100)) + price_of(MODEL).cost_usd(TokenCounts(2, 5, 100))
    )


async def test_an_error_the_cli_writes_itself_leaves_the_session_s_real_calls_priced(tmp_path: Path) -> None:
    config, branch_dir = tmp_path / "claude", tmp_path / "b01"
    usage = _Usage()
    agent = _transcript(config, branch_dir, "s-1")
    synthetic = {"id": "e1", "model": "<synthetic>", "usage": {"input_tokens": 0, "output_tokens": 0}, "content": []}
    error = json.dumps({"type": "assistant", "sessionId": "s-1", "message": synthetic}) + "\n"
    agent.write_text(_message("m1", "s-1", 40) + error)

    await AgentSpend(config, (branch_dir,), tmp_path / "spend.json", "b01", MODEL, usage).record()

    [recorded] = usage.recorded
    assert recorded.cost_usd == pytest.approx(price_of(AGENT_MODEL).cost_usd(TokenCounts(2, 40, 100)))


async def test_an_agent_that_keeps_no_transcript_spends_nothing_recorded(tmp_path: Path) -> None:
    usage = _Usage()

    await AgentSpend(tmp_path / "claude", (tmp_path / "b01",), tmp_path / "spend.json", "b01", MODEL, usage).record()

    assert usage.recorded == []


class _FailingOnce(_Usage):
    """A sink whose ``fail_at``-th write fails once."""

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.calls, self.fail_at = 0, fail_at

    async def record_llm_usage(self, usage) -> None:
        self.calls += 1
        if self.calls == self.fail_at:
            raise OSError("the blackboard is unwritable")
        await super().record_llm_usage(usage)


async def test_spend_the_sink_failed_to_record_is_recorded_at_the_next_read_and_the_rest_once(tmp_path: Path) -> None:
    config, branch_dir = tmp_path / "claude", tmp_path / "b01"
    usage = _FailingOnce(fail_at=2)
    spend = AgentSpend(config, (branch_dir,), tmp_path / "spend.json", "b01", MODEL, usage)
    _transcript(config, branch_dir, "s-1").write_text(_message("m1", "s-1", 40))
    _transcript(config, branch_dir, "s-2").write_text(_message("m2", "s-2", 7))

    with pytest.raises(OSError, match="unwritable"):
        await spend.record()
    await spend.record()
    await spend.record()

    assert sorted((item.launch, item.tokens) for item in usage.recorded) == [
        ("s-1", TokenCounts(2, 40, 100)),
        ("s-2", TokenCounts(2, 7, 100)),
    ]


async def test_a_ledger_write_torn_midway_leaves_the_last_whole_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, branch_dir = tmp_path / "claude", tmp_path / "b01"
    usage = _Usage()
    spend = AgentSpend(config, (branch_dir,), tmp_path / "spend.json", "b01", MODEL, usage)
    agent = _transcript(config, branch_dir, "s-1")
    agent.write_text(_message("m1", "s-1", 40))
    await spend.record()
    with agent.open("a") as handle:
        handle.write(_message("m2", "s-1", 7))
    write_text = Path.write_text

    def torn(path: Path, data: str, *args, **kwargs) -> int:
        write_text(path, data[: len(data) // 2], *args, **kwargs)
        raise OSError("no space left on device")

    with monkeypatch.context() as patched:
        patched.setattr(Path, "write_text", torn)
        with pytest.raises(OSError, match="no space"):
            await spend.record()
    await spend.record()

    # The torn write cost one double count of m2, not every later read.
    assert [item.tokens for item in usage.recorded] == [TokenCounts(2, 40, 100), *[TokenCounts(2, 7, 100)] * 2]


async def test_a_ledger_that_is_no_ledger_is_refused_naming_it_and_records_nothing(tmp_path: Path) -> None:
    config, branch_dir = tmp_path / "claude", tmp_path / "b01"
    usage = _Usage()
    ledger = tmp_path / "spend.json"
    ledger.write_text('{"offsets": {"a": 1')
    _transcript(config, branch_dir, "s-1").write_text(_message("m1", "s-1", 40))

    with pytest.raises(ValueError, match="spend.json"):
        await AgentSpend(config, (branch_dir,), ledger, "b01", MODEL, usage).record()

    assert usage.recorded == [] and ledger.read_text() == '{"offsets": {"a": 1'
