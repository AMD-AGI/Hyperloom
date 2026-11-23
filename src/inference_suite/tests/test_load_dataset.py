# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the fixed load's prompt-dataset client: the requests it sends, the result it saves, and a fixed
round driving it through warmup and measured passes."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from inference_suite.contracts import PromptDataset, WorkloadKind
from inference_suite.load import dataset_client, fixed
from inference_suite.load.fixed import (
    ClientPass,
    ClientTarget,
    ClientTool,
    FixedRound,
    client_argv,
    run_fixed,
)
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.runtime.server import Server
from inference_suite.runtime.watch import RoundGate
from inference_suite.tests.test_config_contract import _FIXED
from inference_suite.tests.test_load import _DEADLINES, _round_config

_TOKENS = ("Hel", "lo", " there", "!")


@dataclass
class FakeServer:
    """An OpenAI-compatible server streaming :data:`_TOKENS` per chat request, the last two as reasoning text;
    ``usage=False`` drops usage."""

    usage: bool = True
    bodies: list[dict] = field(default_factory=list)

    def handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.bodies.append(body | {"path": self.path})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in fake.chunks():
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.002)
                self.wfile.write(b"data: [DONE]\n\n")

        return Handler

    def chunks(self) -> Iterator[dict]:
        yield {"choices": [{"delta": {"role": "assistant", "content": None}}]}
        for index, token in enumerate(_TOKENS):
            yield {"choices": [{"delta": {"reasoning_content" if index >= 2 else "content": token}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        if self.usage:
            yield {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": len(_TOKENS)}}


@pytest.fixture
def server() -> Iterator[tuple[FakeServer, str]]:
    fake = FakeServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield fake, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.jsonl"
    rows = [{"prompt": "alpha"}, {"turns": ["beta", "second turn"]}, {"prompt": "gamma"}, {"prompt": "unused"}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _workload(dataset: Path, **changes: object):
    return replace(_FIXED, concurrency=2, osl=64, num_prompts=6, num_warmups=2, **changes,
                   dataset=PromptDataset(str(dataset), 3))  # fmt: skip


def _run_client(base_url: str, dataset: Path, result: Path) -> int:
    target = ClientTarget(base_url, "/models/m", "m")
    argv = client_argv(ClientTool(sys.executable, "/unused"), target, _workload(dataset), ClientPass(0, 6, result))
    return dataset_client.main(list(argv[2:]))


def test_a_dataset_pass_runs_at_the_workload_s_concurrency_on_its_own_seed(dataset: Path):
    target = ClientTarget("http://h:1", "/models/m", "m")
    argv = client_argv(ClientTool("/py", "/ix"), target, _workload(dataset), ClientPass(7, 5, None))
    assert (argv[argv.index("--max-concurrency") + 1], argv[argv.index("--seed") + 1]) == ("2", "7")


def test_the_client_streams_chat_requests_and_saves_the_result_a_fixed_round_reads(tmp_path, server, dataset):
    fake, base_url = server
    result = tmp_path / "result.json"
    assert _run_client(base_url, dataset, result) == 0

    sent = [(body["path"], body["messages"][0]["content"]) for body in fake.bodies]
    assert sorted(sent) == sorted(("/v1/chat/completions", p) for p in ("alpha", "beta", "gamma") * 2)
    body = fake.bodies[0]
    assert (body["temperature"], body["max_tokens"], body["stream"], body["model"]) == (0.0, 64, True, "m")
    assert "ignore_eos" not in body and body["stream_options"] == {"include_usage": True}

    saved = json.loads(result.read_text())
    assert (saved["completed"], saved["total_input_tokens"], saved["total_output_tokens"]) == (6, 42, 24)
    assert saved["benchmark_outcome"]["status"] == "passed"
    assert 0 < saved["mean_ttft_ms"] < saved["mean_e2el_ms"] and saved["mean_tpot_ms"] > 0


def test_one_seed_sends_one_order_and_another_seed_another():
    pool = [f"p{index}" for index in range(10)]
    order = dataset_client.request_order(pool, 25, seed=0)
    assert order == dataset_client.request_order(pool, 25, seed=0)
    assert order != dataset_client.request_order(pool, 25, seed=1)
    assert sorted(order[:10]) == sorted(pool) and sorted(order[10:20]) == sorted(pool)


def test_a_server_without_usage_or_answering_an_error_fails_its_requests(tmp_path, server, dataset, capsys):
    fake, base_url = server
    fake.usage = False
    result = tmp_path / "result.json"

    assert _run_client(base_url, dataset, result) == 1 and "6 of 6 requests failed" in capsys.readouterr().err
    saved = json.loads(result.read_text())
    assert saved["benchmark_outcome"]["failed"] == 6


@pytest.mark.parametrize("line", ['{"prompt": ""}', '{"turns": []}', '{"text": "x"}', "[1]"])
def test_a_line_without_a_prompt_is_refused(tmp_path: Path, line: str):
    path = tmp_path / "bad.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(dataset_client.DatasetError, match="bad.jsonl:1"):
        dataset_client.read_prompts(path, None)


def test_a_prompt_holding_a_unicode_line_separator_is_one_line_of_the_file(tmp_path: Path):
    path = tmp_path / "prompts.jsonl"
    prompts = ["one\u2028prompt", "another\x85one"]
    lines = [json.dumps({"prompt": prompt}, ensure_ascii=False) for prompt in prompts]
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8", newline="")
    assert dataset_client.read_prompts(path, None) == prompts


def test_a_dataset_the_client_can_send_and_a_workload_without_one_are_not_refused(dataset: Path):
    assert fixed.dataset_refusal(_workload(dataset)) is None
    assert fixed.dataset_refusal(_FIXED) is None


def test_percentiles_interpolate_as_numpy_does():
    values = [4.0, 1.0, 3.0, 2.0, 10.0]
    assert [dataset_client.percentile(values, p) for p in (50, 90, 99)] == pytest.approx([3.0, 7.6, 9.76])


def test_a_fixed_round_drives_the_dataset_client_through_warmup_and_measured_pass(tmp_path, server, dataset):
    fake, base_url = server
    config = _round_config(tmp_path, WorkloadKind.FIXED, fixed=_workload(dataset))
    (tmp_path / "run" / "client").mkdir(parents=True)
    clock = PhaseClock()
    fake_server = Server(base_url, True, None, lambda: True)
    gate = RoundGate(_DEADLINES, lambda: "", time.monotonic())
    outcome = run_fixed(FixedRound(config, ClientTool(sys.executable, "/unused"), fake_server, gate, clock))

    assert outcome.stop is None, outcome.stop
    assert [mark.phase for mark in clock.marks] == [Phase.WARMUP, Phase.MEASURED, Phase.FINISHED]
    assert len(fake.bodies) == 2 + 6
    metrics = outcome.metrics.to_document()
    assert (metrics["requested_requests"], metrics["completed_requests"], metrics["failed_requests"]) == (6, 6, 0)
    assert metrics["total_output_tokens"] == 24 and outcome.timed is not None
    assert not [key for key in fixed._RESULT_KEYS if metrics[key] is None]
