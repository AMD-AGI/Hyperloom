# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the real aiperf parser, record metrics and export writer on constructed replies and print the export lines.

A script, run as ``python -I aiperf_driver.py <aiperf checkout> <artifact dir>`` by an interpreter that has aiperf's
dependencies (``test_aiperf_patch``): nothing of this repository is imported. It prints ``module`` and the path of the
aiperf module it ran, then one ``<scenario>`` and the line of the export the real writer wrote for it, the replies being
the ones a server gives: an answer with text, replies with nothing in them (a usage chunk that says zero tokens, one that
says three, none, no chunk at all), an HTTP error and a refused connection. The client is configured as InferenceX runs
it, to take the token counts from the server's usage chunk.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

START_NS = 1_000_000_000
END_NS = 1_100_000_000
PROMPT_TOKENS = 550
SCENARIOS = (
    "success_with_text",
    "no_content_usage_zero",
    "no_content_usage_three",
    "no_content_no_usage",
    "no_responses_at_all",
    "http_error",
    "unreachable",
)


def _chunk(payload: dict) -> bytes:
    body = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "test-model", **payload}
    return b"data: " + json.dumps(body).encode()


def _reply(usage_tokens: int | None, text: str = "") -> list[bytes]:
    """A streamed chat completion: a role chunk, a finish chunk, a usage chunk when the server counts, the end marker."""
    chunks = [
        _chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]}),
        _chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]
    if usage_tokens is not None:
        usage = {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": usage_tokens}
        chunks.append(_chunk({"choices": [], "usage": {**usage, "total_tokens": PROMPT_TOKENS + usage_tokens}}))
    return [*chunks, b"data: [DONE]"]


def _chat_request_info():
    import orjson
    from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
    from aiperf.common.models import RequestInfo, Text, Turn
    from aiperf.common.models.model_endpoint_info import EndpointInfo, ModelEndpointInfo, ModelInfo, ModelListInfo
    from aiperf.endpoints.openai_chat import ChatEndpoint
    from aiperf.plugin.enums import EndpointType

    endpoint = ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")], model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN
        ),
        endpoint=EndpointInfo(type=EndpointType.CHAT, base_url="http://localhost:8000/v1/test"),
    )
    info = RequestInfo(
        model_endpoint=endpoint,
        turns=[Turn(role="user", texts=[Text(contents=["Hello world"])])],
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="request",
        x_correlation_id="correlation",
        conversation_id="conversation",
    )
    info.payload_bytes = orjson.dumps(ChatEndpoint(model_endpoint=endpoint).format_payload(info))
    return info


def _requests() -> dict:
    """The record the transport hands the parser for each scenario."""
    import aiohttp
    from aiohttp.client_reqrep import ConnectionKey
    from aiperf.common.models import ErrorDetails, RequestRecord, SSEMessage

    def record(chunks: list[bytes], error: ErrorDetails | None = None) -> RequestRecord:
        responses = [SSEMessage.parse(raw, START_NS + 1000 * (i + 1)) for i, raw in enumerate(chunks)]
        return RequestRecord(
            request_info=_chat_request_info(),
            model_name="test-model",
            responses=responses,
            start_perf_ns=START_NS,
            end_perf_ns=END_NS,
            error=error,
        )

    try:
        raise aiohttp.ClientConnectorError(
            ConnectionKey("localhost", 30000, False, True, None, None, None), ConnectionRefusedError(111, "refused")
        )
    except aiohttp.ClientConnectorError as refused:
        unreachable = ErrorDetails.from_exception(refused)
    refusal = ErrorDetails(code=503, type="Service Unavailable", message="overloaded")
    return {
        "success_with_text": record(_reply(7, text="Hello there")),
        "no_content_usage_zero": record(_reply(0)),
        "no_content_usage_three": record(_reply(3)),
        "no_content_no_usage": record(_reply(None)),
        "no_responses_at_all": record([]),
        "http_error": record([], refusal),
        "unreachable": record([], unreachable),
    }


def _communication_init(self, run, **kwargs) -> None:
    from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin

    AIPerfLifecycleMixin.__init__(self, **kwargs)
    self.run = run
    self.comms = MagicMock()
    for method in ("trace_or_debug", "debug", "info", "warning", "error", "exception"):
        setattr(self, method, MagicMock())


def _run(artifacts: Path):
    from aiperf.common.enums import ExportLevel
    from aiperf.config import BenchmarkRun
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.config.flags.resolver import resolve_config
    from aiperf.plugin.enums import EndpointType

    config = CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        streaming=True,
        use_server_token_count=True,
        artifact_directory=artifacts,
        export_level=ExportLevel.RECORDS,
    )
    resolved = resolve_config(config, config.config_file)
    return BenchmarkRun(
        benchmark_id=uuid.uuid4().hex,
        cfg=resolved.benchmark,
        artifact_dir=resolved.benchmark.artifacts.dir,
        random_seed=resolved.random_seed,
        variables=dict(resolved.variables),
    )


async def _export(artifacts: Path) -> list[str]:
    from aiperf.common.enums import CreditPhase
    from aiperf.common.models.record_models import MetricRecordMetadata
    from aiperf.post_processors.metric_record_processor import MetricRecordProcessor
    from aiperf.post_processors.record_export_jsonl_writer import RecordExportJSONLWriter
    from aiperf.records.inference_result_parser import InferenceResultParser

    run = _run(artifacts)
    with patch("aiperf.common.mixins.CommunicationMixin.__init__", _communication_init):
        parser, processor = InferenceResultParser(run=run), MetricRecordProcessor(run=run)
        writer = RecordExportJSONLWriter(service_id="records-manager", run=run)
    await writer.initialize()
    await writer.start()
    requests = _requests()
    for number, name in enumerate(SCENARIOS):
        metadata = MetricRecordMetadata(
            session_num=number,
            x_request_id=name,
            conversation_id="conversation",
            turn_index=0,
            request_start_ns=START_NS,
            request_end_ns=END_NS,
            worker_id="worker",
            record_processor_id="processor",
            benchmark_phase=CreditPhase.PROFILING,
        )
        await writer.process_record(
            await processor.process_record(await parser.parse_request_record(requests[name]), metadata)
        )
    await writer.finalize()
    await writer.stop()
    return writer.output_file.read_text(encoding="utf-8").splitlines()


def main(client: Path, artifacts: Path) -> None:
    sys.path.insert(0, str(client / "src"))
    import aiperf

    print("module", json.dumps(aiperf.__file__), sep="\t")
    for name, line in zip(SCENARIOS, asyncio.run(_export(artifacts)), strict=True):
        print(name, line, sep="\t")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
