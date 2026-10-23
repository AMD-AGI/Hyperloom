# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lines of aiperf's per-request export (``profile_export.jsonl``), as the pinned replay client writes them.

No export of a real AgentX run was reachable when these were written, so every shape is rebuilt from the client's own
source and documentation at the commit InferenceX pins (``SemiAnalysisAI/aiperf`` 754356e): the metadata keys and the
successful and cancelled records are those of ``docs/tutorials/working-with-profile-exports.md``; the error of a refused
request is what ``aiohttp_client.py`` builds (the status as ``code``, its reason phrase as ``type``); the error of a
request that never reached a server is ``ErrorDetails.from_exception`` (the exception's class name as ``type``, its repr
as ``message``); and the reply the client found nothing in is ``InvalidInferenceResultError``, whose ``message`` is the
exception's repr (the notes it carries are not in it) and whose failed record keeps only the input token count
(``inference_result_parser.py``). Fields the writer leaves out when they are ``None`` (``exclude_none``) are left out.
"""

from __future__ import annotations

import json
from typing import Any

BASE_NS = 1_759_813_207_000_000_000
INVALID_REPLY = {
    "type": "InvalidInferenceResultError",
    "message": "InvalidInferenceResultError('Invalid inference result')",
    "cause_chain": ["InvalidInferenceResultError"],
}


def _metadata(number: int, phase: str) -> dict[str, Any]:
    start = BASE_NS + number * 1_000_000_000
    return {
        "session_num": number,
        "x_request_id": f"7609a2e7-aa53-4ab1-98f4-{number:012d}",
        "x_correlation_id": "32ee4f33-cfca-4cfc-988f-79b45408b909",
        "conversation_id": "77aa5b0e-b305-423f-88d5-c00da1892599",
        "turn_index": number % 7,
        "request_start_ns": start,
        "request_end_ns": start + 297_525_228,
        "worker_id": "worker_359d423a",
        "record_processor_id": "record_processor_1fa47cd7",
        "benchmark_phase": phase,
        "was_cancelled": False,
    }


def served(number: int, *, phase: str = "profiling", output_tokens: int = 9) -> dict[str, Any]:
    """A request the server answered and the client counted successful."""
    metadata = {**_metadata(number, phase), "request_ack_ns": BASE_NS + number * 1_000_000_000 + 117_830_613}
    metrics = {
        "input_sequence_length": {"value": 550, "unit": "tokens"},
        "time_to_first_token": {"value": 255.88656799999998, "unit": "ms"},
        "request_latency": {"value": 297.52522799999997, "unit": "ms"},
        "output_token_count": {"value": output_tokens, "unit": "tokens"},
        "output_sequence_length": {"value": output_tokens, "unit": "tokens"},
    }
    return {"metadata": metadata, "metrics": metrics}


def _failed(number: int, phase: str, error: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "metadata": _metadata(number, phase),
        "metrics": {"error_isl": {"value": 550, "unit": "tokens"}, **extra},
        "error": error,
    }


def refused(number: int, *, status: int = 503, reason: str = "Service Unavailable", phase: str = "profiling") -> dict:
    """A request the server answered with an HTTP error status."""
    return _failed(number, phase, {"code": status, "type": reason, "message": "server overloaded"})


def unreachable(number: int, *, phase: str = "profiling") -> dict[str, Any]:
    """A request that never reached a server."""
    error = {
        "type": "ClientConnectorError",
        "message": "ClientConnectorError(ConnectionKey(host='localhost', port=30000, is_ssl=False))",
        "cause_chain": ["ClientConnectorError", "ConnectionRefusedError"],
    }
    return _failed(number, phase, error)


def cancelled(number: int, *, phase: str = "profiling") -> dict[str, Any]:
    """A request the client cancelled before the server answered."""
    record = _failed(number, phase, {"code": 499, "type": "RequestCancellationError", "message": "cancelled"})
    record["metadata"]["was_cancelled"] = True
    return record


def no_content(number: int, *, phase: str = "profiling", output_tokens: int | None = None) -> dict[str, Any]:
    """A reply the client found nothing in. The pinned client exports no output token count for it; a record that has
    one (``output_tokens``) is what a client that kept the count would write."""
    extra = {} if output_tokens is None else {"output_sequence_length": {"value": output_tokens, "unit": "tokens"}}
    return _failed(number, phase, INVALID_REPLY, **extra)


def text(*records: dict[str, Any]) -> list[str]:
    """The export's lines, the way the writer serialises a record (one JSON object per line)."""
    return [json.dumps(record, separators=(",", ":")) for record in records]
