# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn the torch profiler on in a profile round's server, and keep its three trace directories apart.

SGLang reads the trace dir from its environment and opens the window on ``/start_profile`` with the body the client
sends. vLLM takes the dir and the window as ``--profiler-config.*`` flags, and ATOM takes the dir as a flag and bounds
no steps, so its window is the time between ``/start_profile`` and ``/stop_profile``. The layer composes last, so the
dir it names beats any placeholder an earlier layer carries.

TraceLens-patched servers write their CUDA-graph capture sidecars into ``capture_traces`` and
``graph_capture_profile`` below the trace dir; :func:`settle_traces` lifts them beside ``torch_trace``, so a reader
tells a step trace from a sidecar by its directory.

Every framework captures only between ``/start_profile`` and ``/stop_profile``; each load's window sends them through
:func:`server_request` and records what it did as ``profile/capture.json`` (:class:`WindowCapture`). A newer SGLang
refuses both with 403 unless its server holds an admin key, so a profile round sends the key its server holds
(:func:`admin_key`) with both requests: the one its ``--admin-api-key`` names, else a fresh one the round gives it,
which lives only in the server's environment and the window, never in ``launch.json``. An older SGLang ignores the
variable and the header.

A ``/start_profile`` the server answers with an error status ends the round at once (``profile_refused``): the
window would capture nothing, however long the load ran on. One the server does not answer is only recorded, and the
round's server watch decides what became of the server.
"""

from __future__ import annotations

import http.client
import json
import secrets
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from inference_suite.argv import argv_flag_occurrences, last_alias_value
from inference_suite.contracts.config import LaunchDelta, ProfileConfig, SuiteConfig
from inference_suite.contracts.launch import LaunchError
from inference_suite.contracts.layout import CAPTURE_TRACE_DIRS, RunLayout, answered_with_status
from inference_suite.contracts.report import FailedStep
from inference_suite.frameworks import Framework
from inference_suite.runtime.watch import Stop

SGLANG_PROFILER_DIR_ENV = "SGLANG_TORCH_PROFILER_DIR"
#: Where SGLang's server reads its admin key from when ``--admin-api-key`` is not given.
SGLANG_ADMIN_KEY_ENV = "SGLANG_ADMIN_API_KEY"
_SGLANG_ADMIN_KEY_FLAG = frozenset({"admin-api-key"})
#: SGLang refuses to start with an admin key beside its native gRPC listener, which bypasses HTTP auth; the listener is
#: on when either is set.
_SGLANG_GRPC_FLAG = "grpc-port"
_SGLANG_GRPC_ENV = "SGLANG_GRPC_PORT"
ATOM_PROFILER_DIR_FLAG = "--torch-profiler-dir"
#: How long ATOM's server waits for its engine to write the trace before ``/stop_profile`` fails (300 s by default);
#: set to the round's own wait, so a capture the round would still wait for is not cut short by the server.
ATOM_PROFILER_TIMEOUT_ENV = "ATOM_PROFILER_TIMEOUT"
VLLM_PROFILER_CONFIG = "--profiler-config."
#: The ``/start_profile`` body keys that bound SGLang's capture window, in engine forward steps.
SGLANG_WINDOW_KEYS = ("start_step", "num_steps")
START_PROFILE_PATH = "/start_profile"
STOP_PROFILE_PATH = "/stop_profile"
#: ``/stop_profile`` answers once the trace is written, and a large capture serializes for minutes.
_CONTROL_TIMEOUT_SEC = 600.0
#: A loopback request routed through the host's proxy would never reach the server.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
#: What a request the server refused or never answered raises.
SERVER_REQUEST_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError)


def _vllm_args(profile: ProfileConfig, trace_dir: str) -> tuple[str, ...]:
    settings: list[tuple[str, object]] = [("profiler", "torch"), ("torch_profiler_dir", trace_dir)]
    if profile.delay_iters is not None:
        settings += [("delay_iterations", profile.delay_iters), ("max_iterations", profile.max_iters)]
    return tuple(token for name, value in settings for token in (VLLM_PROFILER_CONFIG + name, str(value)))


def profiler_layer(config: SuiteConfig, framework: Framework) -> LaunchDelta | None:
    """The layer that turns the profiler on in ``framework``'s server; ``None`` when the round profiles nothing.

    Raises:
        LaunchError: When ``framework`` runs no server to profile.
    """
    if not config.profile.enabled:
        return None
    trace_dir = str(RunLayout(Path(config.paths.run_dir)).profile_torch_trace)
    if framework is Framework.SGLANG:
        return LaunchDelta(env_set={SGLANG_PROFILER_DIR_ENV: trace_dir})
    if framework is Framework.VLLM:
        return LaunchDelta(args_add=_vllm_args(config.profile, trace_dir))
    if framework is Framework.ATOM:
        return LaunchDelta(
            args_add=(ATOM_PROFILER_DIR_FLAG, trace_dir),
            env_set={ATOM_PROFILER_TIMEOUT_ENV: f"{_CONTROL_TIMEOUT_SEC:g}"},
        )
    raise LaunchError(f"{framework.value} runs no server to profile")


def start_body(profile: ProfileConfig, framework: Framework) -> dict[str, object]:
    """The ``/start_profile`` body the client sends: SGLang's carries the step window, the others take it as given."""
    body = dict(profile.start_body or {})
    if framework is Framework.SGLANG and profile.delay_iters is not None:
        body.update(zip(SGLANG_WINDOW_KEYS, (profile.delay_iters, profile.max_iters)))
    return body


def admin_key(framework: Framework, argv: Sequence[str], env: Mapping[str, str]) -> str | None:
    """The admin key of a profile round's server launched with ``argv`` and ``env``: the one its ``--admin-api-key``
    names, which SGLang takes over the variable, else a fresh one; ``None`` for a server that takes none: not SGLang, or
    SGLang with its gRPC listener on, whose profiler then stays refused."""
    if framework is not Framework.SGLANG:
        return None
    occurrences = argv_flag_occurrences(argv)
    given = last_alias_value(occurrences, _SGLANG_ADMIN_KEY_FLAG)
    if given is not None:
        return given
    if _SGLANG_GRPC_ENV in env or any(name == _SGLANG_GRPC_FLAG for name, _ in occurrences):
        return None
    return secrets.token_urlsafe(32)


def server_request(url: str, body: Mapping[str, object] | None, key: str | None = None) -> bytes:
    """POST ``body`` as JSON to ``url`` (GET when ``body`` is ``None``), with ``key`` as its bearer token when given,
    and return the answer; raises one of :data:`SERVER_REQUEST_ERRORS` when the server refuses the request or does not
    answer."""
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, data=data, headers=headers)
    with _OPENER.open(request, timeout=_CONTROL_TIMEOUT_SEC) as response:
        return response.read()


def start_refusal(error: str) -> Stop | None:
    """The stop a ``/start_profile`` that failed, recorded as ``error``, makes: ``profile_refused`` when the server
    answered it with an error status (:func:`~inference_suite.contracts.layout.answered_with_status`), none when the
    server did not answer."""
    return Stop(FailedStep.PROFILE_REFUSED, error) if answered_with_status(error) else None


@dataclass(frozen=True)
class WindowCapture:
    """What a window did, as ``profile/capture.json`` records it; every instant is Unix seconds.

    ``profiling_start_unix`` is the AgentX replay's first profiling request (``None`` for a fixed load), and ``error``
    the control request the server refused or did not answer.
    """

    profiling_start_unix: float | None
    window_open_unix: float | None
    window_close_unix: float | None
    running_requests_in_window: int | None
    error: str | None

    def to_document(self) -> dict[str, object]:
        return asdict(self)

    def write(self, layout: RunLayout) -> None:
        layout.profile_capture.parent.mkdir(parents=True, exist_ok=True)
        layout.profile_capture.write_text(json.dumps(self.to_document(), indent=2) + "\n", encoding="utf-8")


def make_trace_dirs(layout: RunLayout) -> None:
    for directory in layout.profile_trace_dirs():
        directory.mkdir(parents=True, exist_ok=True)


def settle_traces(layout: RunLayout) -> tuple[str, ...]:
    """Lift the capture sidecars out of ``torch_trace`` beside it; return the trace dirs that hold a file.

    Runs once the server has stopped writing, after :func:`make_trace_dirs`. The paths are relative to the run_dir.
    """
    for name in CAPTURE_TRACE_DIRS:
        nested = layout.profile_torch_trace / name
        if nested.is_dir():
            # rmdir refuses a non-empty target, so two captures are never merged into one directory.
            (layout.profile_dir / name).rmdir()
            nested.rename(layout.profile_dir / name)
    return tuple(
        directory.relative_to(layout.root).as_posix()
        for directory in layout.profile_trace_dirs()
        if any(path.is_file() for path in directory.rglob("*"))
    )


__all__ = [
    "SERVER_REQUEST_ERRORS",
    "SGLANG_ADMIN_KEY_ENV",
    "SGLANG_PROFILER_DIR_ENV",
    "START_PROFILE_PATH",
    "STOP_PROFILE_PATH",
    "WindowCapture",
    "admin_key",
    "make_trace_dirs",
    "profiler_layer",
    "server_request",
    "settle_traces",
    "start_body",
    "start_refusal",
]
