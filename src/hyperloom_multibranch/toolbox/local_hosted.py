# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real toolbox with the local-sandbox executor: one call builds it (handover 3.3, MB1 on one machine).

Every test runs in a fresh local sandbox started from the shipped runner (``hyperloom.measure.local_templates``), is
measured on cards leased from the card pool, and is accepted only as a signed bundle that verified against the key of
that sandbox. At most ``8 / TP`` tests run at once on one node.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from hyperloom.measure.archive import DirResultArchive, ResultArchive
from hyperloom.measure.hosted import HostedConfig, HostedMeasurementBackend
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.measure.local_templates import local_runner_template
from hyperloom.sandbox.local import LocalSandboxProvider
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.acceptance import MeasurementPlanner
from hyperloom_multibranch.toolbox.attempts import DEFAULT_MAX_INFRA_RETRIES
from hyperloom_multibranch.toolbox.hosted_executor import DEFAULT_TEST_TIMEOUT_SEC, HostedExecutor
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.submission_log import SubmissionLog
from hyperloom_multibranch.toolbox.toolbox import RealToolbox

__all__ = ["LocalHostedToolbox", "MAX_CARDS_PER_NODE", "build_local_toolbox"]

#: One node serves at most eight cards, so at most ``8 / TP`` tests run at once (handover 5, item 7).
MAX_CARDS_PER_NODE = 8
_SANDBOXES, _ARCHIVE, _SUBMISSIONS = "sandboxes", "archive", "toolbox/submissions.jsonl"
_POLL_INTERVAL_SEC = 2.0


class LocalHostedToolbox(RealToolbox):
    """A :class:`RealToolbox` that also owns its hosted backend and local sandbox provider; ``close`` releases them."""

    backend: HostedMeasurementBackend
    provider: LocalSandboxProvider

    async def close(self) -> None:
        await super().close()
        await asyncio.to_thread(self.backend.close)
        await asyncio.to_thread(self.provider.close)


def max_parallel_tests(cards: int, tp: int) -> int:
    """How many tests of ``tp`` cards run at once on ``cards`` cards: ``min(cards, 8) // tp``, at least one."""
    if tp < 1:
        raise ValueError("tp must be at least 1")
    if cards < tp:
        raise ValueError(f"{cards} cards cannot run a test of TP={tp}")
    return min(cards, MAX_CARDS_PER_NODE) // tp


def _provider(
    session: Path, cards: tuple[int, ...], tp: int, python: str, envs: Mapping[str, str] | None, hostname: str | None
) -> LocalSandboxProvider:
    template = local_runner_template(f"runner-{tp}", gpus=tp, python=python, envs=envs)
    return LocalSandboxProvider([template], gpus=cards, base_dir=session / _SANDBOXES, hostname=hostname)


def build_local_toolbox(
    *,
    session_dir: Path,
    run_id: str,
    cards: Iterable[int],
    notebook: LocalNotebook,
    repo: ExperimentRepo,
    planner: MeasurementPlanner,
    spec: SpecContext,
    active_image_digest: str | None,
    tp: int = 1,
    policy: QueuePolicy | None = None,
    runner_envs: Mapping[str, str] | None = None,
    archive: ResultArchive | None = None,
    python: str = sys.executable,
    timeout_sec: int = DEFAULT_TEST_TIMEOUT_SEC,
    max_infra_retries: int = DEFAULT_MAX_INFRA_RETRIES,
    hostname: str | None = None,
    poll_interval_sec: float = _POLL_INTERVAL_SEC,
) -> LocalHostedToolbox:
    """Build a :class:`RealToolbox` whose tests run in local sandboxes; call ``await toolbox.start()`` before use.

    Args:
        session_dir: The session directory. Below it: ``sandboxes/`` (the sandboxes), ``archive/`` (verified bundles,
            unless ``archive`` is given), ``hosted/`` (the backend's journal; a restart re-attaches from it),
            ``toolbox/submissions.jsonl`` (the submit log) and ``tests/<request_id>/attempt-<n>`` (landed output).
        run_id: The run's id; the first part of every sandbox's idempotency key.
        cards: The card pool, for example ``[6, 7]``; at most eight are used for concurrency.
        notebook: The notebook; it takes the requests and results, holds the quarantine list and the claims, and is
            searched when a restart recovers unfinished tests.
        repo: The experiment repository the parent recipes are read from.
        planner: Acceptance steps 2 to 4; its ``plan`` must return a ``hyperloom.measure.points.AcceptedMeasurement``
            (``HostedPlanner`` does).
        spec: What the instruction sheet needs besides the call: image, GPU type, commits, blob store.
        active_image_digest: The active framework image; a recipe on another image is refused. ``None`` skips the check.
        tp: The tensor-parallel size of every test, that is the cards of one runner sandbox.
        policy: Quota and priority policy; its ``global_concurrency`` is lowered to ``8 // tp`` when higher.
        runner_envs: Environment the runner template carries (for example ``INFERENCEX_REPO``).
        archive: The result archive; a directory archive below ``session_dir`` when omitted.
        python: The interpreter the runner and the measurement run with.
        timeout_sec: The timeout of one attempt.
        max_infra_retries: Infrastructure retries per test before it ends as ``measurement_failed``.
        hostname: The node name the local provider reports; this machine's by default.
        poll_interval_sec: How often the backend polls a runner for its result.
    """

    session, cards = Path(session_dir), tuple(cards)
    parallel = max_parallel_tests(len(cards), tp)
    config = HostedConfig(
        session_dir=session,
        run_id=run_id,
        spec=spec,
        runner_templates={tp: f"runner-{tp}"},
        archive=archive or DirResultArchive(session / _ARCHIVE),
        max_parallel_tests=parallel,
        poll_interval_sec=poll_interval_sec,
    )
    provider = _provider(session, cards, tp, python, runner_envs, hostname)
    backend = HostedMeasurementBackend(provider, config)
    backend.start()
    base = policy or QueuePolicy()
    toolbox = LocalHostedToolbox(
        log=SubmissionLog(session / _SUBMISSIONS, run_id),
        repo=repo,
        sink=notebook,  # type: ignore[arg-type]
        health=notebook,  # type: ignore[arg-type]
        executor=HostedExecutor(backend, session, timeout_sec=timeout_sec, magpie_python=python),
        planner=planner,
        active_image_digest=active_image_digest,
        policy=replace(base, global_concurrency=min(base.global_concurrency, parallel)),
        claims=notebook,  # type: ignore[arg-type]
        records=notebook,  # type: ignore[arg-type]
        max_infra_retries=max_infra_retries,
    )
    toolbox.backend, toolbox.provider = backend, provider
    return toolbox
