# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The workspace's Experience KB as a multi-branch session uses it: the local service, and the readers of its schemas.

:func:`start_experience_kb` brings the loopback service ``HYPERLOOM_KB_URL`` names to serving with the main flow's
``experience_kb_service ensure --no-restart``, run as a process of its own: the service it starts is then no child of
the scheduler, whose stop kills every child it has (:mod:`~hyperloom_multibranch.scheduling.processes`), so it serves
CLOSE after the stop and the other sessions of the workspace after this one. A service started with other settings
is left serving. A service that cannot start costs the session nothing: reads come back empty and CLOSE's writes wait
in the spool.

:func:`experience_readers` reads the schemas a session's tests and learnings are published under: an AgentX session's
tests all go to ``multibranch-recipe-v1``, an InferenceX session's to ``inference-recipe-v1`` but for the kernel
changes, which go to ``multibranch-recipe-v1`` too, so an InferenceX session reads both
(:class:`~hyperloom_multibranch.blackboard.knowledge.recall.SchemaReaders`). Without ``HYPERLOOM_KB_URL`` there are
none.

:func:`close_targets` is where CLOSE writes: one client of that KB for every schema, with the request timeout of the
main flow's runs and the spool beside the workspace's service, which its next launch delivers. Once one write spools,
the rest spool without a request, so a service that does not answer costs CLOSE one timeout.
"""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404 - runs this interpreter on a module of the installed package.
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom_kb import RemoteClient, RemoteClientError, RemoteConfig, RemoteExperienceKB, is_loopback
from hyperloom_kb.collect import load_mapping
from hyperloom_multibranch.blackboard.knowledge import experience, learnings
from hyperloom_multibranch.blackboard.knowledge.recall import ExperienceReader, SchemaReaders
from hyperloom_multibranch.blackboard.rsi.session import CloseTargets
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.scheduling.layout import SessionLayout

log = logging.getLogger(__name__)

KB_URL_ENV = "HYPERLOOM_KB_URL"
ENSURE_COMMAND = ("-m", "hyperloom.inference_optimizer.experience_kb_service", "ensure", "--no-restart")
#: Past the local service's start (60 s) plus one health check (30 s), so a start fails there first and says why.
ENSURE_TIMEOUT_SECONDS = 120.0
#: The mappings each kind of session publishes its tests through, the schema of each one read.
TEST_MAPPINGS: dict[WorkloadKind, tuple[Path, ...]] = {
    WorkloadKind.AGENTX: (experience.MULTIBRANCH_MAPPING_PATH,),
    WorkloadKind.INFERENCEX: (experience.MAPPING_PATH, experience.MULTIBRANCH_MAPPING_PATH),
}


@dataclass(frozen=True)
class ExperienceReaders:
    """The reader of a session's tests and the one of its learnings; ``None`` without an Experience KB."""

    tests: ExperienceReader | None = None
    learnings: RemoteExperienceKB | None = None


def start_experience_kb(env: Mapping[str, str] | None = None) -> None:
    """Bring the loopback service ``env`` names to serving outside this process's tree; nothing for no URL or one
    that names a service elsewhere. Never raises: a service that does not start is logged."""

    values = os.environ if env is None else env
    url = (values.get(KB_URL_ENV) or "").strip()
    if not url or not is_loopback(url):
        return
    try:
        done = subprocess.run(  # nosec B603 - a fixed argv of this interpreter.
            [sys.executable, *ENSURE_COMMAND],
            env=dict(values),
            capture_output=True,
            text=True,
            timeout=ENSURE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("the Experience KB service did not start (%s); the session runs without it", exc)
        return
    if done.returncode != 0:
        why = (done.stderr or done.stdout).strip()
        log.warning("the Experience KB service did not start (%s); the session runs without it", why)
        return
    log.info("%s", done.stdout.strip())
    if done.stderr.strip():
        log.warning("%s", done.stderr.strip())


def experience_readers(kind: WorkloadKind, env: Mapping[str, str] | None = None) -> ExperienceReaders:
    """The readers of the Experience KB ``env`` names for a session of ``kind``; none when it names none, or names one
    without its token, which is logged."""

    from hyperloom.inference_optimizer.experience_kb_service import REQUEST_TIMEOUT_SECONDS

    try:
        config = RemoteConfig.from_env(env, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    except RemoteClientError as exc:
        log.warning("the session reads no Experience KB: %s", exc)
        return ExperienceReaders()
    if config is None:
        return ExperienceReaders()

    def reader(mapping: Path) -> RemoteExperienceKB:
        return RemoteExperienceKB(RemoteClient(config), load_mapping(mapping).declaration)

    tests = tuple(reader(mapping) for mapping in TEST_MAPPINGS[kind])
    return ExperienceReaders(tests[0] if len(tests) == 1 else SchemaReaders(tests), reader(learnings.MAPPING_PATH))


def close_targets(layout: SessionLayout) -> CloseTargets:
    """Where CLOSE writes ``layout``'s session: the environment's KB, which writes nothing when none is configured
    and fails each step when it is misconfigured, else one client of it for every schema."""

    from hyperloom.inference_optimizer.experience_kb_service import REQUEST_TIMEOUT_SECONDS, spool_root

    reports, judge, blobs = layout.reports, layout.judge, layout.blobs
    try:
        config = RemoteConfig.from_env(spool_root=spool_root(), timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    except RemoteClientError as exc:
        log.warning("CLOSE cannot write to the Experience KB: %s", exc)
        config = None
    if config is None:
        return CloseTargets(receipts=reports, judge_transcript=judge, blobs=blobs)
    client = RemoteClient(config)
    try:
        client.flush_spool()
    except (RemoteClientError, OSError) as exc:
        log.warning("Experience KB spool flush deferred: %s", exc)

    def writer(mapping: Path) -> RemoteExperienceKB:
        return RemoteExperienceKB(client, load_mapping(mapping).declaration)

    return CloseTargets(
        experience_kb=writer(experience.MAPPING_PATH),
        learning_kb=writer(learnings.MAPPING_PATH),
        receipts=reports,
        judge_transcript=judge,
        blobs=blobs,
        multibranch_kb=writer(experience.MULTIBRANCH_MAPPING_PATH),
    )


__all__ = [
    "ENSURE_COMMAND",
    "TEST_MAPPINGS",
    "ExperienceReaders",
    "close_targets",
    "experience_readers",
    "start_experience_kb",
]
