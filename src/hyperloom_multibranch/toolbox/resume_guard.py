# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a resumed session may take over from the toolbox of its earlier leg: the baseline, and the keeps infra ended.

A resume skips PRELUDE and judges every new test against the baseline the earlier leg measured, so that baseline must
still measure what a new test measures. The toolbox records, when it accepts the session's first baseline test, what
that test is measured on (:class:`MeasurementEnvironment`): the build, ``MEASUREMENT_EPOCH``, the node, the image digest,
the card pool and the digest of the frozen base launch, in :data:`BASELINE_ENVIRONMENT_FILE` in its own session
directory. The record is written once and never replaced. :func:`check_baseline_reusable` compares it with what a
toolbox started now in the same directory would measure on, and raises :class:`BaselineNotReusable` with a one-line
reason when anything differs or the record cannot say (a session from before the record, or one whose record lacks a
field). It reads only: it takes no lock and writes nothing.

:func:`infra_ended_keeps` reads the toolbox's submit log for the keeps whose test the infrastructure ended before
anything was measured, so the resume can submit them again; it reads only as well.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest, TestResult, Validity
from hyperloom_common.jsonio import read_json_strict
from hyperloom_common.shared_storage import SHARED_FILE_MODE, publish_shared_bytes
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.build import CodeVersion, code_version
from hyperloom_multibranch.toolbox.submission_log import SUBMISSIONS_FILE, LogEntry, logged_entries

#: The record of what the session's baseline was measured on, in the toolbox's session directory.
BASELINE_ENVIRONMENT_FILE = "baseline-environment.json"


class BaselineNotReusable(Exception):
    """The session's baseline cannot anchor a resumed session; ``reason`` says why in one line."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class MeasurementEnvironment:
    """What the session's tests are measured on: a baseline anchors only tests measured on the same.

    ``build`` is the code the toolbox ran (:func:`~hyperloom_multibranch.contracts.build.code_version`),
    ``measurement_epoch`` the generation of the measurement path
    (:data:`hyperloom.inference_optimizer.cli.bootstrap.MEASUREMENT_EPOCH`), ``node`` the node the tests ran on
    (:func:`hyperloom.sandbox.node.node_name`), ``image_digest`` the image they ran in, ``cards`` how many cards the
    toolbox's pool held and ``base_launch_digest`` the digest of the frozen base launch every test is launched from
    (``""`` for none).
    """

    build: CodeVersion
    measurement_epoch: int
    node: str
    image_digest: str
    cards: int
    base_launch_digest: str

    def to_document(self) -> dict[str, object]:
        return dataclasses.asdict(self)


#: The fields a record must carry, in the order a refusal names them.
RECORDED_FIELDS = tuple(field.name for field in dataclasses.fields(MeasurementEnvironment))


@dataclass(frozen=True)
class MeasuredOn:
    """What a local toolbox knows of where its tests run; :meth:`record` adds the rest and writes the record.

    ``toolbox_dir`` is the toolbox's session directory, where the base launch is frozen and the record is written.
    """

    toolbox_dir: Path
    node: str
    image_digest: str
    cards: int

    def environment(self) -> MeasurementEnvironment:
        """The environment of a test measured now: this build, this epoch and the base launch frozen now.

        Raises:
            BaseLaunchMismatch: The frozen base launch is not the one the session recorded.
            BaseLaunchError: The frozen base launch cannot be read.
        """

        return MeasurementEnvironment(
            build=code_version(),
            measurement_epoch=_measurement_epoch(),
            node=self.node,
            image_digest=self.image_digest,
            cards=self.cards,
            base_launch_digest=_base_launch_digest(self.toolbox_dir),
        )

    def record(self) -> bool:
        """Write the record unless the session has one, which is kept; whether this call wrote it. Blocks."""

        path = self.toolbox_dir / BASELINE_ENVIRONMENT_FILE
        if path.exists():
            return False
        text = json.dumps(self.environment().to_document(), indent=2, sort_keys=True) + "\n"
        return publish_shared_bytes(path, text.encode("utf-8"), mode=SHARED_FILE_MODE, make_parents=True)


def check_baseline_reusable(toolbox_dir: Path, cards: Sequence[int]) -> None:
    """Refuse a resume whose baseline was measured on something else than what the session would measure on now.

    ``toolbox_dir`` is the toolbox's session directory of the session (``SessionLayout.toolbox``), ``cards`` the card
    pool the resumed session hands its toolbox (``machine_cards()``). The record is compared with this build, the current
    ``MEASUREMENT_EPOCH``, this node, the image this process names (``HYPERLOOM_ACTIVE_IMAGE_REF`` and
    ``HYPERLOOM_ACTIVE_IMAGE_DIGEST``), the count of ``cards`` and the base launch the toolbox would launch from now (the
    one frozen in ``toolbox_dir``, read and checked as a start reads it). Reads only; takes no lock and writes nothing.
    Blocks, on git among others.

    Raises:
        BaselineNotReusable: With a one-line reason: the record is missing or unreadable, lacks a field, or differs from
            now on any of them.
    """

    recorded = _recorded(toolbox_dir)
    now = _environment_now(toolbox_dir, len(cards))
    for name in RECORDED_FIELDS:
        was, is_now = recorded[name], getattr(now, name)
        if name == "build":
            was, is_now = _build_text(was), _build_text(dataclasses.asdict(is_now))
        if was != is_now:
            raise BaselineNotReusable(f"the baseline was measured with {name} {was!r} and this run has {is_now!r}")


@dataclass(frozen=True)
class KeepRef:
    """A keep whose test the infrastructure ended before it measured anything.

    ``branch_id`` is the branch that kept it, ``commit`` the commit in ``MB-repo`` the keep was measured against (its
    request's parent), ``request_id`` the test, ``reason`` why it ended and ``request`` the test as it was submitted.
    """

    branch_id: str
    commit: str
    request_id: str
    reason: str
    request: TestRequest


def infra_ended_keeps(toolbox_dir: Path) -> list[KeepRef]:
    """The keeps in the submit log of the toolbox's session directory ``toolbox_dir`` whose last test ended
    ``measurement_failed`` with no measured round, in the order they were submitted; none when there is no log.

    A keep is how a branch's change is measured: a ``test`` of purpose ``explore`` on a branch other than the
    integration branch. The log does not say which kind of branch submitted it, so the keeps of every branch are
    returned and the caller keeps those of the branches it requeues. A keep submitted again, the same change on the same
    branch against the same commit, is judged by its last test. A test that is still waiting, was withdrawn, or whose
    submit failed is no keep that ended. Reads only. Blocks.

    Raises:
        OSError: The log cannot be read.
        ValueError: A line of the log is no entry a replay reads.
    """

    path = toolbox_dir / SUBMISSIONS_FILE
    if not path.exists():
        return []
    last: dict[tuple[str, str, str], tuple[LogEntry, TestRequest]] = {}
    for entry in logged_entries(path):
        if entry.aborted or not _is_keep(entry):
            continue
        request = TestRequest.from_document(entry.request)
        last[(entry.branch, request.parent_commit, request.delta.fingerprint())] = (entry, request)
    keeps = []
    for entry, request in last.values():
        result = _infra_ended(entry)
        if result is not None:
            ended = result.failure_class.value if result.failure_class is not None else "unclassified"
            keeps.append(
                KeepRef(
                    entry.branch, request.parent_commit, entry.request_id, f"{ended}: {result.failure_reason}", request
                )
            )
    return keeps


def _is_keep(entry: LogEntry) -> bool:
    request = entry.request
    return (
        request.get("kind") == TestKind.TEST.value
        and request.get("purpose") == Purpose.EXPLORE.value
        and entry.branch != INTEGRATION_BRANCH
    )


def _infra_ended(entry: LogEntry) -> TestResult | None:
    """The result of ``entry`` when it ended ``measurement_failed`` with no measured round, else ``None``."""

    if entry.outcome is None or "result" not in entry.outcome:
        return None
    result = TestResult.from_document(entry.outcome["result"])
    if result.validity is not Validity.MEASUREMENT_FAILED:
        return None
    return None if any(round_.phase == "measure" for round_ in result.rounds) else result


def _recorded(toolbox_dir: Path) -> Mapping[str, object]:
    path = toolbox_dir / BASELINE_ENVIRONMENT_FILE
    if not path.is_file():
        raise BaselineNotReusable(
            f"{path} does not exist: the session recorded nothing of what its baseline was measured on"
            " (it predates the record, or its tests did not run on this machine's cards)"
        )
    try:
        document = read_json_strict(path, require_dict=True)
    except (OSError, ValueError) as exc:
        raise BaselineNotReusable(f"{path} cannot be read: {exc}") from exc
    missing = [name for name in RECORDED_FIELDS if name not in document]
    if missing:
        raise BaselineNotReusable(f"{path} does not record {', '.join(missing)}, so it cannot be compared")
    return document


def _environment_now(toolbox_dir: Path, cards: int) -> MeasurementEnvironment:
    """What a toolbox started now in ``toolbox_dir`` would measure on; refuses when part of it cannot be named."""

    from hyperloom.measure.pins import ImagePinError, active_image
    from hyperloom.orchestrator.state.base_launch import BaseLaunchMismatch
    from hyperloom.sandbox.node import node_name
    from inference_suite.contracts import BaseLaunchError

    try:
        image = active_image().digest
    except ImagePinError as exc:
        raise BaselineNotReusable(f"the image this run measures in cannot be named: {exc}") from exc
    try:
        return MeasuredOn(toolbox_dir, node_name(), image, cards).environment()
    except (BaseLaunchMismatch, BaseLaunchError) as exc:
        raise BaselineNotReusable(f"the session's frozen base launch cannot be read: {exc}") from exc


def _measurement_epoch() -> int:
    from hyperloom.inference_optimizer.cli.bootstrap import MEASUREMENT_EPOCH

    return MEASUREMENT_EPOCH


def _base_launch_digest(toolbox_dir: Path) -> str:
    from hyperloom.orchestrator.state.base_launch import frozen_base

    frozen = frozen_base(toolbox_dir)
    return "" if frozen is None else frozen.launch.digest()


def _build_text(build: object) -> str:
    """A recorded or a running build as one comparable line."""

    if isinstance(build, Mapping):
        return f"{build.get('package', '')}@{build.get('git_sha', '')}"
    return repr(build)


__all__ = [
    "BASELINE_ENVIRONMENT_FILE",
    "RECORDED_FIELDS",
    "BaselineNotReusable",
    "KeepRef",
    "MeasuredOn",
    "MeasurementEnvironment",
    "check_baseline_reusable",
    "infra_ended_keeps",
]
