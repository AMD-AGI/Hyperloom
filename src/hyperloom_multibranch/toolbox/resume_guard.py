# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a resumed session may take over the baseline the toolbox of its earlier leg measured.

A resume skips PRELUDE and judges every new test against the baseline the earlier leg measured, so that baseline must
still measure what a new test measures. The toolbox records, when it accepts the session's first baseline test, what
that test is measured on (:class:`MeasurementEnvironment`): the build, ``MEASUREMENT_EPOCH``, the node, the image digest,
the card pool and the digest of the frozen base launch, in :data:`BASELINE_ENVIRONMENT_FILE` in its own session
directory. The record is written once and never replaced.

A hosted session's tests run in the grant's sandboxes, not on the node of the application that submits them, and a
restart of a hosted run reschedules that application onto another node, often as another account of the run's group.
:func:`check_hosted_baseline_reusable` is the guard of that restart: it compares the record with what a toolbox started
now in the same directory would measure on, everything but the node (:data:`HOSTED_COMPARED`), with the card count of
the runner template the session's tests run on in place of a pool, and raises :class:`BaselineNotReusable` with a
one-line reason when anything differs or the record cannot say (a session from before the record, or one whose record
lacks a field). It reads only: it takes no lock and writes nothing.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom_common.jsonio import read_json_strict
from hyperloom_common.shared_storage import SHARED_FILE_MODE, publish_shared_bytes
from hyperloom_multibranch.contracts.build import CodeVersion, code_version

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
#: The fields a hosted restart compares: all but ``node``, the application's, which a reschedule changes by design and
#: which measures nothing there (the tests run in the grant's sandboxes).
HOSTED_COMPARED = tuple(name for name in RECORDED_FIELDS if name != "node")


@dataclass(frozen=True)
class MeasuredOn:
    """What a toolbox knows of where its tests run; :meth:`record` adds the rest and writes the record.

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


def check_hosted_baseline_reusable(toolbox_dir: Path, cards: int) -> None:
    """Refuse the resume of a hosted session whose baseline was measured on something else than what its tests would be
    measured on now; the guard of a hosted restart, which may run on another node and as another account of the run's
    group.

    ``toolbox_dir`` is the session's ``SessionLayout.toolbox``, ``cards`` the card count of the runner template the
    resumed session's tests run on (its TP). The record is compared with this build, the current ``MEASUREMENT_EPOCH``,
    the image this process names (``HYPERLOOM_ACTIVE_IMAGE_REF`` and ``HYPERLOOM_ACTIVE_IMAGE_DIGEST``), ``cards`` and
    the base launch the toolbox would launch from now (the one frozen in ``toolbox_dir``, read and checked as a start
    reads it), and never with the node (:data:`HOSTED_COMPARED`). Reads only; takes no lock and writes nothing. Blocks.

    Raises:
        BaselineNotReusable: With a one-line reason: the record is missing or unreadable, lacks a field, differs from
            now on any compared field, or a file the comparison reads is refused to this account.
    """

    try:
        now = _environment_now(toolbox_dir, "", cards)
    except OSError as exc:
        raise BaselineNotReusable(f"the session's frozen base launch cannot be read: {exc}") from exc
    _compare(_recorded(toolbox_dir), now, HOSTED_COMPARED)


def _compare(recorded: Mapping[str, object], now: MeasurementEnvironment, fields: Sequence[str]) -> None:
    for name in fields:
        was, is_now = recorded[name], getattr(now, name)
        if name == "build":
            was, is_now = _build_text(was), _build_text(dataclasses.asdict(is_now))
        if was != is_now:
            raise BaselineNotReusable(f"the baseline was measured with {name} {was!r} and this run has {is_now!r}")


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


def _environment_now(toolbox_dir: Path, node: str, cards: int) -> MeasurementEnvironment:
    """What a toolbox started now in ``toolbox_dir`` on ``node`` would measure on; refuses when part of it cannot be
    named."""

    from hyperloom.measure.pins import ImagePinError, active_image
    from hyperloom.orchestrator.state.base_launch import BaseLaunchMismatch
    from inference_suite.contracts import BaseLaunchError

    try:
        image = active_image().digest
    except ImagePinError as exc:
        raise BaselineNotReusable(f"the image this run measures in cannot be named: {exc}") from exc
    try:
        return MeasuredOn(toolbox_dir, node, image, cards).environment()
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
    "HOSTED_COMPARED",
    "RECORDED_FIELDS",
    "BaselineNotReusable",
    "MeasuredOn",
    "MeasurementEnvironment",
    "check_hosted_baseline_reusable",
]
