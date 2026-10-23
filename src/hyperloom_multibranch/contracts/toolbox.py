# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox as the other multi-branch modules call it: submit a test, cancel it, read the queue and its progress.

A module hands the toolbox a :class:`~hyperloom.contracts.measurement.TestRequest` and gets the ``request_id`` back
once the toolbox accepted it. Acceptance is the toolbox's checks of design section 4.17 (the request validates, the
parent recipe is read, the delta applies, the measurement surface matches), so a request that cannot be measured is
refused here and never queues. The result does not come back through this interface: the toolbox writes it to the
notebook, which pushes it to the submitting branch's inbox. The writer protocols it uses are the notebook's
:class:`~hyperloom_multibranch.contracts.notebook.interfaces.TestRecordSink` (``write_request`` when it accepts,
``write_result`` when the result is in, ``write_withdrawal`` when it takes a test back unmeasured) and
:class:`~hyperloom_multibranch.contracts.notebook.interfaces.NodeHealth` (the quarantine list); both are re-exported
so a toolbox is wired against one module.

Like the notebook's interfaces, every method is a coroutine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

from hyperloom.contracts.measurement import PRIORITY_HIGHEST, PRIORITY_LOWEST, AttachmentRef, TestRequest
from hyperloom.contracts.recipe import KernelArtifact, KernelKind
from hyperloom_multibranch.contracts.notebook.interfaces import NodeHealth, TestRecordSink


class ToolboxError(Exception):
    """Base of every error the toolbox raises."""


class RequestRejected(ToolboxError):
    """The toolbox refused the request; it was not queued and nothing was written. ``reason`` says why.

    ``code`` is a short stable name of the refused check, for a caller that reacts to one of them
    (:data:`REJECTION_CODES`); ``reason`` is the text for a person or a log. ``existing_request_id`` is set by the
    refusals that point at a request already accepted (``profile_exists``): the one the caller waits for instead.
    """

    def __init__(self, reason: str, code: str = "invalid_request", *, existing_request_id: str | None = None) -> None:
        super().__init__(f"{code}: {reason}")
        self.reason = reason
        self.code = code
        self.existing_request_id = existing_request_id


class Backpressure(ToolboxError):
    """The branch has too many tests queued; the request was not accepted and the caller submits it again later."""

    def __init__(self, branch_id: str, queued: int, limit: int) -> None:
        super().__init__(f"branch {branch_id!r} has {queued} tests queued, the limit is {limit}")
        self.branch_id = branch_id
        self.queued = queued
        self.limit = limit


class UnknownTestRequest(ToolboxError, LookupError):
    """The toolbox accepted no request with this ``request_id``."""


class TestWithdrawn(ToolboxError):
    """The test was withdrawn before it produced a result; ``reason`` is what the toolbox wrote to the notebook."""

    __test__ = False

    def __init__(self, request_id: str, reason: str) -> None:
        super().__init__(f"{request_id} was withdrawn: {reason}")
        self.request_id = request_id
        self.reason = reason


#: The checks acceptance can refuse on, as the ``code`` of :class:`RequestRejected`.
REJECTION_CODES = (
    "invalid_request",  # the request breaks the contract of 3.7
    "unknown_parent",  # the parent commit is not in the experiment repository
    "delta_not_applicable",  # the delta does not fit the parent recipe, or a patch does not apply
    "base_image_mismatch",  # the recipe's base image is not the active framework image
    "surface_mismatch",  # the measurement surface differs from the active canonical one
    "unknown_retest",  # ``retest_of`` names no accepted request
    "branch_stopped",  # the queue policy lists the branches that may submit and this one is not among them
    "speculative_context_copy_on_synthetic",  # an InferenceX candidate introduces or changes a context-copy draft
    "speculative_decoding_pinned",  # an AgentX candidate changes the draft or simulated acceptance its parent pins
    "unsupported_kind",  # the toolbox's backend does not run this kind of test (a local backend: profile, repro), or has no cards to lease
    "profile_exists",  # a PROFILE of this commit and recipe is already queued, running or measured: ``existing_request_id``
    "empty_delta",  # an explore test whose delta changes nothing
    "upstream_label_unsupported",  # a delta moves the upstream_ref or upstream_repo of a component the parent has
    "online_tuning_unsupported",  # the candidate runs TunableOp with tuning on: its test never ends in a bounded time
    "context_below_eval",  # an accuracy test whose launch serves less context than its evaluation runs in (every request answers 400)
    "attachment_too_large",  # ``put_attachment``: the data, or what a directory archive expands to, passes the cap
    "attachment_unreadable",  # ``put_attachment``: a ``dir`` attachment is not a tar archive
    "attachment_unsafe",  # ``put_attachment``: a ``dir`` archive has a path that escapes, a link, or too many entries
    "attachments_unsupported",  # ``put_attachment``, ``put_kernel_artifact``: this toolbox has no blob store
    "kernel_artifact_invalid",  # ``put_kernel_artifact``: the table, target or variable cannot be loaded as given
    "kernel_variable_taken",  # ``put_kernel_artifact``: the variable already loads the table of another target
    "target_required",  # ``put_kernel_artifact``: no target given and the variable is bound to none yet in this session
    "lease_leaves_no_measurement",  # ``lease_branch_sandbox``, ``start_analysis kernel_tune``: leasing would leave fewer cards than one test needs
    "lease_exists",  # ``lease_branch_sandbox``: the branch already holds a lease
)


class WorkloadKind(str, Enum):
    """The kind of workload a campaign measures; the judging rule differs (multi-branch design 7.3, K2)."""

    INFERENCEX = "inferencex"  # fixed workload: the rule gate refuses a result without an accuracy score
    AGENTX = "agentx"  # accuracy is a separate ``accuracy`` test after a positive result


class CancelOutcome(str, Enum):
    WITHDRAWN = "withdrawn"  # queued or running and not finished: the toolbox wrote a withdrawal
    FINISHED = "finished"  # the result was already written; nothing was changed


@dataclass(frozen=True)
class QueueState:
    """What the scheduler and the dashboard read: queue lengths by priority, running tests by branch, cold start time.

    ``queued_by_priority`` has a key for every priority of 0 to 4, ``running_by_branch`` only for branches with a test
    running. ``queued_by_branch`` counts what each branch has waiting, the number the backpressure limit applies to.
    ``cold_start_seconds`` is the recent sandbox cold start (``None`` before the first one).
    """

    queued_by_priority: Mapping[int, int] = field(default_factory=lambda: dict.fromkeys(_PRIORITIES, 0))
    running_by_branch: Mapping[str, int] = field(default_factory=dict)
    queued_by_branch: Mapping[str, int] = field(default_factory=dict)
    cold_start_seconds: float | None = None

    @property
    def queued(self) -> int:
        return sum(self.queued_by_priority.values())

    @property
    def running(self) -> int:
        return sum(self.running_by_branch.values())


_PRIORITIES = range(PRIORITY_HIGHEST, PRIORITY_LOWEST + 1)


@dataclass(frozen=True)
class TestProgress:
    """How far a test that has not finished has got, as the toolbox knows it; a field it cannot tell is ``None``.

    A ``queued`` test has ``queued_ahead``, the waiting tests the queue prefers to it by priority and arrival (a test of
    a branch at its K is counted, though it may not start first). A ``running`` test has ``running_seconds`` since it
    left the queue and, when its live output can be read, ``last_output_seconds_ago`` (since its newest server or
    benchmark log was written) and ``last_line`` (that log's last non-empty line, cleaned and shortened).
    """

    __test__ = False

    state: Literal["queued", "running"]
    queued_ahead: int | None = None
    running_seconds: float | None = None
    last_output_seconds_ago: float | None = None
    last_line: str | None = None


@dataclass(frozen=True)
class MeasurementCapacity:
    """How many cards the toolbox can still measure on. The scheduler stops a session when ``usable_cards`` falls below
    the cards one test needs (``scheduling.events.no_measurement_cards``), and bounds how many tests run at once by
    ``measurement_cards`` (``scheduling.widening``).

    ``total_cards`` is the card pool the toolbox owns; ``usable_cards`` is the total minus the cards that are lost for
    good (a card whose holder processes were not provably gone is never leased again). Quarantining a node does not
    reduce ``usable_cards``: quarantine only widens ``avoid_nodes`` and never blocks a test. ``known`` is ``False`` for
    a toolbox that has no card pool of its own to read (both counts are then 0 and mean nothing): the caller does not
    compare against them. ``leased_cards`` are the usable cards any lease holds (branch sandboxes, through
    ``RealToolbox.lease_branch_sandbox``, and tool jobs such as ``kernel_tune``), which no test runs on until they are
    released.
    """

    total_cards: int
    usable_cards: int
    known: bool = True
    leased_cards: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.usable_cards <= self.total_cards:
            raise ValueError(f"usable_cards {self.usable_cards} must be between 0 and total_cards {self.total_cards}")
        if not 0 <= self.leased_cards <= self.usable_cards:
            raise ValueError(f"leased_cards {self.leased_cards} must be between 0 and usable_cards {self.usable_cards}")

    @property
    def measurement_cards(self) -> int:
        """The usable cards tests can run on, busy ones included: those no lease holds."""
        return self.usable_cards - self.leased_cards

    @classmethod
    def unknown(cls) -> MeasurementCapacity:
        return cls(0, 0, known=False)


@dataclass(frozen=True)
class BranchLease:
    """What a branch's sandbox holds from ``RealToolbox.lease_branch_sandbox`` until ``release_branch_sandbox``.

    ``gpu_ids`` are absolute card ids, the ones ``HIP_VISIBLE_DEVICES`` names in ``env``. ``root`` is the lease's own
    working directory below the toolbox's session directory; it stays after the release. ``env`` is the session image's
    environment as a test gets it, with every credential and GPU mask taken out and ``HIP_VISIBLE_DEVICES`` set to the
    leased cards. ``remaining_cards`` is how many cards tests can still run on with this lease held
    (:attr:`MeasurementCapacity.measurement_cards`).
    """

    lease_id: str
    branch_id: str
    gpu_ids: tuple[int, ...]
    root: Path
    env: Mapping[str, str] = field(repr=False)
    remaining_cards: int


#: The scope names of MCP tokens and tools: the host (Claw's front end) and a specialist.
SCOPE_HOST = "host"
SCOPE_SPECIALIST = "specialist"


@dataclass(frozen=True)
class SpecialistGrant:
    """What a specialist's CLI needs to reach the campaign's MCP service: the address, and the token with the name of
    the environment variable it travels in (the token never goes into a configuration file)."""

    url: str
    token_env: str
    token: str = field(repr=False)
    #: Hands over a ``beat()`` the service calls every minute or so while a call of this token is in progress.
    heartbeat_sink: Callable[[Callable[[], None]], None] = field(default=lambda _beat: None, repr=False, compare=False)


class SpecialistTokens(Protocol):
    """Issues and takes back the MCP tokens of the branches' specialists; the specialist runtime calls it.

    The token belongs to the branch: a second ``issue`` for a branch whose token is live with the same ``scopes``
    returns the same grant, so a specialist restarted after it was lost keeps its token. ``revoke`` ends the branch's
    token for good; after it, the next ``issue`` makes a new one. Neither is a coroutine: both are local and quick.
    """

    def issue(self, branch: str, scopes: frozenset[str]) -> SpecialistGrant: ...

    def revoke(self, branch: str) -> None: ...


class Toolbox(Protocol):
    """The toolbox's external interface; the real one and the fake implement it."""

    async def submit_test(self, request: TestRequest) -> str:
        """Accept ``request`` and return its ``request_id``; the result is written to the notebook later.

        Acceptance is synchronous: when this returns, the request is in the toolbox's submit log and in the queue and
        the notebook holds its request row. Raises :class:`RequestRejected` when a check of acceptance refuses it and
        :class:`Backpressure` when the branch's queue is full (nothing is written in either case).
        """

    async def cancel(self, request_id: str) -> CancelOutcome:
        """Take back a test that has not finished; the toolbox writes the withdrawal. Idempotent.

        A finished test stays as it is (:attr:`CancelOutcome.FINISHED`); an unknown id raises
        :class:`UnknownTestRequest`.
        """

    async def queue_state(self) -> QueueState: ...

    async def progress(self, request_id: str) -> TestProgress | None:
        """How far a queued or running test has got; ``None`` when the toolbox is not queueing or running it."""

    async def measurement_capacity(self) -> MeasurementCapacity:
        """The cards this toolbox can still measure on; read-only, see :class:`MeasurementCapacity`."""

    async def put_attachment(self, attachment_id: str, kind: Literal["file", "dir"], data: bytes) -> AttachmentRef:
        """Store ``data`` in the session's blob store and return the reference a repro ``TestRequest`` carries.

        ``dir`` data is a tar archive of the directory. Content-addressed, so the same bytes twice give the same
        reference. Raises :class:`RequestRejected` (``attachment_too_large``, ``attachment_unreadable``,
        ``attachments_unsupported``) and stores nothing when the data is refused.
        """

    async def put_kernel_artifact(
        self, kind: KernelKind, target: str | None, env: str, data: bytes, rows_only: bool = False
    ) -> KernelArtifact:
        """Store the tuned table ``data`` in the session's blob store and return the artifact a delta carries.

        ``env`` is the variable the service reads the table's path from (``metadata["load"]``, ``via`` ``env_file``);
        ``target`` is the operator or shape the table serves, the merge slot of the artifact. The reference is minted
        from the content, ``kernel/<kind>/<target>/<sha256><ext>``, so the same table put twice, by any branch, gives
        the same artifact. A variable loads one table of one target: the first target put for it binds it. ``target``
        ``None`` is the target ``env`` is bound to (the artifact's ``target`` names it); a variable bound to none is
        refused, since nothing says which target to bind it to.
        ``rows_only`` (an aiter table variable only) says ``data`` is just the tuned rows, which the test reads over the
        image's default table for the variable, a shape in both taken from the rows (``metadata["load"]["rows_only"]``).
        Raises
        :class:`RequestRejected` (``kernel_artifact_invalid``, ``kernel_variable_taken``, ``target_required``,
        ``attachments_unsupported``) and stores nothing when the table is refused.
        """


__all__ = [
    "SCOPE_HOST",
    "SCOPE_SPECIALIST",
    "REJECTION_CODES",
    "Backpressure",
    "BranchLease",
    "CancelOutcome",
    "MeasurementCapacity",
    "NodeHealth",
    "QueueState",
    "RequestRejected",
    "SpecialistGrant",
    "SpecialistTokens",
    "TestProgress",
    "TestRecordSink",
    "TestWithdrawn",
    "Toolbox",
    "ToolboxError",
    "UnknownTestRequest",
    "WorkloadKind",
]
