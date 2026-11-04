# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``KernelVariables.put``: which target a put gets, and that the binding is decided in one step.

The put that leaves its target out takes the target its variable is bound to, and is refused (``target_required``) for a
variable that is bound to none. Resolving that target and binding the variable are one critical section, so the puts of
two branches that arrive together are ordered by it: the one that comes second sees the binding of the first. The
races here use real threads and hold one put inside its critical section.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest

from hyperloom.contracts.recipe import KernelArtifact, KernelKind
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.toolbox import kernel_artifacts
from hyperloom_multibranch.toolbox.kernel_artifacts import KernelVariables

ENV = "AITER_CONFIG_GEMM_A8W8"
OTHER_ENV = "AITER_CONFIG_GEMM_A4W4"
FIRST, SECOND = "gemm_a8w8", "gemm_a8w8_small"
TABLE = b"M,N,K,libtype,solidx\n4096,4096,8192,hipblaslt,17\n"
OTHER_TABLE = b"M,N,K,libtype,solidx\n4096,4096,8192,hipblaslt,23\n"
#: How long a put that must be waiting is given to finish if it is not; far more than a put of a table takes.
_NOT_WAITING_SEC = 0.5
_JOIN_SEC = 20


def put(variables: KernelVariables, target: str | None, env: str = ENV, data: bytes = TABLE) -> KernelArtifact:
    return variables.put(KernelKind.GEMM, target, env, data)


def refusal(variables: KernelVariables, target: str | None, env: str = ENV) -> RequestRejected:
    with pytest.raises(RequestRejected) as refused:
        put(variables, target, env)
    return refused.value


@dataclass
class Call:
    """One put on a thread of its own: what it returned, or the refusal that ended it."""

    thread: threading.Thread = field(init=False)
    artifact: KernelArtifact | None = None
    refused: RequestRejected | None = None

    def join(self) -> Call:
        self.thread.join(_JOIN_SEC)
        assert not self.thread.is_alive(), "a put never ended"
        return self


def on_thread(variables: KernelVariables, target: str | None, data: bytes = TABLE, env: str = ENV) -> Call:
    call = Call()

    def work() -> None:
        try:
            call.artifact = put(variables, target, env, data)
        except RequestRejected as exc:
            call.refused = exc

    call.thread = threading.Thread(target=work, daemon=True)
    call.thread.start()
    return call


@dataclass
class Hold:
    """A put of ``FIRST`` held inside its step (its artifact minted, its variable not bound yet): ``entered`` is set when
    it is there, and it goes on when ``release`` is set."""

    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    def reached(self) -> None:
        assert self.entered.wait(_JOIN_SEC), "the first put never reached its step"


@pytest.fixture
def hold(monkeypatch: pytest.MonkeyPatch) -> Hold:
    held = Hold()
    real = kernel_artifacts.mint_kernel_artifact

    def mint(kind: KernelKind, target: str, env: str, data: bytes, rows_only: bool = False) -> KernelArtifact:
        artifact = real(kind, target, env, data, rows_only)
        if target == FIRST and data == TABLE:
            held.entered.set()
            assert held.release.wait(_JOIN_SEC), "the held put was never released"
        return artifact

    monkeypatch.setattr(kernel_artifacts, "mint_kernel_artifact", mint)
    return held


# --- which target a put gets -----------------------------------------------------------------------------------


def test_a_put_without_a_target_takes_the_target_the_variable_is_bound_to() -> None:
    variables = KernelVariables()
    put(variables, FIRST)

    artifact = put(variables, None, data=OTHER_TABLE)

    assert artifact.target == FIRST and artifact.slot == (KernelKind.GEMM, FIRST)
    assert artifact.ref.startswith(f"kernel/gemm/{FIRST}/")


def test_a_put_without_a_target_for_a_variable_bound_to_none_is_refused_by_name() -> None:
    refused = refusal(KernelVariables(), None)

    assert refused.code == kernel_artifacts.CODE_TARGET_REQUIRED == "target_required"
    assert f"{ENV} is bound to none in this session" in refused.reason


def test_a_refused_put_without_a_target_binds_nothing_and_another_variable_is_not_bound_by_this_one() -> None:
    variables = KernelVariables()
    assert refusal(variables, None).code == "target_required"

    assert put(variables, SECOND).target == SECOND, "the variable was left unbound"
    assert refusal(variables, None, OTHER_ENV).code == "target_required", "a variable of its own is not bound by it"


def test_a_put_with_another_target_than_the_bound_one_is_refused_as_before() -> None:
    variables = KernelVariables()
    put(variables, FIRST)

    refused = refusal(variables, SECOND)

    assert refused.code == kernel_artifacts.CODE_VARIABLE_TAKEN
    assert f"{ENV} is bound to target {FIRST!r}" in refused.reason
    assert put(variables, FIRST, data=OTHER_TABLE).target == FIRST, "the bound target is still taken"


def test_a_put_the_measurement_cannot_load_is_refused_before_it_binds_anything() -> None:
    variables = KernelVariables()
    with pytest.raises(RequestRejected) as refused:
        variables.put(KernelKind.GEMM, FIRST, ENV, b"")

    assert refused.value.code == kernel_artifacts.CODE_INVALID
    assert put(variables, SECOND).target == SECOND, "the empty table bound no target"


# --- two puts at once -----------------------------------------------------------------------------------------


def test_a_put_without_a_target_that_arrives_while_another_binds_the_variable_waits_and_takes_its_target(
    hold: Hold,
) -> None:
    variables = KernelVariables()
    binding = on_thread(variables, FIRST)
    hold.reached()

    leaving_out = on_thread(variables, None, OTHER_TABLE)
    leaving_out.thread.join(_NOT_WAITING_SEC)
    assert leaving_out.thread.is_alive(), "it was answered while the other put was still binding the variable"
    hold.release.set()

    assert binding.join().artifact is not None
    second = leaving_out.join()
    assert second.refused is None and second.artifact is not None
    assert second.artifact.target == FIRST, "it saw the binding made while it waited"


def test_two_puts_of_different_targets_at_once_bind_one_and_refuse_the_other_naming_it(hold: Hold) -> None:
    variables = KernelVariables()
    winner = on_thread(variables, FIRST)
    hold.reached()
    loser = on_thread(variables, SECOND, OTHER_TABLE)
    hold.release.set()

    assert winner.join().artifact is not None and winner.artifact.target == FIRST
    assert loser.join().artifact is None and loser.refused is not None
    assert loser.refused.code == kernel_artifacts.CODE_VARIABLE_TAKEN
    assert f"bound to target {FIRST!r}" in loser.refused.reason
    assert put(variables, None, data=OTHER_TABLE).target == FIRST, "the variable is bound to the one that won"


def test_two_puts_without_a_target_at_once_for_a_variable_bound_to_none_are_both_refused() -> None:
    variables = KernelVariables()

    calls = [on_thread(variables, None, data) for data in (TABLE, OTHER_TABLE)]

    for call in calls:
        assert call.join().artifact is None and call.refused is not None
        assert call.refused.code == kernel_artifacts.CODE_TARGET_REQUIRED
    assert put(variables, SECOND).target == SECOND, "neither bound the variable"
