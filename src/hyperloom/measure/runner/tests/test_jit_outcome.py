# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the touched modules decide of a test's outcome: a compile error of a module the patch touched is the candidate's,
one in a module it did not touch is not, and a test that ran a touched module from the image's copy is not a measurement."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.measure.jit.builds import BuildLog
from hyperloom.measure.jit.session import JitCacheSession, KindPlan
from hyperloom.measure.jit.store import DirJitStore
from hyperloom.measure.jit_kinds import AITER_JIT
from hyperloom.measure.returncodes import HOSTED_INFRA_FAILURE_RC
from hyperloom.measure.runner.jit_outcome import COMPILE_ERROR, STALE_MODULE, refine_outcome

SUCCEEDED = {"returncode": 0, "error_class": None, "failure_class": None, "failure_reason": None}
UNCLEAR = {"returncode": 1, "error_class": None, "failure_class": "unclear", "failure_reason": "python_exception: boom"}
EXCERPT = "kernels/k.cu:12: use of undeclared identifier 'x'"


@pytest.fixture
def session(tmp_path: Path) -> JitCacheSession:
    plan = KindPlan(AITER_JIT, "a" * 64, touched=frozenset({"module_gemm", "module_gemm_variant"}))
    made = JitCacheSession(
        DirJitStore(tmp_path / "store"), [plan], private_root=tmp_path / "private", staging=tmp_path / "out"
    )
    made.seed()
    return made


def _builds(failed: dict | None = None, started: set | None = None) -> BuildLog:
    return BuildLog({}, frozenset(started or ()), failed or {})


@pytest.mark.parametrize("document", [UNCLEAR, {**UNCLEAR, "failure_class": "candidate"}])
def test_a_failure_the_logs_left_open_becomes_the_candidates_compile_error_when_a_touched_module_failed(
    session, document
) -> None:
    refined = refine_outcome(document, session, _builds({"module_gemm": EXCERPT}))

    assert (refined["failure_class"], refined["failure_reason"]) == (
        "candidate",
        f"{COMPILE_ERROR}: module_gemm: {EXCERPT}",
    )
    assert refined["returncode"] == 1 and refined["error_class"] is None


def test_a_compile_error_in_a_module_the_patch_did_not_touch_leaves_the_verdict_the_logs_gave(session) -> None:
    assert refine_outcome(UNCLEAR, session, _builds({"module_other": EXCERPT})) == UNCLEAR


def test_an_infrastructure_failure_stays_one_even_when_a_touched_module_failed_to_build(session) -> None:
    infra = {**UNCLEAR, "failure_class": "infra", "failure_reason": "node_fault"}

    assert refine_outcome(infra, session, _builds({"module_gemm": EXCERPT})) == infra


def test_a_test_that_ended_well_but_ran_a_touched_module_from_the_images_copy_is_an_infrastructure_failure(
    session, tmp_path
) -> None:
    image = tmp_path / "image" / "module_gemm.so"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image build")
    jit_dir = Path(session.environment()["AITER_JIT_DIR"])
    (jit_dir / "module_gemm.so").symlink_to(image)

    refined = refine_outcome(SUCCEEDED, session, _builds())

    assert refined["failure_class"] == "infra" and refined["returncode"] == HOSTED_INFRA_FAILURE_RC
    assert refined["failure_reason"] == f"{STALE_MODULE}: module_gemm ran as the image built it"


def test_a_test_whose_touched_module_the_server_built_ends_as_it_did(session) -> None:
    jit_dir = Path(session.environment()["AITER_JIT_DIR"])
    (jit_dir / "module_gemm.so").write_bytes(b"built")

    assert refine_outcome(SUCCEEDED, session, _builds(started={"module_gemm"})) == SUCCEEDED


def test_a_test_with_no_session_or_no_touched_module_is_left_as_it_ended(tmp_path) -> None:
    quiet = JitCacheSession(
        DirJitStore(tmp_path / "s"),
        [KindPlan(AITER_JIT, "a" * 64)],
        private_root=tmp_path / "p",
        staging=tmp_path / "o",
    )

    assert refine_outcome(UNCLEAR, None, _builds({"module_gemm": EXCERPT})) == UNCLEAR
    assert refine_outcome(UNCLEAR, quiet, _builds({"module_gemm": EXCERPT})) == UNCLEAR
