# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.contracts.fork import RepairRequest
from hyperloom_multibranch.contracts.intent import Intent, IntentLayer
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict


def _intent(**overrides) -> Intent:
    fields = {
        "branch": "b1",
        "parent_commit": "c1",
        "target": "MoE GEMM",
        "mechanism": "tune tile sizes",
        "layer": IntentLayer.KERNEL,
    }
    fields.update(overrides)
    return Intent(**fields)


def test_fingerprint_ignores_spelling_branch_and_commit():
    respelled = _intent(branch="b2", parent_commit="c2", target="  moe   gemm", mechanism="Tune Tile Sizes ")
    assert respelled.fingerprint == _intent().fingerprint


@pytest.mark.parametrize(
    "overrides",
    [{"layer": IntentLayer.SOURCE}, {"mechanism": "fuse the epilogue"}, {"target": "attention"}],
)
def test_fingerprint_separates_what_is_tried(overrides):
    assert _intent(**overrides).fingerprint != _intent().fingerprint


def test_fingerprint_fields_do_not_run_together():
    assert _intent(target="a b", mechanism="c").fingerprint != _intent(target="a", mechanism="b c").fingerprint


@pytest.mark.parametrize(("target", "mechanism"), [("", "tune"), ("gemm", "   ")])
def test_an_intent_without_a_target_or_mechanism_is_refused(target, mechanism):
    with pytest.raises(ValueError, match="target and a mechanism"):
        _intent(target=target, mechanism=mechanism)


def _judgement(verdict: Verdict) -> Judgement:
    return Judgement(verdict, 8.0, 2.0, ("r1", "r2"), {"output_throughput": 1080.0}, accuracy_ok=False)


def test_only_a_blocked_gain_is_sent_for_repair():
    delta = RecipeDelta(config=ConfigDelta(server_args="--kv-cache-dtype fp8"))
    request = RepairRequest(_intent(), delta, _judgement(Verdict.BLOCKED), "per-tensor KV scales; scale per head")
    assert request.intent.parent_commit == "c1"
    with pytest.raises(ValueError, match="only a blocked gain"):
        RepairRequest(_intent(), delta, _judgement(Verdict.INCONCLUSIVE), "the defect did not reproduce")
