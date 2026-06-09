# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Content fingerprint regression tests for the ``explore_search`` dedup-ledger key.

Pins: name is not an input; args/envs are order-normalized; env values are
string-coerced; GridVariant/VariantResult fingerprints match the function.
"""

from __future__ import annotations

from inference_optimizer.orchestrator.action_executors._grid_runner import (
    GridVariant,
    VariantResult,
    variant_fingerprint,
)


def test_fingerprint_ignores_name() -> None:
    a = variant_fingerprint("--block-size 128", {"NCCL_ALGO": "Ring"})
    b = variant_fingerprint("--block-size 128", {"NCCL_ALGO": "Ring"})
    assert a == b
    va = GridVariant("A", "--block-size 128", {"NCCL_ALGO": "Ring"})
    vb = GridVariant("totally_different_name", "--block-size 128", {"NCCL_ALGO": "Ring"})
    assert va.fingerprint == vb.fingerprint == a


def test_fingerprint_args_order_independent() -> None:
    fp1 = variant_fingerprint("--block-size 128 --foo bar", {})
    fp2 = variant_fingerprint("--foo bar --block-size 128", {})
    assert fp1 == fp2


def test_fingerprint_envs_order_independent() -> None:
    fp1 = variant_fingerprint("", {"A": "1", "B": "2"})
    fp2 = variant_fingerprint("", {"B": "2", "A": "1"})
    assert fp1 == fp2


def test_fingerprint_env_value_string_coerced() -> None:
    """``"1"`` and ``1`` collide — both end up as the shell string ``"1"``."""
    fp_int = variant_fingerprint("", {"TP": 1})
    fp_str = variant_fingerprint("", {"TP": "1"})
    assert fp_int == fp_str


def test_fingerprint_differs_on_args_change() -> None:
    fp_a = variant_fingerprint("--block-size 128", {})
    fp_b = variant_fingerprint("--block-size 256", {})
    assert fp_a != fp_b


def test_fingerprint_differs_on_env_change() -> None:
    fp_a = variant_fingerprint("", {"NCCL_ALGO": "Ring"})
    fp_b = variant_fingerprint("", {"NCCL_ALGO": "Tree"})
    assert fp_a != fp_b


def test_fingerprint_empty_inputs_stable() -> None:
    fp1 = variant_fingerprint("", {})
    fp2 = variant_fingerprint(None, None)
    assert fp1 == fp2
    assert isinstance(fp1, str)
    assert len(fp1) == 16


def test_fingerprint_unbalanced_quotes_does_not_crash() -> None:
    """Unbalanced quotes fall back to whitespace split — still deterministic."""
    fp1 = variant_fingerprint("--flag 'unterminated", {})
    fp2 = variant_fingerprint("--flag 'unterminated", {})
    assert fp1 == fp2


def test_variant_result_fingerprint_matches_grid_variant() -> None:
    args = "--block-size 128 --foo bar"
    envs = {"NCCL_ALGO": "Ring", "TP": "8"}
    gv = GridVariant("g", args, envs)
    vr = VariantResult(
        name="g",
        extra_server_args=args,
        extra_envs=envs,
        status="succeeded",
    )
    assert gv.fingerprint == vr.fingerprint
    assert gv.fingerprint == variant_fingerprint(args, envs)


def test_variant_result_to_dict_carries_fingerprint() -> None:
    vr = VariantResult(
        name="g",
        extra_server_args="--block-size 128",
        extra_envs={"A": "1"},
        status="succeeded",
    )
    d = vr.to_dict()
    assert d["fingerprint"] == vr.fingerprint
    assert len(d["fingerprint"]) == 16


def test_shared_state_normalizes_explore_search_tested() -> None:
    """SharedState.from_dict shapes the ``explore_search`` ledger with
    defensive defaults and preserves fingerprint-keyed ``tested``."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    fp_a = variant_fingerprint("--A", {})
    raw = {
        "explore_search": {
            "schema_version": 1,
            "tested": {
                fp_a: {
                    "name": "A", "fingerprint": fp_a,
                    "extra_server_args": "--A", "extra_envs": {},
                },
            },
        },
    }
    ss = SharedState.from_dict(raw)
    es = ss.explore_search
    assert es["tested"][fp_a]["name"] == "A"
    assert es["accepted"] == []
    assert es["rejected"] == []
    assert "winners_history" in es
    assert "synergy_attempted" in es
