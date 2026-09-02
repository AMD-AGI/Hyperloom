# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Content fingerprint regression tests for the ``explore_search`` dedup-ledger key."""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    VariantResult,
    variant_fingerprint,
)
from hyperloom.orchestrator.actions.executors._canonical_fingerprint import (
    canonical_fingerprint,
    workload_signature,
)


def test_fingerprint_ignores_name() -> None:
    a = canonical_fingerprint("--block-size 128", {"NCCL_ALGO": "Ring"})
    b = canonical_fingerprint("--block-size 128", {"NCCL_ALGO": "Ring"})
    assert a == b
    va = GridVariant("A", "--block-size 128", {"NCCL_ALGO": "Ring"})
    vb = GridVariant("totally_different_name", "--block-size 128", {"NCCL_ALGO": "Ring"})
    assert va.fingerprint == vb.fingerprint == a


def test_fingerprint_args_order_independent() -> None:
    fp1 = canonical_fingerprint("--block-size 128 --foo bar", {})
    fp2 = canonical_fingerprint("--foo bar --block-size 128", {})
    assert fp1 == fp2


def test_fingerprint_envs_order_independent() -> None:
    fp1 = canonical_fingerprint("", {"A": "1", "B": "2"})
    fp2 = canonical_fingerprint("", {"B": "2", "A": "1"})
    assert fp1 == fp2


def test_fingerprint_env_value_string_coerced() -> None:
    """``"1"`` and ``1`` collide — both end up as the shell string ``"1"``."""
    fp_int = canonical_fingerprint("", {"TP": 1})
    fp_str = canonical_fingerprint("", {"TP": "1"})
    assert fp_int == fp_str


def test_fingerprint_differs_on_args_change() -> None:
    fp_a = canonical_fingerprint("--block-size 128", {})
    fp_b = canonical_fingerprint("--block-size 256", {})
    assert fp_a != fp_b


def test_fingerprint_differs_on_env_change() -> None:
    fp_a = canonical_fingerprint("", {"NCCL_ALGO": "Ring"})
    fp_b = canonical_fingerprint("", {"NCCL_ALGO": "Tree"})
    assert fp_a != fp_b


def test_fingerprint_empty_inputs_stable() -> None:
    fp1 = canonical_fingerprint("", {})
    fp2 = canonical_fingerprint(None, None)
    assert fp1 == fp2
    assert isinstance(fp1, str)
    assert len(fp1) == 16


def test_fingerprint_includes_removal_controls_without_changing_legacy() -> None:
    legacy = variant_fingerprint("", {})
    explicit_append = variant_fingerprint("", {}, args_mode="append")
    remove_flag = variant_fingerprint("", {}, remove_args=["--enable-prefix-caching"])
    unset_env = variant_fingerprint("", {}, unset_envs=["SGLANG_ENABLE_FOO"])
    replace_mode = variant_fingerprint("--max-num-seqs 256", {}, args_mode="replace")
    append_mode = variant_fingerprint("--max-num-seqs 256", {}, args_mode="append")

    assert explicit_append == legacy
    assert remove_flag != legacy
    assert unset_env != legacy
    assert replace_mode != append_mode


def test_grid_variant_fingerprint_carries_removal_controls() -> None:
    a = GridVariant("without_cache", remove_args=["--enable-prefix-caching"])
    b = GridVariant("identity")
    c = GridVariant("without_cache_rename", remove_args=["--enable-prefix-caching"])

    assert a.fingerprint != b.fingerprint
    assert a.fingerprint == c.fingerprint


def test_fingerprint_unbalanced_quotes_does_not_crash() -> None:
    """Unbalanced quotes fall back to whitespace split — still deterministic."""
    fp1 = canonical_fingerprint("--flag 'unterminated", {})
    fp2 = canonical_fingerprint("--flag 'unterminated", {})
    assert fp1 == fp2


def test_fingerprint_value_swap_differs() -> None:
    """Swapping values across different flags must produce distinct fingerprints.

    The flat-token sort bug caused --max-num-seqs 128 --max-model-len 4096 to
    collide with --max-num-seqs 4096 --max-model-len 128 because both produce
    the same sorted token list. Pair-aware hashing preserves the flag->value
    binding.
    """
    fp_a = canonical_fingerprint("--max-num-seqs 128 --max-model-len 4096", {})
    fp_b = canonical_fingerprint("--max-num-seqs 4096 --max-model-len 128", {})
    assert fp_a != fp_b

    fp_c = canonical_fingerprint("--kv-cache-dtype fp8 --block-size 16", {})
    fp_d = canonical_fingerprint("--kv-cache-dtype 16 --block-size fp8", {})
    assert fp_c != fp_d


def test_fingerprint_last_wins_for_repeated_flag() -> None:
    """A repeated flag collapses to its last occurrence."""
    fp_repeat = canonical_fingerprint("--max-num-seqs 128 --max-num-seqs 256", {})
    fp_last = canonical_fingerprint("--max-num-seqs 256", {})
    assert fp_repeat == fp_last


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
    assert gv.fingerprint == canonical_fingerprint(args, envs)


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


def test_workload_signature_stable_and_order_independent() -> None:
    """Same contract fields always digest to the same 12-char value regardless
    of keyword-argument order."""
    sig1 = workload_signature(conc=8, isl=128, osl=256, precision="fp8", tp=1)
    sig2 = workload_signature(tp=1, precision="fp8", osl=256, isl=128, conc=8)
    assert sig1 == sig2
    assert isinstance(sig1, str)
    assert len(sig1) == 12


def test_workload_signature_differs_on_field_change() -> None:
    base = workload_signature(conc=8, isl=128, osl=256, precision="fp8", tp=1)
    changed_conc = workload_signature(conc=16, isl=128, osl=256, precision="fp8", tp=1)
    changed_tp = workload_signature(conc=8, isl=128, osl=256, precision="fp8", tp=2)
    assert base != changed_conc
    assert base != changed_tp


def test_workload_signature_int_and_str_collide() -> None:
    """``8`` and ``"8"`` both coerce to the shell string ``"8"``, so they must
    fingerprint identically -- mirrors the canonical_fingerprint TP behavior."""
    fp_int = workload_signature(conc=8, isl=128, osl=256, precision="fp8", tp=1)
    fp_str = workload_signature(conc="8", isl=128, osl=256, precision="fp8", tp=1)
    assert fp_int == fp_str


def test_workload_signature_falls_back_to_env_vars(monkeypatch) -> None:
    """Omitted args default to the matching $CONC/$ISL/$OSL/$PRECISION/$TP env
    vars so callers can rely on process-wide workload env without threading
    every field through explicitly."""
    monkeypatch.setenv("CONC", "8")
    monkeypatch.setenv("ISL", "128")
    monkeypatch.setenv("OSL", "256")
    monkeypatch.setenv("PRECISION", "fp8")
    monkeypatch.setenv("TP", "1")
    from_env = workload_signature()
    explicit = workload_signature(conc=8, isl=128, osl=256, precision="fp8", tp=1)
    assert from_env == explicit


def test_shared_state_normalizes_explore_search_tested() -> None:
    """SharedState.from_dict shapes the ``explore_search`` ledger with
    defensive defaults and preserves fingerprint-keyed ``tested``."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    fp_a = canonical_fingerprint("--A", {})
    raw = {
        "explore_search": {
            "schema_version": 1,
            "tested": {
                fp_a: {
                    "name": "A",
                    "fingerprint": fp_a,
                    "extra_server_args": "--A",
                    "extra_envs": {},
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


def test_fingerprint_single_dash_flag_differs_from_missing_value() -> None:
    """-x 1 -y 1 and -x 1 -y must hash differently: one flag has a value, the other does not."""
    fp_with = canonical_fingerprint("-x 1 -y 1", {})
    fp_without = canonical_fingerprint("-x 1 -y", {})
    assert fp_with != fp_without


def test_fingerprint_single_dash_flag_order_independent() -> None:
    """-x 1 -y 2 and -y 2 -x 1 must collide: same bindings, different order."""
    fp1 = canonical_fingerprint("-x 1 -y 2", {})
    fp2 = canonical_fingerprint("-y 2 -x 1", {})
    assert fp1 == fp2


def test_fingerprint_negative_number_is_value_not_flag() -> None:
    """A token like -1 is a numeric value, not a flag, and must not cause collisions."""
    fp_flag = canonical_fingerprint("-k -1", {})
    fp_val = canonical_fingerprint("-1 -k", {})
    assert fp_flag != fp_val
