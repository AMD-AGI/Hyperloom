# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Content fingerprint regression tests for the ``explore_search`` dedup-ledger key."""

from __future__ import annotations

from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint


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
    legacy = canonical_fingerprint("", {})
    explicit_append = canonical_fingerprint("", {}, args_mode="append")
    remove_flag = canonical_fingerprint("", {}, remove_args=["--enable-prefix-caching"])
    unset_env = canonical_fingerprint("", {}, unset_envs=["SGLANG_ENABLE_FOO"])
    replace_mode = canonical_fingerprint("--max-num-seqs 256", {}, args_mode="replace")
    append_mode = canonical_fingerprint("--max-num-seqs 256", {}, args_mode="append")

    assert explicit_append == legacy
    assert remove_flag != legacy
    assert unset_env != legacy
    assert replace_mode != append_mode


def test_fingerprint_unbalanced_quotes_does_not_crash() -> None:
    """Unbalanced quotes fall back to whitespace split — still deterministic."""
    fp1 = canonical_fingerprint("--flag 'unterminated", {})
    fp2 = canonical_fingerprint("--flag 'unterminated", {})
    assert fp1 == fp2


def test_fingerprint_value_swap_differs() -> None:
    """Swapping values across different flags must produce distinct fingerprints."""
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


def test_shared_state_normalizes_explore_search_tested() -> None:
    """SharedState.from_dict shapes the ``explore_search`` ledger with defensive defaults and preserves fingerprint-keyed ``tested``."""
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
