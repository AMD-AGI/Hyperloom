"""Content fingerprint regression tests.

The fingerprint is the cross-action dedup-ledger key (used by
``params_search`` and ``backends_search``). It MUST satisfy:

* ``name`` is not an input → renames cannot bypass dedup.
* ``extra_server_args`` is normalized by ``shlex.split`` + sort →
  permutations of the same flags collide.
* ``extra_envs`` is normalized by sorted ``(str(k), str(v))`` pairs →
  dict insertion order and ``"1"`` vs ``1`` collide.
* ``GridVariant.fingerprint`` and ``VariantResult.fingerprint`` agree
  with the standalone function when given identical content.
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


def test_shared_state_migrates_legacy_name_keyed_tested() -> None:
    """SharedState.from_dict re-keys legacy v1 ``params_search.tested``
    entries by content fingerprint and populates ``name_index``."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    legacy = {
        "params_search": {
            "schema_version": 1,
            "accepted": [
                {"name": "A", "extra_server_args": "--A", "extra_envs": {}},
            ],
            "rejected": [
                {"name": "C", "extra_server_args": "--C", "extra_envs": {}},
            ],
            "tested": {
                "A": {"name": "A", "extra_server_args": "--A", "extra_envs": {}},
                "B": {"name": "B", "extra_server_args": "--B", "extra_envs": {}},
                "C": {"name": "C", "extra_server_args": "--C", "extra_envs": {}},
            },
            "cursor": 3,
        },
    }
    ss = SharedState.from_dict(legacy)
    fp_a = variant_fingerprint("--A", {})
    fp_b = variant_fingerprint("--B", {})
    fp_c = variant_fingerprint("--C", {})
    migrated = ss.params_search
    assert migrated["schema_version"] == 2
    assert set(migrated["tested"].keys()) == {fp_a, fp_b, fp_c}
    assert migrated["tested"][fp_a]["name"] == "A"
    assert migrated["name_index"]["A"] == fp_a
    assert migrated["name_index"]["B"] == fp_b
    assert migrated["name_index"]["C"] == fp_c
    # accepted/rejected get fingerprints stamped as well.
    assert migrated["accepted"][0]["fingerprint"] == fp_a
    assert migrated["rejected"][0]["fingerprint"] == fp_c
