from __future__ import annotations

from kernelforge.knowledge.kernel_identity import KernelRecipeIdentity
from kernelforge.knowledge.warmstart_identity import rank_fallback_identities


def _target() -> KernelRecipeIdentity:
    return KernelRecipeIdentity(
        producer="forge-loop",
        kernel_name="softmax",
        framework="vllm",
        framework_version="0.11.3",
        backend="triton",
        gpu="mi355x",
    )


def _row(version: str, gpu: str, *, suffix: str = "", **overrides):
    dimensions = {
        "producer": "forge-loop",
        "kernel_name": "softmax",
        "framework": "vllm",
        "framework_version": version,
        "backend": "triton",
        "gpu": gpu,
        **overrides,
    }
    canonical_id = "kernel:" + ":".join(dimensions.values()) + suffix
    return {
        "canonical_id": canonical_id,
        "dimensions": dimensions,
        "updated_at": "2026-09-15T00:00:00Z",
    }


def test_fuzzy_ranking_accepts_newer_older_and_cross_isa_donors():
    rows = [
        _row("1.0.0", "mi300x"),
        _row("0.10.2", "mi355x"),
        _row("0.11.4", "mi355x"),
    ]

    ranked = rank_fallback_identities(_target(), rows)

    assert ranked == [
        rows[2]["canonical_id"],
        rows[1]["canonical_id"],
        rows[0]["canonical_id"],
    ]


def test_fuzzy_ranking_rejects_unknown_or_unparseable_dimensions():
    rows = [
        _row("unknown", "mi355x"),
        _row("0.11.2", "unknown"),
        _row("not-a-version", "mi355x"),
    ]

    assert rank_fallback_identities(_target(), rows) == []


def test_fuzzy_ranking_never_relaxes_the_other_four_dimensions():
    rows = [
        _row("0.11.2", "mi355x", producer="flydsl"),
        _row("0.11.2", "mi355x", kernel_name="attention"),
        _row("0.11.2", "mi355x", framework="sglang"),
        _row("0.11.2", "mi355x", backend="hip"),
    ]

    assert rank_fallback_identities(_target(), rows) == []


def test_exact_identity_is_not_returned_by_fallback_search():
    exact = _row("0.11.3", "mi355x")

    assert rank_fallback_identities(_target(), [exact]) == []
