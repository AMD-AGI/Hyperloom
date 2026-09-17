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


def _row(version: str, gpu: str, **overrides):
    dimensions = {
        "producer": "forge-loop",
        "kernel_name": "softmax",
        "framework": "vllm",
        "framework_version": version,
        "backend": "triton",
        "gpu": gpu,
        **overrides,
    }
    canonical_id = "kernel:" + ":".join(dimensions.values())
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
        _row("0.11.3", "mi355x"),
    ]

    ranked = rank_fallback_identities(_target(), rows)

    assert ranked == [
        rows[2]["canonical_id"],
        rows[1]["canonical_id"],
        rows[0]["canonical_id"],
    ]


def test_two_spellings_of_one_release_rank_as_that_release():
    # Pages written before the version was canonicalized still carry the spelling
    # their campaign used, so the ranking has to read them as the release they name
    # rather than as a neighbouring one.
    rows = [
        _row("0.11.4", "mi355x"),
        _row("v0.11.3+rocm723", "mi355x"),
    ]

    ranked = rank_fallback_identities(_target(), rows)

    assert ranked[0] == rows[1]["canonical_id"]
    assert len(ranked) == 2


def test_runs_that_both_observed_no_version_can_reach_each_other():
    """A target with no version has four exact dimensions left, and that is enough.

    42% of the store's pages name no version, which used to end the fuzzy tier
    before it started: the same operator, framework and backend sat one word away
    under a different spelling of not knowing, and nothing could read it.
    """
    target = KernelRecipeIdentity(
        producer="forge-loop",
        kernel_name="softmax",
        framework="vllm",
        framework_version="unknown",
        backend="triton",
        gpu="mi355x",
    )
    rows = [
        _row("unspecified", "mi355x"),
        _row("0.11.3", "mi355x"),
    ]

    # The known release is still rejected: how far it sits from an unknown one is
    # not a question either string can answer.
    assert rank_fallback_identities(target, rows) == [rows[0]["canonical_id"]]


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
