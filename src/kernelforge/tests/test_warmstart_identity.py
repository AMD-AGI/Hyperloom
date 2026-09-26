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


def test_a_stored_version_is_taken_as_written_and_ranks_below_the_release_itself():
    """The two sides of the comparison arrive under different guarantees.

    The target's version is whatever the run observed -- ``0.11.3+rocm723`` from
    installed distribution metadata, ``v0.11.3`` from an image tag -- so it is
    resolved here. A stored dimension was resolved when its page was addressed,
    so it is read as written: a page still carrying an unresolved spelling is one
    nothing writes to any more, and ranking it level with the release's own page
    would keep a dead address serving reads.
    """
    rows = [
        _row("0.11.3+rocm723", "mi355x"),
        _row("v0.11.3", "mi355x"),
    ]

    # Read as written means read as PEP 440 reads it, not as the canonicalizer
    # resolves it: the parser drops a tag ``v``, so that page still ties with the
    # release, while a local segment is part of the version and ranks under it.
    # Both used to tie, because both were resolved again on the way in.
    assert rank_fallback_identities(_target(), rows) == [
        rows[1]["canonical_id"],
        rows[0]["canonical_id"],
    ]


def test_a_stored_spelling_that_names_no_release_is_dropped_rather_than_resolved():
    """``unspecified`` is a spelling no address rule produces any more.

    Folding it here would let the page it addresses keep answering reads while
    every write goes to ``unknown`` -- one kernel, two pages, which is the split
    this dimension was canonicalized to close.
    """
    target = KernelRecipeIdentity(
        producer="forge-loop",
        kernel_name="softmax",
        framework="vllm",
        framework_version="unspecified",
        backend="triton",
        gpu="mi355x",
    )
    rows = [
        _row("unspecified", "mi355x"),
        _row("unknown", "mi355x"),
    ]

    assert rank_fallback_identities(target, rows) == [rows[1]["canonical_id"]]


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
        framework_version="unspecified",
        backend="triton",
        gpu="mi355x",
    )
    rows = [
        _row("unknown", "mi355x"),
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
