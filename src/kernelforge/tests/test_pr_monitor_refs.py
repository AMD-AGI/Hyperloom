"""Tests for PR reference rendering, snapshotting and negative caching."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from kernelforge.knowledge.pr_monitor_client import (
    PRContractError,
    PRMonitorClient,
    PRTransportError,
)
from kernelforge.knowledge.pr_monitor_refs import (
    DEFAULT_MAX_BYTES,
    MAX_ENTRY_BYTES,
    UNTRUSTED_PREFIX,
    Snapshot,
    byte_len,
    clip_bytes,
    collect_references,
    commit_snapshot,
    entry_key,
    entry_to_reference,
    identify_repo_by_path,
    is_query_empty,
    load_snapshot,
    merge_references,
    query_key,
    record_empty_query,
    refs_dir,
    render_entry,
    render_index,
    render_reference_set,
    sanitize,
    save_snapshot,
    write_index,
    write_provenance,
)
from kernelforge.knowledge.pr_monitor_search import PRReference
from kernelforge.knowledge.pr_query_context import (
    PR_REPOS_EXPECTED,
    REASON_CONTRACT_ERROR,
    REASON_SKIPPED_DEADLINE,
)


def _ref(number: int = 1, **kwargs) -> PRReference:
    base = dict(
        repo="ROCm/FlyDSL",
        number=number,
        title=f"Optimize kernel {number}",
        hit_via=("file_path",),
        is_merged=True,
        worth_trying=0.6,
        components=("fused_moe", "gemm2"),
        mechanisms=("vectorize",),
        summary=f"Distilled summary {number}",
        head_sha=f"sha{number}",
        schema_version="1",
        n_files=3,
    )
    base.update(kwargs)
    return PRReference(**base)


def test_control_characters_are_removed():
    assert "\x00" not in sanitize("bad\x00text")
    assert sanitize("a\x07b") == "a b"


def test_code_fences_cannot_break_out_of_the_prompt():
    cleaned = sanitize("```python\nimport os\n```")

    assert "```" not in cleaned
    assert "`" not in cleaned


def test_newlines_are_flattened_into_one_line():
    assert sanitize("line one\nline two\n\n  line three") == "line one line two line three"


def test_sanitize_tolerates_none():
    assert sanitize(None) == ""


def test_clip_bytes_never_splits_a_character():
    text = "\u4f60\u597d\u4e16\u754c" * 10

    clipped = clip_bytes(text, 20)

    assert byte_len(clipped) <= 20
    clipped.encode("utf-8").decode("utf-8")


def test_clip_bytes_leaves_short_text_alone():
    assert clip_bytes("short", 100) == "short"


def test_block_opens_with_the_untrusted_data_disclaimer():
    """This text is the only boundary between PR content and system instructions."""
    block = render_reference_set([_ref()])

    assert UNTRUSTED_PREFIX in block
    assert block.index(UNTRUSTED_PREFIX) < block.index("ROCm/FlyDSL#1")


def test_empty_input_renders_nothing_not_a_bare_heading():
    assert render_reference_set([]) == ""


def test_entry_stays_within_the_per_entry_budget():
    huge = _ref(
        title="t" * 4000,
        summary="s" * 4000,
        risk_notes="r" * 4000,
        expected_gain="g" * 4000,
        components=tuple(f"component_{i}" for i in range(40)),
        mechanisms=tuple(f"mechanism_{i}" for i in range(40)),
    )

    assert byte_len(render_entry(huge)) <= MAX_ENTRY_BYTES


def test_entry_budget_keeps_every_actionable_field():
    """Share the entry budget instead of deleting trailing fields."""
    reference = _ref(
        title="t" * 4000,
        summary="s" * 4000,
        risk_notes="r" * 4000,
        expected_gain="g" * 4000,
        components=tuple(f"component_{i}" for i in range(40)),
        mechanisms=tuple(f"mechanism_{i}" for i in range(40)),
    )

    entry = render_entry(reference)

    for field in (
        "title:",
        "summary:",
        "components:",
        "mechanisms:",
        "expected gain:",
        "risk:",
    ):
        assert field in entry


def test_five_entries_fit_inside_the_total_budget():
    """700 B x TOP_K plus the disclaimer must fit, or TOP_K silently shrinks."""
    refs = [
        _ref(
            n,
            title="t" * 300,
            summary="s" * 300,
            components=tuple(f"comp{i}" for i in range(8)),
        )
        for n in range(1, 6)
    ]

    block = render_reference_set(refs)

    assert byte_len(block) <= DEFAULT_MAX_BYTES
    for n in range(1, 6):
        assert f"#{n} " in block


def test_over_budget_drops_whole_entries_never_truncates_one():
    refs = [_ref(n, summary="s" * 600) for n in range(1, 20)]

    unbounded = render_reference_set(refs, max_bytes=1_000_000)
    block = render_reference_set(refs, max_bytes=1500)
    kept = block.count("- ROCm/FlyDSL#")

    assert byte_len(block) <= 1500
    assert 0 < kept < unbounded.count("- ROCm/FlyDSL#")
    # Whatever survived is byte-identical to its unbounded rendering.
    for reference in refs[:kept]:
        assert render_entry(reference) in block


def test_impossibly_small_budget_yields_nothing():
    assert render_reference_set([_ref()], max_bytes=10) == ""


def test_entry_reports_state_score_source_and_size():
    entry = render_entry(_ref(959, worth_trying=0.6, is_merged=True, n_files=3))

    assert "ROCm/FlyDSL#959" in entry
    assert "merged" in entry
    assert "worth 0.60" in entry
    assert "via file_path" in entry
    assert "3 files" in entry


def test_open_pr_is_labelled_open():
    assert "open" in render_entry(_ref(is_merged=False))


def test_unknown_score_is_labelled_not_rendered_as_none():
    entry = render_entry(_ref(worth_trying=None))

    assert "worth unknown" in entry
    assert "None" not in entry


def test_undistilled_reference_is_flagged():
    entry = render_entry(_ref(distill_absent=True))

    assert "not distilled yet" in entry


def test_multi_source_hits_are_shown_joined():
    assert "via file_path+search" in render_entry(_ref(hit_via=("file_path", "search")))


def test_none_valued_optional_fields_do_not_crash_rendering():
    reference = PRReference(repo="r/x", number=1)

    entry = render_entry(reference)

    assert "r/x#1" in entry


def test_environment_overrides_the_total_budget(monkeypatch):
    monkeypatch.setenv("PR_KB_MAX_BYTES", "400")

    block = render_reference_set([_ref(n, summary="s" * 300) for n in range(1, 6)])

    assert byte_len(block) <= 400


def test_entry_key_includes_head_and_schema():
    assert entry_key(_ref(7)) == "ROCm/FlyDSL#7@sha7:1"


def test_entry_key_tolerates_a_missing_head():
    assert entry_key(_ref(7, head_sha="", schema_version="")) == "ROCm/FlyDSL#7@nohead:0"


def test_force_push_produces_a_new_entry_rather_than_a_rewrite():
    snapshot = Snapshot()
    merge_references(snapshot, [_ref(7, head_sha="aaa")])
    merge_references(snapshot, [_ref(7, head_sha="bbb")])

    assert len(snapshot.entries) == 2


def test_merge_is_monotonic_and_reports_only_new_entries():
    snapshot = Snapshot()
    first = merge_references(snapshot, [_ref(1), _ref(2)])
    snapshot.entries[entry_key(_ref(1))]["worth_trying"] = "SENTINEL"
    second = merge_references(snapshot, [_ref(1), _ref(3)])

    assert [ref.number for ref in first] == [1, 2]
    assert [ref.number for ref in second] == [3]
    assert snapshot.entries[entry_key(_ref(1))]["worth_trying"] == "SENTINEL"


def test_snapshot_round_trips_through_disk(tmp_path):
    snapshot = Snapshot()
    merge_references(snapshot, [_ref(1)])
    record_empty_query(snapshot, query_key("search", "ROCm/aiter", "nothing"))

    save_snapshot(str(tmp_path), snapshot)
    restored = load_snapshot(str(tmp_path))

    assert restored.entries == snapshot.entries
    assert restored.empty_queries == snapshot.empty_queries


def test_missing_snapshot_loads_empty(tmp_path):
    snapshot = load_snapshot(str(tmp_path))

    assert snapshot.entries == {}
    assert snapshot.empty_queries == {}


def test_corrupt_snapshot_degrades_instead_of_raising(tmp_path):
    path = refs_dir(str(tmp_path))
    path.mkdir(parents=True, exist_ok=True)
    (path / "snapshot.json").write_text("{not json")

    assert load_snapshot(str(tmp_path)).entries == {}


def test_snapshot_with_wrong_shape_degrades(tmp_path):
    path = refs_dir(str(tmp_path))
    path.mkdir(parents=True, exist_ok=True)
    (path / "snapshot.json").write_text(json.dumps({"entries": "nope"}))

    assert load_snapshot(str(tmp_path)).entries == {}


def test_snapshot_from_a_non_dict_payload_is_rejected():
    with pytest.raises(ValueError, match="snapshot must be an object"):
        Snapshot.from_dict(["unexpected"])


def test_snapshot_write_is_durable_and_leaves_no_temp_file(tmp_path):
    save_snapshot(str(tmp_path), Snapshot())

    directory = refs_dir(str(tmp_path))
    assert (directory / "snapshot.json").is_file()
    assert not [p for p in directory.iterdir() if p.name.endswith(".tmp")]


def test_query_key_is_normalized():
    assert query_key("search", "R/x", "  Fused   RMSNorm ") == "search|R/x|fused rmsnorm"


def test_unseen_query_is_not_cached_as_empty():
    assert is_query_empty(Snapshot(), query_key("search", "R/x", "moe")) is False


def test_recorded_empty_query_is_remembered():
    snapshot = Snapshot()
    key = query_key("file_path", "ROCm/FlyDSL", "a/b.py")
    record_empty_query(snapshot, key)

    assert is_query_empty(snapshot, key) is True


def test_expired_empty_record_is_requeried():
    snapshot = Snapshot()
    key = query_key("search", "R/x", "moe")
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    snapshot.empty_queries[key] = {
        "queried_at": past.isoformat(),
        "empty_until": past.isoformat(),
    }

    assert is_query_empty(snapshot, key) is False


def test_malformed_empty_record_is_ignored():
    snapshot = Snapshot()
    snapshot.empty_queries["k"] = {"empty_until": "not-a-date"}
    snapshot.empty_queries["k2"] = "not-a-dict"

    assert is_query_empty(snapshot, "k") is False
    assert is_query_empty(snapshot, "k2") is False


def test_naive_timestamp_is_treated_as_utc():
    snapshot = Snapshot()
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).replace(tzinfo=None)
    snapshot.empty_queries["k"] = {"empty_until": future.isoformat()}

    assert is_query_empty(snapshot, "k") is True


def test_index_lists_every_surfaced_reference(tmp_path):
    snapshot = Snapshot()
    merge_references(snapshot, [_ref(1), _ref(2, worth_trying=None, is_merged=False)])

    markdown = render_index(snapshot)

    assert "ROCm/FlyDSL#1" in markdown
    assert "ROCm/FlyDSL#2" in markdown
    assert "unknown" in markdown
    assert "entries: 2" in markdown


def test_index_is_written_next_to_the_snapshot(tmp_path):
    snapshot = Snapshot()
    merge_references(snapshot, [_ref(1)])

    path = write_index(str(tmp_path), snapshot)

    assert path.name == "index.md"
    assert "ROCm/FlyDSL#1" in path.read_text()


class _StubClient:
    """Client double recording which candidate queries were actually issued."""

    def __init__(self, *, healthy=True, prs=None, by_file=None, by_query=None):
        self._healthy = healthy
        self._prs = prs or {}
        self.by_file = by_file or {}
        self.by_query = by_query or {}
        self.searched: list[str] = []
        self.path_queries: list[str] = []
        self.timeouts: list[float | None] = []

    def healthz(self, *, timeout_sec=None):
        self.timeouts.append(timeout_sec)
        return self._healthy

    def list_repos(self, *, timeout_sec=None):
        self.timeouts.append(timeout_sec)
        return [
            {"repo_name": name, "is_active": True}
            for name in (
                "ROCm/aiter",
                "ROCm/ATOM",
                "ROCm/FlyDSL",
                "ROCm/hip",
                "ROCm/vllm",
                "sgl-project/sglang",
                "triton-lang/triton",
                "vllm-project/vllm",
            )
        ]

    def pr_request(self, repo, number):
        return (f"/repos/{repo}/prs/{number}", None)

    def get_many(self, requests, *, budget_sec=None):
        from kernelforge.knowledge.pr_monitor_client import FetchOutcome

        outcomes = []
        for path, params in requests:
            params = params or {}
            if "/prs/" in path:
                number = int(path.rsplit("/", 1)[-1])
                outcomes.append(FetchOutcome(path, payload=self._prs.get(number)))
                continue
            if params.get("file_path"):
                self.path_queries.append(params["file_path"])
                numbers = self.by_file.get(params["file_path"], [])
            else:
                self.searched.append(params.get("q", ""))
                numbers = self.by_query.get(params.get("q", ""), [])
            outcomes.append(FetchOutcome(path, payload={"items": [{"number": n} for n in numbers]}))
        return outcomes

    def list_recent_prs(self, repo, *, state="merged", limit=5, timeout_sec=None):
        self.timeouts.append(timeout_sec)
        return []


def _pr_payload(number: int, worth: float = 0.6) -> dict:
    return {
        "summary": {
            "title": f"PR {number}",
            "is_merged": True,
            "pr_updated_at": "2026-08-01T00:00:00Z",
        },
        "files": [{"path": "a.py"}],
        "distill": {
            "status": "ok",
            "worth_trying": worth,
            "components": ["fused_moe"],
            "summary": f"summary {number}",
            "head_sha": f"sha{number}",
            "schema_version": "1",
        },
    }


def test_unreachable_service_with_no_cache_yields_an_empty_fragment(tmp_path):
    result = collect_references(workspace_dir=str(tmp_path), client=_StubClient(healthy=False), kernel_backend="aiter")

    assert result.reason == "service_unreachable"
    assert result.prompt_context == ""


def test_unreachable_service_still_shows_the_cached_references(tmp_path):
    """A transient outage must not retract references mid-campaign."""
    warm = _StubClient(by_query={"moe": [1]}, prs={1: _pr_payload(1)})
    collect_references(workspace_dir=str(tmp_path), client=warm, kernel_backend="aiter", operator_name="moe")

    result = collect_references(
        workspace_dir=str(tmp_path),
        client=_StubClient(healthy=False),
        kernel_backend="aiter",
        operator_name="moe",
    )

    assert result.injected
    assert "ROCm/aiter#1" in result.prompt_context
    assert len(result.references) == result.stats["injected_entries"]
    assert result.reason == "service_unreachable"
    assert result.stats["degraded_reason"] == "service_unreachable"
    assert result.stats["http_calls"] == 0


def test_unresolvable_repo_still_shows_the_cached_references(tmp_path):
    warm = _StubClient(by_query={"moe": [1]}, prs={1: _pr_payload(1)})
    collect_references(workspace_dir=str(tmp_path), client=warm, kernel_backend="aiter", operator_name="moe")

    # A later invocation cannot resolve a repo at all.
    result = collect_references(
        workspace_dir=str(tmp_path),
        client=_StubClient(),
        kernel_backend="ck",
        operator_name="moe",
    )

    assert result.injected
    assert result.stats["degraded_reason"] == "repo_unresolved"


def test_transport_failure_still_shows_the_cached_references(tmp_path):
    warm = _StubClient(by_query={"moe": [1]}, prs={1: _pr_payload(1)})
    collect_references(workspace_dir=str(tmp_path), client=warm, kernel_backend="aiter", operator_name="moe")

    class _Unavailable(_StubClient):
        def list_repos(self, *, timeout_sec=None):
            raise PRTransportError("offline")

    result = collect_references(
        workspace_dir=str(tmp_path),
        client=_Unavailable(),
        kernel_backend="aiter",
        operator_name="moe",
    )

    assert result.injected
    assert result.reason == "service_unreachable"
    assert result.stats["degraded_reason"] == "service_unreachable"


def test_unexpected_client_failure_is_not_silenced(tmp_path):
    class _Broken(_StubClient):
        def list_repos(self, *, timeout_sec=None):
            raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        collect_references(
            workspace_dir=str(tmp_path),
            client=_Broken(),
            kernel_backend="aiter",
            operator_name="moe",
        )


def test_unresolvable_repo_makes_no_query(tmp_path):
    client = _StubClient()

    result = collect_references(workspace_dir=str(tmp_path), client=client, kernel_backend="ck", operator_name="gemm")

    assert result.reason == "repo_unresolved"
    assert client.searched == []


def test_successful_lookup_renders_persists_and_indexes(tmp_path):
    client = _StubClient(by_query={"moe": [1]}, prs={1: _pr_payload(1)})

    result = collect_references(workspace_dir=str(tmp_path), client=client, kernel_backend="aiter", operator_name="moe")

    assert result.injected
    assert UNTRUSTED_PREFIX in result.prompt_context
    assert result.stats["injected_bytes"] == byte_len(result.prompt_context)
    directory = refs_dir(str(tmp_path))
    assert (directory / "snapshot.json").is_file()
    assert "ROCm/aiter#1" in (directory / "index.md").read_text()


def test_deferred_persistence_leaves_the_workspace_untouched(tmp_path):
    """A caller with a guard ahead of it must be able to query read-only."""
    client = _StubClient(by_query={"moe": [1]}, prs={1: _pr_payload(1)})

    result = collect_references(
        workspace_dir=str(tmp_path),
        client=client,
        kernel_backend="aiter",
        operator_name="moe",
        persist=False,
    )

    assert result.injected, "the prompt is still rendered from memory"
    assert result.pending_snapshot["entries"]
    assert not refs_dir(str(tmp_path)).exists()
    assert list(tmp_path.iterdir()) == []


def test_a_deferred_snapshot_is_persisted_on_commit(tmp_path):
    client = _StubClient(by_query={"moe": [1]}, prs={1: _pr_payload(1)})
    result = collect_references(
        workspace_dir=str(tmp_path),
        client=client,
        kernel_backend="aiter",
        operator_name="moe",
        persist=False,
    )

    commit_snapshot(str(tmp_path), result.pending_snapshot)

    directory = refs_dir(str(tmp_path))
    assert (directory / "snapshot.json").is_file()
    assert "ROCm/aiter#1" in (directory / "index.md").read_text()
    assert load_snapshot(str(tmp_path)).entries == result.pending_snapshot["entries"]


def test_committing_nothing_creates_nothing(tmp_path):
    """A degraded lookup leaves no snapshot to commit."""
    commit_snapshot(str(tmp_path), {})

    assert not refs_dir(str(tmp_path)).exists()


def test_snapshot_keeps_all_surfaced_references_beyond_top_k(tmp_path):
    """Persist fetched references that are not shown in the current top-k."""
    numbers = list(range(1, 9))
    client = _StubClient(
        by_query={"moe": numbers},
        prs={number: _pr_payload(number) for number in numbers},
    )

    result = collect_references(
        workspace_dir=str(tmp_path),
        client=client,
        kernel_backend="aiter",
        operator_name="moe",
    )

    assert len(result.references) == 5
    assert len(load_snapshot(str(tmp_path)).entries) == len(numbers)


def test_a_query_proven_empty_is_not_reissued(tmp_path):
    """The negative cache is what keeps refreshes from re-paying for misses."""
    client = _StubClient(by_query={})

    first = collect_references(workspace_dir=str(tmp_path), client=client, kernel_backend="aiter", operator_name="moe")
    issued_first = list(client.searched)
    client.searched.clear()

    second = collect_references(workspace_dir=str(tmp_path), client=client, kernel_backend="aiter", operator_name="moe")

    assert issued_first
    assert client.searched == []
    assert second.stats["skipped_cached_empty"] == len(issued_first)
    assert first.reason and second.reason == "no_candidate"


def test_a_query_that_hit_is_reissued_on_the_next_call(tmp_path):
    client = _StubClient(by_query={"moe": [1]}, prs={1: _pr_payload(1)})

    collect_references(workspace_dir=str(tmp_path), client=client, kernel_backend="aiter", operator_name="moe")
    client.searched.clear()
    collect_references(workspace_dir=str(tmp_path), client=client, kernel_backend="aiter", operator_name="moe")

    assert client.searched


def test_snapshot_is_monotonic_across_two_lookups(tmp_path):
    client = _StubClient(by_query={"moe": [1]}, prs={1: _pr_payload(1, worth=0.6)})
    collect_references(workspace_dir=str(tmp_path), client=client, kernel_backend="aiter", operator_name="moe")

    client._prs = {1: _pr_payload(1, worth=0.1)}
    collect_references(workspace_dir=str(tmp_path), client=client, kernel_backend="aiter", operator_name="moe")

    entries = load_snapshot(str(tmp_path)).entries
    assert len(entries) == 1
    assert next(iter(entries.values()))["worth_trying"] == pytest.approx(0.6)


def test_resume_reinjects_from_the_snapshot_when_a_refresh_finds_nothing(tmp_path):
    """Dropping already-shown references mid-campaign would contradict the lesson."""
    client = _StubClient(by_query={"moe": [1]}, prs={1: _pr_payload(1)})
    first = collect_references(workspace_dir=str(tmp_path), client=client, kernel_backend="aiter", operator_name="moe")
    assert first.injected

    # A later call finds nothing new: different keywords, no hits.
    quiet = _StubClient(by_query={})
    second = collect_references(
        workspace_dir=str(tmp_path),
        client=quiet,
        kernel_backend="aiter",
        operator_name="moe",
    )

    assert second.injected, "the references already shown must keep being shown"
    assert "ROCm/aiter#1" in second.prompt_context
    assert second.stats["from_snapshot"] == 1
    assert "degraded_reason" not in second.stats


def test_service_outage_still_reinjects_what_was_already_shown(tmp_path):
    client = _StubClient(by_query={"moe": [1]}, prs={1: _pr_payload(1)})
    collect_references(workspace_dir=str(tmp_path), client=client, kernel_backend="aiter", operator_name="moe")

    class _Broken(_StubClient):
        def get_many(self, requests, *, budget_sec=None):
            from kernelforge.knowledge.pr_monitor_client import (
                FetchOutcome,
                PRTransportError,
            )

            return [FetchOutcome(path, error=PRTransportError("down")) for path, _ in requests]

    second = collect_references(
        workspace_dir=str(tmp_path),
        client=_Broken(),
        kernel_backend="aiter",
        operator_name="moe",
    )

    assert second.injected
    assert "ROCm/aiter#1" in second.prompt_context
    assert second.stats["degraded_reason"] == "service_unreachable"


def test_all_queries_cached_empty_still_renders_prior_references(tmp_path):
    client = _StubClient(by_file={"a.py": [1]}, by_query={}, prs={1: _pr_payload(1)})
    (tmp_path / "a.py").write_text("x")
    collect_references(
        workspace_dir=str(tmp_path),
        client=client,
        kernel_backend="aiter",
        source_files=["a.py"],
        operator_name="moe",
    )

    # Second call: the keyword query is now a cached miss, path query still hits.
    second = collect_references(
        workspace_dir=str(tmp_path),
        client=client,
        kernel_backend="aiter",
        source_files=["a.py"],
        operator_name="moe",
    )

    assert second.injected
    assert second.stats["skipped_cached_empty"] >= 1


def test_snapshot_entry_round_trips_every_rendered_field(tmp_path):
    """Re-rendering from disk must not lose the distill prose."""
    reference = _ref(
        7,
        summary="detailed distill prose",
        risk_notes="watch occupancy",
        expected_gain="up",
        mechanisms=("vectorize", "prefetch"),
    )
    snapshot = Snapshot()
    merge_references(snapshot, [reference])
    save_snapshot(str(tmp_path), snapshot)

    restored = entry_to_reference(next(iter(load_snapshot(str(tmp_path)).entries.values())))

    assert restored.summary == "detailed distill prose"
    assert restored.risk_notes == "watch occupancy"
    assert restored.expected_gain == "up"
    assert restored.mechanisms == ("vectorize", "prefetch")


def test_unusable_snapshot_entries_are_skipped_on_rebuild():
    assert entry_to_reference({"number": 1}) is None
    assert entry_to_reference({"repo": "r/x"}) is None
    assert entry_to_reference({"repo": "r/x", "number": "abc"}) is None
    assert entry_to_reference("junk") is None


class _PathIndexClient(_StubClient):
    """Serves ?file_path= lookups so only one repo owns a given path."""

    def __init__(self, owner_of: dict, **kwargs):
        super().__init__(**kwargs)
        self.owner_of = owner_of
        self.probed: list[str] = []

    def get_many(self, requests, *, budget_sec=None):
        from kernelforge.knowledge.pr_monitor_client import FetchOutcome

        outcomes = []
        for path, params in requests:
            params = params or {}
            if params.get("file_path") and "/prs/" not in path:
                repo = path[len("/repos/") : -len("/prs")]
                self.probed.append(repo)
                owner = self.owner_of.get(params["file_path"])
                hit = repo in owner if isinstance(owner, set) else owner == repo
                outcomes.append(
                    FetchOutcome(
                        path,
                        payload={"items": [{"number": 1}]} if hit else None,
                    )
                )
                continue
            outcomes.append(super().get_many([(path, params)])[0])
        return outcomes


class _InvalidProbeClient(_PathIndexClient):
    def get_many(self, requests, *, budget_sec=None):
        from kernelforge.knowledge.pr_monitor_client import FetchOutcome

        return [FetchOutcome(path, payload={"unexpected": True}) for path, _ in requests]


class _ContractProbeUnavailableDiscovery(_PathIndexClient):
    def get_many(self, requests, *, budget_sec=None):
        from kernelforge.knowledge.pr_monitor_client import FetchOutcome

        if all((params or {}).get("limit") == 1 for _, params in requests):
            outcomes = []
            failed = False
            for path, _ in requests:
                repo = path[len("/repos/") : -len("/prs")]
                if repo == "ROCm/aiter":
                    outcomes.append(FetchOutcome(path, payload={"items": [{"number": 1}]}))
                elif not failed:
                    outcomes.append(FetchOutcome(path, error=PRContractError("bad payload")))
                    failed = True
                else:
                    outcomes.append(FetchOutcome(path, payload=None))
            return outcomes
        return [FetchOutcome(path, error=PRTransportError("offline")) for path, _ in requests]


def test_fork_upstream_is_identified_by_source_path():
    """Resolve fork ownership from the exact source path."""
    path = "csrc/py_itfs_ck/mha_batch_prefill_kernels.cu"
    client = _PathIndexClient({path: "ROCm/aiter"})

    assert identify_repo_by_path(client, path, PR_REPOS_EXPECTED, hint="carlushuang/aiter-k3") == (
        "ROCm/aiter",
        len(PR_REPOS_EXPECTED),
        "",
    )


def test_a_path_no_repo_owns_identifies_nothing():
    client = _PathIndexClient({})

    assert identify_repo_by_path(client, "no/such/file.cu", PR_REPOS_EXPECTED) == (
        "",
        len(PR_REPOS_EXPECTED),
        "",
    )


def test_name_affinity_resolves_multiple_path_owners():
    path = "a/b.cu"
    client = _PathIndexClient({path: {"ROCm/aiter", "ROCm/ATOM"}})

    assert (
        identify_repo_by_path(
            client,
            path,
            PR_REPOS_EXPECTED,
            hint="someone/aiter-k3",
        )[0]
        == "ROCm/aiter"
    )


def test_identify_needs_both_a_path_and_candidates():
    client = _PathIndexClient({})

    assert identify_repo_by_path(client, "", ("ROCm/aiter",)) == ("", 0, "")
    assert identify_repo_by_path(client, "a/b.cu", ()) == ("", 0, "")


def test_invalid_probe_payload_reports_a_contract_error():
    assert identify_repo_by_path(
        _InvalidProbeClient({}),
        "a/b.cu",
        ("ROCm/aiter",),
    ) == ("", 1, REASON_CONTRACT_ERROR)


def test_untracked_fork_recovers_and_still_injects(tmp_path):
    """End to end: an untracked fork remote must not lose the feature when the
    source path can prove which upstream it belongs to."""
    (tmp_path / "csrc").mkdir()
    (tmp_path / "csrc" / "k.cu").write_text("// kernel")
    client = _PathIndexClient(
        {"csrc/k.cu": "ROCm/aiter"},
        by_query={"mha": [1]},
        prs={1: _pr_payload(1)},
    )

    result = collect_references(
        workspace_dir=str(tmp_path),
        client=client,
        git_remote="https://github.com/carlushuang/aiter-k3.git",
        source_files=["csrc/k.cu"],
        operator_name="mha",
    )

    assert result.repo == "ROCm/aiter"
    assert result.reason != "repo_untracked"


def _fork_workspace(tmp_path):
    (tmp_path / "csrc").mkdir()
    (tmp_path / "csrc" / "k.cu").write_text("// kernel")
    return {
        "workspace_dir": str(tmp_path),
        "git_remote": "https://github.com/carlushuang/aiter-k3.git",
        "source_files": ["csrc/k.cu"],
        "operator_name": "mha",
    }


def test_probe_contract_error_outranks_later_discovery_outage(tmp_path):
    options = _fork_workspace(tmp_path)
    payload = _pr_payload(1)
    payload["distill"]["components"] = ["mha"]
    collect_references(
        workspace_dir=options["workspace_dir"],
        client=_StubClient(by_query={"mha": [1]}, prs={1: payload}),
        kernel_backend="aiter",
        operator_name="mha",
    )

    result = collect_references(
        client=_ContractProbeUnavailableDiscovery({}),
        **options,
    )

    assert result.injected
    assert result.stats["degraded_reason"] == REASON_CONTRACT_ERROR


def test_probe_contract_error_is_reported_without_starting_discovery(tmp_path):
    result = collect_references(
        client=_InvalidProbeClient({}),
        **_fork_workspace(tmp_path),
    )

    assert result.reason == REASON_CONTRACT_ERROR
    assert result.repo == ""
    assert result.stats["http_calls"] == len(PR_REPOS_EXPECTED)


def test_probe_requests_are_counted_as_http_calls(tmp_path):
    """Probing is real traffic. Omitting it understates the cost of the
    untracked-fork path by one request per tracked repo."""
    client = _PathIndexClient({"csrc/k.cu": "ROCm/aiter"}, by_query={"mha": [1]}, prs={1: _pr_payload(1)})

    result = collect_references(client=client, **_fork_workspace(tmp_path))

    assert result.stats["http_calls"] >= len(PR_REPOS_EXPECTED)


def test_probe_requests_are_counted_when_identification_fails(tmp_path):
    """The degraded path spent those requests too."""
    client = _PathIndexClient({})

    result = collect_references(client=client, **_fork_workspace(tmp_path))

    assert result.reason == "repo_untracked"
    assert result.repo == ""
    assert result.stats["http_calls"] == len(PR_REPOS_EXPECTED)


class _BudgetRecordingClient(_PathIndexClient):
    """Records the budget each ``?file_path=`` batch was given."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.budgets: list[float | None] = []

    def get_many(self, requests, *, budget_sec=None):
        if any((params or {}).get("limit") == 1 for _, params in requests):
            self.budgets.append(budget_sec)
        return super().get_many(requests, budget_sec=budget_sec)


def test_the_whole_lookup_fits_inside_one_end_to_end_budget(monkeypatch, tmp_path):
    """Preflight and repository listing spend the caller's seconds too.

    The finding is which stages a budget admits, and the observable for that
    is the requests that were issued -- not how long the call took. Reading it
    off the wall clock made this a race the test lost on a loaded runner: real
    sleeps overshoot, so a pass depended on the scheduler rather than on the
    deadline being honoured. The clock is driven instead, one stage at a time.
    """

    class _Body:
        def __init__(self, payload):
            self._raw = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._raw

    now = 1_000.0
    requested: list[str] = []

    def _now() -> float:
        return now

    def slow(url, timeout=None):
        """Every stage spends half the budget, whatever the socket says."""
        nonlocal now
        requested.append(url.rsplit("/", 1)[-1].split("?")[0])
        now += 0.5
        if url.endswith("/healthz"):
            return _Body({})
        if url.endswith("/repos"):
            return _Body([{"repo_name": "ROCm/aiter", "is_active": True}])
        return _Body([])

    # Both modules read the deadline off their own ``time`` import: refs sets
    # it, search subtracts from it, and a clock patched in one of them only
    # would leave the other reading the real one.
    for module in ("pr_monitor_refs", "pr_monitor_search"):
        monkeypatch.setattr(f"kernelforge.knowledge.{module}.time.monotonic", _now)
    monkeypatch.setattr("kernelforge.knowledge.pr_monitor_client.urllib.request.urlopen", slow)

    collect_references(
        workspace_dir=str(tmp_path),
        client=PRMonitorClient("https://host/pr-monitor"),
        kernel_backend="aiter",
        operator_name="moe",
        budget_sec=1.0,
    )

    # Preflight and the repository listing spend the whole budget between them,
    # so nothing is left to probe a path with. A lookup that charged the caller
    # only for the path probes would have issued a third request here.
    assert requested == ["healthz", "repos"]


def test_a_budget_spent_on_preflight_is_not_reported_as_an_outage(tmp_path):
    """Running out of time and the service being down are different findings."""

    class _SlowPreflight(_StubClient):
        def healthz(self, *, timeout_sec=None):
            time.sleep(0.1)
            return True

    client = _SlowPreflight(by_query={"moe": [1]}, prs={1: _pr_payload(1)})

    result = collect_references(
        workspace_dir=str(tmp_path),
        client=client,
        kernel_backend="aiter",
        operator_name="moe",
        budget_sec=0.05,
    )

    assert result.reason == REASON_SKIPPED_DEADLINE
    assert result.stats["degraded_reason"] == REASON_SKIPPED_DEADLINE


def test_the_deadline_starts_before_local_snapshot_loading(monkeypatch, tmp_path):
    """A slow local read must not leave a fresh budget for the first HTTP call."""

    class _NoHttpClient(_StubClient):
        def __init__(self):
            super().__init__()
            self.health_calls = 0

        def healthz(self, *, timeout_sec=None):
            self.health_calls += 1
            return True

    def slow_snapshot(_workspace_dir):
        """Consume the deadline before returning an empty local cache."""
        time.sleep(0.08)
        return Snapshot()

    monkeypatch.setattr(
        "kernelforge.knowledge.pr_monitor_refs.load_snapshot",
        slow_snapshot,
    )
    client = _NoHttpClient()

    result = collect_references(
        workspace_dir=str(tmp_path),
        client=client,
        kernel_backend="aiter",
        operator_name="moe",
        budget_sec=0.03,
    )

    assert result.reason == REASON_SKIPPED_DEADLINE
    assert client.health_calls == 0


def test_a_budget_spent_before_probing_starts_no_probe(tmp_path):
    class _SlowListing(_BudgetRecordingClient):
        def list_repos(self, *, timeout_sec=None):
            time.sleep(0.1)
            return super().list_repos(timeout_sec=timeout_sec)

    client = _SlowListing({"csrc/k.cu": "ROCm/aiter"}, by_query={"mha": [1]}, prs={1: _pr_payload(1)})

    result = collect_references(client=client, budget_sec=0.05, **_fork_workspace(tmp_path))

    assert result.reason == REASON_SKIPPED_DEADLINE
    assert client.budgets == [], "probing must not start past the deadline"


def test_probing_draws_on_the_caller_budget(tmp_path):
    """Charge path probing to what is left of the caller's deadline."""
    client = _BudgetRecordingClient({"csrc/k.cu": "ROCm/aiter"}, by_query={"mha": [1]}, prs={1: _pr_payload(1)})

    collect_references(client=client, budget_sec=7.0, **_fork_workspace(tmp_path))

    assert len(client.budgets) == 1
    assert 0 < client.budgets[0] <= 7.0


def test_an_unset_budget_does_not_starve_probing(tmp_path):
    """Fall back to the configured default rather than to no deadline at all."""
    client = _BudgetRecordingClient({"csrc/k.cu": "ROCm/aiter"}, by_query={"mha": [1]}, prs={1: _pr_payload(1)})

    result = collect_references(client=client, **_fork_workspace(tmp_path))

    assert len(client.budgets) == 1
    assert 0 < client.budgets[0] <= 30.0
    assert result.repo == "ROCm/aiter"


def test_repo_drift_warns_but_does_not_block(tmp_path, caplog):
    """The tracked set is server-side config; drift is a warning, not a stop."""

    class _Drifted(_StubClient):
        def list_repos(self, *, timeout_sec=None):
            return [{"repo_name": "ROCm/aiter", "is_active": True}]

    client = _Drifted(by_query={"moe": [1]}, prs={1: _pr_payload(1)})

    with caplog.at_level("WARNING"):
        result = collect_references(
            workspace_dir=str(tmp_path),
            client=client,
            kernel_backend="aiter",
            operator_name="moe",
        )

    assert result.injected
    assert any("drift" in record.message for record in caplog.records)


def test_only_existing_source_files_become_path_queries(tmp_path):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "real.py").write_text("x")
    client = _StubClient(by_file={"kernels/real.py": [1]}, prs={1: _pr_payload(1)})

    collect_references(
        workspace_dir=str(tmp_path),
        client=client,
        kernel_backend="flydsl",
        source_files=["kernels/real.py", "kernels/ghost.py"],
    )

    assert client.path_queries == ["kernels/real.py"]


def test_provenance_is_a_sidecar_not_a_manifest_field(tmp_path):
    """Adding a key to the best manifest makes a resumed campaign raise."""
    path = write_provenance(str(tmp_path), {"winning_iteration": 7, "prs": [959]})

    assert path.name == "provenance.json"
    assert json.loads(path.read_text())["prs"] == [959]
