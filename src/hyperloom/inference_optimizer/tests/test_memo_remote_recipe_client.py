"""Unit tests for the MeMo MEM recipe remote and the chained remote wrapper.

Fixtures are verbatim completions captured from a deployed Qwen3-32B MEM, so the
parsers are exercised against the shapes the service actually emits (including
the empty ``<think>`` block SGLang prepends even with thinking disabled).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from hyperloom.orchestrator.knowledge.recipe_kb.chained_remote import (
    ChainedRemoteRecipeClient,
)
from hyperloom.orchestrator.knowledge.recipe_kb.memo_remote_client import (
    MemoRemoteRecipeClient,
    build_memo_remote_from_env,
    hyperloom_model_slug,
    load_coverage_manifest,
    memo_model_key,
    parse_best_config,
    parse_lessons,
    parse_throughput,
    strip_think,
)

CID = "inference:qwen3-32b:mi300x:sglang:qwen3:qwen3forcausallm:0.5.11:fp8"


# ---------------------------------------------------------------- strip_think
def test_strip_think_removes_empty_block():
    raw = "<think>\n\n</think>\n\nThe best measured serving throughput is 5864.7 tokens/second."
    assert strip_think(raw) == "The best measured serving throughput is 5864.7 tokens/second."


def test_strip_think_tolerates_absent_block():
    assert strip_think("plain answer") == "plain answer"
    assert strip_think("") == ""
    assert strip_think(None) == ""


# ---------------------------------------------------------- parse_throughput
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("The best measured serving throughput is 1555.6 tokens/second.", 1555.6),
        ("The best measured serving throughput is 10654.0 tokens/second.", 10654.0),
        ("... on mi300x, the strongest recipe reached 7523.1 tok/s.", 7523.1),
        ("Measured peak throughput: 1,234.5 tokens/second.", 1234.5),
        ("There is no recorded measurement for that stack.", 0.0),
        ("", 0.0),
    ],
)
def test_parse_throughput(answer, expected):
    assert parse_throughput(answer) == pytest.approx(expected)


def test_parse_throughput_rejects_ambiguous_multi_value():
    answer = "Baseline was 100.0 tokens/second and the best reached 200.0 tokens/second."
    assert parse_throughput(answer) == 0.0


# --------------------------------------------------------- parse_best_config
def test_parse_best_config_args_only():
    answer = (
        "The measured recipe for soarailabs-breeze-3b on mi300x with sglang 0.5.11 "
        "at fp8 precision recommends `--quantization fp8` (2 profiling session(s))."
    )
    assert parse_best_config(answer) == {
        "extra_server_args": "--quantization fp8",
        "extra_envs": {},
    }


def test_parse_best_config_with_envs():
    answer = (
        "The measured recipe for qwen3-32b on mi300x with sglang 0.5.11 at fp8 precision "
        "recommends `--quantization fp8 | env: SGLANG_USE_AITER=1 "
        "SGLANG_USE_AITER_FP8_PER_TOKEN=1` (0 profiling session(s))."
    )
    assert parse_best_config(answer) == {
        "extra_server_args": "--quantization fp8",
        "extra_envs": {
            "SGLANG_USE_AITER": "1",
            "SGLANG_USE_AITER_FP8_PER_TOKEN": "1",
        },
    }


def test_parse_best_config_alternate_phrasing():
    answer = (
        "Use `--enable-torch-compile --torch-compile-max-bs 64` for wan-wan-test10-dpo "
        "on mi300x with sglang 0.5.11 at fp8 precision."
    )
    assert parse_best_config(answer)["extra_server_args"] == (
        "--enable-torch-compile --torch-compile-max-bs 64"
    )


def test_parse_best_config_without_backticks_is_empty():
    assert parse_best_config("I have no record for that stack.") == {}
    assert parse_best_config("") == {}


# ------------------------------------------------------------- client wiring
class _FakeCompletions:
    """Minimal stand-in for ``client.chat.completions``."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers
        self.calls: list[str] = []

    def create(self, **kwargs: Any) -> Any:
        question = kwargs["messages"][0]["content"]
        self.calls.append(question)
        text = ""
        for needle, answer in self._answers.items():
            if needle in question:
                text = answer
                break

        class _Msg:
            content = text

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def _client_with(
    answers: dict[str, str], *, allow_config: bool = True
) -> tuple[MemoRemoteRecipeClient, _FakeCompletions]:
    """Build a client whose HTTP layer is replaced by a scripted fake.

    ``allow_config`` defaults to True here so the config parser stays covered;
    the production default is off.
    """
    client = MemoRemoteRecipeClient(
        base_url="http://memo.invalid/v1", model="memo-v5", allow_config=allow_config
    )
    completions = _FakeCompletions(answers)

    class _Chat:
        pass

    chat = _Chat()
    chat.completions = completions

    class _Stub:
        pass

    stub = _Stub()
    stub.chat = chat
    client._client = stub  # noqa: SLF001 - test seam
    return client, completions


def test_get_recipe_builds_flat_arbor_row():
    client, _ = _client_with(
        {
            "best measured serving throughput": (
                "<think>\n\n</think>\n\nThe best measured serving throughput "
                "is 5864.7 tokens/second."
            ),
            "best recorded configuration": (
                "<think>\n\n</think>\n\nThe measured recipe recommends "
                "`--quantization fp8 | env: SGLANG_USE_AITER=1` (3 profiling session(s))."
            ),
        }
    )
    row = client.get_recipe(canonical_id=CID)

    assert row is not None
    assert row["canonical_id"] == CID
    assert row["model"] == "qwen3-32b"
    assert row["hardware"] == "mi300x"
    assert row["framework_name"] == "sglang"
    assert row["framework_version"] == "0.5.11"
    assert row["precision"] == "fp8"
    assert row["best_throughput"] == pytest.approx(5864.7)
    assert row["best_config"]["extra_server_args"] == "--quantization fp8"
    assert row["best_config"]["extra_envs"] == {"SGLANG_USE_AITER": "1"}
    assert row["authority"] == "INFERRED"
    assert row["provenance"]["source"] == "memo"
    assert row["provenance"]["measured"] is False


def test_get_recipe_row_carries_no_v2_envelope_markers():
    """The dispatcher's ``_v2_to_arbor`` must pass the row through untouched."""
    client, _ = _client_with(
        {"best measured serving throughput": "The best measured serving throughput is 1.0 tokens/second."}
    )
    row = client.get_recipe(canonical_id=CID)
    assert row is not None
    for marker in ("body", "labels", "findings", "failures", "gaps", "metrics"):
        assert marker not in row


def test_get_recipe_returns_none_when_nothing_usable():
    client, _ = _client_with({})
    assert client.get_recipe(canonical_id=CID) is None


def test_get_recipe_returns_none_for_malformed_cid_without_raising():
    client, completions = _client_with({})
    assert client.get_recipe(canonical_id="not-a-cid") is None
    assert completions.calls == []  # no request issued


def test_get_recipe_ignores_versioned_reads():
    client, completions = _client_with({})
    assert client.get_recipe(canonical_id=CID, version=2) is None
    assert completions.calls == []


# ------------------------------------------------------------ model spelling
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # HF ids keep the org, matching the keys the MEM was trained on.
        ("Qwen/Qwen3-32B", "qwen-qwen3-32b"),
        ("deepseek-ai/DeepSeek-R1-0528", "deepseek-ai-deepseek-r1-0528"),
        ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama-llama-3.1-8b-instruct"),
        ("mergebench/Llama-3.2-3B_coding", "mergebench-llama-3.2-3b_coding"),
        # Bare names pass through.
        ("gpt-oss-120b", "gpt-oss-120b"),
        # A filesystem path carries no org, so the basename is all there is.
        ("/wekafs/models/Qwen3-32B", "qwen3-32b"),
        ("/a/b/c/Llama-3.1-8B-Instruct/", "llama-3.1-8b-instruct"),
        ("", ""),
    ],
)
def test_memo_model_key(raw, expected):
    assert memo_model_key(raw) == expected


def test_memo_key_differs_from_canonical_slug_for_hf_ids():
    """The whole point of the hint: the two spellings disagree."""
    assert hyperloom_model_slug("Qwen/Qwen3-32B") == "qwen3-32b"
    assert memo_model_key("Qwen/Qwen3-32B") == "qwen-qwen3-32b"


def test_model_hint_rewrites_the_question():
    client, completions = _client_with(
        {"throughput": "The best measured serving throughput is 42.0 tokens/second."}
    )
    client.set_model_hint("Qwen/Qwen3-32B")
    client.get_recipe(canonical_id=CID)

    assert completions.calls, "no question was asked"
    assert "qwen-qwen3-32b" in completions.calls[0]


def test_without_hint_question_uses_canonical_slug():
    client, completions = _client_with(
        {"throughput": "The best measured serving throughput is 42.0 tokens/second."}
    )
    client.get_recipe(canonical_id=CID)

    assert "qwen3-32b" in completions.calls[0]
    assert "qwen-qwen3-32b" not in completions.calls[0]


def test_hint_ignored_for_a_different_identity():
    """A run-scoped hint must not leak into cascade / donor reads."""
    client, completions = _client_with(
        {"throughput": "The best measured serving throughput is 42.0 tokens/second."}
    )
    client.set_model_hint("Qwen/Qwen3-32B")
    other = "inference:llama-3.1-8b:mi300x:vllm:llama:llamaforcausallm:0.11.0:fp8"
    client.get_recipe(canonical_id=other)

    assert "llama-3.1-8b" in completions.calls[0]
    assert "qwen" not in completions.calls[0]


def test_returned_row_keeps_canonical_spelling_not_mem_spelling():
    client, _ = _client_with(
        {"throughput": "The best measured serving throughput is 42.0 tokens/second."}
    )
    client.set_model_hint("Qwen/Qwen3-32B")
    row = client.get_recipe(canonical_id=CID)

    assert row["model"] == "qwen3-32b"  # Hyperloom's keying is untouched
    assert row["canonical_id"] == CID


def test_unusable_hint_falls_back_to_canonical_slug():
    client, completions = _client_with(
        {"throughput": "The best measured serving throughput is 42.0 tokens/second."}
    )
    client.set_model_hint("   ")
    client.get_recipe(canonical_id=CID)
    assert "qwen3-32b" in completions.calls[0]


# --------------------------------------------------------------- local guard
class _FakeLocalStore:
    """Guard store returning a caller-supplied row for known identities."""

    def __init__(self, known: set[str], row: dict | None = None, raises: bool = False) -> None:
        self._known = known
        self._row = row
        self._raises = raises

    def get_recipe(self, *, canonical_id, version=None):
        if self._raises:
            raise RuntimeError("guard exploded")
        if canonical_id not in self._known:
            return None
        base = {"canonical_id": canonical_id}
        base.update(self._row or {"best_throughput": 999.0})
        return base


def test_local_guard_makes_memo_abstain_on_actionable_local_row():
    """RecipeKB reads remote-first, so a measured local row must win."""
    client, completions = _client_with(
        {"best measured serving throughput": "The best measured serving throughput is 1.0 tokens/second."}
    )
    client.set_local_guard(
        _FakeLocalStore(known={CID}, row={"best_throughput": 5000.0, "authority": "MEASURED"})
    )

    assert client.get_recipe(canonical_id=CID) is None
    assert completions.calls == []  # no tokens burned either


def test_bare_t0_anchor_does_not_shadow_the_mem():
    """T0 seeds a bare local row before its cascade reads; it must not block."""
    client, _ = _client_with(
        {"best measured serving throughput": "The best measured serving throughput is 1.0 tokens/second."}
    )
    # Identity + tracing tags only: what T0's put_recipe stamps at anchor time.
    client.set_local_guard(
        _FakeLocalStore(
            known={CID},
            row={
                "best_config": {},
                "best_throughput": 0.0,
                "authority": "EXPERIENTIAL",
                "last_profiled": "",
            },
        )
    )

    row = client.get_recipe(canonical_id=CID)
    assert row is not None, "a bare T0 anchor must not shadow the MEM"
    assert row["best_throughput"] == pytest.approx(1.0)


def test_local_guard_allows_memo_on_local_miss():
    client, _ = _client_with(
        {"best measured serving throughput": "The best measured serving throughput is 1.0 tokens/second."}
    )
    client.set_local_guard(_FakeLocalStore(known=set()))

    row = client.get_recipe(canonical_id=CID)
    assert row is not None
    assert row["best_throughput"] == pytest.approx(1.0)


def test_local_guard_abstains_for_config_only_local_row():
    client, completions = _client_with(
        {"best measured serving throughput": "The best measured serving throughput is 1.0 tokens/second."}
    )
    client.set_local_guard(
        _FakeLocalStore(known={CID}, row={"best_config": {"extra_server_args": "--quantization fp8"}})
    )

    assert client.get_recipe(canonical_id=CID) is None
    assert completions.calls == []


def test_broken_local_guard_does_not_block_reads():
    client, _ = _client_with(
        {"best measured serving throughput": "The best measured serving throughput is 1.0 tokens/second."}
    )
    client.set_local_guard(_FakeLocalStore(known={CID}, raises=True))

    assert client.get_recipe(canonical_id=CID) is not None


def test_search_always_empty():
    client, completions = _client_with({})
    assert client.search(label_match={"model": "x"}, limit=5, prefer=None) == []
    assert completions.calls == []


# ------------------------------------------------------- best_config gating
def test_config_withheld_by_default():
    """A fabricated launch recipe would clear the warm-replay gate, so withhold it."""
    client, completions = _client_with(
        {
            "best measured serving throughput": (
                "The best measured serving throughput is 5864.7 tokens/second."
            ),
            "best recorded configuration": "recommends `--quantization fp8`.",
        },
        allow_config=False,
    )
    row = client.get_recipe(canonical_id=CID)

    assert row is not None
    assert row["best_throughput"] == pytest.approx(5864.7)
    assert row["best_config"] == {}
    # The config question is not even asked, so no tokens are spent on it.
    assert not any("best recorded configuration" in call for call in completions.calls)


def test_config_emitted_when_explicitly_allowed():
    client, completions = _client_with(
        {
            "best measured serving throughput": (
                "The best measured serving throughput is 5864.7 tokens/second."
            ),
            "best recorded configuration": "recommends `--quantization fp8`.",
        },
        allow_config=True,
    )
    row = client.get_recipe(canonical_id=CID)

    assert row["best_config"]["extra_server_args"] == "--quantization fp8"
    assert any("best recorded configuration" in call for call in completions.calls)


def test_no_throughput_and_no_config_is_a_miss():
    client, _ = _client_with({}, allow_config=False)
    assert client.get_recipe(canonical_id=CID) is None


def test_build_from_env_withholds_config_by_default(monkeypatch):
    monkeypatch.setenv("MEMO_KB_URL", "http://memo.invalid/v1")
    monkeypatch.delenv("MEMO_KB_ALLOW_CONFIG", raising=False)
    client = build_memo_remote_from_env()
    assert client._allow_config is False  # noqa: SLF001


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_build_from_env_allows_config_when_opted_in(monkeypatch, value):
    monkeypatch.setenv("MEMO_KB_URL", "http://memo.invalid/v1")
    monkeypatch.setenv("MEMO_KB_ALLOW_CONFIG", value)
    client = build_memo_remote_from_env()
    assert client._allow_config is True  # noqa: SLF001


def test_enabled_reflects_base_url():
    assert MemoRemoteRecipeClient(base_url="http://x/v1").enabled is True
    assert MemoRemoteRecipeClient(base_url="").enabled is False


def test_build_from_env_absent(monkeypatch):
    monkeypatch.delenv("MEMO_KB_URL", raising=False)
    assert build_memo_remote_from_env() is None


def test_build_from_env_present(monkeypatch):
    monkeypatch.setenv("MEMO_KB_URL", "http://memo.invalid/v1")
    monkeypatch.setenv("MEMO_KB_MODEL", "memo-v5")
    monkeypatch.setenv("MEMO_KB_CONFIDENCE", "0.42")
    client = build_memo_remote_from_env()
    assert client is not None
    assert client.enabled is True
    assert client._confidence == pytest.approx(0.42)  # noqa: SLF001


# ------------------------------------------------------------ chained remote
class _StubRemote:
    """A remote returning a fixed row / rows, or raising."""

    def __init__(self, *, row=None, rows=None, raises=False, enabled=True) -> None:
        self._row = row
        self._rows = rows or []
        self._raises = raises
        self.enabled = enabled
        self.get_calls = 0
        self.search_calls = 0
        self.closed = False

    def get_recipe(self, *, canonical_id, version=None):
        self.get_calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return self._row

    def search(self, **kwargs):
        self.search_calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return self._rows

    def close(self):
        self.closed = True


def test_chain_first_hit_short_circuits():
    first = _StubRemote(row={"canonical_id": CID, "model": "first"})
    second = _StubRemote(row={"canonical_id": CID, "model": "second"})
    chain = ChainedRemoteRecipeClient([first, second])

    assert chain.get_recipe(canonical_id=CID)["model"] == "first"
    assert second.get_calls == 0


def test_chain_falls_through_on_miss():
    first = _StubRemote(row=None)
    second = _StubRemote(row={"canonical_id": CID, "model": "second"})
    chain = ChainedRemoteRecipeClient([first, second])

    assert chain.get_recipe(canonical_id=CID)["model"] == "second"
    assert first.get_calls == 1


def test_chain_treats_raise_as_miss():
    first = _StubRemote(raises=True)
    second = _StubRemote(row={"canonical_id": CID, "model": "second"})
    chain = ChainedRemoteRecipeClient([first, second])

    assert chain.get_recipe(canonical_id=CID)["model"] == "second"


def test_chain_all_miss():
    chain = ChainedRemoteRecipeClient([_StubRemote(row=None), _StubRemote(row=None)])
    assert chain.get_recipe(canonical_id=CID) is None
    assert chain.search(label_match={}) == []


def test_chain_drops_disabled_and_none_members():
    live = _StubRemote(row={"canonical_id": CID})
    chain = ChainedRemoteRecipeClient([None, _StubRemote(enabled=False), live])
    assert chain.members == [live]
    assert chain.enabled is True


def test_chain_empty_is_disabled():
    assert ChainedRemoteRecipeClient([]).enabled is False


def test_chain_search_falls_through_and_closes_all():
    first = _StubRemote(rows=[])
    second = _StubRemote(rows=[{"canonical_id": CID}])
    chain = ChainedRemoteRecipeClient([first, second])

    assert chain.search(label_match={}) == [{"canonical_id": CID}]
    chain.close()
    assert first.closed and second.closed


# ------------------------------------------------------------ coverage manifest
_ANSWER = {
    "best measured serving throughput": (
        "The best measured serving throughput is 5864.7 tokens/second."
    )
}
_CID_KEY = "qwen3-32b|mi300x|sglang|0.5.11|fp8"


def test_identity_inside_coverage_is_asked():
    client, completions = _client_with(_ANSWER)
    client._coverage = frozenset({_CID_KEY})  # noqa: SLF001 - test seam

    assert client.get_recipe(canonical_id=CID) is not None
    assert completions.calls


def test_identity_outside_coverage_is_skipped_without_a_round_trip():
    """The point of the manifest: no request, so no fabricated answer to filter."""
    client, completions = _client_with(_ANSWER)
    client._coverage = frozenset({"someone-else|mi300x|sglang|0.5.11|fp8"})  # noqa: SLF001

    assert client.get_recipe(canonical_id=CID) is None
    assert completions.calls == []


def test_coverage_match_ignores_case():
    client, completions = _client_with(_ANSWER)
    client._coverage = frozenset({_CID_KEY.upper().casefold()})  # noqa: SLF001

    assert client.get_recipe(canonical_id=CID) is not None
    assert completions.calls


def test_empty_coverage_asks_every_identity():
    """An absent manifest must not turn the MEM off."""
    client, completions = _client_with(_ANSWER)
    assert client._coverage == frozenset()  # noqa: SLF001
    assert client.get_recipe(canonical_id=CID) is not None
    assert completions.calls


def test_coverage_gate_and_local_guard_stay_independent():
    """A covered identity still abstains when the local store already answers it."""
    client, completions = _client_with(_ANSWER)
    client._coverage = frozenset({_CID_KEY})  # noqa: SLF001
    client.set_local_guard(
        _FakeLocalStore(known={CID}, row={"best_throughput": 5000.0, "authority": "MEASURED"})
    )

    assert client.get_recipe(canonical_id=CID) is None
    assert completions.calls == []


def test_load_coverage_manifest_reads_identity_list(tmp_path):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"count": 2, "identities": [_CID_KEY, "A|B|C|D|E"]}))

    coverage = load_coverage_manifest(str(path))
    assert _CID_KEY in coverage
    assert "a|b|c|d|e" in coverage


@pytest.mark.parametrize("payload", ['{"identities": "not-a-list"}', "not json at all"])
def test_load_coverage_manifest_degrades_to_empty(tmp_path, payload):
    """A broken manifest disables the gate rather than blocking every read."""
    path = tmp_path / "coverage.json"
    path.write_text(payload)
    assert load_coverage_manifest(str(path)) == frozenset()


def test_load_coverage_manifest_tolerates_missing_file():
    assert load_coverage_manifest("/nonexistent/coverage.json") == frozenset()
    assert load_coverage_manifest("") == frozenset()


# ------------------------------------------------------------------- lessons
_POSITIVES = (
    "The highest-value tested changes for qwen3-32b on mi300x with sglang 0.5.11 "
    "at fp8 precision were `--kv-cache-dtype fp8_e4m3 --stream-interval 50` "
    "(+12.8%); `--cuda-graph-max-bs 64` (+2.8%)."
)
_THROUGHPUT_ANSWER = "The best measured serving throughput is 5864.7 tokens/second."


def test_parse_lessons_keeps_deltas_in_order():
    lessons = parse_lessons(_POSITIVES)
    assert [entry["statement"] for entry in lessons] == [
        "--kv-cache-dtype fp8_e4m3 --stream-interval 50",
        "--cuda-graph-max-bs 64",
    ]


def test_parse_lessons_leaves_measured_impact_empty():
    """A correct delta usually carries a wrong gain, so no number is propagated."""
    assert all(entry["measured_impact"] == "" for entry in parse_lessons(_POSITIVES))


def test_parse_lessons_deduplicates_and_tolerates_junk():
    assert parse_lessons("`--a` and again `--a`") == [
        {"statement": "--a", "measured_impact": ""}
    ]
    assert parse_lessons("no backticks here") == []
    assert parse_lessons("") == []


def test_lessons_attached_to_a_qualifying_row():
    client, completions = _client_with(
        {
            "best measured serving throughput": _THROUGHPUT_ANSWER,
            "positive evidence": _POSITIVES,
        },
        allow_config=False,
    )
    row = client.get_recipe(canonical_id=CID)

    assert row is not None
    assert [entry["statement"] for entry in row["lessons"]] == [
        "--kv-cache-dtype fp8_e4m3 --stream-interval 50",
        "--cuda-graph-max-bs 64",
    ]


def test_lessons_alone_never_makes_a_hit():
    """The row must still miss, so the chain reaches a backend with real rows."""
    client, completions = _client_with({"positive evidence": _POSITIVES})

    assert client.get_recipe(canonical_id=CID) is None
    assert not any("positive evidence" in call for call in completions.calls)


def test_lessons_can_be_disabled():
    client, completions = _client_with(
        {"best measured serving throughput": _THROUGHPUT_ANSWER,
         "positive evidence": _POSITIVES},
        allow_config=False,
    )
    client._allow_lessons = False  # noqa: SLF001 - test seam

    row = client.get_recipe(canonical_id=CID)
    assert row is not None and row["lessons"] == []
    assert not any("positive evidence" in call for call in completions.calls)


def test_lessons_empty_when_the_mem_says_nothing_usable():
    client, _ = _client_with(
        {"best measured serving throughput": _THROUGHPUT_ANSWER}, allow_config=False
    )
    row = client.get_recipe(canonical_id=CID)
    assert row is not None and row["lessons"] == []


def test_build_from_env_enables_lessons_by_default(monkeypatch):
    monkeypatch.setenv("MEMO_KB_URL", "http://memo.invalid/v1")
    monkeypatch.delenv("MEMO_KB_ALLOW_LESSONS", raising=False)
    client = build_memo_remote_from_env()
    assert client is not None
    assert client._allow_lessons is True  # noqa: SLF001


@pytest.mark.parametrize(
    ("value", "expected"), [("0", False), ("false", False), ("off", False), ("1", True)]
)
def test_build_from_env_lessons_opt_out(monkeypatch, value, expected):
    monkeypatch.setenv("MEMO_KB_URL", "http://memo.invalid/v1")
    monkeypatch.setenv("MEMO_KB_ALLOW_LESSONS", value)
    client = build_memo_remote_from_env()
    assert client is not None
    assert client._allow_lessons is expected  # noqa: SLF001


def test_build_from_env_wires_the_coverage_manifest(monkeypatch, tmp_path):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"identities": [_CID_KEY]}))
    monkeypatch.setenv("MEMO_KB_URL", "http://memo.invalid/v1")
    monkeypatch.setenv("MEMO_KB_COVERAGE", str(path))

    client = build_memo_remote_from_env()
    assert client is not None
    assert client._coverage == frozenset({_CID_KEY})  # noqa: SLF001
