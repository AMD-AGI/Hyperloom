"""Unit tests for the local-recipe -> gbrain bulk ingest."""
from __future__ import annotations

from typing import Any

from inference_optimizer.recipe_kb.gbrain_ingest import (
    _best_config_split,
    _scalar,
    ingest_local_to_gbrain,
    recipe_to_page,
)
from inference_optimizer.recipe_kb.gbrain_remote_client import _page_to_recipe


def _recipe(**over: Any) -> dict[str, Any]:
    base = {
        "canonical_id": "inference:qwen3-32b:mi300x:sglang:0_5_11:fp8",
        "model": "Qwen3-32B", "hardware": "mi300x", "framework": "sglang",
        "framework_version": "0_5_11", "precision": "fp8",
        "best_config": {"extra_server_args": "--cuda-graph-max-bs 256", "FOO": "1"},
        "best_throughput": 5800.5, "validated_gain_pct": 7.8,
        "authority": "EXPERIENTIAL", "confidence": 0.85,
    }
    base.update(over)
    return base


def test_scalar_quotes_risky_keeps_barewords() -> None:
    # digit-leading underscore token must be quoted (else YAML octal 329)
    assert _scalar("0_5_11") == '"0_5_11"'
    # tokens with ':' / spaces / yaml-keywords get quoted
    assert _scalar("kind:recipe") == '"kind:recipe"'
    assert _scalar("--x 1") == '"--x 1"'
    assert _scalar("no") == '"no"'
    # identifier-ish barewords stay unquoted
    assert _scalar("recipe") == "recipe"
    assert _scalar("mi300x") == "mi300x"
    assert _scalar("unknown_version") == "unknown_version"
    # real numbers emit bare
    assert _scalar(5800.5) == "5800.5"
    assert _scalar(True) == "true"
    assert _scalar(None) == "null"


def test_recipe_to_page_roundtrips_via_reader() -> None:
    slug, content = recipe_to_page(_recipe())
    assert slug == "recipe-snapshot/inference/qwen3-32b/mi300x/sglang/0_5_11/fp8"
    assert content.startswith("---\ntype: recipe\n")
    # the version token must be quoted so it survives YAML parse
    assert 'framework_version: "0_5_11"' in content
    # Simulate the gbrain get_page frontmatter the reader would see.
    fm = {
        "attrs": {
            "model": "Qwen3-32B", "hardware": "mi300x", "framework": "sglang",
            "framework_version": "0_5_11", "precision": "fp8",
            "best_config_args": "--cuda-graph-max-bs 256",
            "best_config_envs": {"FOO": "1"},
            "best_throughput": 5800.5, "validated_gain_pct": 7.8,
        },
        "authority": "EXPERIENTIAL", "confidence": 0.85,
    }
    r = _page_to_recipe(fm)
    assert r["canonical_id"] == "inference:qwen3-32b:mi300x:sglang:0_5_11:fp8"
    assert r["best_config"] == {
        "extra_server_args": "--cuda-graph-max-bs 256",
        "extra_envs": {"FOO": "1"},
    }
    assert r["best_throughput"] == 5800.5


def test_best_config_split_unwraps_nested_extra_envs() -> None:
    # Authoritative local shape (coordinator._build_recipe_payload): args
    # key + NESTED extra_envs dict. The nested dict must be unwrapped, NOT
    # str()'d into a single bogus "extra_envs" env key.
    args, envs = _best_config_split({
        "extra_server_args": "--cuda-graph-max-bs 256",
        "extra_envs": {"SGLANG_X": "1", "FOO": "bar"},
    })
    assert args == "--cuda-graph-max-bs 256"
    assert envs == {"SGLANG_X": "1", "FOO": "bar"}
    assert "extra_envs" not in envs


def test_best_config_split_handles_flat_envs() -> None:
    # Flat shape: each env is a sibling scalar key.
    args, envs = _best_config_split({
        "extra_server_args": "--x 1", "FOO": "1", "BAR": "2",
    })
    assert args == "--x 1"
    assert envs == {"FOO": "1", "BAR": "2"}


def test_best_config_split_skips_passthrough_metadata() -> None:
    # current_best copies name/tput/accuracy into best_config; these are
    # NOT envs and must not be serialized as such.
    args, envs = _best_config_split({
        "extra_server_args": "--x 1",
        "name": "v1", "tput": 5400.0, "accuracy": 0.9,
        "extra_envs": {"E": "1"},
    })
    assert args == "--x 1"
    assert envs == {"E": "1"}


def test_nested_extra_envs_roundtrips_via_reader() -> None:
    # End-to-end: nested-env local recipe -> page -> read must surface the
    # canonical nested ``extra_envs`` (consumable by warm-replay).
    import yaml

    rec = _recipe(best_config={
        "extra_server_args": "--cuda-graph-max-bs 256",
        "extra_envs": {"SGLANG_X": "1"},
    })
    _slug, content = recipe_to_page(rec)
    fm = yaml.safe_load(content.split("---", 2)[1])
    r = _page_to_recipe(fm)
    assert r["best_config"] == {
        "extra_server_args": "--cuda-graph-max-bs 256",
        "extra_envs": {"SGLANG_X": "1"},
    }


def test_negative_knowledge_roundtrips() -> None:
    # Tier 0: pitfalls / lessons / what_failed must survive the full
    # emit -> YAML parse -> read path so a gbrain warm-start keeps its
    # anti-priors. Parse the emitted frontmatter exactly as gbrain would.
    import yaml

    pitfalls = [{"flag": "--num-continuous-decode-steps 16", "reason": "noise -0.3%"}]
    lessons = [{"note": "fp8 baseline already near-optimal"}]
    what_failed = [{"variant": "ncds16", "gain_pct": -0.28}]
    what_worked = [{"flag": "--cuda-graph-max-bs 256", "gain_pct": 0.4}]
    _slug, content = recipe_to_page(_recipe(
        pitfalls=pitfalls, lessons=lessons,
        what_failed=what_failed, what_worked=what_worked,
    ))
    fm = yaml.safe_load(content.split("---", 2)[1])
    r = _page_to_recipe(fm)
    assert r["pitfalls"] == pitfalls
    assert r["lessons"] == lessons
    assert r["what_failed"] == what_failed
    assert r["what_worked"] == what_worked


def test_negative_knowledge_absent_yields_empty() -> None:
    # No regression: a recipe without these fields still reads back empty.
    import yaml

    fm = yaml.safe_load(recipe_to_page(_recipe())[1].split("---", 2)[1])
    r = _page_to_recipe(fm)
    assert r["pitfalls"] == [] and r["lessons"] == []
    assert r["what_failed"] == [] and r["what_worked"] == []


def test_recipe_to_page_skips_bare_anchor() -> None:
    # anchor with no best_config is not worth caching remotely
    assert recipe_to_page(_recipe(best_config={})) is None
    assert recipe_to_page(_recipe(best_config={}, canonical_id="")) is None


class _FakeMcp:
    def __init__(self) -> None:
        self.puts: list[str] = []

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "put_page":
            self.puts.append(args["slug"])
        return {}


def test_ingest_counts_and_gates() -> None:
    recipes = [
        _recipe(),  # has config -> ingest
        _recipe(canonical_id="inference:a:b:c:d:e", best_config={}),  # anchor -> skip
        _recipe(canonical_id="inference:m2:mi355x:vllm:v1:fp16",
                model="m2", hardware="mi355x", framework="vllm"),  # ingest
    ]
    mcp = _FakeMcp()
    stats = ingest_local_to_gbrain(recipes=recipes, mcp=mcp, dry_run=False)
    assert stats["total"] == 3
    assert stats["ingested"] == 2
    assert stats["skipped_no_config"] == 1
    assert stats["errors"] == 0
    assert len(mcp.puts) == 2


def test_ingest_dry_run_writes_nothing() -> None:
    mcp = _FakeMcp()
    stats = ingest_local_to_gbrain(recipes=[_recipe()], mcp=mcp, dry_run=True)
    assert stats["ingested"] == 1 and mcp.puts == []


def test_mirror_recipe_gates_and_writes() -> None:
    from inference_optimizer.recipe_kb.gbrain_ingest import mirror_recipe
    mcp = _FakeMcp()
    assert mirror_recipe(_recipe(), mcp) is True and len(mcp.puts) == 1
    # no concrete config -> skipped
    assert mirror_recipe(_recipe(best_config={}), mcp) is False
    # no mcp -> skipped
    assert mirror_recipe(_recipe(), None) is False
    assert len(mcp.puts) == 1


class _RecordingInner:
    def __init__(self) -> None:
        self.put_calls: list[dict] = []

    def put_recipe(self, **kw):
        self.put_calls.append(kw)
        return {"canonical_id": kw.get("canonical_id"), "version": 1}

    def get_recipe(self, **kw):
        return {"delegated": True}


def test_mirroring_wrapper_local_first_then_mirror() -> None:
    from inference_optimizer.recipe_kb.gbrain_ingest import GbrainMirroringRecipeKB
    inner = _RecordingInner()
    mcp = _FakeMcp()
    kb = GbrainMirroringRecipeKB(inner, mcp)
    r = kb.put_recipe(**_recipe())
    # local write happened + returned, AND gbrain got the mirror
    assert inner.put_calls and r["version"] == 1
    assert len(mcp.puts) == 1
    # reads delegate to inner
    assert kb.get_recipe(canonical_id="x") == {"delegated": True}


def test_mirroring_wrapper_mirror_failure_is_swallowed() -> None:
    from inference_optimizer.recipe_kb.gbrain_ingest import GbrainMirroringRecipeKB

    class _BoomMcp:
        def call(self, *a, **k):
            raise RuntimeError("gbrain down")

    inner = _RecordingInner()
    kb = GbrainMirroringRecipeKB(inner, _BoomMcp())
    # local write still succeeds even though the mirror raises
    r = kb.put_recipe(**_recipe())
    assert inner.put_calls and r["version"] == 1
