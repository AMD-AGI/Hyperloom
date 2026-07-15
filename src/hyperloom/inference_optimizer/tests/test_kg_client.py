# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the knowledge-graph client (fence parse / query / BFS / RMW writes)."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_remote_client import GbrainRemoteError
from hyperloom.orchestrator.knowledge.recipe_kb.kg_client import (
    Fact,
    KGClient,
    format_fact_line,
    generate_knob_candidates_graph_guided,
    generate_variants_graph_guided,
    parse_facts_fence,
)


def _page(body: str, *, frontmatter: str = "---\ntype: recipe\n---\n\n") -> str:
    """Wrap a body in a full page markdown string."""
    return frontmatter + body


class _FakeMcp:
    """Fake MCP serving canned search/get_page and recording put_page."""

    def __init__(self, pages: dict[str, str], *, search_hits: list[str] | None = None) -> None:
        self.pages = pages
        self.search_hits = search_hits if search_hits is not None else list(pages)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        self.calls.append((tool, dict(args)))
        if tool == "list_pages":
            return [{"slug": s} for s in list(self.pages)[: args.get("limit", 100)]]
        if tool == "search":
            return [{"slug": s} for s in self.search_hits]
        if tool == "get_page":
            content = self.pages.get(args.get("slug"))
            return {"content": content} if content is not None else {}
        if tool == "put_page":
            self.pages[args["slug"]] = args["content"]
            return {"ok": True}
        return {}


class _BoomMcp:
    """Fake MCP that always raises a transport error."""

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        raise GbrainRemoteError("boom")


_RECIPE_BODY = """# Recipe

## Facts
- aiter_backend IMPROVES qwen2forcausallm (gain: +35.2%, confidence: 0.95, hw: mi300x, fw: sglang)
- fp8_kernel_patch REVERTED_ON qwen2forcausallm (loss: -4.2%, error: perf_regression, hw: mi300x)
- chunked_prefill CRASHES mi300x (error: cuda_graph_capture, fw: sglang)
- qwen2.5-7b USES_ARCH qwen2forcausallm ()
"""

_ARCH_BODY = """# Arch Family

## Facts
- qwen3-8b USES_ARCH qwen2forcausallm (distance: 1)
- qwen2forcausallm VARIANT_OF llamaforcausallm (distance: 1)
"""

_CONFLICT_BODY = """# Conflicts

## Facts
- aiter_backend CONFLICTS_WITH flash_attention_v2 (severity: hard)
- torch_compile IMPROVES qwen2forcausallm (gain: +8%, hw: mi300x)
"""


def test_parse_facts_fence_extracts_triples() -> None:
    facts = parse_facts_fence(_RECIPE_BODY, source_slug="recipe/x")
    assert len(facts) == 4
    f0 = facts[0]
    assert f0.subject == "aiter_backend"
    assert f0.predicate == "IMPROVES"
    assert f0.object == "qwen2forcausallm"
    assert f0.properties["gain"] == "+35.2%"
    assert f0.gain == pytest.approx(35.2)
    assert f0.confidence == pytest.approx(0.95)
    assert f0.source_slug == "recipe/x"


def test_parse_facts_fence_ignores_non_fact_sections() -> None:
    body = "## Facts\n- a IMPROVES b ()\n\n## Evidence\n- session abc: not a fact line\n"
    facts = parse_facts_fence(body)
    assert len(facts) == 1
    assert facts[0].subject == "a"


def test_parse_facts_fence_empty_when_no_fence() -> None:
    assert parse_facts_fence("# Title\n\nno facts here") == []


def test_format_fact_line_roundtrip() -> None:
    line = format_fact_line("AITER_backend", "improves", "Qwen2ForCausalLM", {"gain": "+5%"})
    assert line == "- aiter_backend IMPROVES qwen2forcausallm (gain: +5%)"
    parsed = parse_facts_fence("## Facts\n" + line + "\n")
    assert parsed[0].subject == "aiter_backend"
    assert parsed[0].predicate == "IMPROVES"


def _client(pages: dict[str, str], **kw: Any) -> tuple[KGClient, _FakeMcp]:
    mcp = _FakeMcp(pages, **kw)
    return KGClient(mcp), mcp


def test_query_facts_filters_by_predicate_alternation() -> None:
    kg, _ = _client({"recipe/x": _page(_RECIPE_BODY)})
    facts = kg.query_facts(predicate="REVERTED_ON|DEGRADES|CRASHES")
    preds = {f.predicate for f in facts}
    assert preds == {"REVERTED_ON", "CRASHES"}


def test_query_facts_filters_by_subject_and_object() -> None:
    kg, _ = _client({"recipe/x": _page(_RECIPE_BODY)})
    facts = kg.query_facts(subject="aiter_backend", object="qwen2forcausallm")
    assert len(facts) == 1
    assert facts[0].predicate == "IMPROVES"


def test_query_facts_object_list_matches_any() -> None:
    kg, _ = _client({"recipe/x": _page(_RECIPE_BODY)})
    facts = kg.query_facts(object=["qwen2forcausallm", "mixtralforcausallm"], predicate="IMPROVES")
    assert len(facts) == 1
    assert facts[0].subject == "aiter_backend"


def test_query_facts_conditions_filter() -> None:
    kg, _ = _client({"recipe/x": _page(_RECIPE_BODY)})
    hit = kg.query_facts(predicate="IMPROVES", conditions={"hw": "mi300x"})
    assert len(hit) == 1
    miss = kg.query_facts(predicate="IMPROVES", conditions={"hw": "mi308x"})
    assert miss == []


def test_query_facts_dedups_across_pages() -> None:
    pages = {"a": _page(_RECIPE_BODY), "b": _page(_RECIPE_BODY)}
    kg, _ = _client(pages)
    facts = kg.query_facts(predicate="IMPROVES")
    assert len(facts) == 1


def test_query_facts_respects_limit() -> None:
    kg, _ = _client({"recipe/x": _page(_RECIPE_BODY)})
    facts = kg.query_facts(limit=2)
    assert len(facts) == 2


def test_graph_traverse_outbound_one_hop() -> None:
    kg, _ = _client({"arch": _page(_ARCH_BODY)})
    nodes = kg.graph_traverse(start_entity="qwen3-8b", predicate_filter=["USES_ARCH", "VARIANT_OF"], max_hops=1)
    assert any(n.entity == "qwen2forcausallm" and n.depth == 1 for n in nodes)


def test_graph_traverse_two_hops_reaches_base_arch() -> None:
    kg, _ = _client({"arch": _page(_ARCH_BODY)})
    nodes = kg.graph_traverse(
        start_entity="qwen3-8b", predicate_filter=["USES_ARCH", "VARIANT_OF"], max_hops=2
    )
    entities = {n.entity for n in nodes}
    assert "qwen2forcausallm" in entities
    assert "llamaforcausallm" in entities


def test_find_conflicts_detects_pair_in_stack() -> None:
    kg, _ = _client({"c": _page(_CONFLICT_BODY)})
    conflicts = kg.find_conflicts(knobs=["aiter_backend", "flash_attention_v2", "torch_compile"])
    assert len(conflicts) == 1
    assert conflicts[0]["conflicts_with"] == "aiter_backend"
    assert conflicts[0]["knob"] == "flash_attention_v2"


def test_find_conflicts_skips_when_endpoint_not_in_stack() -> None:
    kg, _ = _client({"c": _page(_CONFLICT_BODY)})
    conflicts = kg.find_conflicts(knobs=["aiter_backend", "torch_compile"])
    assert conflicts == []


def test_find_conflicts_detected_even_with_hw_fw_conditions() -> None:
    # CONFLICTS_WITH facts carry no hw/fw props; passing hardware/framework must
    # not filter them out.
    kg, _ = _client({"c": _page(_CONFLICT_BODY)})
    conflicts = kg.find_conflicts(
        knobs=["aiter_backend", "flash_attention_v2"],
        hardware="mi300x",
        framework="sglang",
    )
    assert len(conflicts) == 1
    assert conflicts[0]["knob"] == "flash_attention_v2"


def test_emit_fact_appends_to_existing_fence() -> None:
    pages = {"recipe/x": _page(_RECIPE_BODY)}
    kg, mcp = _client(pages)
    wrote = kg.emit_fact(
        page_slug="recipe/x",
        subject="torch_compile",
        predicate="IMPROVES",
        object="qwen2forcausallm",
        properties={"gain": "+8%"},
    )
    assert wrote is True
    facts = parse_facts_fence(mcp.pages["recipe/x"])
    assert any(f.subject == "torch_compile" and f.predicate == "IMPROVES" for f in facts)


def test_emit_fact_idempotent_on_duplicate() -> None:
    pages = {"recipe/x": _page(_RECIPE_BODY)}
    kg, mcp = _client(pages)
    wrote = kg.emit_fact(
        page_slug="recipe/x", subject="aiter_backend", predicate="IMPROVES", object="qwen2forcausallm"
    )
    assert wrote is False
    assert not any(t == "put_page" for t, _ in mcp.calls)


def test_emit_fact_creates_fence_when_absent() -> None:
    pages = {"p": _page("# Title\n\nbody only\n")}
    kg, mcp = _client(pages)
    wrote = kg.emit_fact(page_slug="p", subject="a", predicate="IMPROVES", object="b")
    assert wrote is True
    assert "## Facts" in mcp.pages["p"]


def test_retract_fact_removes_matching_line() -> None:
    pages = {"recipe/x": _page(_RECIPE_BODY)}
    kg, mcp = _client(pages)
    removed = kg.retract_fact(
        page_slug="recipe/x", subject="fp8_kernel_patch", predicate="REVERTED_ON", object="qwen2forcausallm"
    )
    assert removed is True
    facts = parse_facts_fence(mcp.pages["recipe/x"])
    assert not any(f.subject == "fp8_kernel_patch" for f in facts)


def test_retract_fact_noop_when_absent() -> None:
    pages = {"recipe/x": _page(_RECIPE_BODY)}
    kg, mcp = _client(pages)
    removed = kg.retract_fact(page_slug="recipe/x", subject="nope", predicate="IMPROVES", object="x")
    assert removed is False


class _StructuredMcp:
    """Fake MCP whose get_page returns parsed {frontmatter: dict, body}."""

    def __init__(self) -> None:
        self.frontmatter = {
            "type": "recipe",
            "tags": ["kind:recipe", "model:qwen3"],
            "confidence": 0.85,
            "attrs": {"model": "qwen3", "best_config_args": "--tp 8"},
        }
        self.body = "# Recipe\n\n## Facts\n- a IMPROVES b ()\n"
        self.put_content: str | None = None

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "list_pages":
            return [{"slug": "recipe/x"}]
        if tool == "search":
            return [{"slug": "recipe/x"}]
        if tool == "get_page":
            return {"frontmatter": dict(self.frontmatter), "body": self.body}
        if tool == "put_page":
            self.put_content = args["content"]
            return {"ok": True}
        return {}


def test_emit_fact_preserves_frontmatter_on_structured_page() -> None:
    # When get_page returns {frontmatter: dict, body} with no raw content field,
    # emit_fact must not write a body-only page (which would drop type/tags/attrs).
    mcp = _StructuredMcp()
    kg = KGClient(mcp)
    wrote = kg.emit_fact(
        page_slug="recipe/x", subject="torch_compile", predicate="IMPROVES", object="qwen3"
    )
    assert wrote is True
    assert mcp.put_content is not None
    assert mcp.put_content.startswith("---")
    assert "type: recipe" in mcp.put_content
    assert "best_config_args" in mcp.put_content  # attrs preserved
    assert "torch_compile IMPROVES qwen3" in mcp.put_content


def test_search_cache_avoids_repeat_search() -> None:
    pages = {"recipe/x": _page(_RECIPE_BODY)}
    kg, mcp = _client(pages)
    kg.query_facts(predicate="IMPROVES")
    kg.query_facts(predicate="IMPROVES")  # identical -> served from cache
    assert sum(1 for t, _ in mcp.calls if t == "search") == 1


def test_emit_fact_invalidates_cache() -> None:
    pages = {"recipe/x": _page(_RECIPE_BODY)}
    kg, mcp = _client(pages)
    kg.query_facts(subject="torch_compile")  # warms cache (empty result)
    kg.emit_fact(page_slug="recipe/x", subject="torch_compile", predicate="IMPROVES", object="qwen2forcausallm")
    facts = kg.query_facts(subject="torch_compile")  # re-search post-write
    assert any(f.subject == "torch_compile" for f in facts)


def test_query_facts_safe_returns_empty_on_error() -> None:
    kg = KGClient(_BoomMcp())
    assert kg.query_facts_safe(predicate="IMPROVES") == []


def test_graph_traverse_safe_returns_empty_on_error() -> None:
    kg = KGClient(_BoomMcp())
    assert kg.graph_traverse_safe(start_entity="x") == []


def test_emit_fact_safe_returns_false_on_error() -> None:
    kg = KGClient(_BoomMcp())
    assert kg.emit_fact_safe(page_slug="x", subject="a", predicate="IMPROVES", object="b") is False


def test_is_available_true_and_false() -> None:
    kg_ok, _ = _client({"x": _page(_RECIPE_BODY)})
    assert kg_ok.is_available() is True
    assert KGClient(_BoomMcp()).is_available() is False


_VARIANT_BODY = """# KG

## Facts
- aiter_backend IMPROVES qwen2forcausallm (gain: +35%, confidence: 0.95, hw: mi300x, fw: sglang)
- torch_compile IMPROVES qwen2forcausallm (gain: +8%, hw: mi300x, fw: sglang)
- decode_patch IMPROVES qwen2forcausallm (gain: +20%, hw: mi300x, fw: sglang)
- decode_patch CONFLICTS_WITH flash_attn (severity: hard)
- needs_dep IMPROVES qwen2forcausallm (gain: +40%, hw: mi300x, fw: sglang)
- needs_dep REQUIRES chunked_prefill ()
"""


def test_graph_guided_orders_by_gain_and_excludes_in_stack() -> None:
    kg, _ = _client({"kg": _page(_VARIANT_BODY)})
    out = generate_variants_graph_guided(
        kg,
        architectures=["Qwen2ForCausalLM"],
        hardware="mi300x",
        framework="sglang",
        in_stack=["aiter_backend"],  # exclude top knob
    )
    knobs = [v["knob"] for v in out]
    assert "aiter_backend" not in knobs
    assert "needs_dep" not in knobs
    assert knobs[0] == "decode_patch"  # +20% before torch_compile +8%
    assert "torch_compile" in knobs


def test_graph_guided_rejects_conflict_with_stack() -> None:
    kg, _ = _client({"kg": _page(_VARIANT_BODY)})
    out = generate_variants_graph_guided(
        kg,
        architectures=["Qwen2ForCausalLM"],
        hardware="mi300x",
        framework="sglang",
        in_stack=["flash_attn"],  # decode_patch conflicts with this
    )
    knobs = [v["knob"] for v in out]
    assert "decode_patch" not in knobs


def test_graph_guided_dependency_met() -> None:
    kg, _ = _client({"kg": _page(_VARIANT_BODY)})
    out = generate_variants_graph_guided(
        kg,
        architectures=["Qwen2ForCausalLM"],
        hardware="mi300x",
        framework="sglang",
        in_stack=["chunked_prefill"],  # satisfies needs_dep REQUIRES
    )
    knobs = [v["knob"] for v in out]
    assert "needs_dep" in knobs
    assert knobs[0] == "needs_dep"  # +40% top


class _StubKnobKG:
    """Stub KG returning canned facts keyed by predicate (knob path)."""

    def __init__(self, facts_by_pred: dict[str, list[Fact]]) -> None:
        self._by = facts_by_pred
        self.queries: list[dict[str, Any]] = []

    def query_facts_safe(self, **kwargs: Any) -> list[Fact]:
        self.queries.append(kwargs)
        pred = kwargs.get("predicate")
        key = pred[0] if isinstance(pred, (list, tuple)) else pred
        return list(self._by.get(str(key), []))


def _knob_fact(subject: str, *, gain: str = "+10%", name: str = "k", args: str = "--x 1",
               envs: str = "", keep_n: str = "2") -> Fact:
    props = {"gain": gain, "name": name, "args": args, "keep_n": keep_n}
    if envs:
        props["envs"] = envs
    return Fact(subject=subject, predicate="KNOB_IMPROVES", object="llamaforcausallm+bf16", properties=props)


def test_knob_guided_orders_by_gain_and_surfaces_args_envs() -> None:
    kg = _StubKnobKG({
        "KNOB_IMPROVES": [
            _knob_fact("fp_low", gain="+5%", args="--a 1"),
            _knob_fact("fp_high", gain="+30%", args="--moe-runner-backend aiter",
                       envs='{"VLLM_USE_AITER":"1"}', name="moe-aiter"),
        ],
    })
    out = generate_knob_candidates_graph_guided(
        kg, architectures=["LlamaForCausalLM"], precision="bf16", hardware="mi300x", framework="sglang",
    )
    assert [v["knob"] for v in out] == ["fp_high", "fp_low"]
    assert out[0]["args"] == "--moe-runner-backend aiter"
    assert out[0]["envs"] == {"VLLM_USE_AITER": "1"}
    assert out[0]["name"] == "moe-aiter"
    assert out[0]["source"] == "kg_knob"
    # Object node folds precision in.
    assert kg.queries[0]["object"] == ["llamaforcausallm+bf16"]


def test_knob_guided_drops_reverted_and_tried() -> None:
    kg = _StubKnobKG({
        "KNOB_IMPROVES": [_knob_fact("fp_bad"), _knob_fact("fp_tried"), _knob_fact("fp_ok")],
        "KNOB_REVERTED_ON": [Fact(subject="fp_bad", predicate="KNOB_REVERTED_ON", object="llamaforcausallm+bf16")],
    })
    out = generate_knob_candidates_graph_guided(
        kg, architectures=["LlamaForCausalLM"], precision="bf16", tried=["fp_tried"],
    )
    knobs = [v["knob"] for v in out]
    assert "fp_bad" not in knobs  # KNOB_REVERTED_ON blocked
    assert "fp_tried" not in knobs  # already tried
    assert "fp_ok" in knobs


def test_knob_guided_no_precision_uses_bare_arch() -> None:
    kg = _StubKnobKG({"KNOB_IMPROVES": []})
    generate_knob_candidates_graph_guided(kg, architectures=["LlamaForCausalLM"])
    # The REVERTED_ON probe runs first; both anchor on the bare (normalized) arch node.
    assert kg.queries[0]["object"] == ["llamaforcausallm"]


def test_knob_guided_no_archs_returns_empty() -> None:
    kg = _StubKnobKG({"KNOB_IMPROVES": [_knob_fact("fp_ok")]})
    assert generate_knob_candidates_graph_guided(kg, architectures=[]) == []


class _LinkGraphMcp:
    """Fake MCP modeling gbrain's native link graph.

    ``add_link`` requires both endpoint pages to exist, edges are unique on
    ``(from, to, link_type)`` (re-add upserts context), and ``remove_link``
    deletes every edge between a pair.
    """

    def __init__(self, pages: list[str] | None = None, edges: list[dict[str, Any]] | None = None) -> None:
        self.pages: set[str] = set(pages or [])
        self.edges: list[dict[str, Any]] = list(edges or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        self.calls.append((tool, dict(args)))
        if tool == "list_pages":
            return [{"slug": s} for s in self.pages]
        if tool == "get_page":
            slug = args.get("slug")
            return {"slug": slug, "body": "x"} if slug in self.pages else {"error": "page_not_found"}
        if tool == "put_page":
            self.pages.add(args["slug"])
            return {"slug": args["slug"], "status": "created_or_updated"}
        if tool == "add_link":
            f, t, lt = args["from"], args["to"], args.get("link_type", "")
            if f not in self.pages or t not in self.pages:
                return {"error": "internal_error", "message": "addLink failed: page not found"}
            for e in self.edges:
                if e["from_slug"] == f and e["to_slug"] == t and e["link_type"] == lt:
                    e["context"] = args.get("context", "{}")
                    return {"status": "ok"}
            self.edges.append(
                {"from_slug": f, "to_slug": t, "link_type": lt, "context": args.get("context", "{}")}
            )
            return {"status": "ok"}
        if tool == "remove_link":
            f, t = args["from"], args["to"]
            self.edges = [e for e in self.edges if not (e["from_slug"] == f and e["to_slug"] == t)]
            return {"status": "ok"}
        if tool == "get_links":
            return [dict(e) for e in self.edges if e["from_slug"] == args["slug"]]
        if tool == "get_backlinks":
            return [dict(e) for e in self.edges if e["to_slug"] == args["slug"]]
        if tool == "traverse_graph":
            return self._traverse(args)
        return {}

    def _traverse(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        start, depth = args["slug"], int(args.get("depth", 5))
        direction, link_type = args.get("direction", "out"), args.get("link_type")
        out: list[dict[str, Any]] = []
        frontier = {start}
        for hop in range(1, depth + 1):
            nxt: set[str] = set()
            for e in self.edges:
                if link_type and e["link_type"] != link_type:
                    continue
                if direction in ("out", "both") and e["from_slug"] in frontier:
                    out.append({**e, "depth": hop})
                    nxt.add(e["to_slug"])
                if direction in ("in", "both") and e["to_slug"] in frontier:
                    out.append({**e, "depth": hop})
                    nxt.add(e["from_slug"])
            frontier = nxt
            if not frontier:
                break
        return out


def test_native_query_facts_uses_get_links() -> None:
    mcp = _LinkGraphMcp(
        pages=["aiter_backend", "qwen2forcausallm"],
        edges=[
            {
                "from_slug": "aiter_backend",
                "to_slug": "qwen2forcausallm",
                "link_type": "improves",
                "context": '{"gain":"+35.2%","confidence":"0.95","hw":"mi300x"}',
            }
        ],
    )
    kg = KGClient(mcp, use_native_kg=True)
    facts = kg.query_facts(subject="AITER_backend", predicate="IMPROVES")
    assert any(t == "get_links" for t, _ in mcp.calls)
    assert len(facts) == 1
    assert facts[0].subject == "aiter_backend"
    assert facts[0].object == "qwen2forcausallm"
    assert facts[0].gain == pytest.approx(35.2)


def test_native_query_facts_object_anchor_uses_backlinks() -> None:
    mcp = _LinkGraphMcp(
        pages=["aiter_backend", "qwen2forcausallm"],
        edges=[
            {
                "from_slug": "aiter_backend",
                "to_slug": "qwen2forcausallm",
                "link_type": "improves",
                "context": "{}",
            }
        ],
    )
    kg = KGClient(mcp, use_native_kg=True)
    facts = kg.query_facts(object="Qwen2ForCausalLM", predicate=["IMPROVES"])
    assert any(t == "get_backlinks" for t, _ in mcp.calls)
    assert facts[0].subject == "aiter_backend"


def test_native_query_facts_conditions_filter() -> None:
    mcp = _LinkGraphMcp(
        pages=["a", "arch"],
        edges=[{"from_slug": "a", "to_slug": "arch", "link_type": "improves", "context": '{"hw":"mi300x"}'}],
    )
    kg = KGClient(mcp, use_native_kg=True)
    assert kg.query_facts(subject="a", conditions={"hw": "mi300x"})
    assert kg.query_facts(subject="a", conditions={"hw": "mi308x"}) == []


def test_native_query_facts_predicate_only_returns_empty() -> None:
    mcp = _LinkGraphMcp(pages=["a"], edges=[])
    kg = KGClient(mcp, use_native_kg=True)
    assert kg.query_facts(predicate="IMPROVES") == []


def test_native_graph_traverse_two_hops() -> None:
    mcp = _LinkGraphMcp(
        pages=["qwen3-8b", "qwen2forcausallm", "llamaforcausallm"],
        edges=[
            {"from_slug": "qwen3-8b", "to_slug": "qwen2forcausallm", "link_type": "uses_arch", "context": "{}"},
            {"from_slug": "qwen2forcausallm", "to_slug": "llamaforcausallm", "link_type": "variant_of", "context": "{}"},
        ],
    )
    kg = KGClient(mcp, use_native_kg=True)
    nodes = kg.graph_traverse(
        start_entity="qwen3-8b", predicate_filter=["USES_ARCH", "VARIANT_OF"], max_hops=2
    )
    entities = {n.entity for n in nodes}
    assert "qwen2forcausallm" in entities
    assert "llamaforcausallm" in entities


def test_native_emit_fact_materializes_nodes_and_links() -> None:
    mcp = _LinkGraphMcp(pages=[])  # neither node exists yet
    kg = KGClient(mcp, use_native_kg=True)
    wrote = kg.emit_fact(
        page_slug="ignored",
        subject="torch_compile",
        predicate="IMPROVES",
        object="qwen3",
        properties={"gain": "+8%"},
    )
    assert wrote is True
    assert "torch_compile" in mcp.pages and "qwen3" in mcp.pages
    assert len(mcp.edges) == 1
    assert mcp.edges[0]["link_type"] == "improves"
    assert kg.query_facts(subject="torch_compile")[0].gain == pytest.approx(8.0)


def test_native_emit_fact_idempotent() -> None:
    mcp = _LinkGraphMcp(pages=["a", "b"])
    kg = KGClient(mcp, use_native_kg=True)
    kg.emit_fact(page_slug="p", subject="a", predicate="IMPROVES", object="b")
    kg.emit_fact(page_slug="p", subject="a", predicate="IMPROVES", object="b")
    assert len(mcp.edges) == 1


def test_native_retract_preserves_other_link_types() -> None:
    mcp = _LinkGraphMcp(
        pages=["a", "b"],
        edges=[
            {"from_slug": "a", "to_slug": "b", "link_type": "keep_knob", "context": "{}"},
            {"from_slug": "a", "to_slug": "b", "link_type": "conflicts_with", "context": "{}"},
        ],
    )
    kg = KGClient(mcp, use_native_kg=True)
    removed = kg.retract_fact(page_slug="p", subject="a", predicate="KEEP_KNOB", object="b")
    assert removed is True
    remaining = {e["link_type"] for e in mcp.edges}
    assert remaining == {"conflicts_with"}


def test_native_retract_noop_when_absent() -> None:
    mcp = _LinkGraphMcp(
        pages=["a", "b"],
        edges=[{"from_slug": "a", "to_slug": "b", "link_type": "keep_knob", "context": "{}"}],
    )
    kg = KGClient(mcp, use_native_kg=True)
    assert kg.retract_fact(page_slug="p", subject="a", predicate="IMPROVES", object="b") is False
    assert len(mcp.edges) == 1


def test_native_emit_ignores_page_slug() -> None:
    # The subject node is the edge source; ``page_slug`` is neither created as a
    # page nor used as the edge anchor.
    mcp = _LinkGraphMcp(pages=[])
    kg = KGClient(mcp, use_native_kg=True)
    kg.emit_fact(page_slug="recipe/some-page", subject="a", predicate="IMPROVES", object="b")
    assert "recipe/some-page" not in mcp.pages
    assert mcp.edges[0]["from_slug"] == "a"


class _AddLinkFailsMcp(_LinkGraphMcp):
    """Link-graph fake whose ``add_link`` always reports an in-band error."""

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "add_link":
            self.calls.append((tool, dict(args)))
            return {"error": "internal_error", "message": "addLink failed"}
        return super().call(tool, args)


def test_native_emit_returns_false_on_add_link_error() -> None:
    # A failed add_link (reported in-band) must not parse as success.
    mcp = _AddLinkFailsMcp(pages=["a", "b"])
    kg = KGClient(mcp, use_native_kg=True)
    assert kg.emit_fact(page_slug="p", subject="a", predicate="IMPROVES", object="b") is False
    assert mcp.edges == []


class _PutPageFailsMcp(_LinkGraphMcp):
    """Link-graph fake whose ``put_page`` reports an in-band error."""

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "put_page":
            self.calls.append((tool, dict(args)))
            return {"error": "internal_error", "message": "putPage failed"}
        return super().call(tool, args)


def test_native_emit_returns_false_when_node_creation_fails() -> None:
    # A missing endpoint that cannot be materialized aborts the edge write.
    mcp = _PutPageFailsMcp(pages=[])
    kg = KGClient(mcp, use_native_kg=True)
    assert kg.emit_fact(page_slug="p", subject="a", predicate="IMPROVES", object="b") is False
    assert mcp.edges == []


def test_native_graph_traverse_breadth_capped() -> None:
    # A high-fan-in hub must not blow up the foreground warm-start path.
    from hyperloom.orchestrator.knowledge.recipe_kb import kg_client as kgmod

    hub = "hub"
    leaves = [f"leaf{i}" for i in range(kgmod._MAX_TRAVERSE_NODES + 50)]
    mcp = _LinkGraphMcp(
        pages=[hub, *leaves],
        edges=[{"from_slug": hub, "to_slug": l, "link_type": "improves", "context": "{}"} for l in leaves],
    )
    kg = KGClient(mcp, use_native_kg=True)
    nodes = kg.graph_traverse(start_entity=hub, max_hops=1)
    assert len(nodes) == kgmod._MAX_TRAVERSE_NODES
