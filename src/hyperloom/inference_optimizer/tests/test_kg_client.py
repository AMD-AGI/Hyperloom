# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the knowledge-graph client (native link graph)."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_mcp import GbrainRemoteError
from hyperloom.orchestrator.knowledge.recipe_kb.kg_client import (
    Fact,
    KGClient,
    generate_knob_candidates_graph_guided,
    generate_variants_graph_guided,
)


class _LinkGraphMcp:
    """Fake MCP modeling the native link graph.

    ``add_link`` requires both endpoint pages to exist and edges are unique on
    ``(from, to, link_type)`` (re-add upserts context).
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
            self.edges.append({"from_slug": f, "to_slug": t, "link_type": lt, "context": args.get("context", "{}")})
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


class _BoomMcp:
    """Fake MCP that always raises a transport error."""

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        raise GbrainRemoteError("boom")


def test_query_facts_uses_get_links() -> None:
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
    kg = KGClient(mcp)
    facts = kg.query_facts(subject="AITER_backend", predicate="IMPROVES")
    assert any(t == "get_links" for t, _ in mcp.calls)
    assert len(facts) == 1
    assert facts[0].subject == "aiter_backend"
    assert facts[0].object == "qwen2forcausallm"
    assert facts[0].gain == pytest.approx(35.2)


def test_query_facts_object_anchor_uses_backlinks() -> None:
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
    kg = KGClient(mcp)
    facts = kg.query_facts(object="Qwen2ForCausalLM", predicate=["IMPROVES"])
    assert any(t == "get_backlinks" for t, _ in mcp.calls)
    assert facts[0].subject == "aiter_backend"


def test_query_facts_reads_legacy_punctuation_slug() -> None:
    mcp = _LinkGraphMcp(
        pages=["foo=1", "target:2"],
        edges=[
            {
                "from_slug": "foo=1",
                "to_slug": "target:2",
                "link_type": "improves",
                "context": "{}",
            }
        ],
    )
    kg = KGClient(mcp)

    facts = kg.query_facts(subject="foo=1", object="target:2")

    assert len(facts) == 1
    assert facts[0].subject == "foo_1"
    assert facts[0].object == "target_2"
    queried = [args["slug"] for tool, args in mcp.calls if tool == "get_links"]
    assert queried == ["foo=1", "foo_1"]


def test_query_facts_conditions_filter() -> None:
    mcp = _LinkGraphMcp(
        pages=["a", "arch"],
        edges=[{"from_slug": "a", "to_slug": "arch", "link_type": "improves", "context": '{"hw":"mi300x"}'}],
    )
    kg = KGClient(mcp)
    assert kg.query_facts(subject="a", conditions={"hw": "mi300x"})
    assert kg.query_facts(subject="a", conditions={"hw": "mi308x"}) == []


def test_query_facts_predicate_only_returns_empty() -> None:
    mcp = _LinkGraphMcp(pages=["a"], edges=[])
    kg = KGClient(mcp)
    assert kg.query_facts(predicate="IMPROVES") == []


def test_query_facts_filters_by_subject_and_object() -> None:
    mcp = _LinkGraphMcp(
        pages=["aiter_backend", "qwen2forcausallm", "mi300x"],
        edges=[
            {"from_slug": "aiter_backend", "to_slug": "qwen2forcausallm", "link_type": "improves", "context": "{}"},
            {"from_slug": "aiter_backend", "to_slug": "mi300x", "link_type": "reverted_on", "context": "{}"},
        ],
    )
    kg = KGClient(mcp)
    facts = kg.query_facts(subject="aiter_backend", object="qwen2forcausallm")
    assert len(facts) == 1
    assert facts[0].predicate == "IMPROVES"


def test_graph_traverse_two_hops() -> None:
    mcp = _LinkGraphMcp(
        pages=["qwen3-8b", "qwen2forcausallm", "llamaforcausallm"],
        edges=[
            {"from_slug": "qwen3-8b", "to_slug": "qwen2forcausallm", "link_type": "uses_arch", "context": "{}"},
            {
                "from_slug": "qwen2forcausallm",
                "to_slug": "llamaforcausallm",
                "link_type": "variant_of",
                "context": "{}",
            },
        ],
    )
    kg = KGClient(mcp)
    nodes = kg.graph_traverse(start_entity="qwen3-8b", predicate_filter=["USES_ARCH", "VARIANT_OF"], max_hops=2)
    entities = {n.entity for n in nodes}
    assert "qwen2forcausallm" in entities
    assert "llamaforcausallm" in entities


def test_graph_traverse_reads_legacy_punctuation_slug() -> None:
    mcp = _LinkGraphMcp(
        pages=["foo=1", "target:2"],
        edges=[
            {
                "from_slug": "foo=1",
                "to_slug": "target:2",
                "link_type": "improves",
                "context": "{}",
            }
        ],
    )
    kg = KGClient(mcp)

    nodes = kg.graph_traverse(start_entity="foo=1", max_hops=1)

    assert [node.entity for node in nodes] == ["target_2"]
    traversed = [
        args["slug"] for tool, args in mcp.calls if tool == "traverse_graph"
    ]
    assert traversed == ["foo_1", "foo=1"]


def test_find_conflicts_detects_pair_in_stack() -> None:
    mcp = _LinkGraphMcp(
        pages=["aiter_backend", "flash_attention_v2", "torch_compile"],
        edges=[
            {"from_slug": "aiter_backend", "to_slug": "flash_attention_v2", "link_type": "conflicts_with", "context": '{"severity":"hard"}'},
        ],
    )
    kg = KGClient(mcp)
    conflicts = kg.find_conflicts(knobs=["aiter_backend", "flash_attention_v2", "torch_compile"])
    assert len(conflicts) == 1
    assert conflicts[0]["conflicts_with"] == "aiter_backend"
    assert conflicts[0]["knob"] == "flash_attention_v2"


def test_find_conflicts_skips_when_endpoint_not_in_stack() -> None:
    mcp = _LinkGraphMcp(
        pages=["aiter_backend", "flash_attention_v2", "torch_compile"],
        edges=[
            {"from_slug": "aiter_backend", "to_slug": "flash_attention_v2", "link_type": "conflicts_with", "context": '{"severity":"hard"}'},
        ],
    )
    kg = KGClient(mcp)
    conflicts = kg.find_conflicts(knobs=["aiter_backend", "torch_compile"])
    assert conflicts == []


def test_find_conflicts_detected_even_with_hw_fw_conditions() -> None:
    mcp = _LinkGraphMcp(
        pages=["aiter_backend", "flash_attention_v2"],
        edges=[
            {"from_slug": "aiter_backend", "to_slug": "flash_attention_v2", "link_type": "conflicts_with", "context": '{"severity":"hard"}'},
        ],
    )
    kg = KGClient(mcp)
    conflicts = kg.find_conflicts(
        knobs=["aiter_backend", "flash_attention_v2"],
        hardware="mi300x",
        framework="sglang",
    )
    assert len(conflicts) == 1
    assert conflicts[0]["knob"] == "flash_attention_v2"


def test_emit_fact_materializes_nodes_and_links() -> None:
    mcp = _LinkGraphMcp(pages=[])
    kg = KGClient(mcp)
    wrote = kg.emit_fact(
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


def test_emit_fact_idempotent() -> None:
    mcp = _LinkGraphMcp(pages=["a", "b"])
    kg = KGClient(mcp)
    kg.emit_fact(subject="a", predicate="IMPROVES", object="b")
    kg.emit_fact(subject="a", predicate="IMPROVES", object="b")
    assert len(mcp.edges) == 1


class _AddLinkFailsMcp(_LinkGraphMcp):
    """Link-graph fake whose ``add_link`` always reports an in-band error."""

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "add_link":
            self.calls.append((tool, dict(args)))
            return {"error": "internal_error", "message": "addLink failed"}
        return super().call(tool, args)


def test_emit_returns_false_on_add_link_error() -> None:
    mcp = _AddLinkFailsMcp(pages=["a", "b"])
    kg = KGClient(mcp)
    assert kg.emit_fact(subject="a", predicate="IMPROVES", object="b") is False
    assert mcp.edges == []


class _PutPageFailsMcp(_LinkGraphMcp):
    """Link-graph fake whose ``put_page`` reports an in-band error."""

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "put_page":
            self.calls.append((tool, dict(args)))
            return {"error": "internal_error", "message": "putPage failed"}
        return super().call(tool, args)


def test_emit_returns_false_when_node_creation_fails() -> None:
    mcp = _PutPageFailsMcp(pages=[])
    kg = KGClient(mcp)
    assert kg.emit_fact(subject="a", predicate="IMPROVES", object="b") is False
    assert mcp.edges == []


def test_query_facts_safe_returns_empty_on_error() -> None:
    kg = KGClient(_BoomMcp())
    assert kg.query_facts_safe(predicate="IMPROVES") == []


def test_graph_traverse_safe_returns_empty_on_error() -> None:
    kg = KGClient(_BoomMcp())
    assert kg.graph_traverse_safe(start_entity="x") == []


def test_emit_fact_safe_returns_false_on_error() -> None:
    kg = KGClient(_BoomMcp())
    assert kg.emit_fact_safe(subject="a", predicate="IMPROVES", object="b") is False


def test_is_available_true_and_false() -> None:
    mcp = _LinkGraphMcp(pages=["x"])
    assert KGClient(mcp).is_available() is True
    assert KGClient(_BoomMcp()).is_available() is False


def test_graph_traverse_breadth_capped() -> None:
    from hyperloom.orchestrator.knowledge.recipe_kb import kg_client as kgmod

    hub = "hub"
    leaves = [f"leaf{i}" for i in range(kgmod._MAX_TRAVERSE_NODES + 50)]
    mcp = _LinkGraphMcp(
        pages=[hub, *leaves],
        edges=[{"from_slug": hub, "to_slug": l, "link_type": "improves", "context": "{}"} for l in leaves],
    )
    kg = KGClient(mcp)
    nodes = kg.graph_traverse(start_entity=hub, max_hops=1)
    assert len(nodes) == kgmod._MAX_TRAVERSE_NODES


_IMPROVES_EDGES = [
    {"from_slug": "aiter_backend", "to_slug": "qwen2forcausallm", "link_type": "improves", "context": '{"gain":"+35%","confidence":"0.95","hw":"mi300x","fw":"sglang"}'},
    {"from_slug": "torch_compile", "to_slug": "qwen2forcausallm", "link_type": "improves", "context": '{"gain":"+8%","hw":"mi300x","fw":"sglang"}'},
    {"from_slug": "decode_patch", "to_slug": "qwen2forcausallm", "link_type": "improves", "context": '{"gain":"+20%","hw":"mi300x","fw":"sglang"}'},
    {"from_slug": "decode_patch", "to_slug": "flash_attn", "link_type": "conflicts_with", "context": '{"severity":"hard"}'},
    {"from_slug": "needs_dep", "to_slug": "qwen2forcausallm", "link_type": "improves", "context": '{"gain":"+40%","hw":"mi300x","fw":"sglang"}'},
    {"from_slug": "needs_dep", "to_slug": "chunked_prefill", "link_type": "requires", "context": "{}"},
]


def _variants_mcp() -> _LinkGraphMcp:
    pages = ["aiter_backend", "torch_compile", "decode_patch", "flash_attn", "needs_dep", "chunked_prefill", "qwen2forcausallm"]
    return _LinkGraphMcp(pages=pages, edges=_IMPROVES_EDGES)


def test_graph_guided_orders_by_gain_and_excludes_in_stack() -> None:
    kg = KGClient(_variants_mcp())
    out = generate_variants_graph_guided(
        kg,
        architectures=["Qwen2ForCausalLM"],
        hardware="mi300x",
        framework="sglang",
        in_stack=["aiter_backend"],
    )
    knobs = [v["knob"] for v in out]
    assert "aiter_backend" not in knobs
    assert "needs_dep" not in knobs
    assert knobs[0] == "decode_patch"
    assert "torch_compile" in knobs


def test_graph_guided_rejects_conflict_with_stack() -> None:
    kg = KGClient(_variants_mcp())
    out = generate_variants_graph_guided(
        kg,
        architectures=["Qwen2ForCausalLM"],
        hardware="mi300x",
        framework="sglang",
        in_stack=["flash_attn"],
    )
    knobs = [v["knob"] for v in out]
    assert "decode_patch" not in knobs


def test_graph_guided_dependency_met() -> None:
    kg = KGClient(_variants_mcp())
    out = generate_variants_graph_guided(
        kg,
        architectures=["Qwen2ForCausalLM"],
        hardware="mi300x",
        framework="sglang",
        in_stack=["chunked_prefill"],
    )
    knobs = [v["knob"] for v in out]
    assert "needs_dep" in knobs
    assert knobs[0] == "needs_dep"


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


def _knob_fact(
    subject: str, *, gain: str = "+10%", name: str = "k", args: str = "--x 1", envs: str = "", keep_n: str = "2"
) -> Fact:
    props = {"gain": gain, "name": name, "args": args, "keep_n": keep_n}
    if envs:
        props["envs"] = envs
    return Fact(subject=subject, predicate="KNOB_IMPROVES", object="llamaforcausallm+bf16", properties=props)


def test_knob_guided_orders_by_gain_and_surfaces_args_envs() -> None:
    kg = _StubKnobKG(
        {
            "KNOB_IMPROVES": [
                _knob_fact("fp_low", gain="+5%", args="--a 1"),
                _knob_fact(
                    "fp_high",
                    gain="+30%",
                    args="--moe-runner-backend aiter",
                    envs='{"VLLM_USE_AITER":"1"}',
                    name="moe-aiter",
                ),
            ],
        }
    )
    out = generate_knob_candidates_graph_guided(
        kg,
        architectures=["LlamaForCausalLM"],
        precision="bf16",
        hardware="mi300x",
        framework="sglang",
    )
    assert [v["knob"] for v in out] == ["fp_high", "fp_low"]
    assert out[0]["args"] == "--moe-runner-backend aiter"
    assert out[0]["envs"] == {"VLLM_USE_AITER": "1"}
    assert out[0]["name"] == "moe-aiter"
    assert out[0]["source"] == "kg_knob"
    assert kg.queries[0]["object"] == ["llamaforcausallm+bf16"]


def test_knob_guided_drops_reverted_and_tried() -> None:
    kg = _StubKnobKG(
        {
            "KNOB_IMPROVES": [_knob_fact("fp_bad"), _knob_fact("fp_tried"), _knob_fact("fp_ok")],
            "KNOB_REVERTED_ON": [Fact(subject="fp_bad", predicate="KNOB_REVERTED_ON", object="llamaforcausallm+bf16")],
        }
    )
    out = generate_knob_candidates_graph_guided(
        kg,
        architectures=["LlamaForCausalLM"],
        precision="bf16",
        tried=["fp_tried"],
    )
    knobs = [v["knob"] for v in out]
    assert "fp_bad" not in knobs
    assert "fp_tried" not in knobs
    assert "fp_ok" in knobs


def test_knob_guided_no_precision_uses_bare_arch() -> None:
    kg = _StubKnobKG({"KNOB_IMPROVES": []})
    generate_knob_candidates_graph_guided(kg, architectures=["LlamaForCausalLM"])
    assert kg.queries[0]["object"] == ["llamaforcausallm"]


def test_knob_guided_no_archs_returns_empty() -> None:
    kg = _StubKnobKG({"KNOB_IMPROVES": [_knob_fact("fp_ok")]})
    assert generate_knob_candidates_graph_guided(kg, architectures=[]) == []
