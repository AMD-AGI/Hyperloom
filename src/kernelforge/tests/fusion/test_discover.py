# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for LLM-autonomous discovery (no LLM — injected fake llm_fn)."""

from __future__ import annotations

import gzip
import json

from kernelforge.fusion.diagnose import diagnose_from_shares
from kernelforge.fusion.discover import (
    build_discovery_prompt,
    discover_recipes,
    existing_operator_hints_from_knowledge,
    hot_kernels_from_trace,
    ordered_fusion_boundaries_from_trace,
    parse_discovered_recipes,
)


def _candidate_diag(shares=None, busy=0.21):
    shares = shares or {"gemm": 0.4, "add": 0.14, "elementwise": 0.14, "cast": 0.13, "mul": 0.08}
    return diagnose_from_shares(shares, busy_fraction_of_wall=busy)


def _write_trace(path, events, gz=False):
    payload = {"traceEvents": events}
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


class TestHotKernels:
    def test_ranks_and_filters_compute(self, tmp_path):
        p = tmp_path / "d.trace.json"
        _write_trace(
            p,
            [
                {"cat": "kernel", "name": "Cijk_gemm", "ts": 0, "dur": 100},  # gemm -> filtered
                {"cat": "kernel", "name": "vectorized_elementwise mul", "ts": 100, "dur": 40},
                {"cat": "kernel", "name": "bfloat16tofloat32_copy", "ts": 200, "dur": 30},
                {"cat": "kernel", "name": "rotary_embedding_kernel", "ts": 300, "dur": 10},
            ],
        )
        hot = hot_kernels_from_trace(p, launch_bound_only=True)
        names = [h["name"] for h in hot]
        assert not any("Cijk_gemm" in n for n in names)  # compute filtered out
        assert hot[0]["category"] == "mul"  # highest launch-bound share first
        assert all(0 <= h["share"] <= 1 for h in hot)

    def test_missing_trace_safe(self, tmp_path):
        assert hot_kernels_from_trace(tmp_path / "nope.json") == []


class TestOrderedFusionBoundaries:
    def test_keeps_compute_endpoints_around_repeated_epilogue(self, tmp_path):
        p = tmp_path / "d.trace.json"
        events = []
        for base in (0, 1000):
            events.extend(
                [
                    {
                        "cat": "kernel",
                        "name": "Cijk_gate_up_gemm",
                        "ts": base,
                        "dur": 40,
                        "pid": 2,
                        "tid": 0,
                    },
                    {
                        "cat": "kernel",
                        "name": "act_and_mul_kernel silu",
                        "ts": base + 40,
                        "dur": 5,
                        "pid": 2,
                        "tid": 0,
                    },
                    {
                        "cat": "kernel",
                        "name": "Cijk_down_gemm",
                        "ts": base + 45,
                        "dur": 20,
                        "pid": 2,
                        "tid": 0,
                    },
                ]
            )
        _write_trace(p, events)

        boundaries = ordered_fusion_boundaries_from_trace(p)

        match = next(row for row in boundaries if row["categories"] == ["gemm", "activation", "gemm"])
        assert match["count"] == 2
        assert match["boundary_kind"] == "epilogue"
        assert match["launches_removed_upper_bound"] == 1

    def test_keeps_long_qk_postprocess_chain_through_cache_write(self, tmp_path):
        p = tmp_path / "d.trace.json"
        names = [
            "Cijk_qkv_gemm",
            "elementwise direct_copy_kernel q",
            "add_rmsnorm_quant_kernel q",
            "elementwise direct_copy_kernel k",
            "add_rmsnorm_quant_kernel k",
            "rotary_embedding_kernel",
            "store_kvcache",
            "_fwd_grouped_kernel_stage1 attention",
        ]
        events = []
        for repeat in range(2):
            for index, name in enumerate(names):
                events.append(
                    {
                        "cat": "kernel",
                        "name": name,
                        "ts": repeat * 1000 + index * 5,
                        "dur": 5,
                        "pid": 2,
                        "tid": 0,
                    }
                )
        _write_trace(p, events)

        boundaries = ordered_fusion_boundaries_from_trace(p)

        assert any(
            row["categories"]
            == [
                "gemm",
                "elementwise",
                "rmsnorm",
                "elementwise",
                "rmsnorm",
                "rope",
                "copy",
                "attention",
            ]
            and row["count"] == 2
            for row in boundaries
        )

    def test_marks_terminal_attention_outside_the_fusable_span(self, tmp_path):
        """The chain really does run into attention, so the boundary keeps it as
        adjacency evidence. But the fusable part is the prologue before it: the
        native operator fuses norm + RoPE + cache write, never the attention
        kernel itself. The boundary must say so explicitly."""
        p = tmp_path / "d.trace.json"
        names = [
            "Cijk_qkv_gemm",
            "add_rmsnorm_quant_kernel q",
            "rotary_embedding_kernel",
            "store_kvcache",
            "_fwd_grouped_kernel_stage1 attention",
        ]
        events = []
        for repeat in range(2):
            for index, name in enumerate(names):
                events.append(
                    {
                        "cat": "kernel",
                        "name": name,
                        "ts": repeat * 1000 + index * 5,
                        "dur": 5,
                        "pid": 2,
                        "tid": 0,
                    }
                )
        _write_trace(p, events)

        boundaries = ordered_fusion_boundaries_from_trace(p)
        row = next(b for b in boundaries if b["categories"][-1] == "attention")

        assert row["terminal_compute"] == "attention"
        assert "attention" not in row["fusable_categories"]
        assert row["fusable_categories"] == ["rmsnorm", "rope", "copy"]


class TestExistingOperatorHints:
    def test_recalls_semantic_operators_for_observed_boundaries(self, tmp_path):
        knowledge = tmp_path / "knowledge"
        knowledge.mkdir()
        (knowledge / "kv.md").write_text(
            "QK norm, RoPE, and cache write use `fused_qk_norm_rope_cache_pts_quant_shuffle`.\n",
            encoding="utf-8",
        )
        (knowledge / "gemm.md").write_text(
            "Gate/up GEMM with SiLU uses `gemm_a16w16_gated`.\n",
            encoding="utf-8",
        )
        boundaries = [
            {
                "categories": ["rmsnorm", "rope", "copy"],
                "kernels": ["rmsnorm", "rotary", "store_kvcache"],
            },
            {
                "categories": ["gemm", "activation"],
                "kernels": ["Cijk_gate_up_gemm", "act_and_mul_kernel silu"],
            },
        ]

        hints = existing_operator_hints_from_knowledge(knowledge, boundaries)
        operators = {row["operator"] for row in hints}

        assert "fused_qk_norm_rope_cache_pts_quant_shuffle" in operators
        assert "gemm_a16w16_gated" in operators

    def test_does_not_recall_on_substring_or_prefix_collisions(self, tmp_path):
        """Terms must match whole words, not substrings or prefixes.

        ``add`` is a substring of ``padding`` and ``norm`` is a prefix of
        ``normalization``; neither implies the documented operator performs the
        observed operation. A false recall is worse than no recall, because the
        author is then told to integrate an unrelated operator.
        """
        knowledge = tmp_path / "knowledge"
        knowledge.mkdir()
        (knowledge / "padding.md").write_text(
            "`fused_padding_workspace_helper` allocates padding for copying "
            "buffers that are materialized ahead of the launch.\n",
            encoding="utf-8",
        )
        (knowledge / "stats.md").write_text(
            "`fused_layer_normalization_stats` collects normalization statistics for offline calibration.\n",
            encoding="utf-8",
        )
        boundaries = [
            {
                "categories": ["elementwise", "rmsnorm"],
                "kernels": ["elementwise_add_kernel", "rmsnorm_kernel"],
            },
        ]

        hints = existing_operator_hints_from_knowledge(knowledge, boundaries)
        operators = {row["operator"] for row in hints}

        assert "fused_padding_workspace_helper" not in operators
        assert "fused_layer_normalization_stats" not in operators

    def test_hints_carry_a_score_so_weak_matches_are_distinguishable(self, tmp_path):
        knowledge = tmp_path / "knowledge"
        knowledge.mkdir()
        (knowledge / "kv.md").write_text(
            "---\noperator: fused_qk_norm_rope_cache\n---\n\n"
            "QK norm, RoPE, and cache write use `fused_qk_norm_rope_cache`.\n",
            encoding="utf-8",
        )
        (knowledge / "attn.md").write_text(
            "Paged attention decode uses `fused_attention_decode`.\n",
            encoding="utf-8",
        )
        boundaries = [
            {
                "categories": ["rmsnorm", "rope", "copy"],
                "kernels": ["rmsnorm", "rotary", "store_kvcache"],
            },
        ]

        hints = existing_operator_hints_from_knowledge(knowledge, boundaries)

        assert hints, "expected at least one recalled operator"
        assert all("score" in row for row in hints)
        scores = [float(row["score"]) for row in hints]
        assert scores == sorted(scores, reverse=True)
        # The declared operator of a matching card must outrank an incidental one.
        assert hints[0]["operator"] == "fused_qk_norm_rope_cache"

    def test_falls_back_to_hot_kernels_when_boundaries_are_absent(self, tmp_path):
        """Boundaries need min_repeats=2 to materialize. A short trace can leave
        them empty while the hot-kernel table still proves a launch-bound chain,
        so retrieval must not silently go dark."""
        knowledge = tmp_path / "knowledge"
        knowledge.mkdir()
        (knowledge / "gemm.md").write_text(
            "Gate/up GEMM with SiLU uses `gemm_a16w16_gated`.\n",
            encoding="utf-8",
        )

        hints = existing_operator_hints_from_knowledge(
            knowledge,
            [],
            fallback_categories=["gemm", "activation"],
            fallback_kernel_names=["Cijk_gate_up_gemm", "act_and_mul_kernel silu"],
        )

        assert "gemm_a16w16_gated" in {row["operator"] for row in hints}


class TestDiscoveryPrompt:
    def test_prompt_has_profile_source_and_no_answer_encoding(self):
        d = _candidate_diag()
        hot = [{"name": "vectorized_elementwise mul", "category": "mul", "share": 0.14, "count": 60, "avg_us": 3.2}]
        p = build_discovery_prompt(
            model_type="zaya",
            framework="sglang",
            source_text="class CCA:\n    def _normalize_qk(self, q, k): ...\n",
            diagnosis=d,
            hot_kernels=hot,
            shapes={"hidden_size": 2048},
        )
        # Includes the measured profile + the real source.
        assert "launch_bound_share" in p and "vectorized_elementwise mul" in p
        assert "_normalize_qk" in p  # the real source is embedded
        assert "ZAYA_FUSED_" in p  # asks for a model-prefixed flag
        # Must NOT hand the model the answer (no template names).
        low = p.lower()
        for leak in ("residual_add_rmsnorm", "swiglu", "silu", " residualscaling", "cca qk", "grouped-mean"):
            assert leak not in low, f"answer-encoding leaked into prompt: {leak}"

    def test_prompt_includes_ordered_boundaries_and_existing_operator_evidence(self):
        d = _candidate_diag()
        prompt = build_discovery_prompt(
            model_type="toy",
            framework="sglang",
            source_text="def forward(x): return self.mlp(x)\n",
            diagnosis=d,
            hot_kernels=[],
            shapes={"hidden_size": 4096},
            ordered_boundaries=[
                {
                    "signature": "gemm -> activation -> gemm",
                    "count": 36,
                    "total_us": 1700.0,
                    "boundary_kind": "epilogue",
                    "launches_removed_upper_bound": 1,
                }
            ],
            existing_operator_hints=[
                {
                    "operator": "gemm_a16w16_gated",
                    "path": "knowledge/gemm.md",
                    "evidence": "Gate/up GEMM with SiLU",
                }
            ],
        )

        assert "Ordered fusion boundaries" in prompt
        assert "gemm -> activation -> gemm" in prompt
        assert "Existing ROCm operator evidence" in prompt
        assert "gemm_a16w16_gated" in prompt
        assert "integration" in prompt.lower()

    def test_prompt_excludes_terminal_attention_from_the_fusable_span(self):
        prompt = build_discovery_prompt(
            model_type="toy",
            framework="sglang",
            source_text="def forward(x): return x\n",
            diagnosis=_candidate_diag(),
            hot_kernels=[],
            shapes={},
            ordered_boundaries=[
                {
                    "signature": "gemm -> rmsnorm -> rope -> copy -> attention",
                    "fusable_categories": ["rmsnorm", "rope", "copy"],
                    "terminal_compute": "attention",
                    "count": 28,
                    "total_us": 900.0,
                    "boundary_kind": "compute_boundary",
                    "launches_removed_upper_bound": 3,
                }
            ],
        )

        assert "fusable-span=rmsnorm -> rope -> copy" in prompt
        assert "do NOT include the terminal attention kernel" in prompt

    def test_prompt_shows_recall_scores(self):
        prompt = build_discovery_prompt(
            model_type="toy",
            framework="sglang",
            source_text="def forward(x): return x\n",
            diagnosis=_candidate_diag(),
            hot_kernels=[],
            shapes={},
            existing_operator_hints=[
                {
                    "operator": "gemm_a16w16_gated",
                    "path": "knowledge/gemm.md",
                    "evidence": "Gate/up GEMM with SiLU",
                    "score": 812.5,
                },
                {
                    "operator": "fused_attention_decode",
                    "path": "knowledge/attn.md",
                    "evidence": "Paged attention decode",
                    "score": 30.0,
                },
            ],
        )

        assert "score=812.5" in prompt
        assert "score=30.0" in prompt
        # The prompt must say what the score means, or it is noise to the model.
        assert "higher score" in prompt.lower()


class TestParse:
    def test_parses_fenced_json_and_prefixes_flag(self):
        payload = json.dumps(
            [
                {
                    "name": "cca_qk",
                    "env_flag": "FUSED_QK",
                    "op_chain": "_add_grouped_qk_means + _normalize_qk",
                    "source_anchors": ["_normalize_qk", "_add_grouped_qk_means"],
                    "fusion_math": "grouped mean then rmsnorm then temp",
                    "eager_reference": "import CCA._normalize_qk",
                    "priority": 0.9,
                    "rationale": "dominant elementwise tail",
                }
            ]
        )
        text = "Here is my analysis.\n```json\n" + payload + "\n```\n"
        recipes = parse_discovered_recipes(
            text, model_type="zaya", framework="sglang", source_file="/sgl/models/zaya.py", shapes={"hidden_size": 2048}
        )
        assert len(recipes) == 1
        r = recipes[0]
        assert r.pattern_id == "llm:cca_qk"
        assert r.env_flag == "ZAYA_FUSED_QK"  # normalized + model-prefixed
        assert "_normalize_qk" in r.source_hints
        assert r.source_confirmed is True
        assert r.trigger_share == 0.9

    def test_ranks_by_priority(self):
        text = '[{"name":"a","priority":0.3},{"name":"b","priority":0.8}]'
        rs = parse_discovered_recipes(text, model_type="zaya", framework="sglang", source_file="/x.py", shapes={})
        assert [r.pattern_id for r in rs] == ["llm:b", "llm:a"]

    def test_preserves_existing_operator_integration_plan(self):
        text = json.dumps(
            [
                {
                    "name": "gate_epilogue",
                    "candidate_kind": "integration",
                    "existing_operator": "gemm_a16w16_gated",
                    "priority": 0.9,
                }
            ]
        )

        recipe = parse_discovered_recipes(
            text,
            model_type="toy",
            framework="sglang",
            source_file="/x.py",
            shapes={},
        )[0]

        assert recipe.candidate_kind == "integration"
        assert recipe.existing_operator == "gemm_a16w16_gated"
        assert recipe.to_dict()["candidate_kind"] == "integration"

    def test_integration_without_operator_is_downgraded(self):
        """``integration`` only means something when an operator is named.

        The authoring prompt injects its "benchmark the existing operator first"
        block only when both fields are set, so an operator-less integration
        recipe would claim the kind while silently skipping the constraint.
        """
        text = json.dumps(
            [
                {
                    "name": "vague_plan",
                    "candidate_kind": "integration",
                    "existing_operator": "   ",
                    "priority": 0.9,
                }
            ]
        )

        recipe = parse_discovered_recipes(
            text,
            model_type="toy",
            framework="sglang",
            source_file="/x.py",
            shapes={},
        )[0]

        assert recipe.candidate_kind == "new_fusion"
        assert recipe.existing_operator == ""

    def test_unknown_kind_with_operator_becomes_integration(self):
        text = json.dumps(
            [
                {
                    "name": "odd_kind",
                    "candidate_kind": "banana",
                    "existing_operator": "gemm_a16w16_gated",
                    "priority": 0.9,
                }
            ]
        )

        recipe = parse_discovered_recipes(
            text,
            model_type="toy",
            framework="sglang",
            source_file="/x.py",
            shapes={},
        )[0]

        assert recipe.candidate_kind == "integration"
        assert recipe.existing_operator == "gemm_a16w16_gated"

    def test_salvages_objects_from_truncated_array(self):
        # Response cut off at max_tokens mid-3rd-object: array never closes, but
        # the 2 complete objects before the cut must still be recovered.
        text = (
            "```json\n[\n"
            '  {"name": "a", "env_flag": "ZAYA_FUSED_A", "priority": 0.9},\n'
            '  {"name": "b", "env_flag": "ZAYA_FUSED_B", "priority": 0.5},\n'
            '  {"name": "c", "env_flag": "ZAYA_FUSED_C", "rationale": "this got cut o'
        )
        rs = parse_discovered_recipes(text, model_type="zaya", framework="sglang", source_file="/x.py", shapes={})
        assert [r.pattern_id for r in rs] == ["llm:a", "llm:b"]  # 2 salvaged, ranked
        assert rs[0].env_flag == "ZAYA_FUSED_A"

    def test_no_json_returns_empty(self):
        assert (
            parse_discovered_recipes(
                "no json here", model_type="zaya", framework="sglang", source_file="/x.py", shapes={}
            )
            == []
        )

    def test_bare_array_without_fence(self):
        rs = parse_discovered_recipes(
            'prose [{"name":"x","env_flag":"ZAYA_FUSED_X"}] tail',
            model_type="zaya",
            framework="sglang",
            source_file="/x.py",
            shapes={},
        )
        assert len(rs) == 1 and rs[0].env_flag == "ZAYA_FUSED_X"


class TestDiscoverRecipes:
    def test_end_to_end_with_fake_llm(self, tmp_path):
        src = tmp_path / "zaya.py"
        src.write_text(
            "class CCA:\n    def _normalize_qk(self): ...\n    def _add_grouped_qk_means(self): ...\n", encoding="utf-8"
        )
        trace = tmp_path / "d.trace.json"
        _write_trace(trace, [{"cat": "kernel", "name": "vectorized_elementwise mul", "ts": 0, "dur": 40}])
        d = _candidate_diag()

        captured = {}

        def fake_llm(prompt: str) -> str:
            captured["prompt"] = prompt
            return (
                '[{"name":"cca_qk","env_flag":"FUSED_QK",'
                '"op_chain":"_add_grouped_qk_means + _normalize_qk",'
                '"source_anchors":["_normalize_qk"],"fusion_math":"m",'
                '"eager_reference":"import CCA._normalize_qk","priority":0.9}]'
            )

        recipes = discover_recipes(
            d,
            model_type="zaya",
            framework="sglang",
            source_file=str(src),
            shapes={"hidden_size": 2048},
            trace_path=str(trace),
            llm_fn=fake_llm,
        )
        assert len(recipes) == 1
        assert recipes[0].env_flag == "ZAYA_FUSED_QK"
        assert "_normalize_qk" in captured["prompt"]  # real source reached the LLM

    def test_recalls_via_hot_kernels_when_trace_has_no_repeats(self, tmp_path):
        """A single-decode trace yields no ordered boundary (min_repeats=2), but
        the hot-kernel table still names the chain, so retrieval must still run
        and the recalled operator must reach the prompt and the Recipe."""
        src = tmp_path / "toy.py"
        src.write_text("def mlp(x): return act(gate_up(x))\n", encoding="utf-8")
        trace = tmp_path / "d.trace.json"
        _write_trace(
            trace,
            [
                {"cat": "kernel", "name": "Cijk_gate_up_gemm", "ts": 0, "dur": 30},
                {"cat": "kernel", "name": "act_and_mul_kernel silu", "ts": 40, "dur": 20},
            ],
        )
        knowledge = tmp_path / "knowledge"
        knowledge.mkdir()
        (knowledge / "gemm.md").write_text("Gate/up GEMM with SiLU uses `gemm_a16w16_gated`.\n", encoding="utf-8")
        captured = {}

        def fake_llm(prompt: str) -> str:
            captured["prompt"] = prompt
            return json.dumps(
                [
                    {
                        "name": "gate_up_epilogue",
                        "env_flag": "FUSED_GATE_UP",
                        "op_chain": "gate_up + act",
                        "candidate_kind": "integration",
                        "existing_operator": "gemm_a16w16_gated",
                        "priority": 0.9,
                    }
                ]
            )

        recipes = discover_recipes(
            _candidate_diag(),
            model_type="toy",
            framework="sglang",
            source_file=str(src),
            shapes={},
            trace_path=str(trace),
            llm_fn=fake_llm,
            knowledge_root=knowledge,
        )

        assert ordered_fusion_boundaries_from_trace(trace) == []
        assert "gemm_a16w16_gated" in captured["prompt"]
        assert recipes[0].candidate_kind == "integration"
        assert recipes[0].existing_operator == "gemm_a16w16_gated"
        assert recipes[0].to_dict()["existing_operator"] == "gemm_a16w16_gated"

    def test_max_fusions_is_configurable_and_reaches_the_prompt(self, tmp_path):
        src = tmp_path / "toy.py"
        src.write_text("def f(x): return x\n", encoding="utf-8")
        trace = tmp_path / "d.trace.json"
        _write_trace(trace, [{"cat": "kernel", "name": "mul", "ts": 0, "dur": 10}])
        captured = {}

        def fake_llm(prompt: str) -> str:
            captured["prompt"] = prompt
            return "[]"

        discover_recipes(
            _candidate_diag(),
            model_type="toy",
            framework="sglang",
            source_file=str(src),
            shapes={},
            trace_path=str(trace),
            llm_fn=fake_llm,
            max_fusions=3,
        )

        assert "identify up to 3 CONTIGUOUS op chains" in captured["prompt"]

    def test_max_fusions_default_is_bounded_and_env_overridable(self, tmp_path, monkeypatch):
        src = tmp_path / "toy.py"
        src.write_text("def f(x): return x\n", encoding="utf-8")
        trace = tmp_path / "d.trace.json"
        _write_trace(trace, [{"cat": "kernel", "name": "mul", "ts": 0, "dur": 10}])
        captured = {}

        def fake_llm(prompt: str) -> str:
            captured["prompt"] = prompt
            return "[]"

        def run():
            discover_recipes(
                _candidate_diag(),
                model_type="toy",
                framework="sglang",
                source_file=str(src),
                shapes={},
                trace_path=str(trace),
                llm_fn=fake_llm,
            )
            return captured["prompt"]

        monkeypatch.delenv("FORGE_MAX_FUSIONS", raising=False)
        assert "identify up to 4 CONTIGUOUS op chains" in run()

        monkeypatch.setenv("FORGE_MAX_FUSIONS", "6")
        assert "identify up to 6 CONTIGUOUS op chains" in run()

        monkeypatch.setenv("FORGE_MAX_FUSIONS", "not-a-number")
        assert "identify up to 4 CONTIGUOUS op chains" in run()

    def test_non_candidate_skips_llm(self, tmp_path):
        d = diagnose_from_shares({"gemm": 0.9}, busy_fraction_of_wall=0.8)
        called = {"n": 0}

        def fake_llm(prompt):
            called["n"] += 1
            return "[]"

        assert (
            discover_recipes(
                d,
                model_type="zaya",
                framework="sglang",
                source_file="/x.py",
                shapes={},
                trace_path="/x",
                llm_fn=fake_llm,
            )
            == []
        )
        assert called["n"] == 0  # not a candidate -> LLM never called
