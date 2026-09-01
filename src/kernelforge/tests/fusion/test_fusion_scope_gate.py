# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scope gate: a discovered fusion must fit inside ONE framework file.

A fusion is wired by REPLACING one call site in the source file the discovery
prompt embedded. A proposal claiming ops that file never performs therefore has
no wireable call site, and the campaign spent authoring it ends in an orphan
module -- which is what ``fused_symbol_invocation_evidence`` catches at the far
end of the pipeline, after the cost has been paid. This gate catches the same
defect before the campaign starts.

Provenance: a real Qwen3-14B-FP8 run proposed four recipes against vLLM's
``qwen3.py``. Two of them crossed a boundary and neither was wireable:

* ``qknorm_rope_kvcache`` folded in the KV-cache write. In vLLM v1 that happens
  inside the attention backend, so ``key_cache`` / ``slot_mapping`` are not
  names ``Qwen3Attention.forward`` can reach. It won the run: 37.16x
  microbench, SNR 52.1 dB, SERVING SMOKE OK -- and a framework edit that was one
  ``# noqa: F401`` import, for exactly zero end-to-end gain.
* ``reduce_act_mul_fp8_quant`` fused the MLP activation chain, which lives in
  ``qwen2.py`` (``qwen3.py`` only does ``from .qwen2 import Qwen2MLP``).

Its anchors were all present in the file, so an anchor-presence check would have
passed both. What separates them is the ops they claim versus the ops the file
performs.
"""

from __future__ import annotations

import json

from kernelforge.fusion.discover import parse_discovered_recipes
from kernelforge.fusion.locate import out_of_scope_terms

# The shape that matters: a model file that norms and applies RoPE, imports its
# MLP from a sibling module, and never touches the KV cache (the attention
# backend does that, several frames below).
_MODEL_SOURCE = """
from .other_model import OtherMLP as MyMLP


class MyAttention(nn.Module):
    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v))[0]


class MyDecoderLayer(nn.Module):
    def __init__(self):
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = MyMLP()
"""


def _proposal(name: str, ops: list[str], traits: list[str]) -> dict:
    """One discovery JSON object, minimal but complete enough to parse."""
    return {
        "name": name,
        "ops": ops,
        "traits": traits,
        "op_chain": name,
        "fusion_math": name,
        "eager_reference": "MyAttention.forward",
        "source_anchors": ["MyAttention.forward"],
        "priority": 0.9,
    }


def _survivors(tmp_path, *proposals) -> list[str]:
    """Run proposals through the real parser against a written-out source file."""
    source = tmp_path / "my_model.py"
    source.write_text(_MODEL_SOURCE, encoding="utf-8")
    recipes = parse_discovered_recipes(
        json.dumps(list(proposals)),
        model_type="my_model",
        framework="vllm",
        source_file=str(source),
        shapes={},
        category_shares=None,
    )
    return [r.pattern_id for r in recipes]


def test_a_fusion_folding_in_the_kv_cache_write_is_dropped(tmp_path):
    """The recipe that won the real run, and could not be wired anywhere."""
    survivors = _survivors(
        tmp_path,
        _proposal("qknorm_rope_kvcache", ["copy", "rmsnorm", "rope"], ["qk_norm", "kvcache"]),
    )
    assert survivors == []


def test_a_fusion_reaching_into_an_imported_module_is_dropped(tmp_path):
    """``MyMLP`` is imported, so its activation chain is not this file's to replace."""
    survivors = _survivors(
        tmp_path,
        _proposal("act_mul_quant", ["activation", "mul"], ["fp8", "quant"]),
    )
    assert survivors == []


def test_a_chain_wholly_inside_the_shown_file_survives(tmp_path):
    """QK-norm + RoPE is right there in ``MyAttention.forward`` -- keep it."""
    survivors = _survivors(
        tmp_path,
        _proposal("qk_norm_rope", ["rmsnorm", "rope"], ["qk_norm"]),
    )
    assert survivors == ["llm:qk_norm_rope"]


def test_two_wireable_chains_stay_two_separate_recipes(tmp_path):
    """Two modules that each fuse internally give two patches, never one.

    The loop attempts recipes one at a time and stops at the first KEEP, so
    "two patches" means two runs; what this gate guarantees is that each recipe
    stays self-contained rather than being merged into one unwireable proposal.
    """
    survivors = _survivors(
        tmp_path,
        _proposal("qk_norm_rope", ["rmsnorm", "rope"], ["qk_norm"]),
        _proposal("attn_out_oproj", ["copy", "mul"], []),
    )
    assert survivors == ["llm:qk_norm_rope", "llm:attn_out_oproj"]


def test_an_unreadable_source_is_not_judged():
    """No source is no evidence: the gate fails open like every other check."""
    assert out_of_scope_terms("", ["kvcache", "activation"]) == []


def test_a_term_with_no_unambiguous_spelling_is_not_judged():
    """``add`` / ``mul`` / ``copy`` / ``reduce`` take too many source shapes.

    A gate that fires on a spelling is worse than no gate, so these terms
    deliberately have no entry and can never trigger a drop.
    """
    assert out_of_scope_terms(_MODEL_SOURCE, ["add", "mul", "copy", "reduce"]) == []


def test_the_gate_reports_every_term_that_is_out_of_scope():
    """The log line names all of them, so the drop is diagnosable from one line."""
    assert out_of_scope_terms(_MODEL_SOURCE, ["rope", "kvcache", "activation", "moe"]) == [
        "kvcache",
        "activation",
        "moe",
    ]
