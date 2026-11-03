"""Kernel categories from kernel names, one ordered rule list.

A rule is (category, pattern, unless): the first rule whose pattern matches
the lower-cased demangled name, and whose `unless` (if any) does not, names
the category. Order matters where names carry several hints:

- collectives first (`limits.COLLECTIVE`, the classifier every ruppell output
  uses): FlashInfer's `trtllm_mnnvl_allreduce` and vLLM's
  `cross_device_reduce` would otherwise read as attention or reduce;
- copies and fills next (`__amd_rocclr_copyBuffer`, `memcpy`, `memset`);
- Triton's generated kernels (`triton_` prefix: Inductor's `triton_poi_`,
  `triton_red_`, `triton_per_`), before any rule reads `add` or `sum` in
  their fused names;
- sampling before MoE (`TopKTopPSamplingFromProbKernel` is a sampler, not
  MoE routing);
- MoE before GEMM and attention (`fused_moe`, `fmoe`, `moe::dev::*`, and
  `topk` as the routing step: the rule cannot tell an attention indexer's
  top-k from MoE routing, so `_dsv4_topk_kernel` lands in MoE);
- attention before GEMM (`deep_gemm::sm100_paged_mqa_logits` is attention
  scores; FlashInfer's gated-delta-net kernels are built on cutlass);
- norm before elementwise and quantization (`_q_kv_norm_quant_kernel`,
  `vectorized_layer_norm_kernel`), unless the name is a GEMM
  (`deep_gemm::sm100_tf32_hc_prenorm_gemm_impl`);
- quantization (`mxfp8_quantize`, `*_quant*`) is elementwise unless the name
  is a GEMM (`kernel_cutlass_...quantization...` would otherwise match
  `cutlass`);
- GEMM (`gemm`, `cutlass`, `cublas`, `nvjet`, hipBLASLt `Cijk_`, `bmm`,
  `matmul`, `_mm`, `*linear_kernel`);
- elementwise, then reduce;
- anything else is `other`. A hand-written Triton kernel is named after its
  Python function (`_zero_kv_blocks_kernel`) and cannot be told apart.

`tests/test_birdeye_offline.py` checks the rules on kernel names taken from
real traces (B200 vLLM runs, the PyTorch demo, MI350 DSV4.1 runs).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ruppell.limits import COLLECTIVE

CATEGORIES = ["GEMM", "attention", "MoE", "norm", "elementwise", "reduce", "comm", "memcpy", "memset", "triton",
              "other"]

_GEMM_HINT = r"gemm|matmul|cublas|nvjet|cijk_|_mm_|_mm$|bmm_"

RULES: List[Tuple[str, re.Pattern, Optional[re.Pattern]]] = [
    ("comm", COLLECTIVE, None),
    ("memcpy", re.compile(r"memcpy|copybuffer|memory_copy|^copy_(dtod|htod|dtoh)"), None),
    ("memset", re.compile(r"memset|fillbuffer"), None),
    ("triton", re.compile(r"^triton_"), None),
    ("reduce", re.compile(r"sampling|sampler"), None),
    ("MoE", re.compile(r"moe|fmoe|grouped_gemm|groupedgemm|topk|top_k|expert"), None),
    ("attention", re.compile(
        r"fmha|flash_?attn|flash_fwd|flash_bwd|flashattn|flashmla|flash_mla|sdpa|scaled_dot_product|"
        r"paged_?attention|paged_attn|unified_attention|attention|_attn|attn_|(^|[^a-z])mla|_mla|mla_|mha_|_mha|"
        r"mqa|gqa|batchdecode|batchprefill|decode_attention|prefill_attention|sparse_attn|"
        r"reshape_and_cache|concat_and_cache|delta_rule|gated_delta|flashinfergdn"), None),
    ("norm", re.compile(r"rms_?norm|layer_?norm|group_?norm|batch_?norm|instance_?norm|norm"),
     re.compile(r"gemm")),
    ("elementwise", re.compile(r"quantiz|_quant|quant_"), re.compile(_GEMM_HINT)),
    ("GEMM", re.compile(_GEMM_HINT + r"|cutlass|hipblaslt|wgmma|xmma|_linear_kernel|linear_kernel|"
                        r"splitkreduce|tensorop|^cijk|mfma_gemm"), None),
    ("elementwise", re.compile(
        r"elementwise|vectorized|unrolled|act_and_mul|silu|gelu|swiglu|relu|sigmoid|tanh|rotary|(^|_)rope|rope_|"
        r"_rope|cast|copy_kernel|direct_copy|fill|index_select|indexselect|gather|scatter|"
        r"add_kernel|mul_kernel|_add_|_mul_"), None),
    ("reduce", re.compile(r"reduce|softmax|argmax|argmin|cumsum|scan|sort|radix|mean|_sum|sum_"), None),
]


def categorize(name: str) -> str:
    """The category of one kernel (its demangled name, full or short)."""
    s = (name or "").strip().lower()
    if s.startswith("void "):
        s = s[5:]
    for cat, pat, unless in RULES:
        if pat.search(s) and not (unless is not None and unless.search(s)):
            return cat
    return "other"


def categorize_many(names: List[str]) -> List[str]:
    cache: Dict[str, str] = {}
    out = []
    for n in names:
        c = cache.get(n)
        if c is None:
            c = cache[n] = categorize(n)
        out.append(c)
    return out


# aten ops (the CPU op tree, or a PyTorch profiler trace) whose category is
# the op's own, as TraceLens's categorizer does it; any other op takes its
# kernels' category
ATEN_CATEGORY = [
    ("GEMM", re.compile(r"^aten::(mm|addmm|bmm|baddbmm|addbmm|matmul|linear|_scaled_mm|_grouped_mm|"
                        r"_scaled_grouped_mm|einsum|tensordot)$")),
    ("attention", re.compile(r"scaled_dot_product|flash_attention|efficient_attention|cudnn_attention|"
                             r"_flash_attention|unified_attention|attention")),
    ("norm", re.compile(r"^aten::(native_)?(layer_norm|group_norm|batch_norm|rms_norm|_fused_rms_norm)|rms_norm")),
    ("MoE", re.compile(r"moe")),
]


def op_category(op: Optional[str]) -> Optional[str]:
    """The category an op name alone decides (aten GEMMs, attention, norms,
    MoE ops; registered ops named `namespace::op` only, never a module class
    such as `DeepseekV4FlashMLAAttention`), or None to fall back to its
    kernels' categories."""
    if not op or "::" not in op:
        return None
    for cat, pat in ATEN_CATEGORY:
        if pat.search(op):
            return cat
    return None
