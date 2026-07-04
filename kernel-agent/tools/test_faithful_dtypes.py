"""Faithful per-arg dtype propagation.

TraceLens records each kernel arg's dtype INLINE in the analysis.md ``Args``
column ("(64,5120) bf16"). HL parsed the shape but left ``input_dtypes`` empty,
so the GEAK harness could not allocate correct-dtype tensors (fp8 weight vs bf16
activation). These tests verify HL now surfaces the real, ordered per-arg dtypes.
"""
import sys
from argparse import Namespace

sys.argv = ["x"]
import tracelens_analysis as tla  # noqa: E402 - module reads argv at import time.


def test_split_shape_dtype():
    assert tla._split_shape_dtype("(64,5120) bf16") == ("(64,5120)", "bf16")
    assert tla._split_shape_dtype("(161,512,5120) fp8") == ("(161,512,5120)", "fp8")
    assert tla._split_shape_dtype("()") == ("()", "")          # scalar, arity kept
    assert tla._split_shape_dtype("(64,9) int") == ("(64,9)", "int")
    assert tla._split_shape_dtype("(1792,) bf16") == ("(1792,)", "bf16")
    assert tla._split_shape_dtype({"shape": "(8,8) fp16"}) == ("(8,8)", "fp16")


def test_dtypes_from_shapes_aligned():
    shapes = ["(64,5120) bf16", "(161,512,5120) fp8", "()", "(64,9) int"]
    assert tla._dtypes_from_shapes(shapes) == ["bf16", "fp8", "", "int"]


def test_dtypes_from_shapes_none_present():
    # no dtype on any entry -> [] (preserves prior empty-default behaviour)
    assert tla._dtypes_from_shapes(["(1,2)", "(3,4)"]) == []
    assert tla._dtypes_from_shapes([]) == []


def _args():
    return Namespace(framework="sglang", model_name="glm", target_platform="MI300X",
                     analysis_mode="inference", runtime_env="local")


def test_enrich_populates_input_dtypes_from_inline_shapes():
    cand = {"name": "fused_moe",
            "shapes": ["(64,5120) bf16", "(161,512,5120) fp8", "(64,9) int"]}
    tla.enrich_candidates_with_runtime_metadata([cand], _args())
    assert cand["input_dtypes"] == ["bf16", "fp8", "int"]


def test_enrich_preserves_explicit_dtypes():
    cand = {"name": "x", "shapes": ["(1,2) bf16"], "input_dtypes": ["fp16"]}
    tla.enrich_candidates_with_runtime_metadata([cand], _args())
    assert cand["input_dtypes"] == ["fp16"]


def test_enrich_no_dtype_stays_empty():
    cand = {"name": "y", "shapes": ["(1,2)", "(3,4)"]}
    tla.enrich_candidates_with_runtime_metadata([cand], _args())
    assert cand["input_dtypes"] == []


# --- analysis_remapped.md parse tolerance (extra/inserted column) ---
import tracelens_skill_runner as tlr  # noqa: E402


_REMAP_TABLE = (
    "<!-- impact-begin kind=p_item category=gemm mid=4.0 low=2.0 high=8.0 -->\n"
    "\n## Detailed Analysis\n\n### Compute Kernel Insights\n\n"
    "<!-- reasoning-candidate tier=compute rank=1 -->\n"
    "#### 🔴 P1: Inserted Source Path column (Tensile)\n\n"
    "**Identification:** stub\n**Data:**\n"
    # NOTE: 'Source Path' INSERTED between 'Kernel Path' and 'Time (ms)'.
    "| Operation | Args | Kernel Path | Source Path | Time (ms) | %E2E | Count | "
    "FLOPS/Byte | Efficiency | Bound |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    "| stub_op | (64,5120) bf16 | /x/k.cu | /x/launch.py | 1.0 | 5 | 10 | 1000 | "
    "40% of 708 TFLOPS | compute-bound |\n"
    "**Reasoning for Slowdown:** stub\n**Resolution:** stub\n"
)


def test_parse_tolerates_inserted_source_path_column(tmp_path):
    md = tmp_path / "analysis_remapped.md"
    md.write_text(_REMAP_TABLE, encoding="utf-8")
    cands = tlr.parse_analysis_md(md, top_k=10)
    assert len(cands) == 1
    c = cands[0]
    assert c["name"] == "stub_op"
    # name-keyed read still maps the right cells despite the inserted column:
    # the 'Kernel Path' cell is captured (under tracelens_launcher_path) and the
    # Args shape parses — neither is corrupted by the extra 'Source Path' column.
    assert c.get("tracelens_launcher_path") == "/x/k.cu"
    assert c["shapes"] == ["(64,5120) bf16"]
