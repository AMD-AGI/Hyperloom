"""Faithful per-arg dtype propagation.

TraceLens records each kernel arg's dtype INLINE in the analysis.md ``Args``
column ("(64,5120) bf16"). HL parsed the shape but left ``input_dtypes`` empty,
so the GEAK harness could not allocate correct-dtype tensors (fp8 weight vs bf16
activation). These tests verify HL now surfaces the real, ordered per-arg dtypes.
"""
import sys
from argparse import Namespace

sys.argv = ["x"]
import tracelens_analysis as tla


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
