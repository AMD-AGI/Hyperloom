"""Tests for ``orchestrator.score_priors`` — IMPL-CHECKLIST §3.31‒3.38."""
from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.score_priors import (
    INITIAL_PRIORS,
    ModelClass,
    classify_model,
    prior_for,
)


# ---------------------------------------------------------------------------
# Initial-prior table snapshot
# ---------------------------------------------------------------------------
def test_dense_priors_match_design_table():
    row = INITIAL_PRIORS[ModelClass.DENSE]
    assert row["backends"] == 3.0
    assert row["params"] == 5.0
    assert row["kernel-opt"] == 8.0
    assert row["torch.compile"] == 7.0
    assert row["sweep"] == 1.0


def test_moe_mla_priors_match_design_table():
    row = INITIAL_PRIORS[ModelClass.MOE_MLA]
    assert row["backends"] == 9.0
    assert row["params"] == 6.0
    assert row["kernel-opt"] == 2.0
    assert row["torch.compile"] == 0.0


def test_moe_swa_priors_match_design_table():
    row = INITIAL_PRIORS[ModelClass.MOE_SWA]
    assert row["backends"] == 8.0
    assert row["params"] == 7.0


def test_moe_mla_nsa_top_score_for_backends():
    row = INITIAL_PRIORS[ModelClass.MOE_MLA_NSA]
    assert row["backends"] == 10.0


def test_unknown_class_provides_neutral_prior():
    row = INITIAL_PRIORS[ModelClass.UNKNOWN]
    assert row["backends"] == 5.0
    assert row["kernel-opt"] == 5.0


# ---------------------------------------------------------------------------
# prior_for
# ---------------------------------------------------------------------------
def test_prior_for_returns_table_value():
    assert prior_for(ModelClass.DENSE, "kernel-opt") == 8.0


def test_prior_for_accepts_string_class():
    assert prior_for("dense", "kernel-opt") == 8.0


def test_prior_for_unknown_action_returns_default():
    assert prior_for(ModelClass.DENSE, "totally-new-action") == 1.0


def test_prior_for_underscore_variant_resolves():
    """``prior_for`` should accept both 'kernel-opt' and 'kernel_opt'."""
    assert prior_for(ModelClass.DENSE, "kernel_opt") == 8.0


def test_prior_for_hyphen_variant_resolves():
    # The table key is 'torch.compile'; underscore lookup should fail back
    # to the default. But hyphen variants should work where applicable.
    assert prior_for(ModelClass.DENSE, "torch.compile") == 7.0


def test_prior_for_unknown_class_falls_back_to_unknown_row():
    assert prior_for("totally-mysterious", "backends") == 5.0


def test_prior_for_custom_default():
    assert prior_for(ModelClass.DENSE, "nonexistent", default=99.0) == 99.0


# ---------------------------------------------------------------------------
# classify_model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path,expected",
    [
        ("/srv/models/llama-3-8b-instruct", ModelClass.DENSE),
        ("openai/gpt-oss-20b", ModelClass.DENSE),
        ("Qwen/Qwen2-7B-Dense", ModelClass.DENSE),
        ("Qwen/qwen-dense-7b", ModelClass.DENSE),
        ("microsoft/phi-3-mini", ModelClass.DENSE),
        ("mistralai/Mistral-7b-instruct", ModelClass.DENSE),
    ],
)
def test_classify_dense(path: str, expected: ModelClass):
    assert classify_model(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "deepseek-ai/DeepSeek-V2-Lite",
        "deepseek-ai/DeepSeek-V3-Coder",
        "/models/DeepSeek_V3_0324",
    ],
)
def test_classify_moe_mla(path: str):
    assert classify_model(path) == ModelClass.MOE_MLA


@pytest.mark.parametrize(
    "path",
    [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen2-MoE-A2.7B",
        "Qwen/qwen-moe-tiny",
    ],
)
def test_classify_moe_swa(path: str):
    assert classify_model(path) == ModelClass.MOE_SWA


@pytest.mark.parametrize(
    "path",
    [
        "moonshot-ai/Kimi-Plus",
        "/models/some_NSA_v2",
    ],
)
def test_classify_moe_mla_nsa(path: str):
    assert classify_model(path) == ModelClass.MOE_MLA_NSA


def test_classify_empty_returns_unknown():
    assert classify_model("") == ModelClass.UNKNOWN


def test_classify_unknown_returns_unknown():
    assert classify_model("anthropic/claude-future-7") == ModelClass.UNKNOWN


def test_classify_is_case_insensitive():
    assert classify_model("DEEPSEEK-V3") == ModelClass.MOE_MLA
