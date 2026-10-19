# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Settings prelude planning and source-splicing tests."""

from __future__ import annotations

import ast

import pytest

from hyperloom.orchestrator.deployment.settings_prelude import (
    _TORCH_API_TRANSLATIONS,
    SettingsPlan,
    apply_settings_prelude,
    plan_settings,
    render_prelude,
)


def test_classification_buckets() -> None:
    plan = plan_settings(
        {
            "HL_STALE": "1",
            "RUN_EVAL": "1",
            "PYTORCH_TUNABLEOP_ENABLED": "1",
            "HSA_ENABLE_SDMA": "0",
            "PYTORCH_TUNABLEOP_FILENAME": "/scratch/run/tuned.csv",
            "HL_FUSE_ATTN": "1",
        },
        unset_envs=("HL_STALE",),
    )

    assert plan.unset_env == ["HL_STALE"]
    assert plan.excluded == [{"name": "RUN_EVAL", "value": "1", "reason": "benchmark_control"}]
    assert plan.applied_api == ["torch.cuda.tunable.enable(True)"]
    assert plan.applied_env["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert plan.untranslated == [
        {"name": "HSA_ENABLE_SDMA", "value": "0", "reason": "pre_context_init"},
        {"name": "PYTORCH_TUNABLEOP_FILENAME", "value": "/scratch/run/tuned.csv", "reason": "unportable_absolute_path"},
    ]
    assert plan.applied_env["HL_FUSE_ATTN"] == "1"


def test_classification_is_case_insensitive_but_preserves_original_casing() -> None:
    plan = plan_settings({"Run_Eval": "1", "Hl_Fuse_Attn": "1"})

    assert plan.excluded == [{"name": "Run_Eval", "value": "1", "reason": "benchmark_control"}]
    assert plan.applied_env == {"Hl_Fuse_Attn": "1"}


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("PYTORCH_TUNABLEOP_ENABLED", "yes", "torch.cuda.tunable.enable(True)"),
        ("PYTORCH_TUNABLEOP_TUNING", "0", "torch.cuda.tunable.tuning_enable(False)"),
        (
            "PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS",
            "250",
            "torch.cuda.tunable.set_max_tuning_duration(250)",
        ),
        (
            "PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS",
            17,
            "torch.cuda.tunable.set_max_tuning_iterations(17)",
        ),
    ],
)
def test_tunableop_translates_to_api_calls(name: str, value: object, expected: str) -> None:
    assert name in _TORCH_API_TRANSLATIONS
    plan = plan_settings({name: value})
    assert plan.applied_api == [expected]
    assert plan.applied_env == {name: str(value)}


def test_tunableop_record_untuned_has_no_api_translation() -> None:
    # No real torch.cuda.tunable.record_untuned_enable exists; must fall through to tier 1.
    assert "PYTORCH_TUNABLEOP_RECORD_UNTUNED" not in _TORCH_API_TRANSLATIONS
    plan = plan_settings({"PYTORCH_TUNABLEOP_RECORD_UNTUNED": "1"})
    assert plan.applied_api == []
    assert plan.applied_env == {"PYTORCH_TUNABLEOP_RECORD_UNTUNED": "1"}


def test_render_prelude_is_empty_for_empty_plan() -> None:
    assert render_prelude(SettingsPlan()) == ""


def test_render_prelude_is_deterministic() -> None:
    first = plan_settings(
        {
            "HL_Z": "last",
            "PYTORCH_TUNABLEOP_TUNING": "false",
            "HL_A": "first",
        },
        unset_envs=("HL_UNSET_Z", "HL_UNSET_A"),
    )
    second = plan_settings(
        {
            "HL_A": "first",
            "PYTORCH_TUNABLEOP_TUNING": "false",
            "HL_Z": "last",
        },
        unset_envs=("HL_UNSET_A", "HL_UNSET_Z"),
    )

    assert render_prelude(first).encode() == render_prelude(second).encode()


def test_unset_envs_emit_pop_and_never_also_set() -> None:
    plan = plan_settings({"HL_STALE": "1"}, unset_envs=("HL_STALE",))
    prelude = render_prelude(plan)

    assert plan.unset_env == ["HL_STALE"]
    assert "HL_STALE" not in plan.applied_env
    assert '_hl_os.environ.pop(_hl_k, None)' in prelude
    assert '_hl_os.environ[_hl_k] = _hl_v' not in prelude


def test_prelude_composes_env_and_api_together() -> None:
    plan = plan_settings({"HL_FUSE_ATTN": "1", "PYTORCH_TUNABLEOP_ENABLED": "1"})
    rendered = render_prelude(plan)
    result = apply_settings_prelude(b"def hyperloom_optimize(model):\n    return model\n", plan)

    ast.parse(result)
    assert '("HL_FUSE_ATTN", "1")' in rendered
    assert "torch.cuda.tunable.enable(True)" in rendered
    assert rendered.index("for _hl_k, _hl_v") < rendered.index("def _hl_apply_runtime_settings")


def test_apply_prelude_preserves_trailing_source_bytes() -> None:
    source = b"def hyperloom_optimize(model):\n    return model\n"
    result = apply_settings_prelude(source, plan_settings({"HL_FUSE_ATTN": "1"}))

    assert result.endswith(source)


def test_apply_prelude_rejects_module_without_hook() -> None:
    with pytest.raises(ValueError, match="valid module"):
        apply_settings_prelude(b"def other(model):\n    return model\n", plan_settings({"HL_FUSE_ATTN": "1"}))


def test_apply_prelude_inserts_after_docstring_and_future_import() -> None:
    source = b'"""module doc"""\nfrom __future__ import annotations\n\ndef hyperloom_optimize(model):\n    return model\n'
    result = apply_settings_prelude(source, plan_settings({"HL_FUSE_ATTN": "1"}))
    text = result.decode()

    assert text.index('"""module doc"""') < text.index("from __future__ import annotations")
    assert text.index("from __future__ import annotations") < text.index("# --- hyperloom accepted settings")
    ast.parse(result)


def test_credential_shaped_name_raises() -> None:
    with pytest.raises(ValueError, match="EXPERIMENT_API_KEY"):
        plan_settings({"EXPERIMENT_API_KEY": "not-written"})
