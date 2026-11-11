# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Settings prelude planning and source-splicing tests."""

from __future__ import annotations

import ast
import os
import sys
import types

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
        {"name": "PYTORCH_TUNABLEOP_FILENAME", "value": "/scratch/run/tuned.csv", "reason": "unportable_absolute_path"},
    ]
    assert plan.applied_env["HL_FUSE_ATTN"] == "1"
    assert plan.applied_env["HSA_ENABLE_SDMA"] == "0"


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


def test_render_prelude_includes_import_timing_guard_for_unprotected_env() -> None:
    # TORCHINDUCTOR_MAX_AUTOTUNE (and similar) have no _TORCH_API_TRANSLATIONS entry, so
    # they rely purely on being set before torch._inductor/_dynamo first get imported.
    # The rendered prelude must guard against a too-late import instead of silently
    # measuring/shipping a setting that never took effect.
    plan = plan_settings({"TORCHINDUCTOR_MAX_AUTOTUNE": "1"})
    rendered = render_prelude(plan)

    assert "torch._inductor" in rendered
    assert "torch._dynamo" in rendered
    assert "raise RuntimeError" in rendered
    ast.parse(rendered)


def test_render_prelude_omits_import_timing_guard_when_fully_api_protected() -> None:
    # Every env entry here has a matching _TORCH_API_TRANSLATIONS poke, so nothing is at
    # risk from import timing -- the guard would be a pure false positive.
    plan = plan_settings({"PYTORCH_TUNABLEOP_ENABLED": "1"})
    rendered = render_prelude(plan)

    assert "raise RuntimeError" not in rendered


def test_render_prelude_omits_import_timing_guard_when_env_is_empty() -> None:
    plan = plan_settings({}, unset_envs=("HL_STALE",))
    rendered = render_prelude(plan)

    assert "raise RuntimeError" not in rendered


def test_import_timing_guard_raises_only_when_inductor_or_dynamo_already_imported() -> None:
    plan = plan_settings({"TORCHINDUCTOR_MAX_AUTOTUNE": "1"})
    module_bytes = apply_settings_prelude(
        b"def hyperloom_optimize(model):\n    return model\n", plan, "hyperloom_optimize"
    )
    namespace: dict[str, object] = {"__name__": "generated_hook_under_test"}

    sys.modules["torch._inductor"] = types.ModuleType("torch._inductor")
    try:
        with pytest.raises(RuntimeError, match="generated_hook_under_test was imported after"):
            exec(compile(module_bytes, "<generated>", "exec"), namespace)
    finally:
        del sys.modules["torch._inductor"]


@pytest.mark.parametrize("initialized", [True, False])
def test_import_timing_guard_raises_once_the_gpu_is_initialized(
    monkeypatch: pytest.MonkeyPatch, initialized: bool
) -> None:
    plan = plan_settings({"HSA_ENABLE_SDMA": "0"})
    module_bytes = apply_settings_prelude(b"def apply(model):\n    return model\n", plan, "apply")
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: initialized))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("HSA_ENABLE_SDMA", raising=False)
    namespace: dict[str, object] = {"__name__": "encoder_opt"}

    if initialized:
        with pytest.raises(RuntimeError, match="Move its import above `import torch`"):
            exec(compile(module_bytes, "<generated>", "exec"), namespace)
    else:
        exec(compile(module_bytes, "<generated>", "exec"), namespace)
        assert os.environ["HSA_ENABLE_SDMA"] == "0"


def test_import_timing_guard_does_not_raise_when_nothing_pre_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = plan_settings({"TORCHINDUCTOR_MAX_AUTOTUNE": "1"})
    module_bytes = apply_settings_prelude(
        b"def hyperloom_optimize(model):\n    return model\n", plan, "hyperloom_optimize"
    )
    assert "torch._inductor" not in sys.modules
    assert "torch._dynamo" not in sys.modules
    monkeypatch.delenv("TORCHINDUCTOR_MAX_AUTOTUNE", raising=False)

    namespace: dict[str, object] = {"__name__": "generated_hook_under_test"}
    exec(compile(module_bytes, "<generated>", "exec"), namespace)  # must not raise

    assert os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1"


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
    assert "_hl_os.environ.pop(_hl_k, None)" in prelude
    assert "_hl_os.environ[_hl_k] = _hl_v" not in prelude


def test_prelude_composes_env_and_api_together() -> None:
    plan = plan_settings({"HL_FUSE_ATTN": "1", "PYTORCH_TUNABLEOP_ENABLED": "1"})
    rendered = render_prelude(plan)
    result = apply_settings_prelude(b"def hyperloom_optimize(model):\n    return model\n", plan, "hyperloom_optimize")

    compile(result, "<string>", "exec")
    assert '("HL_FUSE_ATTN", "1")' in rendered
    assert "torch.cuda.tunable.enable(True)" in rendered
    assert rendered.index("for _hl_k, _hl_v") < rendered.index("def _hl_apply_runtime_settings")


def test_apply_prelude_preserves_trailing_source_bytes() -> None:
    source = b"def hyperloom_optimize(model):\n    return model\n"
    result = apply_settings_prelude(source, plan_settings({"HL_FUSE_ATTN": "1"}), "hyperloom_optimize")

    assert result.endswith(source)


def test_apply_prelude_rejects_module_without_hook() -> None:
    with pytest.raises(ValueError, match="valid module"):
        apply_settings_prelude(
            b"def other(model):\n    return model\n", plan_settings({"HL_FUSE_ATTN": "1"}), "hyperloom_optimize"
        )


def test_apply_prelude_inserts_after_docstring_and_future_import() -> None:
    source = (
        b'"""module doc"""\nfrom __future__ import annotations\n\ndef hyperloom_optimize(model):\n    return model\n'
    )
    result = apply_settings_prelude(source, plan_settings({"HL_FUSE_ATTN": "1"}), "hyperloom_optimize")
    text = result.decode()

    assert text.index('"""module doc"""') < text.index("from __future__ import annotations")
    assert text.index("from __future__ import annotations") < text.index("# --- hyperloom accepted settings")
    compile(result, "<string>", "exec")


def test_credential_shaped_name_raises() -> None:
    with pytest.raises(ValueError, match="EXPERIMENT_API_KEY"):
        plan_settings({"EXPERIMENT_API_KEY": "not-written"})


@pytest.mark.parametrize(
    ("name", "char"),
    [("formfeed", "\x0c"), ("vtab", "\x0b"), ("nel", "\x85"), ("line-separator", "\u2028")],
)
def test_a_docstring_line_break_python_does_not_count_keeps_the_prelude_in_place(name: str, char: str) -> None:
    """str.splitlines breaks on these and ast does not; the two must not disagree."""
    source = (
        f'"""Doc{char}line."""\n'
        "from __future__ import annotations\n\n\ndef hyperloom_optimize(model):\n    return model\n"
    ).encode()
    out = apply_settings_prelude(source, plan_settings({"TORCHINDUCTOR_FREEZING": "1"}), "hyperloom_optimize")
    # compile, not ast.parse: a prelude spliced above `from __future__` still parses.
    compile(out, "<string>", "exec")
    assert b"TORCHINDUCTOR_FREEZING" in out


@pytest.mark.parametrize(
    "name",
    ["AWS_ACCESS_KEY_ID", "MY_PRIVATE_KEY", "GATEWAY_SIGNATURE", "SOME_AUTH"],
    ids=["access-key-id", "private-key", "signature", "auth"],
)
def test_a_credential_the_log_redaction_check_misses_is_still_refused(name: str) -> None:
    """These pass is_secret_shaped_env_name, so the looser check would bake them into the module.

    The failure that matters is the pair: AWS_SECRET_ACCESS_KEY is refused either way, so
    under the looser check an operator passing both would get a refusal for one half while
    the other half shipped.
    """
    with pytest.raises(ValueError, match="require an external credential"):
        plan_settings({name: "value"})


@pytest.mark.parametrize(
    "name",
    [
        "PYTORCH_HIP_ALLOC_CONF",
        "TORCHINDUCTOR_FREEZING",
        "OMP_NUM_THREADS",
        "PYTORCH_TUNABLEOP_ENABLED",
        "HIP_VISIBLE_DEVICES",
        "TORCH_BLAS_PREFER_HIPBLASLT",
    ],
)
def test_the_settings_this_feature_carries_are_not_mistaken_for_credentials(name: str) -> None:
    """The stricter check must not start refusing the knobs the prelude exists to carry."""
    plan_settings({name: "1"})


import hashlib
from hyperloom.orchestrator.deployment.export import export_custom_inference


def test_export_bakes_accepted_switches_into_module(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"HL_FUSE_ATTN": "1", "HL_HOIST": "1"}

    result = export_custom_inference(session, state)

    exported = session / "deployment" / hook.name
    assert result["status"] == "exported"
    assert "HL_FUSE_ATTN" in exported.read_text()
    assert "HL_HOIST" in exported.read_text()
    assert hook.read_bytes() == b"def hyperloom_optimize(model):\n    return model\n"
    assert result["files"][hook.name] == hashlib.sha256(exported.read_bytes()).hexdigest()
    assert result["settings"]["base_sha256"] == hashlib.sha256(hook.read_bytes()).hexdigest()


def test_export_excludes_benchmark_control_envs(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"RUN_EVAL": "false", "CONC": "1"}

    result = export_custom_inference(session, state)

    exported_text = (session / "deployment" / hook.name).read_text()
    assert result["status"] == "exported"
    assert {"name": "RUN_EVAL", "value": "false", "reason": "benchmark_control"} in result["settings"]["excluded"]
    assert {"name": "CONC", "value": "1", "reason": "benchmark_control"} in result["settings"]["excluded"]
    assert "RUN_EVAL" not in exported_text
    assert "CONC" not in exported_text


def test_export_refuses_credential_shaped_setting(workload):
    _, _, session, state = workload
    state["current_best"]["extra_envs"] = {"HF_TOKEN": "x"}

    result = export_custom_inference(session, state)

    assert result["status"] == "incomplete"
    assert "HF_TOKEN" in result["reasons"][0]


def test_export_reports_untranslatable_settings(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"PYTORCH_TUNABLEOP_FILENAME": "/scratch/run/tuned.csv"}

    result = export_custom_inference(session, state)

    exported_text = (session / "deployment" / hook.name).read_text()
    assert result["status"] == "exported"
    assert {
        "name": "PYTORCH_TUNABLEOP_FILENAME",
        "value": "/scratch/run/tuned.csv",
        "reason": "unportable_absolute_path",
    } in result["settings"]["untranslated"]
    assert "PYTORCH_TUNABLEOP_FILENAME" not in exported_text
    assert len(result["warnings"]) == 1
    assert "PYTORCH_TUNABLEOP_FILENAME" in result["warnings"][0]


def test_export_ships_settings_read_at_gpu_init(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}

    result = export_custom_inference(session, state)

    exported_text = (session / "deployment" / hook.name).read_text()
    assert result["status"] == "exported"
    assert result["settings"]["applied_env"] == {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}
    assert result["settings"]["untranslated"] == []
    assert "PYTORCH_HIP_ALLOC_CONF" in exported_text
    assert "is_initialized()" in exported_text
    assert result["warnings"] == []
