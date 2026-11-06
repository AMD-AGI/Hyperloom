# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A tuning value a patch introduces reaches explore as a setting it sweeps, not as the author's fixed pick."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors._integrate_refusals import (
    UNTUNABLE_VALUE_ADVISORY,
    _untunable_value_refusal,
)
from hyperloom.orchestrator.actions.executors._framework_switch_manifest import (
    MAX_SETTING_VALUES,
    SETTINGS_KEY,
    parse_settings,
    setting_variants,
    untunable_values,
)
from hyperloom.orchestrator.actions.executors.explore import (
    _framework_lever_attributions,
    _grid_variants_from_payload,
    framework_lever_grid,
)
from hyperloom.orchestrator.loop._integrate_promotion import IntegratePromotionMixin
from hyperloom.orchestrator.prompts.specialist_prompt_builder import SpecialistPromptInputs, _section_output_protocol
from hyperloom.orchestrator.specialists.domains import get_domain
from hyperloom.orchestrator.state.shared_state import SharedState

from .test_framework_switch_manifest import _run_rewrite_integrate

# The added lines of the tile patch a GLM-5.3 run kept: the 1024-token cutoff and the new heads setting were the
# author's picks, and explore never saw either.
_KEPT_TILE_PATCH = """\
--- a/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py
+++ b/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py
@@ -1,1 +1,4 @@
+_FP8_WIDE_HEAD_TILE_MIN_TOKENS = 1024
+def _fp8_heads_per_block(seq: int, num_heads: int) -> int:
+    if seq < _FP8_WIDE_HEAD_TILE_MIN_TOKENS:
+        return 16
--- a/python/sglang/srt/environ.py
+++ b/python/sglang/srt/environ.py
@@ -1,1 +1,2 @@
+    SGLANG_DSA_TILELANG_FP8_PREFILL_HEADS_PER_BLOCK = EnvInt(None)
"""

_THRESHOLD_PATCH = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,3 @@
+_WIDE_TILE_MIN_TOKENS = 1024
 def f():
-    return 1
+    return 2
"""

_SETTING_PATCH = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,4 @@
+import os
+_WIDE_TILE_MIN_TOKENS = int(os.environ.get("HL_WIDE_TILE_MIN_SEQS", "1024"))
 def f():
-    return 1
+    return 2
"""

_SETTING = {"setting": "hl_wide_tile_min_seqs", "values": ["1", "4096"], "target": "src.py", "evidence": "decode"}


@pytest.fixture(autouse=True)
def _allowlist_tmp_framework_roots(monkeypatch, tmp_path):
    """Let integrate_patch treat the test's temp checkout as framework source."""
    from .conftest import patch_integrate_patch_roots

    patch_integrate_patch_roots(monkeypatch, tmp_path)


def _patch(tmp_path: Path, body: str, name: str = "001.patch") -> list[Path]:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return [path]


def _diff(target: str, *added: str) -> str:
    return f"--- a/{target}\n+++ b/{target}\n@@ -1,1 +1,{len(added)} @@\n" + "".join(f"+{line}\n" for line in added)


# what counts as a fixed tuning value


def test_the_kept_tile_patch_fixes_both_of_its_tuning_values(tmp_path):
    patches = _patch(tmp_path, _KEPT_TILE_PATCH)

    assert untunable_values(patches, declared=set()) == [
        "SGLANG_DSA_TILELANG_FP8_PREFILL_HEADS_PER_BLOCK",
        "_FP8_WIDE_HEAD_TILE_MIN_TOKENS",
    ]
    assert untunable_values(patches, declared={"SGLANG_DSA_TILELANG_FP8_PREFILL_HEADS_PER_BLOCK"}) == [
        "_FP8_WIDE_HEAD_TILE_MIN_TOKENS"
    ]


@pytest.mark.parametrize(
    "line",
    [
        "SPLIT_KV_THRESHOLD = 8192",
        "_MAX_BATCH: int = 256  # measured on MI355X",
        "_TILE_SIZES = (16, 32, 64)",
        '    min_tokens = int(os.environ.get("HL_MIN_TOKENS", "4096"))',
        "    HL_RESERVE_MB_PER_TOKEN = EnvFloat(1.5)",
    ],
)
def test_a_threshold_constant_or_numeric_setting_is_a_fixed_tuning_value(tmp_path, line):
    assert untunable_values(_patch(tmp_path, _diff("pkg/kernel.py", line)), declared=set())


@pytest.mark.parametrize(
    "line",
    [
        "_LOG2E = 1.4426950408889634",
        "    _TILE_MIN_TOKENS = 1024",
        "_WIDE_TILE_MIN_TOKENS = envs.SGLANG_WIDE_TILE_MIN_TOKENS.get()",
        '    if os.environ.get("HL_FUSED_INTER", "0") == "1":',
        '    on = os.getenv("HL_FAST_PATH", "1")',
        "    HL_FAST_PATH = EnvBool(False)",
    ],
)
def test_facts_gates_and_settings_already_read_are_not_fixed_tuning_values(tmp_path, line):
    assert untunable_values(_patch(tmp_path, _diff("pkg/kernel.py", line)), declared=set()) == []


@pytest.mark.parametrize("target", ["pkg/tests/test_kernel.py", "pkg/test_kernel.py", "pkg/kernel.cpp"])
def test_only_runtime_python_is_scanned(tmp_path, target):
    assert untunable_values(_patch(tmp_path, _diff(target, "SPLIT_KV_THRESHOLD = 8192")), declared=set()) == []


def test_an_unreadable_patch_fixes_nothing(tmp_path):
    assert untunable_values([tmp_path / "missing.patch"], declared=set()) == []


# the refusal


def test_an_enablement_round_or_benchmark_owned_env_is_never_refused(tmp_path):
    patches = _patch(tmp_path, _diff("pkg/kernel.py", '    n = int(os.getenv("HL_SPLITS", "8"))'))

    assert _untunable_value_refusal({}, patch_paths=patches, tunable_settings=[]) == "HL_SPLITS"
    assert _untunable_value_refusal({"enablement": True}, patch_paths=patches, tunable_settings=[]) == ""
    owned = {"base_extra_envs": {"HL_SPLITS": "4"}}
    assert _untunable_value_refusal(owned, patch_paths=patches, tunable_settings=[]) == ""
    declared = parse_settings([{"setting": "HL_SPLITS", "values": ["4", "16"]}], reserved=set())
    assert _untunable_value_refusal({}, patch_paths=patches, tunable_settings=declared) == ""


@pytest.mark.asyncio
async def test_a_hardcoded_threshold_is_refused_before_the_bench(tmp_path, monkeypatch):
    result, repo, _, legs = await _run_rewrite_integrate(
        tmp_path, monkeypatch, delta_pct=8.0, switches=[], patch_body=_THRESHOLD_PATCH
    )

    assert (result["status"], result["error_class"]) == ("apply_failed", "tuning_value_hardcoded")
    assert result["error"] == "patch fixes tuning value(s) explore cannot change: _WIDE_TILE_MIN_TOKENS"
    assert SETTINGS_KEY in result["advisory"]
    assert legs == []
    assert (repo / "src.py").read_text().endswith("return 1\n")


# a declared setting rides the verdict to explore


@pytest.mark.asyncio
async def test_a_declared_setting_is_benched_at_its_default_and_registered_with_a_keep(tmp_path, monkeypatch):
    result, _, benched_envs, _ = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        switches=[],
        patch_body=_SETTING_PATCH,
        extra_done={SETTINGS_KEY: [_SETTING]},
    )

    assert result["status"] == "kept"
    assert "HL_WIDE_TILE_MIN_SEQS" not in benched_envs
    assert result[SETTINGS_KEY] == [
        {
            "switch": "HL_WIDE_TILE_MIN_SEQS",
            "values": ["1", "4096"],
            "depends_on": [],
            "target": "src.py",
            "evidence": "decode",
        }
    ]


@pytest.mark.asyncio
async def test_a_setting_behind_a_dormant_switch_stays_with_the_inert_keep(tmp_path, monkeypatch):
    result, _, benched_envs, _ = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=0.2,
        patch_body=_SETTING_PATCH,
        extra_done={SETTINGS_KEY: [{**_SETTING, "depends_on": ["hl_hoist"]}]},
    )

    assert result["status"] == "kept_inert"
    assert benched_envs == {"HL_HOIST": "1", "HL_CACHE": "1"}
    assert [entry["depends_on"] for entry in result[SETTINGS_KEY]] == [["HL_HOIST"]]


def test_settings_are_parsed_like_switches_and_capped():
    raw = [
        {"setting": "hl_splits", "values": [4, "8", "8", *range(16, 26)], "depends_on": "hl_gate"},
        {"setting": "HL_SPLITS", "values": ["2"]},
        {"setting": "HL_NO_VALUES", "values": []},
        {"setting": "not an env", "values": ["1"]},
        {"setting": "HL_API_TOKEN", "values": ["1"]},
        {"setting": "HL_OWNED", "values": ["1"]},
        "HL_SCALAR",
    ]

    settings = parse_settings(raw, reserved={"HL_OWNED"})

    assert [entry["switch"] for entry in settings] == ["HL_SPLITS"]
    assert settings[0]["values"] == ["4", "8", "16", "17", "18", "19"]
    assert len(settings[0]["values"]) == MAX_SETTING_VALUES
    assert settings[0]["depends_on"] == ["HL_GATE"]


def test_a_token_named_setting_is_dropped_so_the_author_is_told_to_rename_it():
    assert parse_settings([{"setting": "HL_WIDE_TILE_MIN_TOKENS", "values": ["1"]}], reserved=set()) == []
    assert "(say _MIN_SEQS, not _MIN_TOKENS)" in UNTUNABLE_VALUE_ADVISORY


# explore sweeps each value and keeps the best


def _ledger(*, setting_default_on: bool = False) -> SharedState:
    state = SharedState()
    switches = [{"switch": "HL_GATE", "value": "1", "enables": [], "depends_on": []}]
    state.record_authored_framework_levers(switches, default_on=False)
    setting = parse_settings([{**_SETTING, "depends_on": ["HL_GATE"]}], reserved=set())
    state.record_authored_framework_levers(setting, default_on=setting_default_on)
    return state


def test_a_registered_setting_keeps_its_values_on_the_lever_row():
    rows = {row["switch"]: row for row in _ledger().authored_framework_levers}

    assert rows["HL_WIDE_TILE_MIN_SEQS"]["values"] == ["1", "4096"]
    assert "values" not in rows["HL_GATE"]


def test_the_promotion_registers_a_kept_patchs_settings():
    state = SharedState()
    owner = SimpleNamespace(shared_state=state)
    result = {"status": "kept", SETTINGS_KEY: parse_settings([_SETTING], reserved=set())}

    assert IntegratePromotionMixin._record_authored_levers(owner, result) is True
    assert [row["switch"] for row in state.authored_framework_levers] == ["HL_WIDE_TILE_MIN_SEQS"]


def test_explore_benches_every_value_with_the_gate_it_needs_on():
    payload = framework_lever_grid(_ledger())
    settings = [v for v in payload if v["framework_lever_source"] == "setting"]

    assert [v["extra_envs"] for v in settings] == [
        {"HL_GATE": "1", "HL_WIDE_TILE_MIN_SEQS": "1"},
        {"HL_GATE": "1", "HL_WIDE_TILE_MIN_SEQS": "4096"},
    ]
    assert [v["name"] for v in settings] == [
        "fwsetting_hl_wide_tile_min_seqs_1",
        "fwsetting_hl_wide_tile_min_seqs_4096",
    ]
    assert len(_grid_variants_from_payload(payload)) == len(payload)
    assert [v["framework_lever_source"] for v in payload] == ["setting", "setting", "additive"]


def test_an_attributed_setting_is_not_swept_again():
    state = _ledger()
    state.record_framework_lever_attribution("HL_WIDE_TILE_MIN_SEQS", gain_pct=3.0)

    assert [v["framework_lever_source"] for v in framework_lever_grid(state)] == ["additive"]


def test_a_setting_is_attributed_its_best_value():
    payload = setting_variants(parse_settings([{**_SETTING, "values": ["1", "256", "4096"]}], reserved=set()))
    gains = {"1": 4.0, "256": 1.5, "4096": -0.5}
    outcomes = [
        {"variant_name": v["name"], "outcome": "REVERT", "metrics": {"gain_pct": gains[v["framework_lever_value"]]}}
        for v in payload
    ]
    switch_seed = {"name": "fwlever_hl_gate", "framework_lever": "HL_GATE", "framework_lever_source": "additive"}
    outcomes.append({"variant_name": "fwlever_hl_gate", "outcome": "KEEP", "metrics": {"gain_pct": 2.0}})

    attributed = _framework_lever_attributions(outcomes, [*payload, switch_seed])

    assert [(row["switch"], row["gain_pct"], row["variant_name"]) for row in attributed] == [
        ("HL_GATE", 2.0, "fwlever_hl_gate"),
        ("HL_WIDE_TILE_MIN_SEQS", 4.0, "fwsetting_hl_wide_tile_min_seqs_1"),
    ]


# the author is told


@pytest.mark.parametrize(("mode", "told"), [("patch", True), ("research", False)])
def test_every_patch_author_is_told_to_ship_tuning_values_as_settings(mode, told):
    inp = SpecialistPromptInputs(domain=get_domain("serving_specialist"), mode=mode)
    text = "\n".join(_section_output_protocol(inp))

    assert (f"- ``{SETTINGS_KEY}``: a tuning value you introduce" in text) is told
