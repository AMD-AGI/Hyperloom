# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The single-file deliverable, driven end to end through ``single_file_deliverable``."""

from __future__ import annotations

import json
from hyperloom.orchestrator.deployment.deliverable import single_file_deliverable
from .conftest import _landed_without_snapshot, accept_source


def test_a_config_only_keep_still_exports(workload):
    """It touches no file, so recording no snapshot is correct rather than a gap -- the
    refusal must key on the layer having landed code, not on the snapshot being absent."""
    _, hook, session, state = workload
    state["optimization_stack"] = [accept_source(session, hook), {"scope": "source_patch", "source_snapshot": ""}]

    assert single_file_deliverable(session, state)["status"] == "exported"


def test_a_refused_export_replaces_the_previous_success(workload):
    """The rmtree removes the exported hook; the metadata beside it must not still claim one."""
    _, hook, session, state = workload
    assert single_file_deliverable(session, state)["status"] == "exported"
    assert (session / "reports" / "deployment.json").is_file()

    hook.write_text("def hyperloom_optimize(model):\n    return None  # never benchmarked\n")
    result = single_file_deliverable(session, state)

    assert result["status"] == "incomplete"
    written = json.loads((session / "reports" / "deployment.json").read_text())
    assert written["status"] == "incomplete", "the refusal must land where the success did"
    assert not (session / "deployment" / hook.name).exists()


def test_dirty_source_does_not_export_as_accepted(workload):
    _, hook, session, state = workload
    hook.write_text("def hyperloom_optimize(model):\n    return None\n")
    result = single_file_deliverable(session, state)
    assert result["status"] == "incomplete"
    assert not (session / "deployment" / hook.name).exists()


def test_export_allows_stdlib_and_unrelated_scratch_files(workload):
    """Scratch files nothing imports are harmless; the old git check rejected them."""
    root, hook, session, state = workload
    (root / "scratch_probe.py").write_text("# left behind by a specialist\n")
    hook.write_text("import json\nimport torch\n\n\ndef hyperloom_optimize(model):\n    return model\n")
    state["optimization_stack"] = [accept_source(session, hook, "accepted1")]

    assert single_file_deliverable(session, state)["status"] == "exported"


def test_export_contains_only_module_and_metadata(workload):
    _, hook, session, state = workload
    result = single_file_deliverable(session, state)
    assert result["status"] == "exported"
    assert result["reasons"] == []
    assert result["kind"] == "single_file"
    assert set(result["files"]) == {hook.name}
    assert {p.name for p in (session / "deployment").iterdir()} == {hook.name}
    assert json.loads((session / "reports/deployment.json").read_text())["kind"] == "single_file"
    assert (session / "deployment" / hook.name).read_bytes() == hook.read_bytes()


def test_export_is_idempotent_across_closes(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"HL_FUSE_ATTN": "1"}

    first_result = single_file_deliverable(session, state)
    first_bytes = (session / "deployment" / hook.name).read_bytes()
    second_result = single_file_deliverable(session, state)
    second_bytes = (session / "deployment" / hook.name).read_bytes()

    assert first_result["status"] == "exported"
    assert second_result["status"] == "exported"
    assert first_result["files"] == second_result["files"]
    assert first_bytes == second_bytes


def test_export_refuses_a_hook_importing_a_sibling(workload):
    """Only the hook ships, so a sibling import would deploy broken."""
    root, hook, session, state = workload
    (root / "_kernels.py").write_text("X = 1\n")
    hook.write_text("import _kernels\n\n\ndef hyperloom_optimize(model):\n    return model\n")
    state["optimization_stack"] = [accept_source(session, hook, "accepted1")]

    result = single_file_deliverable(session, state)

    assert result["status"] == "incomplete"
    assert "imports '_kernels' from beside itself" in result["reasons"][0]


def test_export_refuses_a_layer_that_landed_code_without_a_snapshot(workload):
    """``missing_source_snapshot`` already names this; until now it was only logged at lift
    time, long after the export had read past the entry and shipped the tree."""
    _, hook, session, state = workload
    state["optimization_stack"] = [
        accept_source(session, hook),
        _landed_without_snapshot(variant_name="spec-9"),
    ]

    result = single_file_deliverable(session, state)

    assert result["status"] != "exported"
    assert any("no snapshot" in str(r) for r in result["reasons"]), result["reasons"]
    assert not (session / "deployment" / hook.name).exists()


def test_export_refuses_a_module_without_its_declared_function(named_workload):
    _, hook, session, state = named_workload
    hook.write_text("def hyperloom_optimize(model):\n    return model\n")
    state["optimization_stack"] = [accept_source(session, hook, "renamed")]
    result = single_file_deliverable(session, state)
    assert result["status"] != "exported"
    assert not (session / "deployment" / hook.name).exists()


def test_export_refuses_a_namespace_package_sibling(workload):
    """A directory needs no __init__.py to import; ship one file and it is gone."""
    root, hook, session, state = workload
    (root / "kernels").mkdir()
    (root / "kernels" / "fast.py").write_text("Y = 2\n")
    hook.write_text("import kernels.fast\n\n\ndef hyperloom_optimize(model):\n    return model\n")
    state["optimization_stack"] = [accept_source(session, hook, "accepted1")]

    result = single_file_deliverable(session, state)

    assert result["status"] == "incomplete"
    assert "imports 'kernels' from beside itself" in result["reasons"][0]


def test_export_refuses_a_relative_import(workload):
    root, hook, session, state = workload
    hook.write_text("from . import helpers\n\n\ndef hyperloom_optimize(model):\n    return model\n")
    state["optimization_stack"] = [accept_source(session, hook, "accepted1")]

    assert single_file_deliverable(session, state)["status"] == "incomplete"


def test_export_refuses_when_no_snapshot_covers_the_hook(workload):
    """A config-only KEEP records no snapshot; exporting then would verify nothing."""
    _, hook, session, state = workload
    state["optimization_stack"] = [{"stack_entry_id": "env0", "source_snapshot": ""}]
    hook.write_text("def hyperloom_optimize(model):\n    return None  # never benchmarked\n")

    result = single_file_deliverable(session, state)

    assert result["status"] == "incomplete"
    assert "nothing proves what was benchmarked" in result["reasons"][0]
    assert not (session / "deployment" / hook.name).exists()


def test_export_refuses_a_stack_that_changed_more_than_the_module(workload):
    """The backstop under the apply-time artifact refusal, for anything that gets past it.

    A snapshot naming a second file means the accepted configuration is not one file, so
    shipping the module alone would hand the operator something that was never measured.
    """
    root, hook, session, state = workload
    (root / "tuned_gemm.csv").write_text("gfx,cu_num,us\n")
    layer = accept_source(session, hook, "accepted1")
    snapshot = session / "optimization_stack" / "src" / "accepted1"
    (snapshot / "files" / "tuned_gemm.csv").write_text("gfx,cu_num,us\n")
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "complete": True,
                "files": [{"rel": hook.name, "op": "upsert"}, {"rel": "tuned_gemm.csv", "op": "upsert"}],
            }
        )
    )
    state["optimization_stack"] = [layer]

    result = single_file_deliverable(session, state)

    assert result["status"] == "incomplete"
    assert "tuned_gemm.csv" in result["reasons"][0]
    assert not (session / "deployment" / hook.name).exists()


def test_pycache_does_not_block_export(workload):
    root, _, session, state = workload
    pycache = root / "__pycache__"
    pycache.mkdir()
    (pycache / "hyperloom_optimize.cpython-312.pyc").write_bytes(b"junk")

    result = single_file_deliverable(session, state)

    assert result["status"] == "exported"
