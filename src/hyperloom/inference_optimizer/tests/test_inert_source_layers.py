# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An inert KEEP's source is recorded beside the stack, not on it.

A patch committed with every switch off lands code the export must still account for: a
later explore entry switches it on, and the export refuses a tree no snapshot covers. It
is not an adoption, so putting it on the optimization stack would publish it as an
unvalidated KEEP and skew every length and attribution count taken over that stack.
"""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.source_snapshot import source_layers_in_order


class _Recorder:
    """The promotion surface ``_record_inert_source_layer`` touches."""

    def __init__(self, framework: str = "custom", stack_len: int = 0) -> None:
        from hyperloom.orchestrator.state.shared_state import SharedState

        self.shared_state = SharedState()
        self.shared_state.framework = framework
        self.shared_state.optimization_stack = [{"name": f"keep-{i}"} for i in range(stack_len)]

    def record(self, result: dict[str, Any]) -> bool:
        from hyperloom.orchestrator.loop._integrate_promotion import IntegratePromotionMixin

        return IntegratePromotionMixin._record_inert_source_layer(self, result)


def _result(**over: Any) -> dict[str, Any]:
    return {
        "status": "kept_inert",
        "source_snapshot": "/snapshots/a",
        "source_manifest": "/snapshots/a/manifest.json",
        "framework_root": "/repo",
        "target_files": ["hyperloom_optimize.py"],
        "specialist_task_id": "spec-7",
        **over,
    }


def test_an_inert_keep_records_its_source_beside_the_stack() -> None:
    rec = _Recorder(stack_len=2)
    assert rec.record(_result()) is True
    assert rec.shared_state.optimization_stack == [{"name": "keep-0"}, {"name": "keep-1"}]
    (layer,) = rec.shared_state.inert_source_layers
    assert layer["scope"] == "source_patch"
    assert layer["after_stack_len"] == 2
    assert layer["source_snapshot"] == "/snapshots/a"
    assert layer["variant_name"] == "spec-7:inert"


def test_an_inert_keep_with_no_snapshot_records_nothing() -> None:
    """There is nothing for the export to verify against, so there is nothing to keep."""
    rec = _Recorder()
    assert rec.record(_result(source_snapshot="")) is False
    assert rec.shared_state.inert_source_layers == []


@pytest.mark.parametrize("framework", ["vllm", "sglang", ""])
def test_only_a_custom_session_records_the_layer(framework: str) -> None:
    """Custom is the one framework whose export reads it."""
    rec = _Recorder(framework=framework)
    assert rec.record(_result()) is False
    assert rec.shared_state.inert_source_layers == []


def test_an_anonymous_specialist_still_names_the_layer() -> None:
    rec = _Recorder()
    rec.record(_result(specialist_task_id=""))
    assert rec.shared_state.inert_source_layers[0]["variant_name"] == "inert"


def test_layers_are_ordered_by_where_they_landed_between_the_stack_entries() -> None:
    """The export checks the source in the order it reached the tree, not stack-then-inert."""
    stack = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    inert = [{"name": "i0", "after_stack_len": 0}, {"name": "i2", "after_stack_len": 2}]
    assert [e["name"] for e in source_layers_in_order(stack, inert)] == ["i0", "a", "b", "i2", "c"]


def test_an_inert_layer_without_a_position_sorts_first() -> None:
    assert [e["name"] for e in source_layers_in_order([{"name": "a"}], [{"name": "i"}])] == ["i", "a"]


class _Replayer:
    """The ``build_env_spec`` surface ``_inert_layers_in`` reads."""

    def __init__(self, *layers: int) -> None:
        from hyperloom.orchestrator.state.shared_state import SharedState

        self.shared_state = SharedState()
        self.shared_state.inert_source_layers = [{"name": f"i{k}", "after_stack_len": k} for k in layers]

    def layers_in(self, stack_len: int) -> list[str]:
        from hyperloom.orchestrator.loop._launch_identity import LaunchIdentityMixin

        stack = [{"name": f"keep-{i}"} for i in range(stack_len)]
        return [layer["name"] for layer in LaunchIdentityMixin._inert_layers_in(self, stack)]


def test_a_layer_that_landed_before_the_lift_is_replayed() -> None:
    """``after_stack_len`` counts the stack entries that preceded the layer, so anything
    below the lifted length was already in the tree the number was measured on."""
    assert _Replayer(0, 1).layers_in(2) == ["i0", "i1"]


def test_a_layer_that_landed_after_the_lift_is_not_replayed() -> None:
    """A ``current_best`` with two entries was measured as the second one landed. A layer
    recorded at ``after_stack_len == 2`` arrived strictly afterwards, so replaying it
    would reconstruct a tree the accepted number never described. This is the boundary
    that makes ``<`` the right comparison rather than ``<=``."""
    assert _Replayer(2, 3).layers_in(2) == []


def test_the_boundary_splits_a_mixed_set() -> None:
    assert _Replayer(0, 2, 1, 5).layers_in(2) == ["i0", "i1"]


def test_an_empty_lift_replays_nothing() -> None:
    """Nothing has been measured yet, so no layer can predate the measurement."""
    assert _Replayer(0).layers_in(0) == []


def test_the_export_takes_every_layer_unfiltered() -> None:
    """``_verify_accepted_source`` checks the tree *on disk*, which holds every inert
    layer ever committed -- they are never reverted. ``build_env_spec`` reconstructs a
    point in history and must filter. The asymmetry is the point, not an oversight."""
    import inspect

    from hyperloom.orchestrator.deployment import export

    body = inspect.getsource(export._verify_accepted_source)
    assert 'state.get("inert_source_layers", [])' in body
    assert "after_stack_len" not in body


import json
import pytest
from pathlib import Path
from hyperloom.orchestrator.deployment.export import export_custom_inference
from .conftest import _landed_without_snapshot, _lift_coordinator


def _custom_coordinator(tmp_path):
    coord = _lift_coordinator(tmp_path)
    coord.shared_state.framework = "custom"
    return coord


def _inert_layer(tmp_path, name, content, after_stack_len):
    return {"source_snapshot": _snapshot(tmp_path, name, content), "after_stack_len": after_stack_len}


def _inert_result(snapshot):
    return {
        "status": "kept_inert",
        "specialist_task_id": "spec",
        "workspace": "/w",
        "source_snapshot": snapshot,
        "source_manifest": snapshot + "/manifest.json",
        "source_snapshot_complete": True,
        "target_files": ["hook.py"],
    }


def _lift_keep(coord, tmp_path, name, content, tput):
    snapshot = _snapshot(tmp_path, name, content)
    assert coord.writeback.lift_to_current_best(
        "integrate_patch",
        tput,
        {
            "name": name,
            "extra_server_args": "",
            "output_throughput": tput,
            "scope": "source_patch",
            "source_snapshot": snapshot,
        },
    )
    return snapshot


def _snapshot(root, name, content):
    snapshot = root / name
    (snapshot / "files").mkdir(parents=True)
    (snapshot / "files" / "hook.py").write_text(content)
    (snapshot / "manifest.json").write_text(
        json.dumps({"complete": True, "files": [{"rel": "hook.py", "op": "upsert"}]})
    )
    return str(snapshot)


def _tree(tmp_path, content):
    tree = tmp_path / "tree"
    tree.mkdir(exist_ok=True)
    (tree / "hook.py").write_text(content)
    return tree


def _validated_keep(coord, tmp_path, name, content, tput):
    snapshot = _lift_keep(coord, tmp_path, name, content, tput)
    assert coord.writeback.validate(tput, {"output_throughput": tput})
    return snapshot


def test_an_inert_keep_leaves_the_stack_and_the_validation_alone(tmp_path):
    """The layer is not an adoption: the stack, its gain mirror, current_best and the validation stamp hold."""
    from hyperloom.inference_optimizer.breakdown.session_facts import recovery_block

    coord = _custom_coordinator(tmp_path)
    _validated_keep(coord, tmp_path, "kept", "v1\n", 1100.0)
    state = coord.shared_state
    before = (list(state.optimization_stack), list(state.gain_per_stack_entry), dict(state.current_best))
    assert not state.optimization_stack_has_unvalidated_keeps()

    assert coord.writeback._record_inert_source_layer(_inert_result(_snapshot(tmp_path, "inert", "v2\n")))

    assert (state.optimization_stack, state.gain_per_stack_entry, state.current_best) == before
    assert [(layer["variant_name"], layer["after_stack_len"]) for layer in state.inert_source_layers] == [
        ("spec:inert", 1)
    ]
    assert not state.optimization_stack_has_unvalidated_keeps()
    assert not recovery_block(state)["resume_pending_revalidation"]


def test_a_keep_after_an_inert_keep_still_needs_validation(tmp_path):
    coord = _custom_coordinator(tmp_path)
    _validated_keep(coord, tmp_path, "kept", "v1\n", 1100.0)
    coord.writeback._record_inert_source_layer(_inert_result(_snapshot(tmp_path, "inert", "v2\n")))

    _lift_keep(coord, tmp_path, "kept2", "v3\n", 1200.0)

    assert coord.shared_state.optimization_stack_has_unvalidated_keeps()
    assert coord.writeback.validate(1200.0, {"output_throughput": 1200.0})
    assert not coord.shared_state.optimization_stack_has_unvalidated_keeps()


def test_an_inert_keep_before_any_validation_does_not_stamp_one(tmp_path):
    coord = _custom_coordinator(tmp_path)

    assert coord.writeback._record_inert_source_layer(_inert_result(_snapshot(tmp_path, "inert", "v2\n")))

    state = coord.shared_state
    assert not state.optimization_stack
    assert (state.cumulative_gain_validated_stack_len, state.cumulative_gain_validated_ts) == (0, "")


def test_an_inert_keep_is_not_recorded_without_a_snapshot_or_outside_custom(tmp_path):
    coord = _custom_coordinator(tmp_path)
    assert not coord.writeback._record_inert_source_layer({"status": "kept_inert", "source_snapshot": ""})

    for framework in ("vllm", "xdit"):
        other = _lift_coordinator(tmp_path)
        other.shared_state.framework = framework
        assert not other.writeback._record_inert_source_layer(
            _inert_result(_snapshot(tmp_path, f"inert-{framework}", "v2\n"))
        )
        assert not other.shared_state.inert_source_layers and not other.shared_state.optimization_stack


def test_env_spec_takes_an_inert_layer_only_once_current_best_was_measured_on_it(tmp_path):
    """A layer that landed after the lift is not in current_best's tree, so its launch identity must not move."""
    coord = _custom_coordinator(tmp_path)
    keep1 = _validated_keep(coord, tmp_path, "kept", "v1\n", 1100.0)
    spec = coord.writeback.build_env_spec()

    inert = _snapshot(tmp_path, "inert", "v2\n")
    coord.writeback._record_inert_source_layer(_inert_result(inert))

    assert coord.writeback.build_env_spec()["launch_identity"] == spec["launch_identity"]

    keep2 = _validated_keep(coord, tmp_path, "kept2", "v3\n", 1200.0)
    layers = [Path(layer["snapshot_dir"]).parent for layer in coord.writeback.build_env_spec()["source_snapshots"]]
    assert layers == [Path(keep1), Path(inert), Path(keep2)]


def test_export_accepts_a_tree_an_inert_keep_changed_only_once_it_is_recorded(tmp_path):
    """A KEEP, then an inert KEEP that rewrites the same file: the tree matches the later snapshot only."""
    from hyperloom.orchestrator.deployment.export import _verify_accepted_source

    tree = _tree(tmp_path, "v2\n")
    stack = [{"source_snapshot": _snapshot(tmp_path, "keep", "v1\n")}]

    with pytest.raises(ValueError, match="differs from the accepted stack"):
        _verify_accepted_source({"optimization_stack": stack}, tree)

    state = {"optimization_stack": stack, "inert_source_layers": [_inert_layer(tmp_path, "inert", "v2\n", 1)]}
    assert _verify_accepted_source(state, tree) == {"hook.py"}


def test_export_reads_inert_layers_in_the_order_they_landed(tmp_path):
    """KEEP v1, inert v2, KEEP v3, inert v4: the last layer to land is the one the tree must match."""
    from hyperloom.orchestrator.deployment.export import _verify_accepted_source

    stack = [
        {"source_snapshot": _snapshot(tmp_path, "keep1", "v1\n")},
        {"source_snapshot": _snapshot(tmp_path, "keep2", "v3\n")},
        {"action": "explore"},
    ]
    inert = [_inert_layer(tmp_path, "inert1", "v2\n", 1)]
    assert _verify_accepted_source({"optimization_stack": stack, "inert_source_layers": inert}, _tree(tmp_path, "v3\n"))
    with pytest.raises(ValueError, match="differs from the accepted stack"):
        _verify_accepted_source({"optimization_stack": stack, "inert_source_layers": inert}, _tree(tmp_path, "v2\n"))

    inert.append(_inert_layer(tmp_path, "inert2", "v4\n", 2))
    assert _verify_accepted_source({"optimization_stack": stack, "inert_source_layers": inert}, _tree(tmp_path, "v4\n"))
    with pytest.raises(ValueError, match="differs from the accepted stack"):
        _verify_accepted_source({"optimization_stack": stack, "inert_source_layers": inert}, _tree(tmp_path, "v5\n"))


def test_inert_layers_before_the_first_entry_and_at_one_length_keep_their_landing_order():
    from hyperloom.orchestrator.source_snapshot import source_layers_in_order

    stack = [{"id": "keep1"}, {"id": "keep2"}]
    inert = [
        {"id": "first", "after_stack_len": 0},
        {"id": "a", "after_stack_len": 1},
        {"id": "b", "after_stack_len": 1},
        {"id": "last", "after_stack_len": 2},
    ]

    assert [layer["id"] for layer in source_layers_in_order(stack, inert)] == [
        "first",
        "keep1",
        "a",
        "b",
        "keep2",
        "last",
    ]


def test_an_inert_layer_that_landed_code_without_a_snapshot_is_refused(workload):
    """Inert layers reach the same check: they are in the tree being captured too."""
    _, hook, session, state = workload
    state["inert_source_layers"] = [_landed_without_snapshot(variant_name="spec-3:inert")]

    result = export_custom_inference(session, state)

    assert result["status"] != "exported"
    assert any("no snapshot" in str(r) for r in result["reasons"]), result["reasons"]
