# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author-time recording of the v6 ``metadata`` section.

Metadata used to be re-derived at export from ``state.json`` and
``manifest.json``, which meant the exporting process had to re-probe its own
environment for facts the launching process already knew. These pin the
recorded path: each producer writes only the keys it owns, the singleton
deep-merges rather than replaces, and the exporter prefers a recorded leaf over
a projected one without losing the projection's fallbacks.
"""

from __future__ import annotations

from types import SimpleNamespace

from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_metadata
from hyperloom.inference_optimizer.breakdown.recorder import (
    assemble_parts,
    record_metadata_identity,
    record_metadata_langfuse,
    recorder_for,
    section_shape,
    snapshot_metadata,
)
from hyperloom.inference_optimizer.breakdown.recorder.session_metadata import SECTION

_MANIFEST = {
    "session_id": "sess-1",
    "claw_session_id": "claw-9",
    "created_at_utc": "2026-09-01T00:00:00+00:00",
    "session_dir": "/data/sess-1",
    "user_data_path": "/data",
    "code_revision": "abc1234",
    "host": "node-7",
    "pid": 4242,
    "image": "registry.example.com/team/hyperloom:v3",
    "max_minutes": 120,
    "model_name": "DeepSeek-V3",
    "model_path": "/models/dsv3",
    "framework": "sglang",
    "gpu_type": "MI300X",
    "tp": 8,
    "workload": {"conc": 64, "isl": 1024, "osl": 512, "precision": "fp8", "max_model_len": 4096},
    "objective": {"kind": "time_only", "value": None},
}


def _state(**overrides):
    base = dict(
        session_id="sess-1",
        start_ts="2026-09-01T00:00:00+00:00",
        stop_ts="2026-09-01T02:00:00+00:00",
        stop_reason="target_reached",
        max_minutes=120,
        tick=37,
        model_name="DeepSeek-V3",
        model_path="/models/dsv3",
        model_class="moe",
        framework="sglang",
        gpu_type="MI300X",
        tp=8,
        conc=64,
        isl=1024,
        osl=512,
        precision="fp8",
        max_model_len=4096,
        operator_extra_env={"HSA_NO_SCRATCH_RECLAIM": "1"},
        operator_server_args="--enable-torch-compile",
        server_args="",
        crash_count=0,
        crash_timestamps=[],
        degraded_mode=False,
        resume_pending_revalidation=False,
        last_tick_exception=None,
        model_info={
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "is_moe": True,
            "num_experts": 256,
            "torch_dtype": "bfloat16",
        },
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---- section registration ----


def test_metadata_is_a_registered_singleton():
    """An unregistered section is silently dropped by the recorder."""
    assert section_shape(SECTION) == "singleton"


# ---- identity ----


def test_the_manifest_stamp_records_identity_and_image(tmp_path):
    record_metadata_identity(tmp_path, _MANIFEST)
    session = assemble_parts(tmp_path)[SECTION]["session"]
    assert session["session_id"] == "sess-1"
    assert session["host"] == "node-7"
    assert session["pid"] == 4242
    assert session["image"] == "registry.example.com/team/hyperloom:v3"
    assert session["image_id"] == "hyperloom:v3"


def test_identity_carries_the_launch_shape(tmp_path):
    record_metadata_identity(tmp_path, _MANIFEST)
    task_config = assemble_parts(tmp_path)[SECTION]["task_config"]
    assert task_config["model_name"] == "DeepSeek-V3"
    assert task_config["framework_name"] == "sglang"
    assert (task_config["tp"], task_config["conc"], task_config["isl"]) == (8, 64, 1024)


def test_an_empty_manifest_records_nothing(tmp_path):
    record_metadata_identity(tmp_path, {})
    assert SECTION not in assemble_parts(tmp_path)


# ---- lifecycle snapshot ----


def test_a_state_snapshot_carries_the_budget_anchor_and_its_end(tmp_path):
    rec = recorder_for(tmp_path, producer="coordinator")
    snapshot_metadata(rec, _state())
    session = assemble_parts(tmp_path)[SECTION]["session"]
    assert session["start_ts"] == "2026-09-01T00:00:00+00:00"
    assert session["ended_at_utc"].startswith("2026-09-01T02:00:00")
    assert session["tick_count"] == 37


def test_a_running_session_records_no_end(tmp_path):
    """The end is only stamped alongside a reason, so a resume cannot leave a stale one."""
    rec = recorder_for(tmp_path, producer="coordinator")
    snapshot_metadata(rec, _state(stop_reason=""))
    assert assemble_parts(tmp_path)[SECTION]["session"]["ended_at_utc"] == ""


def test_the_snapshot_carries_the_whole_architecture_not_a_digest(tmp_path):
    rec = recorder_for(tmp_path, producer="coordinator")
    snapshot_metadata(rec, _state())
    architecture = assemble_parts(tmp_path)[SECTION]["task_config"]["architecture"]
    assert architecture["model_class"] == "moe"
    assert architecture["hidden_size"] == 7168
    assert architecture["torch_dtype"] == "bfloat16"
    assert architecture["num_experts"] == 256


def test_a_non_transformers_model_records_only_the_derived_class(tmp_path):
    rec = recorder_for(tmp_path, producer="coordinator")
    snapshot_metadata(rec, _state(model_info={}, model_class=""))
    task_config = assemble_parts(tmp_path)[SECTION]["task_config"]
    assert "architecture" not in task_config


def test_crash_timestamps_are_recorded_as_iso(tmp_path):
    """State keeps epoch seconds; every reader would otherwise convert them itself."""
    rec = recorder_for(tmp_path, producer="coordinator")
    snapshot_metadata(rec, _state(crash_count=2, crash_timestamps=[1_788_220_800.0, "bogus"]))
    recovery = assemble_parts(tmp_path)[SECTION]["session"]["recovery"]
    assert recovery["recovered"] is True
    assert recovery["crash_count"] == 2
    # The unparseable entry is skipped rather than failing the whole snapshot.
    assert recovery["crash_timestamps"] == ["2026-09-01T00:00:00+00:00"]


def test_a_tick_exception_keeps_the_header_and_drops_the_traceback(tmp_path):
    rec = recorder_for(tmp_path, producer="coordinator")
    snapshot_metadata(
        rec,
        _state(last_tick_exception={"tick": 12, "stage": "dispatch", "message": "x" * 900, "traceback": "y" * 5000}),
    )
    recorded = assemble_parts(tmp_path)[SECTION]["session"]["recovery"]["last_tick_exception"]
    assert recorded["tick"] == 12
    assert recorded["stage"] == "dispatch"
    assert len(recorded["message"]) == 500
    assert "traceback" not in recorded


def test_a_state_without_a_session_id_records_nothing(tmp_path):
    rec = recorder_for(tmp_path, producer="coordinator")
    snapshot_metadata(rec, _state(session_id=""))
    assert SECTION not in assemble_parts(tmp_path)


# ---- langfuse ----


def test_a_disabled_emitter_still_records_why(tmp_path):
    record_metadata_langfuse(tmp_path, {"enabled": False, "disabled_reason": "no_credentials"})
    langfuse = assemble_parts(tmp_path)[SECTION]["langfuse"]
    assert langfuse["enabled"] is False
    assert langfuse["disabled_reason"] == "no_credentials"


def test_the_trace_url_is_resolved_from_host_and_trace_id(tmp_path):
    record_metadata_langfuse(
        tmp_path,
        {"enabled": True, "trace_id": "tr-1", "config": {"host": "https://lf.example.com/"}, "counts": {"spans": 12}},
    )
    langfuse = assemble_parts(tmp_path)[SECTION]["langfuse"]
    assert langfuse["trace_url"] == "https://lf.example.com/trace/tr-1"
    assert langfuse["counts"] == {"spans": 12}


# ---- singleton merge across producers ----


def test_each_producer_contributes_only_its_own_keys(tmp_path):
    """A later partial write must not erase what an earlier one recorded."""
    record_metadata_identity(tmp_path, _MANIFEST)
    rec = recorder_for(tmp_path, producer="coordinator")
    snapshot_metadata(rec, _state())
    record_metadata_langfuse(tmp_path, {"enabled": True, "trace_id": "tr-1"})
    metadata = assemble_parts(tmp_path)[SECTION]
    # Identity-only, lifecycle-only, and langfuse-only facts all survive.
    assert metadata["session"]["image_id"] == "hyperloom:v3"
    assert metadata["session"]["tick_count"] == 37
    assert metadata["langfuse"]["trace_id"] == "tr-1"


# ---- export overlay ----


def _collect(recorded=None, **overrides):
    kwargs = dict(
        exported_at_utc="2026-09-01T02:00:05+00:00",
        session={"session_id": "sess-1", "code_revision": "abc1234", "pid": 1, "image": "collected:v1"},
        workload={"framework_name": "sglang", "model_class": "moe"},
        model_info={"model_type": "deepseek_v3", "hidden_size": 7168},
        langfuse={"enabled": False},
        versions={"geak": {"tool": "geak", "commit": "dead", "root_dir": "/opt/geak", "version": "1.2"}},
        state={},
        warnings=["w"],
        recorded=recorded,
    )
    kwargs.update(overrides)
    return collect_v6_metadata(**kwargs)


def test_a_recorded_leaf_beats_the_projection(tmp_path):
    metadata = _collect(recorded={"session": {"pid": 4242, "image": "recorded:v2"}})
    assert metadata["session"]["pid"] == 4242
    assert metadata["session"]["image"] == "recorded:v2"


def test_an_empty_recorded_leaf_does_not_erase_a_projected_one():
    """Absence of evidence is not evidence of absence."""
    metadata = _collect(recorded={"session": {"pid": 0, "code_revision": ""}})
    assert metadata["session"]["pid"] == 1
    assert metadata["session"]["code_revision"] == "abc1234"


def test_the_projection_supplies_blocks_the_fragment_never_wrote():
    metadata = _collect(recorded={"langfuse": {"enabled": True}})
    assert metadata["task_config"]["framework_name"] == "sglang"
    assert metadata["versions"]["schema_version"]


def test_tool_provenance_keeps_commit_and_root_dir():
    """A bare version string cannot tell you which checkout produced a result."""
    tools = _collect()["versions"]["tools"]
    assert tools["geak"]["commit"] == "dead"
    assert tools["geak"]["root_dir"] == "/opt/geak"


def test_export_facts_are_never_taken_from_a_fragment():
    metadata = _collect(recorded={"exported_at_utc": "1999-01-01T00:00:00+00:00", "warnings": ["stale"]})
    assert metadata["exported_at_utc"] == "2026-09-01T02:00:05+00:00"
    assert metadata["warnings"] == ["w"]


def test_elapsed_is_measured_from_the_resolved_anchor_and_end():
    """The recorder writes both ends; the span between them is only known here."""
    metadata = _collect(
        recorded={
            "session": {
                "start_ts": "2026-09-01T00:00:00+00:00",
                "ended_at_utc": "2026-09-01T01:30:00+00:00",
            }
        }
    )
    assert metadata["session"]["elapsed_minutes"] == 90.0


def test_the_recovery_block_is_carried_whole():
    metadata = _collect(
        session={
            "session_id": "sess-1",
            "recovery": {
                "recovered": True,
                "crash_count": 1,
                "crash_timestamps": ["2026-09-01T00:10:00+00:00"],
                "resume_pending_revalidation": True,
                "last_tick_exception": {"tick": 4},
            },
        }
    )
    recovery = metadata["session"]["recovery"]
    assert recovery["crash_timestamps"] == ["2026-09-01T00:10:00+00:00"]
    assert recovery["resume_pending_revalidation"] is True
    assert recovery["last_tick_exception"] == {"tick": 4}
