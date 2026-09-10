# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The KERNEL entry as a sequence of lanes that all report the same way."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.phases.kernel import FORGE_LANES, ForgeLane, KernelPhase, lane_failure


class _Bus:
    def __init__(self) -> None:
        self.posted: list[dict] = []

    async def append_and_seq(self, message):
        self.posted.append(dict(message.payload))
        return message


class _Phase:
    """Only what ``_run_forge_lane`` touches, so the sequencing is what is under test."""

    def __init__(self) -> None:
        self.bus = _Bus()
        self.evidence: dict[str, dict] = {}
        self.reprofiles = 0

    async def _maybe_reprofile_for_kernel(self) -> None:
        self.reprofiles += 1

    def _record_phase_entry_evidence(self, **kvs) -> None:
        self.evidence.update(kvs)

    async def run(self, lane: ForgeLane) -> None:
        await KernelPhase._run_forge_lane(self, lane)


def _lane(name: str, *, gate=lambda _p: True, run=None) -> ForgeLane:
    async def _ok(_phase):
        return {"status": "ok"}

    return ForgeLane(name=name, response_kind=f"{name}_done", gate=gate, run=run or _ok)


class TestTheOrderIsTheContract:
    def test_gemm_tunes_before_fusion_authors_before_the_controller_rewrites(self):
        # GEMM tunes the tables the later lanes measure against, and fusion
        # changes the decode path the controller then reads a trace of.
        assert [lane.name for lane in FORGE_LANES] == ["gemm_tuning", "fusion", "kernel_rewrite_controller"]

    def test_the_rewrite_controller_has_no_gate(self):
        # Choosing operators is its job, so nothing upstream decides for it.
        rewrite = next(lane for lane in FORGE_LANES if lane.name == "kernel_rewrite_controller")

        assert rewrite.gate(object()) is True


class TestOneShapeForEveryLane:
    @pytest.mark.asyncio
    async def test_a_lane_that_runs_refreshes_the_snapshot_and_reports_twice(self):
        phase = _Phase()

        await phase.run(_lane("fusion"))

        assert phase.reprofiles == 1
        assert phase.evidence["fusion"]["status"] == "done"
        assert [post["kind"] for post in phase.bus.posted] == ["fusion_done"]
        assert phase.bus.posted[0]["source"] == "kernel_entry_auto"

    @pytest.mark.asyncio
    async def test_a_gated_off_lane_costs_nothing_and_says_nothing(self):
        phase = _Phase()

        await phase.run(_lane("fusion", gate=lambda _p: False))

        assert phase.reprofiles == 0
        assert phase.evidence == {}
        assert phase.bus.posted == []

    @pytest.mark.asyncio
    async def test_a_crash_is_reported_in_the_same_shape_as_a_result(self):
        async def _boom(_phase):
            raise RuntimeError("lane boom")

        phase = _Phase()

        await phase.run(_lane("gemm_tuning", run=_boom))

        posted = phase.bus.posted[0]
        assert posted["status"] == "failed"
        assert posted["result"]["error_class"] == "RuntimeError"
        assert posted["result"]["decision"] == "REVERT"
        assert phase.evidence["gemm_tuning"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_the_evidence_row_carries_what_the_lane_measured(self):
        async def _kept(_phase):
            return {"status": "ok", "best_speedup": 1.4, "kept": True, "patch_count": 2}

        phase = _Phase()

        await phase.run(_lane("kernel_rewrite_controller", run=_kept))

        row = phase.evidence["kernel_rewrite_controller"]
        assert row["best_speedup"] == 1.4
        assert row["kept"] is True
        assert row["patch_count"] == 2


class TestTheFailureEnvelope:
    def test_every_lane_names_the_reason_the_same_way(self):
        envelope = lane_failure(ValueError("bad shape"))

        assert envelope["status"] == "failed"
        assert envelope["decision"] == "REVERT"
        assert envelope["error_class"] == "ValueError"
        assert "bad shape" in envelope["error"]

    def test_a_lane_may_add_its_own_engine_tag(self):
        assert lane_failure(RuntimeError("x"), engine="forge_fusion")["engine"] == "forge_fusion"
