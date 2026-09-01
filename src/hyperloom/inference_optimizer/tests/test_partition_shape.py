# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the launch-time check on a session's compute-partition shape.

The module under test never changes the card, so these tests are about what it
concludes from what it reads. Two behaviours carry the weight: a declared mode
that cannot be verified must refuse the session rather than assume it holds, and
a workload that provably will not fit must be refused in milliseconds at launch
rather than discovered as an out-of-memory crash hours in.
"""

from __future__ import annotations

import pytest

from hyperloom.common.gpu_partition import PartitionLayout, layout_for
from hyperloom.orchestrator.actions.executors import _partition_shape as ps


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in (
        ps.PARTITION_GPU_ENV,
        ps.PARTITION_MODE_ENV,
        ps.PARTITION_COUNT_ENV,
        ps.PARTITION_CU_ENV,
        ps.PARTITION_STREAMS_ENV,
        ps.PARTITION_TOTAL_STREAMS_ENV,
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def card(monkeypatch):
    """Present a card in a given shape, or none at all."""

    def present(layout):
        monkeypatch.setattr(ps, "observe_partition", lambda gpu_id, gpu_type=None: layout)

    return present


def _layout(mode, cu, gib=None, probed=True):
    """Build a layout directly, so a table-derived one can be posed as such."""
    if probed:
        return layout_for(mode, cu_per_partition=cu, gib_per_partition=gib)
    derived = layout_for(mode, cu_per_partition=cu, gib_per_partition=gib)
    return PartitionLayout(
        mode=derived.mode,
        partitions=derived.partitions,
        cu_per_partition=derived.cu_per_partition,
        gib_per_partition=derived.gib_per_partition,
        probed=False,
    )


CPX_36 = _layout("CPX", 32, 36.0)
SPX_288 = _layout("SPX", 256, 288.0)


class TestUnreadableCard:
    def test_a_declared_mode_that_cannot_be_verified_refuses(self, card):
        """The flag exists to catch a set that did not take; unverified is not satisfied."""
        card(None)
        verdict = ps.validate_session_shape(declared_mode="CPX")

        assert verdict.ok is False
        assert "could not be read" in verdict.refusal
        assert "it does not set it" in verdict.refusal

    def test_nothing_declared_on_an_unreadable_card_runs_as_before(self, card):
        """A host without amd-smi is the ordinary case, not a failure."""
        card(None)
        verdict = ps.validate_session_shape()

        assert verdict.ok is True
        assert verdict.layout is None
        assert verdict.warnings == ()


class TestDeclaredMode:
    def test_a_mismatch_refuses_and_names_both_modes(self, card):
        card(SPX_288)
        verdict = ps.validate_session_shape(declared_mode="CPX")

        assert verdict.ok is False
        assert "is in SPX, not the declared CPX" in verdict.refusal

    def test_the_refusal_says_the_optimizer_will_not_fix_it(self, card):
        """Otherwise the natural reading is that the flag applies the mode."""
        card(SPX_288)
        verdict = ps.validate_session_shape(declared_mode="CPX")
        assert "Nothing in the optimizer changes the mode" in verdict.refusal

    def test_a_match_proceeds(self, card):
        card(CPX_36)
        verdict = ps.validate_session_shape(declared_mode="CPX", params={"peak_gib_per_stream": 4.0})
        assert verdict.ok is True

    def test_no_declaration_accepts_whatever_the_card_is_in(self, card):
        card(CPX_36)
        verdict = ps.validate_session_shape(params={"peak_gib_per_stream": 4.0})
        assert verdict.ok is True
        assert verdict.layout is not None
        assert verdict.layout.mode == "CPX"


class TestUnpartitionedCard:
    def test_spx_needs_no_feasibility_check(self, card):
        """One partition is the whole card, which every session already assumes."""
        card(SPX_288)
        verdict = ps.validate_session_shape(params={"peak_gib_per_stream": 200.0}, streams=4)

        assert verdict.ok is True
        assert not any("Streams per partition" in note for note in verdict.notes)

    def test_the_shape_is_still_recorded(self, card):
        card(SPX_288)
        verdict = ps.validate_session_shape()
        assert any("SPX" in note for note in verdict.notes)


class TestFeasibility:
    def test_the_measured_mi355x_case_is_refused_at_launch(self, card):
        """20.7 GiB per stream, two streams, a 36 GiB CPX partition."""
        card(CPX_36)
        verdict = ps.validate_session_shape(streams=2, params={"peak_gib_per_stream": 20.7})

        assert verdict.ok is False
        assert "2 x 20.7 GiB = 41.4 GiB needed per partition" in verdict.refusal
        assert "36.0 GiB available" in verdict.refusal

    def test_a_measured_refusal_says_it_was_measured(self, card):
        """Measured and weights-only call for different responses from an operator."""
        card(CPX_36)
        verdict = ps.validate_session_shape(streams=2, params={"peak_gib_per_stream": 20.7})
        assert "a measured per-stream peak" in verdict.refusal

    def test_the_same_workload_at_one_stream_is_allowed(self, card):
        card(CPX_36)
        verdict = ps.validate_session_shape(streams=1, params={"peak_gib_per_stream": 20.7})
        assert verdict.ok is True

    def test_a_fit_reports_the_headroom_it_checked(self, card):
        card(CPX_36)
        verdict = ps.validate_session_shape(streams=2, params={"peak_gib_per_stream": 10.0})

        assert verdict.ok is True
        assert any("20.0 GiB needed of 36.0 GiB" in note for note in verdict.notes)

    def test_streams_are_multiplied_in_not_ignored(self, card):
        """Each stream holds its own copy, so the one-stream figure is the trap."""
        card(CPX_36)
        assert ps.validate_session_shape(streams=2, params={"peak_gib_per_stream": 17.0}).ok is True
        assert ps.validate_session_shape(streams=3, params={"peak_gib_per_stream": 17.0}).ok is False

    def test_an_unknown_footprint_warns_and_runs(self, card):
        """Refusing here would ground every session that cannot be sized."""
        card(CPX_36)
        verdict = ps.validate_session_shape(streams=2)

        assert verdict.ok is True
        assert any("out-of-memory failure rather than a refusal" in w for w in verdict.warnings)

    def test_unreported_partition_memory_warns_and_runs(self, card):
        card(_layout("CPX", 32, gib=None))
        verdict = ps.validate_session_shape(streams=2, params={"peak_gib_per_stream": 20.7})

        assert verdict.ok is True
        assert any("reported no usable per-partition memory" in w for w in verdict.warnings)

    def test_a_zero_capacity_is_unknown_rather_than_a_limit_that_refuses_everything(self, card):
        """The guard here and the one in ``fits_in_partition`` must not disagree.

        A ``0.0`` capacity used to pass this side's ``is None`` test and then hit
        the other side's falsiness test, so the arithmetic was skipped while the
        warning that explains why was not printed. Both now ask
        ``capacity_known``.
        """
        card(_layout("CPX", 32, gib=0.0))
        verdict = ps.validate_session_shape(streams=2, params={"peak_gib_per_stream": 20.7})

        assert verdict.ok is True
        assert any("reported no usable per-partition memory" in w for w in verdict.warnings)
        assert not any("GiB needed of" in n for n in verdict.notes)


class TestStreamsAreRefusedNotDefaulted:
    """``streams=0`` must not become the default, here or at the CLI.

    The CLI already refuses it, and says in a comment why falsiness is the wrong
    test. This entry point used ``streams or DEFAULT`` anyway, so the same value
    exited 2 through one door and reported "Streams per partition: 2" through the
    other.
    """

    @pytest.mark.parametrize("streams", [0, -1, -8])
    def test_a_value_below_one_refuses(self, card, streams):
        card(CPX_36)
        verdict = ps.validate_session_shape(streams=streams, params={"peak_gib_per_stream": 1.0})

        assert verdict.ok is False
        assert "must be >= 1" in verdict.refusal
        assert str(streams) in verdict.refusal

    def test_it_refuses_before_the_card_is_read(self, monkeypatch):
        """A usage error about the request needs no probe to decide."""

        def _fail(*_args, **_kwargs):
            raise AssertionError("the card must not be read to refuse streams=0")

        monkeypatch.setattr(ps, "observe_partition", _fail)
        assert ps.validate_session_shape(streams=0).ok is False

    def test_omitted_streams_still_take_the_default(self, card):
        """``None`` means "not named" and is the only thing that may default."""
        card(CPX_36)
        for verdict in (
            ps.validate_session_shape(params={"peak_gib_per_stream": 1.0}),
            ps.validate_session_shape(streams=None, params={"peak_gib_per_stream": 1.0}),
        ):
            assert verdict.ok is True
            assert any(f"Streams per partition: {ps.DEFAULT_STREAMS_PER_PARTITION}" in n for n in verdict.notes)

    def test_a_non_numeric_request_refuses_rather_than_raises(self, card):
        card(CPX_36)
        assert ps.validate_session_shape(streams="two").ok is False  # type: ignore[arg-type]


class TestFitCheckNeedsAFanOut:
    """A partitioned card met by a session that will not place work on partitions.

    The refusal multiplies a footprint by streams sharing one partition. With no
    fan-out that premise is false twice over: nothing places a second stream,
    and nothing pins the benchmark to a partition at all -- whole cards
    enumerate first, so the run may land on a whole card.
    """

    def test_a_workload_too_big_for_a_partition_is_not_refused(self, card):
        card(CPX_36)
        verdict = ps.validate_session_shape(
            streams=2,
            params={"peak_gib_per_stream": 20.7},
            fanout_expected=False,
        )
        assert verdict.ok is True

    def test_the_shape_is_still_recorded(self, card):
        """Provenance is the point; withholding the refusal must not lose the mode."""
        card(CPX_36)
        verdict = ps.validate_session_shape(fanout_expected=False)

        assert verdict.layout is not None
        assert verdict.layout.mode == "CPX"
        assert any("CPX" in note for note in verdict.notes)

    def test_it_says_the_numbers_belong_to_an_unknown_fraction_of_the_card(self, card):
        card(CPX_36)
        verdict = ps.validate_session_shape(fanout_expected=False)
        assert any("unknown fraction of the card" in w for w in verdict.warnings)

    def test_no_stream_note_is_made_for_streams_nothing_will_place(self, card):
        card(CPX_36)
        verdict = ps.validate_session_shape(streams=4, fanout_expected=False)
        assert not any("Streams per partition" in note for note in verdict.notes)

    def test_a_mode_mismatch_is_still_refused(self, card):
        """The assertion is about the card, so no fan-out is needed to check it."""
        card(SPX_288)
        verdict = ps.validate_session_shape(declared_mode="CPX", fanout_expected=False)
        assert verdict.ok is False

    def test_an_unpartitioned_card_is_unaffected(self, card):
        card(SPX_288)
        assert ps.validate_session_shape(fanout_expected=False).ok is True


class TestTableFallbackIsSurfaced:
    def test_a_derived_cu_count_warns_because_selection_matches_on_it(self, card):
        card(_layout("DPX", 152, 144.0, probed=False))
        verdict = ps.validate_session_shape(params={"peak_gib_per_stream": 4.0})

        assert verdict.ok is True
        assert any("built-in board table" in w for w in verdict.warnings)

    def test_a_probed_count_warns_about_nothing(self, card):
        card(CPX_36)
        verdict = ps.validate_session_shape(params={"peak_gib_per_stream": 4.0})
        assert verdict.warnings == ()


class TestPerStreamFootprint:
    def test_an_explicit_param_is_taken_as_measured(self):
        assert ps.per_stream_footprint_gib({"peak_gib_per_stream": 12.5}) == (12.5, "measured")

    def test_a_prior_run_supplies_it(self):
        state = type("S", (), {"current_best": {"peak_gib_per_stream": 9.0}, "model_path": ""})()
        assert ps.per_stream_footprint_gib(None, state) == (9.0, "measured")

    def test_a_param_overrides_a_prior_run(self):
        state = type("S", (), {"current_best": {"peak_gib_per_stream": 9.0}, "model_path": ""})()
        assert ps.per_stream_footprint_gib({"peak_gib_per_stream": 3.0}, state)[0] == 3.0

    @pytest.mark.parametrize("bad", ["n/a", None, 0, -1, [1]])
    def test_an_unusable_reading_is_treated_as_unknown(self, bad):
        """The contract is "refuse nothing when unknown", and malformed is unknown."""
        assert ps.per_stream_footprint_gib({"peak_gib_per_stream": bad}) == (0.0, "")

    def test_no_model_and_no_measurement_is_unknown(self):
        assert ps.per_stream_footprint_gib({}, None) == (0.0, "")

    def test_an_unreadable_checkpoint_is_unknown_not_fatal(self, monkeypatch):
        state = type("S", (), {"current_best": {}, "model_path": "/nonexistent/model"})()
        assert ps.per_stream_footprint_gib(None, state) == (0.0, "")


class TestRuntimeEnv:
    def test_publishes_the_shape_the_entrypoint_fans_out_on(self):
        env = ps.runtime_env(CPX_36, 2)
        assert env == {
            ps.PARTITION_MODE_ENV: "CPX",
            ps.PARTITION_COUNT_ENV: "8",
            ps.PARTITION_CU_ENV: "32",
            ps.PARTITION_STREAMS_ENV: "2",
            ps.PARTITION_TOTAL_STREAMS_ENV: "16",
        }

    def test_without_a_fan_out_the_topology_is_still_published(self):
        """The platform fingerprint reads it back from here, on the crash path."""
        env = ps.runtime_env(CPX_36, 2, fanout=False)
        assert env == {
            ps.PARTITION_MODE_ENV: "CPX",
            ps.PARTITION_COUNT_ENV: "8",
            ps.PARTITION_CU_ENV: "32",
        }

    def test_without_a_fan_out_no_concurrency_is_stated(self):
        """Streams are directions to a benchmark that places work per partition."""
        env = ps.runtime_env(CPX_36, 2, fanout=False)
        assert ps.PARTITION_STREAMS_ENV not in env
        assert ps.PARTITION_TOTAL_STREAMS_ENV not in env

    def test_it_publishes_no_device_list(self):
        """HIP enumerates whole cards before partitions; only the GPU process can check."""
        assert not any("DEVICE" in key or "VISIBLE" in key for key in ps.runtime_env(CPX_36, 2))

    def test_total_streams_is_the_concurrency_a_fanned_out_entrypoint_should_drive(self):
        assert ps.runtime_env(CPX_36, 3)[ps.PARTITION_TOTAL_STREAMS_ENV] == "24"


class TestSessionShapeSummary:
    def test_an_unknown_shape_is_absent_rather_than_a_second_schema(self):
        """No ``layout is None`` branch: an unknown shape is ``{}`` at the caller.

        The branch that used to answer ``None`` returned four keys where the live
        path returns seven, so ``cu_probed`` and ``fanout_expected`` were missing
        rather than false -- and a missing provenance key is how the report came
        to claim a board-table derivation it had not made.
        """
        with pytest.raises(AttributeError):
            ps.session_shape_summary(None, 2)  # type: ignore[arg-type]
        assert ps.session_shape_summary(SPX_288, 2)["mode"] == "SPX"

    def test_every_key_is_present_on_every_call(self):
        expected = {
            "mode",
            "partitions",
            "cu_per_partition",
            "gib_per_partition",
            "streams_per_partition",
            "cu_probed",
            "fanout_expected",
        }
        assert set(ps.session_shape_summary(SPX_288, 2)) == expected
        assert set(ps.session_shape_summary(CPX_36, 1, fanout_expected=False)) == expected

    def test_it_records_whether_the_cu_count_was_probed(self):
        assert ps.session_shape_summary(CPX_36, 2)["cu_probed"] is True
        assert ps.session_shape_summary(_layout("DPX", 152, probed=False), 2)["cu_probed"] is False

    def test_it_is_json_safe(self):
        import json

        json.dumps(ps.session_shape_summary(CPX_36, 2))

    def test_it_records_whether_anything_will_fan_out(self):
        assert ps.session_shape_summary(CPX_36, 2, fanout_expected=False)["fanout_expected"] is False
        assert ps.session_shape_summary(CPX_36, 2)["fanout_expected"] is True


class TestReportedProvenance:
    """What the report says about where the CU count came from.

    A false provenance line is the one failure this feature exists to prevent,
    so the rendering gets the same scrutiny as the probe.
    """

    @staticmethod
    def _render(shape):
        from hyperloom.orchestrator.actions.executors.report import (
            _format_compute_partition_section,
        )

        return "\n".join(_format_compute_partition_section({"compute_partition": shape}))

    def test_a_probed_count_is_reported_as_probed(self):
        assert "32 (from the device)" in self._render(ps.session_shape_summary(CPX_36, 2))

    def test_a_derived_count_is_reported_as_derived(self):
        shape = ps.session_shape_summary(_layout("CPX", 32, 36.0, probed=False), 2)
        assert "32 (derived from the board table)" in self._render(shape)

    def test_an_unknown_provenance_claims_neither(self):
        """A shape recovered from the published env knows the count, not its origin.

        Truthiness on an absent key read that as the board table, so a fresh
        launch that probed the device reported a guess it never made.
        """
        rendered = self._render({"mode": "CPX", "partitions": 8, "cu_per_partition": 32})

        assert "CU per partition  : 32" in rendered
        assert "board table" not in rendered
        assert "from the device" not in rendered

    def test_a_session_that_cannot_fan_out_is_told_the_figure_is_one_device(self):
        shape = ps.session_shape_summary(CPX_36, 2, fanout_expected=False)
        assert "does not place work on individual partitions" in self._render(shape)

    def test_a_session_that_cannot_fan_out_claims_no_stream_placement(self):
        """A stated total concurrency above a paragraph denying it is a contradiction."""
        shape = ps.session_shape_summary(CPX_36, 2, fanout_expected=False)
        assert "streams/partition" not in self._render(shape)
        assert "streams/partition" in self._render(ps.session_shape_summary(CPX_36, 2))

    def test_an_unpartitioned_card_renders_nothing(self):
        assert self._render(ps.session_shape_summary(SPX_288, 2)) == ""


class TestRecordedShapeIsProvenanceNotADecision:
    def test_the_topology_cannot_be_rewritten_by_update_state(self):
        """Locked for the same reason as model_path: the report prints whatever it says."""
        from hyperloom.orchestrator.policy.gate import CORE_STATE_FIELDS

        assert "compute_partition" in CORE_STATE_FIELDS

    def test_the_published_env_is_a_lossy_subset_of_the_verdict(self):
        """Why the seed is handed the verdict instead of re-reading the environment."""
        from hyperloom.common.gpu_partition import published_shape

        verdict = ps.session_shape_summary(CPX_36, 2)
        from_env = published_shape(ps.runtime_env(CPX_36, 2)) or {}

        assert set(verdict) - set(from_env) == {"cu_probed", "gib_per_partition", "fanout_expected"}


class TestEnvReaders:
    @pytest.mark.parametrize(("raw", "expected"), [("3", 3), ("-1", 0), ("", 0), ("x", 0)])
    def test_the_gpu_id_is_clamped_and_never_raises(self, monkeypatch, raw, expected):
        monkeypatch.setenv(ps.PARTITION_GPU_ENV, raw)
        assert ps.partition_gpu_id() == expected

    @pytest.mark.parametrize("raw", ["abc", "-1", "0x3", "2.5"])
    def test_an_unusable_gpu_id_is_warned_about_not_swallowed(self, monkeypatch, caplog, raw):
        """Card 0's topology filed as the session's is the mislabelling this prevents.

        The reader returns 0 so a bad value cannot crash a launch, but it has to
        say so: the fallback reads a different card, and every number the session
        files afterwards carries that card's shape.
        """
        monkeypatch.setenv(ps.PARTITION_GPU_ENV, raw)
        with caplog.at_level("WARNING"):
            assert ps.partition_gpu_id() == 0

        messages = [r.getMessage() for r in caplog.records]
        assert any(ps.PARTITION_GPU_ENV in m for m in messages)
        assert any(repr(raw) in m for m in messages)
        assert any("may not be this session's" in m for m in messages)

    @pytest.mark.parametrize("raw", ["0", "7", ""])
    def test_a_usable_gpu_id_says_nothing(self, monkeypatch, caplog, raw):
        monkeypatch.setenv(ps.PARTITION_GPU_ENV, raw)
        with caplog.at_level("WARNING"):
            ps.partition_gpu_id()
        assert caplog.records == []

    def test_the_module_exports_no_reader_without_a_caller(self):
        """Two env readers shipped with only their own tests as callers; both are gone."""
        assert not hasattr(ps, "expected_mode")
        assert not hasattr(ps, "streams_per_partition")
        assert "EXPECTED_MODE_ENV" not in ps.__all__
        assert "STREAMS_PER_PARTITION_ENV" not in ps.__all__


class TestFrameworkFanout:
    """The finding this fixes: ``--framework`` defaults to ``None``."""

    @pytest.mark.parametrize("scriptable", ["xdit", "custom", "XDiT"])
    def test_a_scriptable_framework_can_place_work_per_partition(self, scriptable):
        from hyperloom.inference_optimizer.cli import _partition_fanout_supported

        assert _partition_fanout_supported(scriptable) == (True, "")

    @pytest.mark.parametrize("serving", ["vllm", "sglang", "atom"])
    def test_a_serving_framework_is_refused_with_a_reason(self, serving):
        from hyperloom.inference_optimizer.cli import _partition_fanout_supported

        supported, detail = _partition_fanout_supported(serving)
        assert supported is False
        assert "runs a server" in detail

    @pytest.mark.parametrize("unresolved", [None, "", "   "])
    def test_an_unresolved_framework_says_so_rather_than_passing_silently(self, unresolved):
        """A truthiness guard here reads as "checked and fine" while meaning "not checked"."""
        from hyperloom.inference_optimizer.cli import _partition_fanout_supported

        supported, detail = _partition_fanout_supported(unresolved)
        assert supported is False
        assert "not resolved yet" in detail


class TestCliExport:
    """The launch entry point: validates, publishes, or exits 2."""

    @pytest.fixture
    def export(self, card, monkeypatch):
        from hyperloom.inference_optimizer import cli

        monkeypatch.setattr(cli.os, "environ", dict(cli.os.environ), raising=False)

        def call(layout=CPX_36, **kw):
            card(layout)
            kw.setdefault("declared_mode", None)
            kw.setdefault("streams_per_partition", None)
            kw.setdefault("framework", "xdit")
            return cli._export_partition_shape(**kw)

        return call

    def test_a_readable_card_is_published_without_any_flag(self, export):
        """The shape is a measurement property, so it is recorded either way."""
        assert export()["mode"] == "CPX"

    def test_an_unreadable_card_without_a_flag_publishes_nothing(self, export):
        assert export(layout=None) == {}

    def test_a_mode_mismatch_exits_two(self, export):
        with pytest.raises(SystemExit) as exit_info:
            export(layout=SPX_288, declared_mode="CPX")
        assert exit_info.value.code == 2

    def test_a_declared_mode_on_an_unreadable_card_exits_two(self, export):
        with pytest.raises(SystemExit) as exit_info:
            export(layout=None, declared_mode="CPX")
        assert exit_info.value.code == 2

    def test_a_misspelled_mode_exits_two_at_launch(self, export):
        with pytest.raises(SystemExit) as exit_info:
            export(declared_mode="OPX")
        assert exit_info.value.code == 2

    def test_an_infeasible_workload_exits_two(self, export, monkeypatch):
        """The whole point: a refusal at launch instead of an OOM three hours in."""
        monkeypatch.setattr(ps, "per_stream_footprint_gib", lambda *a, **k: (20.7, "measured"))
        with pytest.raises(SystemExit) as exit_info:
            export(streams_per_partition=2)
        assert exit_info.value.code == 2

    def test_the_workload_is_actually_sized_at_launch(self, export, monkeypatch):
        """Without the model path reaching the resolver, the check can only ever warn."""
        seen: dict = {}

        def spy(params=None, shared_state=None):
            seen.update(params or {})
            return 0.0, ""

        monkeypatch.setattr(ps, "per_stream_footprint_gib", spy)
        export(model_path="/models/flux", precision="bf16")
        assert seen.get("model_path") == "/models/flux"
        assert seen.get("precision") == "bf16"

    def test_a_resume_can_use_the_measured_peak(self, export, monkeypatch):
        """A prior measurement rules out a mode the weights alone would fit."""
        state = type("S", (), {"current_best": {"peak_gib_per_stream": 20.7}, "model_path": "", "precision": ""})()
        with pytest.raises(SystemExit) as exit_info:
            export(streams_per_partition=2, shared_state=state)
        assert exit_info.value.code == 2

    @pytest.mark.parametrize("invalid", [0, -1])
    def test_a_non_positive_stream_count_exits_rather_than_defaulting(self, export, invalid):
        """``0 or DEFAULT`` is DEFAULT, which would honour a mistake as the default."""
        with pytest.raises(SystemExit) as exit_info:
            export(streams_per_partition=invalid)
        assert exit_info.value.code == 2

    def test_an_omitted_stream_count_takes_the_default(self, export):
        assert export()["streams_per_partition"] == ps.DEFAULT_STREAMS_PER_PARTITION

    def test_a_resume_re_checks_rather_than_trusts_the_archive(self, export):
        """A card can be repartitioned while a session is stopped."""
        from hyperloom.inference_optimizer import cli

        args = type("A", (), {"compute_partition_mode": None, "streams_per_partition": None})()
        state = type("S", (), {"compute_partition": {"mode": "CPX", "streams_per_partition": 4}})()
        cli._restore_partition_shape_from_state(args, state)

        assert (args.compute_partition_mode, args.streams_per_partition) == ("CPX", 4)
        with pytest.raises(SystemExit):
            export(layout=SPX_288, declared_mode=args.compute_partition_mode)

    def test_a_resume_flag_overrides_the_archive(self, export):
        from hyperloom.inference_optimizer import cli

        args = type("A", (), {"compute_partition_mode": "DPX", "streams_per_partition": 1})()
        state = type("S", (), {"compute_partition": {"mode": "CPX", "streams_per_partition": 4}})()
        cli._restore_partition_shape_from_state(args, state)

        assert (args.compute_partition_mode, args.streams_per_partition) == ("DPX", 1)

    def test_a_resume_of_an_unpartitioned_session_restores_nothing(self, export):
        from hyperloom.inference_optimizer import cli

        args = type("A", (), {"compute_partition_mode": None, "streams_per_partition": None})()
        cli._restore_partition_shape_from_state(args, type("S", (), {"compute_partition": {}})())

        assert (args.compute_partition_mode, args.streams_per_partition) == ("", None)

    def test_a_serving_session_on_a_split_card_is_not_refused_without_flags(self, export, monkeypatch):
        """The reported bug: a plain sglang run exited 2 on a card someone else left in CPX.

        No flags, so no fan-out was ever asked for, and a serving benchmark
        cannot do one. Refusing on ``2 x footprint`` was arithmetic about a
        shape the session was never going to run in.
        """
        monkeypatch.setattr(ps, "per_stream_footprint_gib", lambda *a, **k: (20.7, "weights"))
        shape = export(framework="sglang")

        assert shape["mode"] == "CPX"
        assert shape["fanout_expected"] is False

    def test_the_same_session_is_still_refused_when_the_operator_asks_for_the_shape(self, export, monkeypatch):
        """Naming the flags asserts the shape, and an assertion is held to."""
        monkeypatch.setattr(ps, "per_stream_footprint_gib", lambda *a, **k: (20.7, "weights"))
        with pytest.raises(SystemExit) as exit_info:
            export(framework="sglang", streams_per_partition=2)
        assert exit_info.value.code == 2

    def test_a_scriptable_session_is_still_refused_without_flags(self, export, monkeypatch):
        """Where the fan-out is real, the default of two streams is a real premise."""
        monkeypatch.setattr(ps, "per_stream_footprint_gib", lambda *a, **k: (20.7, "weights"))
        with pytest.raises(SystemExit) as exit_info:
            export(framework="xdit")
        assert exit_info.value.code == 2

    def test_the_runtime_handoff_is_published_only_where_something_reads_it(self, export):
        from hyperloom.inference_optimizer import cli

        export(framework="xdit")
        assert cli.os.environ.get(ps.PARTITION_MODE_ENV) == "CPX"

    def test_a_serving_framework_is_handed_no_fan_out_instruction(self, export):
        """Stating a concurrency nothing will drive is the contradiction here."""
        from hyperloom.inference_optimizer import cli

        assert export(framework="sglang")["mode"] == "CPX"
        assert ps.PARTITION_STREAMS_ENV not in cli.os.environ
        assert ps.PARTITION_TOTAL_STREAMS_ENV not in cli.os.environ

    def test_a_serving_framework_still_publishes_the_topology(self, export):
        """Otherwise the platform fingerprint loses the mode -- the whole point."""
        from hyperloom.inference_optimizer import cli

        export(framework="sglang")
        assert cli.os.environ.get(ps.PARTITION_MODE_ENV) == "CPX"

    def test_the_summary_carries_what_the_env_cannot(self, export):
        """The published variables are a lossy subset: no provenance, no memory."""
        shape = export(framework="xdit")
        assert shape["cu_probed"] is True
        assert shape["gib_per_partition"] == 36.0

    def test_a_multi_node_session_records_no_shape(self, export):
        """The card this process can read is not the card the benchmark ran on."""
        assert export(nodes=2) == {}

    def test_a_multi_node_session_publishes_nothing(self, export):
        """Not even the topology: it would be the wrong node's."""
        from hyperloom.inference_optimizer import cli

        export(nodes=4)
        assert ps.PARTITION_MODE_ENV not in cli.os.environ

    def test_a_multi_node_assertion_exits_rather_than_going_unchecked(self, export):
        """Same rule as an unreadable card: unverifiable is not satisfied."""
        with pytest.raises(SystemExit) as exit_info:
            export(nodes=2, declared_mode="CPX")
        assert exit_info.value.code == 2

    def test_an_explicit_zero_on_resume_still_reaches_the_guard(self, export):
        """Falsiness here would read the mistake as "omitted" and paper over it."""
        from hyperloom.inference_optimizer import cli

        args = type("A", (), {"compute_partition_mode": None, "streams_per_partition": 0})()
        state = type("S", (), {"compute_partition": {"mode": "CPX", "streams_per_partition": 4}})()
        cli._restore_partition_shape_from_state(args, state)

        assert args.streams_per_partition == 0
        with pytest.raises(SystemExit) as exit_info:
            export(streams_per_partition=args.streams_per_partition)
        assert exit_info.value.code == 2

    def test_a_stale_shape_cannot_be_inherited_from_the_shell(self, export, monkeypatch):
        """A second session in the same shell must not adopt the first one's shape."""
        monkeypatch.setenv(ps.PARTITION_MODE_ENV, "CPX")
        monkeypatch.setenv(ps.PARTITION_COUNT_ENV, "8")
        export(layout=None)
        assert ps.PARTITION_MODE_ENV not in ps.os.environ
