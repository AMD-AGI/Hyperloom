# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for reading a card's compute-partition shape.

Every entry point in the module under test is a read, so the tests supply
``amd-smi`` payloads rather than asserting on commands issued. The payload
shapes are the real ones: a partitioned device reports its *partition's* CU
count and VRAM, which is the fact the module is built around.
"""

from __future__ import annotations

import pytest

from hyperloom.common import gpu_partition as gp


@pytest.fixture
def smi(monkeypatch):
    """Route every amd-smi read through a scripted payload table."""
    payloads: dict[str, object] = {}
    calls: list[list[str]] = []

    def fake(args, timeout_s=gp._READ_TIMEOUT_S):
        argv = list(args)
        calls.append(argv)
        for key, payload in payloads.items():
            if key in " ".join(argv):
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise gp.PartitionError(f"no scripted payload for {argv}")

    monkeypatch.setattr(gp, "_amd_smi_json", fake)
    return type("Smi", (), {"payloads": payloads, "calls": calls})()


def _partition_payload(mode: str, gpu_id: int = 0) -> dict:
    return {"current_partition": [{"gpu_id": gpu_id, "accelerator_type": mode}]}


def _vram_payload(value: float, unit: str = "MB") -> dict:
    return {"gpu_data": [{"vram": {"size": {"value": value, "unit": unit}}}]}


def _asic_payload(cu: int) -> dict:
    return {"gpu_data": [{"asic": {"num_compute_units": cu}}]}


class TestParseMode:
    def test_canonicalizes_case_and_whitespace(self):
        assert gp.parse_mode("  cpx ") == "CPX"

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_nothing_named_is_empty(self, raw):
        assert gp.parse_mode(raw) == ""

    def test_an_unknown_mode_is_a_parse_error(self):
        """A typo must fail at parse time, not as an assertion that never holds."""
        with pytest.raises(gp.PartitionError, match="unknown compute-partition mode"):
            gp.parse_mode("opx")

    def test_the_error_lists_what_is_valid(self):
        with pytest.raises(gp.PartitionError, match="SPX"):
            gp.parse_mode("octuple")


class TestReadPartitionMode:
    def test_reads_the_mode_of_the_named_card(self, smi):
        smi.payloads["partition"] = {
            "current_partition": [
                {"gpu_id": 0, "accelerator_type": "SPX"},
                {"gpu_id": 3, "accelerator_type": "CPX"},
            ]
        }
        assert gp.read_partition_mode(3) == "CPX"

    def test_a_card_absent_from_the_report_raises(self, smi):
        smi.payloads["partition"] = _partition_payload("SPX", gpu_id=0)
        with pytest.raises(gp.PartitionError, match="no state for GPU 7"):
            gp.read_partition_mode(7)

    def test_an_empty_report_raises_rather_than_returning_nothing(self, smi):
        smi.payloads["partition"] = {"current_partition": []}
        with pytest.raises(gp.PartitionError, match="reported no compute-partition state"):
            gp.read_partition_modes()

    def test_rows_without_a_usable_gpu_id_are_skipped(self, smi):
        smi.payloads["partition"] = {
            "current_partition": [
                {"gpu_id": "n/a", "accelerator_type": "SPX"},
                {"gpu_id": 1, "accelerator_type": "DPX"},
            ]
        }
        assert gp.read_partition_modes() == {1: "DPX"}


class TestReadDeviceCu:
    def test_reads_the_devices_own_cu_count(self, smi):
        smi.payloads["--asic"] = _asic_payload(32)
        assert gp.read_device_cu(0) == 32

    @pytest.mark.parametrize("key", ["num_compute_units", "compute_units", "num_cu", "cu_count"])
    def test_each_field_name_in_use_is_accepted(self, smi, key):
        """The schema has moved between releases; a miss sends us to the table."""
        smi.payloads["--asic"] = {"gpu_data": [{"asic": {key: 64}}]}
        assert gp.read_device_cu(0) == 64

    def test_a_flat_payload_without_an_asic_block_still_parses(self, smi):
        smi.payloads["--asic"] = {"gpu_data": [{"num_compute_units": 256}]}
        assert gp.read_device_cu(0) == 256

    def test_an_unreadable_probe_is_none_and_warns(self, smi, caplog):
        smi.payloads["--asic"] = gp.PartitionError("amd-smi absent")
        with caplog.at_level("WARNING"):
            assert gp.read_device_cu(0) is None
        assert "fall back to the board table" in caplog.text

    def test_a_missing_count_is_none_and_warns(self, smi, caplog):
        smi.payloads["--asic"] = {"gpu_data": [{"asic": {"market_name": "MI355X"}}]}
        with caplog.at_level("WARNING"):
            assert gp.read_device_cu(0) is None
        assert "no compute-unit count" in caplog.text

    @pytest.mark.parametrize("bad", [0, -8, "n/a", None])
    def test_an_unusable_value_is_not_taken(self, smi, bad):
        smi.payloads["--asic"] = {"gpu_data": [{"asic": {"num_compute_units": bad}}]}
        assert gp.read_device_cu(0) is None


class TestReadDeviceGib:
    def test_mebibytes_are_the_binary_kind(self, smi):
        """An MI355X's 288 GiB is reported as 294896 "MB"."""
        smi.payloads["--vram"] = _vram_payload(294896, "MB")
        assert gp.read_device_gib(0) == pytest.approx(288.0, abs=0.1)

    def test_a_partitioned_device_reports_its_partitions_memory(self, smi):
        """Under CPX a device *is* a partition, so this needs no dividing."""
        smi.payloads["--vram"] = _vram_payload(36862, "MB")
        assert gp.read_device_gib(0) == pytest.approx(36.0, abs=0.1)

    @pytest.mark.parametrize(
        ("value", "unit", "expected"),
        [(288, "GIB", 288.0), (288, "GB", 288.0), (1, "TIB", 1024.0)],
    )
    def test_other_units_scale(self, smi, value, unit, expected):
        smi.payloads["--vram"] = _vram_payload(value, unit)
        assert gp.read_device_gib(0) == pytest.approx(expected)

    def test_an_unreadable_probe_warns_rather_than_debugs(self, smi, caplog):
        """Losing this figure skips the feasibility check, which is the louder event."""
        smi.payloads["--vram"] = gp.PartitionError("amd-smi failed")
        with caplog.at_level("WARNING"):
            assert gp.read_device_gib(0) is None
        assert "feasibility will not be checked" in caplog.text

    def test_an_unknown_unit_is_not_guessed(self, smi):
        smi.payloads["--vram"] = _vram_payload(288, "parsecs")
        assert gp.read_device_gib(0) is None


class TestLayoutFor:
    def test_a_probed_cu_count_is_used_verbatim(self):
        layout = gp.layout_for("CPX", cu_per_partition=32, gib_per_partition=36.0)
        assert (layout.partitions, layout.cu_per_partition, layout.probed) == (8, 32, True)

    def test_a_probed_count_does_not_consult_the_board_table(self):
        """The table conflates boards that share an ISA; the device does not."""
        layout = gp.layout_for("DPX", gpu_type="mi308x", cu_per_partition=40)
        assert layout.cu_per_partition == 40
        assert layout.probed is True

    def test_the_table_is_the_fallback_and_says_so(self):
        layout = gp.layout_for("DPX", gpu_type="mi300x")
        assert (layout.partitions, layout.cu_per_partition) == (2, 152)
        assert layout.probed is False
        assert "derived" in layout.describe()

    def test_an_unsizable_board_without_a_probe_raises(self):
        with pytest.raises(gp.PartitionError, match="not in the board table"):
            gp.layout_for("CPX", gpu_type="some-new-board")

    def test_an_uneven_division_raises_rather_than_flooring(self, monkeypatch):
        """A floored CU count matches no device, and reports the wrong cause.

        Every board in the table divides evenly today, so the guard is exercised
        against an injected entry -- it exists for the one that does not.
        """
        monkeypatch.setitem(gp.AMD_GPU_DISPATCH_IDENTITIES, "oddball", ("gfx950", 300))
        with pytest.raises(gp.PartitionError, match="does not divide"):
            gp.layout_for("CPX", gpu_type="oddball")

    def test_an_unknown_mode_raises(self):
        with pytest.raises(gp.PartitionError, match="unknown compute-partition mode"):
            gp.layout_for("OPX", cu_per_partition=32)

    def test_spx_is_one_partition_and_not_partitioned(self):
        layout = gp.layout_for("SPX", cu_per_partition=256)
        assert layout.partitions == 1
        assert layout.partitioned is False


class TestObservePartition:
    def test_describes_the_live_topology_from_the_device(self, smi):
        smi.payloads["partition"] = _partition_payload("CPX")
        smi.payloads["--asic"] = _asic_payload(32)
        smi.payloads["--vram"] = _vram_payload(36862, "MB")

        layout = gp.observe_partition(0)

        assert layout is not None
        assert (layout.mode, layout.partitions, layout.cu_per_partition) == ("CPX", 8, 32)
        assert layout.gib_per_partition == pytest.approx(36.0, abs=0.1)
        assert layout.probed is True

    def test_an_unreadable_card_is_none_not_an_error(self, smi):
        """A host without amd-smi is the ordinary case, not a failure."""
        smi.payloads["partition"] = gp.PartitionError("amd-smi not found")
        assert gp.observe_partition(0) is None

    def test_a_mode_this_build_does_not_know_is_refused_loudly(self, smi, caplog):
        smi.payloads["partition"] = _partition_payload("OPX")
        with caplog.at_level("WARNING"):
            assert gp.observe_partition(0) is None
        assert "does not know" in caplog.text

    def test_falls_back_to_the_table_when_the_cu_probe_fails(self, smi):
        smi.payloads["partition"] = _partition_payload("DPX")
        smi.payloads["--asic"] = gp.PartitionError("no asic block")
        smi.payloads["--vram"] = _vram_payload(294896, "MB")

        layout = gp.observe_partition(0, gpu_type="mi300x")

        assert layout is not None
        assert layout.cu_per_partition == 152
        assert layout.probed is False


class TestFitsInPartition:
    def test_the_measured_mi355x_case_does_not_fit_cpx_in_pairs(self):
        """20.7 GiB per stream, two streams, 36 GiB partition."""
        layout = gp.layout_for("CPX", cu_per_partition=32, gib_per_partition=36.0)
        assert gp.fits_in_partition(20.7, layout, 2) is False

    def test_the_same_footprint_fits_a_single_stream(self):
        """Which is why gating on the one-stream figure is the trap."""
        layout = gp.layout_for("CPX", cu_per_partition=32, gib_per_partition=36.0)
        assert gp.fits_in_partition(20.7, layout, 1) is True

    def test_unknown_capacity_does_not_refuse(self):
        layout = gp.layout_for("CPX", cu_per_partition=32)
        assert gp.fits_in_partition(999.0, layout, 2) is True

    def test_unknown_footprint_does_not_refuse(self):
        layout = gp.layout_for("CPX", cu_per_partition=32, gib_per_partition=36.0)
        assert gp.fits_in_partition(0.0, layout, 2) is True

    def test_exactly_filling_a_partition_fits(self):
        layout = gp.layout_for("CPX", cu_per_partition=32, gib_per_partition=36.0)
        assert gp.fits_in_partition(18.0, layout, 2) is True

    def test_a_zero_capacity_is_unknown_rather_than_a_limit(self):
        """Shared with the caller's guard, which used to test ``is None`` instead."""
        layout = gp.PartitionLayout(mode="CPX", partitions=8, cu_per_partition=32, gib_per_partition=0.0)
        assert layout.capacity_known is False
        assert gp.fits_in_partition(999.0, layout, 2) is True


class TestCapacityKnown:
    """One predicate for "is this capacity worth checking against".

    Two call sites asked the question with different tests -- ``is None`` in the
    validator and falsiness in the arithmetic -- so a zero took opposite paths
    through them.
    """

    @pytest.mark.parametrize(
        ("gib", "known"),
        [(36.0, True), (0.1, True), (None, False), (0.0, False), (-1.0, False)],
    )
    def test_only_a_positive_capacity_is_known(self, gib, known):
        layout = gp.PartitionLayout(mode="CPX", partitions=8, cu_per_partition=32, gib_per_partition=gib)
        assert layout.capacity_known is known


class TestPartitionDevicePredicate:
    def test_matches_a_partition_and_rejects_a_whole_card(self):
        """HIP lists whole cards first, so index selection measures the wrong device."""
        is_partition = gp.partition_device_predicate(32)
        assert is_partition(32) is True
        assert is_partition(256) is False

    def test_its_docstring_admits_it_has_no_in_tree_caller(self):
        """Otherwise the next reader goes looking for the consumer and finds none."""
        assert "No caller in this repository" in (gp.partition_device_predicate.__doc__ or "")


class TestPublicSurface:
    def test_the_dead_unpartitioned_mode_constant_is_gone(self):
        """Defined and exported, referenced nowhere -- including by its own tests."""
        assert not hasattr(gp, "UNPARTITIONED_MODE")
        assert "UNPARTITIONED_MODE" not in gp.__all__

    @pytest.mark.parametrize(
        "name",
        ["read_partition_mode", "read_partition_modes", "read_device_cu", "read_device_gib"],
    )
    def test_the_probe_helpers_behind_observe_partition_are_not_advertised(self, name):
        """``observe_partition`` calls itself the single entry point a caller needs.

        Listing the four steps it takes contradicted that: they are its call
        graph, not the module's interface. They stay importable for the tests
        that exercise each amd-smi payload shape.
        """
        assert name not in gp.__all__
        assert callable(getattr(gp, name))

    def test_everything_advertised_exists(self):
        for name in gp.__all__:
            assert hasattr(gp, name), name


class TestPublishedShape:
    def test_reads_back_what_a_launch_published(self):
        env = {
            gp.PARTITION_MODE_ENV: "CPX",
            gp.PARTITION_COUNT_ENV: "8",
            gp.PARTITION_CU_ENV: "32",
            gp.PARTITION_STREAMS_ENV: "2",
        }
        assert gp.published_shape(env) == {
            "mode": "CPX",
            "partitions": 8,
            "cu_per_partition": 32,
            "streams_per_partition": 2,
        }

    def test_no_published_mode_is_none(self):
        """Distinct from an unpartitioned card, which is a known SPX."""
        assert gp.published_shape({}) is None

    def test_unparseable_numbers_are_omitted_not_zeroed(self):
        env = {gp.PARTITION_MODE_ENV: "DPX", gp.PARTITION_COUNT_ENV: "lots"}
        assert gp.published_shape(env) == {"mode": "DPX"}

    def test_it_reads_no_device(self, monkeypatch):
        """This runs on the crash path, where spawning a probe is unacceptable."""

        def explode(*args, **kwargs):
            raise AssertionError("published_shape must not run amd-smi")

        monkeypatch.setattr(gp, "_amd_smi_json", explode)
        assert gp.published_shape({gp.PARTITION_MODE_ENV: "SPX"}) == {"mode": "SPX"}
