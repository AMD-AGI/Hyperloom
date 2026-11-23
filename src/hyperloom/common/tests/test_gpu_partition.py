# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for reading a card's compute-partition shape."""

from __future__ import annotations

import pytest

from hyperloom.common import gpu_partition as gp
from hyperloom_common.gpu_card import GpuCard


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


class TestParseMode:
    def test_canonicalizes_case_and_whitespace(self):
        assert gp.parse_mode("  cpx ") == "CPX"

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_nothing_named_is_empty(self, raw):
        assert gp.parse_mode(raw) == ""

    def test_an_unknown_mode_is_a_parse_error(self):
        """A typo must fail at parse time, not as an assertion that never holds."""
        with pytest.raises(gp.PartitionError, match="unknown compute-partition mode .*expected one of .*SPX"):
            gp.parse_mode("opx")


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


class TestLayoutFor:
    def test_the_device_cu_count_is_one_partitions(self):
        """A partitioned card presents each partition as a device of its own."""
        layout = gp.layout_for("CPX", cu_per_partition=32, gib_per_partition=36.0)
        assert (layout.partitions, layout.cu_per_partition, layout.gib_per_partition) == (8, 32, 36.0)
        assert layout.describe() == "CPX (8 x 32 CU, 36 GiB)"

    def test_an_unknown_mode_raises(self):
        with pytest.raises(gp.PartitionError, match="unknown compute-partition mode"):
            gp.layout_for("OPX", cu_per_partition=32)

    def test_spx_is_one_partition_and_not_partitioned(self):
        layout = gp.layout_for("SPX", cu_per_partition=256)
        assert layout.partitions == 1
        assert layout.partitioned is False


#: One CPX partition of an MI355X, as amd-smi describes the device: 32 CUs and an eighth of 288 GiB.
_CPX_PARTITION = GpuCard("AMD Instinct MI355X", "75a3", "gfx950", 32, 36862, 1000.0, 2400)


class TestObservePartition:
    def test_the_partitions_are_sized_by_the_session_card(self, smi):
        smi.payloads["partition"] = _partition_payload("CPX")

        layout = gp.observe_partition(0, card=_CPX_PARTITION)

        assert layout is not None
        assert (layout.mode, layout.partitions, layout.cu_per_partition) == ("CPX", 8, 32)
        assert layout.gib_per_partition == pytest.approx(36.0, abs=0.1)
        assert [call[0] for call in smi.calls] == ["partition"]

    @pytest.mark.parametrize("card", [None, GpuCard("", "", "gfx950", None, 294912, None, None)])
    def test_a_card_in_a_mode_without_a_compute_unit_count_is_unsized_and_says_so(self, smi, caplog, card):
        smi.payloads["partition"] = _partition_payload("DPX")
        with caplog.at_level("WARNING"):
            assert gp.observe_partition(0, card=card) is None
        assert "no compute-unit count" in caplog.text

    def test_an_unreadable_card_is_none_not_an_error(self, smi):
        """A host without amd-smi is the ordinary case, not a failure."""
        smi.payloads["partition"] = gp.PartitionError("amd-smi not found")
        assert gp.observe_partition(0, card=_CPX_PARTITION) is None

    def test_a_mode_this_build_does_not_know_is_refused_loudly(self, smi, caplog):
        smi.payloads["partition"] = _partition_payload("OPX")
        with caplog.at_level("WARNING"):
            assert gp.observe_partition(0, card=_CPX_PARTITION) is None
        assert "does not know" in caplog.text


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
    """One predicate for \"is this capacity worth checking against\"."""

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
