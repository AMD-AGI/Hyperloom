# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free tests for ``scripts/partition_mode_sweep.py``.

The script sits outside the coverage denominator because ``scripts/`` is not
shipped as a package, but the logic tested here decides which silicon a
benchmark runs on -- and getting that wrong produces a plausible number
attributed to the wrong mode. So the parsing, device selection and aggregation
are pinned here. Nothing below sets a partition mode, shells out, or needs a
GPU.

Payload shapes are the ones observed on an 8-card MI355X node rather than
invented: ``amd-smi`` reporting an idle card's ``process_list`` as a bare
string, ``rocminfo`` putting ``BDFID`` before ``Compute Unit`` in each agent
block, and HSA enumerating whole cards ahead of partitions.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "partition_mode_sweep.py"


def _load():
    spec = importlib.util.spec_from_file_location("partition_mode_sweep", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: the module's dataclasses resolve their annotations
    # through sys.modules, and a loader that skips this raises AttributeError on
    # the first field with a default_factory.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pms = _load()


#: One MI355X (card 0, PCI bus 0x09) in CPX beside seven untouched cards, as
#: ``rocminfo`` actually printed it. Two properties matter and both are load
#: bearing: ``BDFID`` precedes ``Compute Unit``, and the whole cards come first.
ROCMINFO_CPX = """
Agent 1
  Name:                    AMD EPYC 9575F 64-Core Processor
  Device Type:             CPU
  Compute Unit:            128
Agent 2
  Name:                    AMD EPYC 9575F 64-Core Processor
  Device Type:             CPU
  Compute Unit:            128
Agent 3
  Name:                    gfx950
  Device Type:             GPU
  BDFID:                   31744
  Compute Unit:            256
Agent 4
  Name:                    gfx950
  Device Type:             GPU
  BDFID:                   26880
  Compute Unit:            256
Agent 5
  Name:                    gfx950
  Device Type:             GPU
  BDFID:                   2304
  Compute Unit:            32
Agent 6
  Name:                    gfx950
  Device Type:             GPU
  BDFID:                   2305
  Compute Unit:            32
"""


# ------------------------------------------------------- rocminfo parsing


class TestParseHsaAgents:
    def test_cu_is_not_shifted_by_the_field_order(self):
        """BDFID precedes Compute Unit, so a line-at-a-time parser is off by one.

        Printing on ``BDFID`` with the last-seen CU attributes each agent the
        previous one's count -- which, on the node this came from, silently
        turned the first CPX partition into a 256-CU card and would have pointed
        the benchmark at whole silicon while labelling it CPX.
        """
        agents = pms.parse_hsa_agents(ROCMINFO_CPX)
        by_bdf = {a.bdf: a.cu for a in agents}
        assert by_bdf == {"7c:00.0": 256, "69:00.0": 256, "09:00.0": 32, "09:00.1": 32}

    def test_cpu_agents_are_not_gpus(self):
        agents = pms.parse_hsa_agents(ROCMINFO_CPX)
        assert len(agents) == 4
        assert all(a.cu in (32, 256) for a in agents)

    def test_indices_are_positions_among_gpus_only(self):
        """The index has to be the one ROCR_VISIBLE_DEVICES uses, so CPUs cannot count."""
        agents = pms.parse_hsa_agents(ROCMINFO_CPX)
        assert [a.index for a in agents] == [0, 1, 2, 3]

    def test_bdfid_decodes_to_bus_device_function(self):
        agents = pms.parse_hsa_agents(ROCMINFO_CPX)
        first = next(a for a in agents if a.bdf == "09:00.0")
        assert (first.bus, first.device, first.function) == (0x09, 0x00, 0)

    def test_partitions_of_one_card_share_a_bus_and_differ_by_function(self):
        agents = pms.parse_hsa_agents(ROCMINFO_CPX)
        on_card = [a for a in agents if a.bus == 0x09]
        assert {a.function for a in on_card} == {0, 1}

    def test_an_agent_missing_its_fields_is_dropped_not_guessed(self):
        assert pms.parse_hsa_agents("Agent 1\n  Device Type:             GPU\n") == ()

    def test_empty_input_is_no_agents(self):
        assert pms.parse_hsa_agents("") == ()


# --------------------------------------------------------- device selection


def _agents(*specs: tuple[int, int, int]) -> tuple:
    """Build agents from ``(cu, bus, function)`` triples in enumeration order."""
    return tuple(pms.HsaAgent(index=i, cu=cu, bus=bus, device=0, function=fn) for i, (cu, bus, fn) in enumerate(specs))


class TestSelectPartitionDevices:
    def test_whole_cards_enumerate_first_so_indices_are_not_zero_based(self):
        """The measured trap: under DPX the partitions are HIP devices 7 and 8.

        Seven untouched cards take indices 0-6, so a driver that assumed the
        partitions of the card it just split start at 0 would benchmark a
        neighbour and report the number as DPX.
        """
        agents = _agents(*[(256, 0x19 + i, 0) for i in range(7)], (128, 0x09, 0), (128, 0x09, 1))
        layout = pms.layout_for("DPX", cu_per_partition=128)
        assert pms.select_partition_devices(agents, layout, bus=0x09) == (7, 8)

    def test_spx_does_not_collect_the_untouched_neighbours(self):
        """SPX's "partition" is a whole card, so CU alone matches all eight."""
        agents = _agents(*[(256, 0x19 + i, 0) for i in range(7)], (256, 0x09, 0))
        layout = pms.layout_for("SPX", cu_per_partition=256)
        assert pms.select_partition_devices(agents, layout, bus=0x09) == (7,)

    def test_a_set_that_did_not_take_is_an_error_not_a_short_list(self):
        """Finding fewer partitions than the mode implies means the topology is stale."""
        agents = _agents((256, 0x09, 0))
        layout = pms.layout_for("CPX", cu_per_partition=32)
        with pytest.raises(pms.SweepError, match="CPX implies 8 partitions"):
            pms.select_partition_devices(agents, layout, bus=0x09)

    def test_the_error_names_what_was_actually_on_the_bus(self):
        agents = _agents((256, 0x09, 0))
        layout = pms.layout_for("CPX", cu_per_partition=32)
        with pytest.raises(pms.SweepError, match=r"09:00\.0=256CU"):
            pms.select_partition_devices(agents, layout, bus=0x09)

    def test_nothing_on_the_bus_says_so(self):
        layout = pms.layout_for("SPX", cu_per_partition=256)
        with pytest.raises(pms.SweepError, match="nothing on that bus"):
            pms.select_partition_devices(_agents((256, 0x19, 0)), layout, bus=0x09)

    def test_selection_uses_the_shared_predicate(self):
        """The rule lives in gpu_partition; this script must not re-invent it."""
        agents = _agents((256, 0x09, 0), (32, 0x09, 1))
        assert pms.partition_device_predicate(32)(32) is True
        assert pms.partition_device_predicate(32)(256) is False
        layout = pms.layout_for("DPX", cu_per_partition=32)
        with pytest.raises(pms.SweepError):
            # Two devices on the bus, but only one is 32 CU: DPX wants two.
            pms.select_partition_devices(agents, layout, bus=0x09)


class TestPartitionCuOnBus:
    def test_a_settled_card_reports_one_width(self):
        assert pms.partition_cu_on_bus(_agents((32, 0x09, 0), (32, 0x09, 1)), 0x09) == 32

    def test_neighbours_do_not_contribute(self):
        assert pms.partition_cu_on_bus(_agents((32, 0x09, 0), (256, 0x19, 0)), 0x09) == 32

    def test_mixed_widths_mean_the_read_was_mid_transition(self):
        with pytest.raises(pms.SweepError, match="mid-transition"):
            pms.partition_cu_on_bus(_agents((32, 0x09, 0), (256, 0x09, 1)), 0x09)

    def test_an_empty_bus_is_an_error(self):
        with pytest.raises(pms.SweepError, match="no GPU on bus"):
            pms.partition_cu_on_bus(_agents((256, 0x19, 0)), 0x09)


# ------------------------------------------------------------ amd-smi payloads


class TestResidentProcesses:
    def test_an_idle_card_reports_a_string_not_an_empty_list(self):
        """The observed payload. Counting it as a process refuses every free node."""
        payload = [{"gpu": 0, "process_list": [{"process_info": "No running processes detected"}]}]
        assert pms.resident_processes(payload) == {0: 0}

    def test_a_real_process_is_counted(self):
        payload = [{"gpu": 0, "process_list": [{"process_info": {"name": "python", "pid": 42}}]}]
        assert pms.resident_processes(payload) == {0: 1}

    def test_a_nameless_entry_is_not_a_process(self):
        payload = [{"gpu": 0, "process_list": [{"process_info": {"name": "  "}}]}]
        assert pms.resident_processes(payload) == {0: 0}

    def test_an_unexpected_string_is_counted_rather_than_ignored(self):
        """Fail closed: an unrecognised sentinel might really be a process."""
        payload = [{"gpu": 0, "process_list": [{"process_info": "something new"}]}]
        assert pms.resident_processes(payload) == {0: 1}

    def test_cards_are_reported_separately(self):
        payload = [
            {"gpu": 0, "process_list": [{"process_info": "No running processes detected"}]},
            {"gpu": 1, "process_list": [{"process_info": {"name": "vllm"}}]},
        ]
        assert pms.resident_processes(payload) == {0: 0, 1: 1}

    def test_the_idle_sentinel_is_also_recognised_unwrapped(self):
        """Observed at both depths: as the list, and as the entry's process_info."""
        assert pms.resident_processes([{"gpu": 0, "process_list": "No running processes detected"}]) == {0: 0}

    def test_a_process_is_counted_without_the_process_info_wrapper(self):
        assert pms.resident_processes([{"gpu": 0, "process_list": [{"name": "python", "pid": 42}]}]) == {0: 1}

    def test_an_empty_list_is_an_idle_card(self):
        assert pms.resident_processes([{"gpu": 0, "process_list": []}]) == {0: 0}


class TestResidentProcessesRefusesDrift:
    """Schema drift must raise, never read as "nobody is using the node".

    This count is the only thing between an ``amd-smi`` payload this parser does
    not understand and a partition set that evicts whatever is running. Every
    case below used to return ``{}`` or silently skip the row, which the sweep
    read as an idle node and acted on.
    """

    @pytest.mark.parametrize(
        "payload",
        [
            {"process": [{"gpu": 0, "process_list": []}]},  # a wrapper key appears
            "No running processes detected",  # the whole payload degrades to a string
            None,  # the query answers nothing
            42,
        ],
    )
    def test_a_payload_that_is_not_a_list_of_rows_is_refused(self, payload):
        with pytest.raises(pms.SweepError, match="amd-smi process returned"):
            pms.resident_processes(payload)

    def test_a_row_that_is_not_a_mapping_is_refused(self):
        with pytest.raises(pms.SweepError, match="where a GPU row was expected"):
            pms.resident_processes([["gpu", 0]])

    @pytest.mark.parametrize("row", [{"process_list": []}, {"gpu": 0}, {}])
    def test_a_row_missing_either_field_is_refused(self, row):
        with pytest.raises(pms.SweepError, match="without gpu|without process_list|without gpu or process_list"):
            pms.resident_processes([row])

    def test_a_renamed_gpu_id_is_refused_rather_than_skipped(self):
        """A row filed under a name this parser cannot read is not an absent row."""
        with pytest.raises(pms.SweepError, match="not a number"):
            pms.resident_processes([{"gpu": "card0", "process_list": []}])

    def test_a_process_list_of_an_unmodelled_type_is_refused(self):
        with pytest.raises(pms.SweepError, match="process_list of type dict"):
            pms.resident_processes([{"gpu": 0, "process_list": {"pid": 42}}])

    def test_a_process_entry_of_an_unmodelled_type_is_refused(self):
        with pytest.raises(pms.SweepError, match="process entry of type NoneType"):
            pms.resident_processes([{"gpu": 0, "process_list": [None]}])


class TestCurrentModes:
    def test_reads_the_observed_payload(self):
        payload = {
            "current_partition": [
                {"gpu_id": 0, "memory": "NPS1", "accelerator_type": "SPX"},
                {"gpu_id": 1, "memory": "NPS1", "accelerator_type": "CPX"},
            ]
        }
        assert pms.current_modes(payload) == {0: "SPX", 1: "CPX"}

    def test_the_current_marker_asterisk_is_stripped(self):
        payload = {"current_partition": [{"gpu_id": 0, "accelerator_type": "SPX*"}]}
        assert pms.current_modes(payload) == {0: "SPX"}

    def test_not_available_is_absent_rather_than_a_mode(self):
        payload = {"current_partition": [{"gpu_id": 0, "accelerator_type": "N/A"}]}
        assert pms.current_modes(payload) == {}


class TestSupportedModes:
    def test_only_named_rows_describe_a_profile(self):
        """The profile table continues each profile in rows with blank identity."""
        payload = {
            "partition_profiles": [
                {"profile_index": 0, "accelerator_type": "SPX*", "num_partitions": 1},
                {"profile_index": None, "accelerator_type": "", "resource_type": "DMA"},
                {"profile_index": 3, "accelerator_type": "CPX", "num_partitions": 8},
            ]
        }
        assert pms.supported_modes(payload) == ("SPX", "CPX")

    def test_an_unprivileged_query_reports_nothing_not_everything(self):
        payload = {"partition_profiles": [{"accelerator_type": "N/A"}]}
        assert pms.supported_modes(payload) == ()

    def test_duplicates_collapse(self):
        payload = {
            "partition_profiles": [
                {"accelerator_type": "DPX"},
                {"accelerator_type": "DPX"},
            ]
        }
        assert pms.supported_modes(payload) == ("DPX",)


class TestCardBus:
    def test_reads_the_bus_from_a_domain_qualified_address(self):
        assert pms.card_bus([{"gpu": 0, "bdf": "0000:09:00.0"}], 0) == 0x09

    def test_reads_the_bus_without_a_domain(self):
        assert pms.card_bus([{"gpu": 1, "bdf": "7c:00.0"}], 1) == 0x7C

    def test_a_missing_card_is_an_error(self):
        with pytest.raises(pms.SweepError, match="no PCI address"):
            pms.card_bus([{"gpu": 0, "bdf": "0000:09:00.0"}], 3)


# -------------------------------------------------------------- the fan-out


class TestBuildPartitionCommand:
    def test_substitutes_the_per_partition_values(self):
        layout = pms.layout_for("CPX", cu_per_partition=32)
        cmd = pms.build_partition_command(
            ["bench", "--gpu", "{device}", "--out", "{output_dir}", "--mode", "{mode}"],
            device=9,
            output_dir=Path("/tmp/out"),
            layout=layout,
            partition_index=2,
        )
        assert cmd == ["bench", "--gpu", "9", "--out", "/tmp/out", "--mode", "CPX"]

    def test_a_path_with_a_space_stays_one_argument(self):
        """Substitution is per already-split token, so no shell can re-split it."""
        layout = pms.layout_for("SPX", cu_per_partition=256)
        cmd = pms.build_partition_command(
            ["bench", "{output_dir}"],
            device=0,
            output_dir=Path("/tmp/my runs"),
            layout=layout,
            partition_index=0,
        )
        assert cmd == ["bench", "/tmp/my runs"]

    def test_shape_placeholders_are_available(self):
        layout = pms.layout_for("QPX", cu_per_partition=64)
        cmd = pms.build_partition_command(
            ["b", "{partitions}", "{partition_index}", "{cu}"],
            device=0,
            output_dir=Path("/x"),
            layout=layout,
            partition_index=3,
        )
        assert cmd == ["b", "4", "3", "64"]

    def test_a_template_without_placeholders_is_left_alone(self):
        layout = pms.layout_for("SPX", cu_per_partition=256)
        assert pms.build_partition_command(
            ["b", "-v"], device=0, output_dir=Path("/x"), layout=layout, partition_index=0
        ) == [
            "b",
            "-v",
        ]


class TestPartitionEnv:
    def test_pins_the_process_to_its_partition(self):
        layout = pms.layout_for("CPX", cu_per_partition=32)
        env = pms.partition_env({}, layout, device=11, streams_per_partition=2)
        assert env["ROCR_VISIBLE_DEVICES"] == "11"

    def test_an_inherited_hip_mask_is_removed_not_kept(self):
        """Two masks apply in sequence, the second indexing into the first.

        A stale HIP_VISIBLE_DEVICES=0 alongside ROCR would send every
        partition's work to one device, and the sweep would report the mode as
        uniformly slow with every process apparently succeeding.
        """
        layout = pms.layout_for("CPX", cu_per_partition=32)
        env = pms.partition_env(
            {"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"},
            layout,
            device=11,
            streams_per_partition=2,
        )
        assert "HIP_VISIBLE_DEVICES" not in env
        assert "CUDA_VISIBLE_DEVICES" not in env

    def test_publishes_the_same_contract_the_optimizer_publishes(self):
        """An entrypoint written against a session's env must work here unchanged."""
        layout = pms.layout_for("CPX", cu_per_partition=32)
        env = pms.partition_env({}, layout, device=7, streams_per_partition=2)
        assert env[pms.PARTITION_MODE_ENV] == "CPX"
        assert env[pms.PARTITION_COUNT_ENV] == "8"
        assert env[pms.PARTITION_CU_ENV] == "32"
        assert env[pms.PARTITION_STREAMS_ENV] == "2"
        assert env[pms.PARTITION_TOTAL_STREAMS_ENV] == "16"

    def test_the_caller_environment_is_not_mutated(self):
        base = {"PATH": "/usr/bin"}
        layout = pms.layout_for("SPX", cu_per_partition=256)
        pms.partition_env(base, layout, device=0, streams_per_partition=1)
        assert base == {"PATH": "/usr/bin"}


# ------------------------------------------------------------- aggregation


def _run(index: int, *, throughput: float | None = 100.0, latency: float | None = 10.0, ok: bool = True):
    measurement: dict = {}
    if throughput is not None:
        measurement["output_throughput"] = throughput
    if latency is not None:
        measurement["e2el_mean_ms"] = latency
    return pms.PartitionRun(
        partition_index=index,
        hsa_device=7 + index,
        output_dir=Path("/x"),
        returncode=0 if ok else 1,
        measurement=measurement if ok else {},
        error="" if ok else "exited 1",
    )


class TestModeResult:
    def test_throughput_is_the_sum_over_partitions(self):
        """The whole point: a mode's number is every partition, loaded together."""
        result = pms.ModeResult(mode="CPX", runs=[_run(i, throughput=320.0) for i in range(8)])
        assert result.total() == pytest.approx(2560.0)

    def test_a_partially_reporting_mode_is_unmeasured_not_slow(self):
        """Summing six of eight understates by a quarter while looking like a result."""
        runs = [_run(i) for i in range(6)] + [_run(6, ok=False), _run(7, ok=False)]
        result = pms.ModeResult(mode="CPX", runs=runs)
        assert result.measured is False

    def test_every_partition_reporting_is_measured(self):
        result = pms.ModeResult(mode="DPX", runs=[_run(0), _run(1)])
        assert result.measured is True

    def test_a_mode_that_never_ran_is_not_measured(self):
        assert pms.ModeResult(mode="CPX", skipped="will not fit").measured is False

    def test_a_field_missing_on_one_partition_blocks_the_sum(self):
        """A partial sum is a wrong number, not an approximate one."""
        result = pms.ModeResult(mode="DPX", runs=[_run(0), _run(1, throughput=None)])
        assert result.total() is None

    def test_the_throughput_field_falls_back_in_preference_order(self):
        runs = [
            pms.PartitionRun(0, 7, Path("/x"), returncode=0, measurement={"total_token_throughput": 5.0}),
            pms.PartitionRun(1, 8, Path("/x"), returncode=0, measurement={"total_token_throughput": 7.0}),
        ]
        assert pms.ModeResult(mode="DPX", runs=runs).total() == pytest.approx(12.0)

    def test_latency_is_the_worst_partition_not_the_mean_of_them(self):
        """A request landing on the slow partition is not consoled by the average."""
        runs = [_run(0, latency=10.0), _run(1, latency=90.0)]
        assert pms.ModeResult(mode="DPX", runs=runs).worst_latency_ms() == pytest.approx(90.0)

    def test_latency_accepts_the_alternate_spellings(self):
        runs = [pms.PartitionRun(0, 7, Path("/x"), returncode=0, measurement={"mean_e2el_ms": 12.0})]
        assert pms.ModeResult(mode="SPX", runs=runs).worst_latency_ms() == pytest.approx(12.0)

    def test_latency_missing_on_one_partition_is_not_reported(self):
        runs = [_run(0, latency=10.0), _run(1, latency=None)]
        assert pms.ModeResult(mode="DPX", runs=runs).worst_latency_ms() is None


class TestToFloat:
    """The canonical coercion from hyperloom.common.coerce, not a local copy.

    Pinned here because the aggregation depends on its rejections: a stray bool
    becoming 1.0, or an inf reaching the sum, produces a throughput comparison
    that is wrong rather than absent.
    """

    def test_a_bool_is_not_a_measurement(self):
        assert pms.to_float(True) is None

    @pytest.mark.parametrize("bad", ["", "abc", None, float("inf"), float("nan")])
    def test_unusable_values_are_none(self, bad):
        assert pms.to_float(bad) is None

    def test_a_numeric_string_is_accepted(self):
        assert pms.to_float("12.5") == pytest.approx(12.5)


class TestCardTotalGib:
    def test_a_split_mode_reading_is_scaled_back_to_the_card(self):
        """36 GiB on a CPX partition is a 288 GiB MI355X."""
        assert pms.card_total_gib(36.0, "CPX") == pytest.approx(288.0)

    def test_an_unpartitioned_reading_is_already_the_card(self):
        assert pms.card_total_gib(288.0, "SPX") == pytest.approx(288.0)

    @pytest.mark.parametrize("bad", [None, 0.0, -1.0])
    def test_an_unknown_reading_stays_unknown(self, bad):
        assert pms.card_total_gib(bad, "SPX") is None


class TestFeasibilityGate:
    def test_the_mi355x_case_that_motivated_the_gate(self):
        """20.7 GiB x 2 streams does not fit a 36 GiB CPX partition."""
        layout = pms.layout_for("CPX", cu_per_partition=32, gib_per_partition=36.0)
        assert pms.fits_in_partition(20.7, layout, 2) is False

    def test_one_stream_of_the_same_workload_does_fit(self):
        """Which is why gating on the single-stream figure is the trap."""
        layout = pms.layout_for("CPX", cu_per_partition=32, gib_per_partition=36.0)
        assert pms.fits_in_partition(20.7, layout, 1) is True

    def test_the_whole_card_holds_it_comfortably(self):
        layout = pms.layout_for("SPX", cu_per_partition=256, gib_per_partition=288.0)
        assert pms.fits_in_partition(20.7, layout, 2) is True


# ----------------------------------------------------------- mode resolution


class TestResolveModes:
    def test_defaults_to_what_the_card_reports(self):
        assert pms.resolve_modes("", ("SPX", "DPX", "QPX", "CPX")) == ("SPX", "DPX", "QPX", "CPX")

    def test_an_explicit_list_keeps_its_order(self):
        assert pms.resolve_modes("CPX,SPX", ("SPX", "DPX", "QPX", "CPX")) == ("CPX", "SPX")

    def test_whitespace_and_commas_both_separate(self):
        assert pms.resolve_modes("spx dpx", ("SPX", "DPX")) == ("SPX", "DPX")

    def test_duplicates_collapse(self):
        assert pms.resolve_modes("SPX,SPX", ("SPX",)) == ("SPX",)

    def test_a_misspelled_mode_is_refused(self):
        with pytest.raises(pms.SweepError, match="unknown compute-partition mode"):
            pms.resolve_modes("SPXX", ("SPX",))

    def test_a_mode_the_card_disclaims_is_refused_before_anything_is_set(self):
        with pytest.raises(pms.SweepError, match="does not report support for CPX"):
            pms.resolve_modes("CPX", ("SPX", "DPX"))

    def test_an_unqueryable_card_is_not_second_guessed(self):
        """No profile table means "unknown", so the request stands and the set judges it."""
        assert pms.resolve_modes("CPX", ()) == ("CPX",)

    def test_no_request_and_no_profiles_says_which_flag_to_pass(self):
        with pytest.raises(pms.SweepError, match="--modes"):
            pms.resolve_modes("", ())


# --------------------------------------------------------------- reporting


class TestRender:
    def test_a_skipped_mode_is_shown_with_its_reason(self):
        results = [
            pms.ModeResult(mode="SPX", layout=pms.layout_for("SPX", cu_per_partition=256), runs=[_run(0)]),
            pms.ModeResult(mode="CPX", skipped="2 x 20.7 GiB will not fit a 36.0 GiB partition"),
        ]
        text = "\n".join(pms.render(results, entry_mode="SPX"))
        assert "will not fit" in text
        assert "CPX" in text

    def test_the_winner_is_named(self):
        results = [
            pms.ModeResult(
                mode="SPX", layout=pms.layout_for("SPX", cu_per_partition=256), runs=[_run(0, throughput=100.0)]
            ),
            pms.ModeResult(
                mode="DPX",
                layout=pms.layout_for("DPX", cu_per_partition=128),
                runs=[_run(0, throughput=90.0), _run(1, throughput=90.0)],
            ),
        ]
        text = "\n".join(pms.render(results, entry_mode="SPX"))
        assert "Fastest measured mode: DPX" in text
        assert "1.80x" in text

    def test_nothing_measured_says_so_rather_than_naming_a_winner(self):
        results = [pms.ModeResult(mode="CPX", error="setting CPX failed")]
        text = "\n".join(pms.render(results, entry_mode="SPX"))
        assert "No mode produced a measurement" in text

    def test_the_report_says_the_number_is_a_sum(self):
        """The claim has to travel with the figure or it gets read as a single run."""
        results = [pms.ModeResult(mode="SPX", layout=pms.layout_for("SPX", cu_per_partition=256), runs=[_run(0)])]
        text = "\n".join(pms.render(results, entry_mode="SPX"))
        assert "sum over a mode's partitions" in text

    def test_it_points_at_the_assertion_flag_for_the_session_that_follows(self):
        results = [pms.ModeResult(mode="SPX", layout=pms.layout_for("SPX", cu_per_partition=256), runs=[_run(0)])]
        text = "\n".join(pms.render(results, entry_mode="SPX"))
        assert "--compute-partition-mode" in text


class TestSummaryJson:
    def test_records_the_shape_and_the_per_partition_devices(self):
        result = pms.ModeResult(
            mode="DPX",
            layout=pms.layout_for("DPX", cu_per_partition=128, gib_per_partition=144.0),
            runs=[_run(0), _run(1)],
        )
        payload = pms.summary_json([result], entry_mode="SPX", gpu_id=0)
        assert payload["entry_mode"] == "SPX"
        mode = payload["modes"][0]
        assert mode["partitions"] == 2
        assert mode["cu_per_partition"] == 128
        assert [r["hsa_device"] for r in mode["partition_runs"]] == [7, 8]

    def test_a_skipped_mode_records_why_rather_than_a_null_row(self):
        payload = pms.summary_json([pms.ModeResult(mode="CPX", skipped="will not fit")], entry_mode="SPX", gpu_id=0)
        assert payload["modes"][0]["skipped"] == "will not fit"
        assert payload["modes"][0]["measured"] is False


class TestMagpieCommand:
    def test_the_output_dir_is_left_as_a_placeholder_for_the_fan_out(self):
        cmd = pms.magpie_command(Path("/cfg/bench.yaml"), python_exe="/usr/bin/python3")
        assert "{output_dir}" in cmd
        assert "/cfg/bench.yaml" in cmd

    def test_it_is_the_local_run_mode(self):
        assert "local" in pms.magpie_command(Path("/cfg/bench.yaml"))


class TestBusyDetection:
    @pytest.mark.parametrize(
        "detail",
        ["AMDSMI_STATUS_BUSY", "Device busy, try again", "resident process present"],
    )
    def test_a_transient_refusal_is_retried(self, detail):
        assert pms._is_busy(detail) is True

    @pytest.mark.parametrize("detail", ["permission denied", "unknown partition mode", ""])
    def test_a_permanent_failure_fails_fast(self, detail):
        """Retrying these for two minutes turns a clear error into a hang."""
        assert pms._is_busy(detail) is False


class TestSudoPrefix:
    def test_privilege_is_opt_in(self):
        assert pms._sudo_prefix(False) == []

    def test_the_prefix_never_prompts(self):
        """An unattended sweep that stops at a password prompt hangs silently."""
        assert pms._sudo_prefix(True) == ["sudo", "-n"]


# ------------------------------------------------------------ control flow
#
# The tests above are pure functions. What follows drives main() end to end
# against a fake node, because the decisions being pinned -- which card's
# processes block a set, and what happens on the way out of a failure -- live in
# its control flow and cannot be reached any other way. Nothing here touches a
# GPU: every call that would is replaced.


IDLE = "No running processes detected"


class FakeNode:
    """One MI355X-shaped card on PCI bus 0x09, plus an untouched neighbour."""

    def __init__(self, tmp_path):
        self.mode = "SPX"
        self.sets: list[str] = []
        self.processes: object = [{"gpu": 0, "process_list": IDLE}, {"gpu": 1, "process_list": IDLE}]
        self.output_dir = tmp_path / "out"
        #: Raised by the next set of this mode, once. Simulates a mid-sweep fault.
        self.fail_set_on: dict[str, BaseException] = {}
        #: Raised instead of enumerating agents for this mode.
        self.fail_agents_on: dict[str, BaseException] = {}

    def summary(self) -> dict:
        return json.loads((self.output_dir / "sweep_summary.json").read_text())


@pytest.fixture
def node(monkeypatch, tmp_path):
    fake = FakeNode(tmp_path)

    def _amd_smi_json(args, timeout_s=None, *, sudo=False):
        if args[0] == "list":
            return [{"gpu": 0, "bdf": "0000:09:00.0"}, {"gpu": 1, "bdf": "0000:0a:00.0"}]
        if args[0] == "partition":
            return []  # no profile table, so the mode list is taken as given
        if args[0] == "process":
            return fake.processes
        raise AssertionError(f"unexpected amd-smi call: {args}")

    def _set_mode(gpu_id, mode, *, sudo=False):
        boom = fake.fail_set_on.pop(mode, None)
        if boom is not None:
            raise boom
        fake.sets.append(mode)
        fake.mode = mode
        return mode

    def _read_hsa_agents():
        boom = fake.fail_agents_on.pop(fake.mode, None)
        if boom is not None:
            raise boom
        partitions = pms.MODE_PARTITION_COUNTS[fake.mode]
        cu = 256 // partitions
        return tuple(pms.HsaAgent(index=i, cu=cu, bus=0x09, device=0, function=i) for i in range(partitions))

    def _run_mode(layout, devices, **kwargs):
        return [
            pms.PartitionRun(i, d, tmp_path, returncode=0, measurement={"output_throughput": 100.0})
            for i, d in enumerate(devices)
        ]

    monkeypatch.setattr(pms, "_amd_smi_json", _amd_smi_json)
    monkeypatch.setattr(pms, "read_mode", lambda gpu_id: fake.mode)
    monkeypatch.setattr(pms, "set_mode", _set_mode)
    monkeypatch.setattr(pms, "read_hsa_agents", _read_hsa_agents)
    monkeypatch.setattr(pms, "read_device_gib", lambda gpu_id: 288.0)
    monkeypatch.setattr(pms, "run_mode", _run_mode)
    # main() installs handlers and never removes them; leaving pytest's own
    # SIGINT replaced for the rest of the session is not this test's business.
    monkeypatch.setattr(pms.signal, "signal", lambda *a, **k: None)
    return fake


def _sweep(node, *extra, modes="SPX,DPX", gpu=0):
    return pms.main(
        [
            "--benchmark-command",
            "bench --gpu {device}",
            "--modes",
            modes,
            "--gpu",
            str(gpu),
            "--output-dir",
            str(node.output_dir),
            *extra,
        ]
    )


class TestBusyCheckScope:
    """Only the swept card is repartitioned, so only its contexts are at risk.

    The check was node-wide while the set is per-card, so any busy card on a
    shared node blocked sweeping an idle one -- and the only way past it,
    ``--allow-busy``, also gave up the protection on the target card itself.
    """

    def test_a_busy_neighbour_does_not_block_an_idle_target(self, node):
        """The set never touches card 1, so card 1's tenant is not this sweep's business."""
        node.processes = [
            {"gpu": 0, "process_list": IDLE},
            {"gpu": 1, "process_list": [{"process_info": {"name": "vllm"}}]},
        ]
        assert _sweep(node, gpu=0) == 0
        assert node.sets  # the sweep actually ran

    def test_a_process_on_the_target_card_still_refuses(self, node):
        node.processes = [{"gpu": 0, "process_list": [{"process_info": {"name": "vllm"}}]}]
        assert _sweep(node, gpu=0) == 2
        assert node.sets == []

    def test_the_refusal_names_the_card_and_the_count(self, node, capsys):
        node.processes = [{"gpu": 0, "process_list": [{"process_info": {"name": "vllm"}}]}]
        _sweep(node, gpu=0)
        assert "1 process(es) still hold a context on GPU 0" in capsys.readouterr().err

    def test_a_target_missing_from_the_listing_is_refused_not_assumed_idle(self, node):
        """An absent row is "unknown", and unknown does not license an eviction."""
        node.processes = [{"gpu": 1, "process_list": IDLE}]
        assert _sweep(node, gpu=0) == 2
        assert node.sets == []

    def test_allow_busy_proceeds_on_the_target_card(self, node):
        node.processes = [{"gpu": 0, "process_list": [{"process_info": {"name": "vllm"}}]}]
        assert _sweep(node, "--allow-busy", gpu=0) == 0

    def test_an_unreadable_payload_refuses_by_default(self, node):
        node.processes = {"process": [{"gpu": 0, "process_list": []}]}
        assert _sweep(node, gpu=0) == 2
        assert node.sets == []

    def test_allow_busy_is_the_way_past_an_unreadable_payload(self, node):
        """The check cannot change what happens next, so it is not consulted."""
        node.processes = {"process": [{"gpu": 0, "process_list": []}]}
        assert _sweep(node, "--allow-busy", gpu=0) == 0

    def test_a_dry_run_does_not_consult_it_at_all(self, node):
        """--dry-run sets nothing, so nothing it might evict is at stake."""
        node.processes = {"unparseable": True}
        assert _sweep(node, "--dry-run", gpu=0) == 0
        assert node.sets == []


class TestExitCodeContract:
    def test_a_clean_sweep_reports_and_restores(self, node):
        assert _sweep(node) == 0
        assert node.mode == "SPX"
        assert node.summary()["entry_mode"] == "SPX"

    def test_an_expected_failure_moves_on_to_the_next_mode(self, node):
        node.fail_agents_on["DPX"] = pms.SweepError("rocminfo timed out")
        assert _sweep(node) == 0
        assert node.summary()["modes"][0]["mode"] == "SPX"

    def test_an_unexpected_error_still_renders_and_writes_the_summary(self, node):
        """The finding: anything not a SweepError escaped main() past the report."""
        node.fail_agents_on["DPX"] = KeyError("compute_partition")
        assert _sweep(node) == 4
        assert node.summary()["modes"][0]["mode"] == "SPX"

    def test_an_unexpected_error_does_not_lose_the_modes_already_measured(self, node, capsys):
        """SPX measured before DPX broke; escaping main() threw that away."""
        node.fail_agents_on["DPX"] = KeyError("compute_partition")
        _sweep(node)
        out = capsys.readouterr().out
        assert "Fastest measured mode: SPX" in out
        assert node.summary()["modes"][0]["aggregate_throughput"] == pytest.approx(100.0)

    def test_an_unexpected_error_still_restores_the_card(self, node):
        node.fail_agents_on["DPX"] = KeyError("compute_partition")
        _sweep(node)
        assert node.mode == "SPX"

    def test_an_unexpected_error_stops_the_sweep_rather_than_setting_more_modes(self, node):
        """An assumption this script does not understand is broken; stop mutating."""
        node.fail_agents_on["DPX"] = KeyError("compute_partition")
        _sweep(node, modes="SPX,DPX,QPX")
        assert "QPX" not in node.sets

    def test_the_unexpected_error_is_named_not_swallowed(self, node, capsys):
        node.fail_agents_on["DPX"] = KeyError("compute_partition")
        _sweep(node)
        assert "KeyError" in capsys.readouterr().err

    def test_a_failed_restore_after_a_clean_sweep_exits_3(self, node):
        # Entry mode is SPX and only DPX is swept, so the one set of SPX is the
        # restore itself.
        node.fail_set_on["SPX"] = pms.SweepError("permission denied")
        assert _sweep(node, modes="DPX") == 3

    def test_a_failed_restore_outranks_an_unexpected_error(self, node):
        """A card left in the wrong shape mislabels whatever runs next, so it wins."""
        node.fail_agents_on["QPX"] = KeyError("compute_partition")
        node.fail_set_on["SPX"] = pms.SweepError("permission denied")
        assert _sweep(node, modes="DPX,QPX") == 3

    def test_an_unexpected_error_in_the_restore_is_still_a_restore_failure(self, node):
        """Not a SweepError, so it used to propagate out of the finally and exit 1."""
        node.fail_set_on["SPX"] = RuntimeError("amd-smi vanished")
        assert _sweep(node, modes="DPX") == 3

    def test_the_reported_reproduction_a_key_error_from_set_mode(self, node):
        """As reviewed: KeyError out of set_mode escaped main() and exited 1."""
        node.fail_set_on["DPX"] = KeyError("compute_partition")
        assert _sweep(node) == 4
        assert node.mode == "SPX"
        assert node.summary()["modes"][0]["measured"] is True

    def test_an_unwritable_report_does_not_cost_the_restore_failure_its_code(self, node, monkeypatch):
        """The exit code is the one thing a caller cannot reconstruct itself."""
        node.fail_set_on["SPX"] = pms.SweepError("permission denied")
        monkeypatch.setattr(pms, "render", lambda results, *, entry_mode: (_ for _ in ()).throw(OSError("no space")))
        assert _sweep(node, modes="DPX") == 3

    def test_an_unwritable_report_is_not_reported_as_a_clean_sweep(self, node, monkeypatch):
        monkeypatch.setattr(pms, "render", lambda results, *, entry_mode: (_ for _ in ()).throw(OSError("no space")))
        assert _sweep(node) == 4

    def test_a_sweep_that_measures_nothing_exits_1(self, node, monkeypatch):
        monkeypatch.setattr(pms, "run_mode", lambda layout, devices, **kw: [])
        assert _sweep(node) == 1

    def test_an_unexpected_error_before_anything_is_set_does_not_report_a_refusal(self, node, monkeypatch):
        """Exit 2 means this script decided to refuse; a bug is not a decision."""
        monkeypatch.setattr(pms, "read_device_gib", lambda gpu_id: (_ for _ in ()).throw(KeyError("vram")))
        assert _sweep(node) == 4
        assert node.sets == []
