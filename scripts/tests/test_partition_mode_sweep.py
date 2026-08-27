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


class TestAsFloat:
    def test_a_bool_is_not_a_measurement(self):
        assert pms._as_float(True) is None

    @pytest.mark.parametrize("bad", ["", "abc", None, float("inf"), float("nan")])
    def test_unusable_values_are_none(self, bad):
        assert pms._as_float(bad) is None

    def test_a_numeric_string_is_accepted(self):
        assert pms._as_float("12.5") == pytest.approx(12.5)


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
