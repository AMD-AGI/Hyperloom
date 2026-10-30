# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The phases of a round, read off the marks its Magpie runs leave beside their server logs.

The marks are written the way the launcher writes them (``server_ready_at``, ``kv_metrics.json``); the cases are the
rounds a test is made of: a graded test's warmup round and its measured round, a round with a warmup run in front of its
measured one, an AgentX round, a round whose server never got ready.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import PhaseTiming
from hyperloom.measure.phases import PhaseRecorder
from hyperloom.measure.round_marks import (
    ROLE_BOOT,
    ROLE_MEASURE,
    ROLE_WARMUP,
    Boundary,
    BootParts,
    RoundShape,
    RunMarks,
    boot_parts,
    read_runs,
    round_boundaries,
    server_log_of,
)

T0 = 1_000.0
WINDOW = (T0, T0 + 300.0)


def _run_dir(root: Path, round_name: str, *, warmup: bool = False) -> Path:
    directory = root / round_name / "warmup_round" if warmup else root / round_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_run(
    directory: Path,
    ready: float,
    boot: float,
    *,
    marks: dict[str, float] | None = None,
    ended: float | None = None,
    log: str | None = None,
) -> None:
    """What a Magpie run leaves: the ready stamp, the watchdog's phase marks and, when it ended, the file's time."""
    (directory / "server_ready_at").write_text(f"{ready:.3f} {boot:.3f}\n", encoding="utf-8")
    if marks is not None or ended is not None:
        phase_marks = [{"phase": phase, "mono": 1.0, "ts": at} for phase, at in (marks or {}).items()]
        kv = directory / "kv_metrics.json"
        kv.write_text(json.dumps({"phase_marks": phase_marks, "samples": []}), encoding="utf-8")
        if ended is not None:
            os.utime(kv, (ended, ended))
    if log is not None:
        (directory / "server.log").write_text(log, encoding="utf-8")


def _phases(boundaries: list[Boundary], window: tuple[float, float] = WINDOW) -> list[tuple[str, float]]:
    """The phases the boundaries make, as a recorder stamps them, ending at the window's end."""
    recorder = PhaseRecorder()
    for boundary in boundaries:
        recorder.begin(boundary.name, at=boundary.at)
    phases = recorder.finish(at=window[1])
    assert sum(p.seconds for p in phases) == pytest.approx(window[1] - window[0]), "the phases tile the window"
    return [(p.name, round(p.seconds, 3)) for p in phases]


def _shape(role: str = ROLE_MEASURE, **overrides: object) -> RoundShape:
    fields: dict[str, object] = {"agentx": False, "reuses_server": False, "failed": False}
    return RoundShape(role, **{**fields, **overrides})  # type: ignore[arg-type]


def _run(**fields: object) -> RunMarks:
    return RunMarks(Path("/nowhere"), **{"ready": T0 + 90.0, **fields})  # type: ignore[arg-type]


# -- reading the marks --------------------------------------------------------------------------------------------


def test_the_runs_of_a_round_are_read_in_the_order_they_were_ready_from_their_stamps_and_kv_marks(
    tmp_path: Path,
) -> None:
    _write_run(
        _run_dir(tmp_path, "variant_00_test", warmup=True),
        T0 + 90.0,
        82.221,
        marks={"eval": T0 + 120.0},
        ended=T0 + 150.0,
    )
    _write_run(_run_dir(tmp_path, "variant_00_test"), T0 + 160.0, 0.0, marks={"measured": T0 + 160.0}, ended=T0 + 200.0)

    warmup, measured = read_runs(tmp_path, WINDOW)

    assert (warmup.ready, warmup.boot_seconds, warmup.eval_at, warmup.ended) == (
        T0 + 90.0,
        82.221,
        T0 + 120.0,
        T0 + 150.0,
    )
    assert (measured.ready, measured.boot_seconds, measured.eval_at, measured.ended) == (
        T0 + 160.0,
        0.0,
        None,
        T0 + 200.0,
    )
    assert warmup.directory == tmp_path / "variant_00_test" / "warmup_round"


def test_a_stamp_an_earlier_round_left_in_the_directory_is_not_this_rounds(tmp_path: Path) -> None:
    _write_run(_run_dir(tmp_path, "variant_00_c128"), T0 - 400.0, 80.0, marks={"eval": T0 - 300.0}, ended=T0 - 200.0)

    assert read_runs(tmp_path, WINDOW) == []


def test_a_mark_or_file_time_from_before_the_round_is_not_this_rounds_either(tmp_path: Path) -> None:
    directory = _run_dir(tmp_path, "variant_00_c128")
    _write_run(directory, T0 + 5.0, 0.0, marks={"eval": T0 - 50.0, "measured": T0 + 5.0}, ended=T0 - 10.0)

    (run,) = read_runs(tmp_path, WINDOW)

    assert (run.eval_at, run.client_measured_at, run.ended) == (None, T0 + 5.0, None)


def test_a_run_with_no_kv_file_or_a_broken_one_has_a_stamp_and_no_instants(tmp_path: Path) -> None:
    _write_run(_run_dir(tmp_path, "variant_00_a"), T0 + 5.0, 0.0)
    broken = _run_dir(tmp_path, "variant_00_b")
    _write_run(broken, T0 + 6.0, 0.0)
    (broken / "kv_metrics.json").write_text("{not json", encoding="utf-8")
    weird = _run_dir(tmp_path, "variant_00_c")
    _write_run(weird, T0 + 7.0, 0.0)
    (weird / "kv_metrics.json").write_text(
        json.dumps({"phase_marks": [None, {"phase": "eval", "ts": "x"}, 3]}), encoding="utf-8"
    )

    runs = read_runs(tmp_path, WINDOW)

    assert [(r.eval_at, r.ended) for r in runs] == [(None, None), (None, None), (None, runs[2].ended)]
    assert runs[2].ended is not None, "a readable file that holds no usable mark still says when it was written"


@pytest.mark.parametrize("stamp", ["10**400", "1e999", "NaN", "-Infinity", "true"], ids=lambda s: s[:9])
def test_a_phase_mark_that_is_no_finite_number_a_float_can_hold_is_no_mark(tmp_path: Path, stamp: str) -> None:
    directory = _run_dir(tmp_path, "variant_00_a")
    _write_run(directory, T0 + 5.0, 0.0)
    text = '{"phase_marks": [{"phase": "eval", "ts": %s}, {"phase": "measured", "ts": %f}]}' % (stamp, T0 + 7.0)
    (directory / "kv_metrics.json").write_text(text.replace("10**400", "1" + "0" * 400), encoding="utf-8")

    (run,) = read_runs(tmp_path, WINDOW)

    assert (run.eval_at, run.client_measured_at) == (None, T0 + 7.0), "the mark beside it is still read"


def test_an_output_directory_that_does_not_exist_holds_no_runs(tmp_path: Path) -> None:
    assert read_runs(tmp_path / "missing", WINDOW) == []


# -- the boundaries of each kind of round -------------------------------------------------------------------------


def test_the_warmup_round_of_a_graded_test_boots_warms_up_evaluates_and_collects_with_the_server_left_up() -> None:
    run = _run(boot_seconds=82.221, eval_at=T0 + 150.0, ended=T0 + 200.0)
    shape = _shape(ROLE_WARMUP, tail="round_collect")

    boundaries = round_boundaries(WINDOW, [run], shape)

    assert _phases(boundaries) == [
        ("round_setup", 7.779),
        ("server_boot", 82.221),
        ("warmup", 60.0),
        ("accuracy_eval", 50.0),
        ("round_collect", 100.0),
    ]
    assert [b.name for b in boundaries][0] == "round_setup" and boundaries[0].at == T0


def test_the_measured_round_of_a_graded_test_found_its_server_ready_and_ends_in_the_teardown() -> None:
    run = _run(ready=T0 + 4.0, ended=T0 + 60.0)

    assert _phases(round_boundaries(WINDOW, [run], _shape())) == [
        ("round_setup", 4.0),
        ("measured", 56.0),
        ("teardown", 240.0),
    ]


def test_a_round_with_a_discarded_warmup_run_ahead_of_its_measured_one_names_each_part() -> None:
    warmup = _run(ready=T0 + 90.0, boot_seconds=80.0, ended=T0 + 130.0)
    measured = _run(ready=T0 + 160.0, ended=T0 + 200.0)

    assert _phases(round_boundaries(WINDOW, [warmup, measured], _shape())) == [
        ("round_setup", 10.0),
        ("server_boot", 80.0),
        ("warmup", 40.0),
        ("round_collect", 30.0),
        ("measured", 40.0),
        ("teardown", 100.0),
    ]


def test_an_evaluation_in_the_measured_run_of_a_test_with_no_warmup_is_split_from_the_bench() -> None:
    run = _run(ready=T0 + 60.0, boot_seconds=50.0, eval_at=T0 + 100.0, ended=T0 + 140.0)

    assert _phases(round_boundaries(WINDOW, [run], _shape())) == [
        ("round_setup", 10.0),
        ("server_boot", 50.0),
        ("measured", 40.0),
        ("accuracy_eval", 40.0),
        ("teardown", 160.0),
    ]


def test_an_agentx_measured_run_is_split_at_the_phases_its_client_began_and_ended() -> None:
    run = _run(
        ready=T0 + 10.0,
        client_warmup_at=T0 + 25.0,
        client_measured_at=T0 + 55.0,
        client_finished_at=T0 + 255.0,
        ended=T0 + 270.0,
    )

    assert _phases(round_boundaries(WINDOW, [run], _shape(agentx=True))) == [
        ("round_setup", 10.0),
        ("client_start", 15.0),
        ("warmup", 30.0),
        ("measured", 200.0),
        ("round_collect", 15.0),
        ("teardown", 30.0),
    ]


def test_an_agentx_run_that_never_saw_its_measured_phase_is_measured_whole() -> None:
    run = _run(ready=T0 + 10.0, client_warmup_at=T0 + 25.0, ended=T0 + 270.0)

    assert _phases(round_boundaries(WINDOW, [run], _shape(agentx=True))) == [
        ("round_setup", 10.0),
        ("measured", 260.0),
        ("teardown", 30.0),
    ]


def test_the_discarded_agentx_warmup_run_is_all_warmup_whatever_its_client_did() -> None:
    discarded = _run(
        ready=T0 + 90.0,
        boot_seconds=80.0,
        client_warmup_at=T0 + 100.0,
        client_measured_at=T0 + 110.0,
        ended=T0 + 140.0,
    )
    measured = _run(ready=T0 + 150.0, client_measured_at=T0 + 160.0, ended=T0 + 290.0)

    names = [name for name, _ in _phases(round_boundaries(WINDOW, [discarded, measured], _shape(agentx=True)))]

    assert names == ["round_setup", "server_boot", "warmup", "round_collect", "client_start", "measured", "teardown"]


def test_a_round_that_only_boots_the_server_counts_what_follows_its_ready_as_warmup() -> None:
    run = _run(ready=T0 + 90.0, boot_seconds=80.0, ended=T0 + 100.0)

    assert _phases(round_boundaries(WINDOW, [run], _shape(ROLE_BOOT, tail="round_collect")))[2:] == [
        ("warmup", 10.0),
        ("round_collect", 200.0),
    ]


def test_a_round_whose_server_never_got_ready_is_all_boot() -> None:
    assert _phases(round_boundaries(WINDOW, [], _shape(failed=True))) == [("server_boot", 300.0)]


@pytest.mark.parametrize(
    "shape",
    [_shape(), _shape(failed=True, reuses_server=True), _shape(ROLE_WARMUP)],
    ids=["succeeded", "failed on a ready server", "graded warmup that succeeded"],
)
def test_a_round_the_marks_cannot_split_is_one_round_and_not_a_guess(shape: RoundShape) -> None:
    assert _phases(round_boundaries(WINDOW, [], shape)) == [("round", 300.0)]


def test_a_mark_outside_the_window_moves_to_its_edge_so_the_phases_still_tile_it() -> None:
    run = _run(ready=T0 + 90.0, boot_seconds=80.0, eval_at=T0 + 900.0, ended=T0 - 50.0)

    phases = _phases(round_boundaries(WINDOW, [run], _shape()))

    assert phases[0] == ("round_setup", 10.0) and sum(seconds for _, seconds in phases) == 300.0


# -- the parts of a boot ------------------------------------------------------------------------------------------

VLLM_LOG = """\
INFO 10-22 04:00:01 [config.py:1] Initializing a V1 LLM engine
(EngineCore_0 pid=9) INFO 10-22 04:00:30 [gpu_model_runner.py:1] Loading weights took 41.23 seconds
(EngineCore_0 pid=9) INFO 10-22 04:00:31 [gpu_model_runner.py:1] Model loading took 15.20 GiB and 42.10 seconds
(EngineCore_0 pid=9) INFO 10-22 04:01:00 [monitor.py:1] torch.compile takes 14.19 s in total
(EngineCore_0 pid=9) INFO 10-22 04:01:20 [gpu_model_runner.py:1] Graph capturing finished in 12 secs, took 0.56 GiB
INFO:     Application startup complete.
"""
SGLANG_LOG = """\
[2026-10-22 04:00:01 TP0] Load weight begin. avail mem=250.00 GB
[2026-10-22 04:00:01 TP1] Load weight begin. avail mem=250.00 GB
[2026-10-22 04:00:41 TP0] Load weight end. type=Qwen3ForCausalLM, dtype=torch.bfloat16, avail mem=235.00 GB
[2026-10-22 04:00:43 TP1] Load weight end. type=Qwen3ForCausalLM, dtype=torch.bfloat16, avail mem=235.00 GB
[2026-10-22 04:00:50 TP0] Capture cuda graph begin. This can take up to several minutes. avail mem=200.00 GB
[2026-10-22 04:01:02 TP0] Capture cuda graph end. Time elapsed: 11.90 s. mem usage=1.20 GB. avail mem=198.00 GB
[2026-10-22 04:01:03 TP1] Capture cuda graph end. Time elapsed: 12.40 s. mem usage=1.20 GB. avail mem=198.00 GB
The server is fired up and ready to roll!
"""


def test_a_vllm_boot_says_how_long_the_weights_the_compile_and_the_graphs_took(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(VLLM_LOG, encoding="utf-8")

    assert boot_parts(log) == BootParts(weights_load=42.10, compile=14.19, graph_capture=12.0)


def test_an_sglang_boot_is_read_from_the_instants_it_began_and_ended_loading_and_the_longest_rank_wins(
    tmp_path: Path,
) -> None:
    log = tmp_path / "server.log"
    log.write_text(SGLANG_LOG, encoding="utf-8")

    assert boot_parts(log) == BootParts(weights_load=42.0, compile=None, graph_capture=12.40)


def test_a_log_that_says_nothing_of_the_boot_or_is_not_there_gives_no_parts(tmp_path: Path) -> None:
    quiet = tmp_path / "server.log"
    quiet.write_text("INFO: Application startup complete.\n", encoding="utf-8")

    assert boot_parts(quiet) == boot_parts(tmp_path / "absent.log") == BootParts()
    assert BootParts().known() == []


def test_a_booted_run_is_split_into_what_its_log_accounts_for_in_the_order_an_engine_does_it(tmp_path: Path) -> None:
    directory = _run_dir(tmp_path, "variant_00_warmup")
    _write_run(directory, T0 + 90.0, 80.0, log=VLLM_LOG)
    (run,) = read_runs(tmp_path, WINDOW)

    phases = _phases(round_boundaries(WINDOW, [run], _shape(ROLE_WARMUP)))

    assert [name for name, _ in phases[:5]] == [
        "round_setup",
        "server_boot.other",
        "server_boot.weights_load",
        "server_boot.compile",
        "server_boot.graph_capture",
    ]
    assert [seconds for _, seconds in phases[1:5]] == [
        pytest.approx(80.0 - 42.1 - 14.19 - 12.0, abs=1e-3),
        42.1,
        14.19,
        12.0,
    ]
    assert sum(seconds for name, seconds in phases if name.startswith("server_boot")) == pytest.approx(80.0)


def test_parts_that_outrun_the_boot_are_cut_to_it_so_the_boot_is_never_longer_than_it_was(tmp_path: Path) -> None:
    directory = _run_dir(tmp_path, "variant_00_warmup")
    _write_run(directory, T0 + 30.0, 20.0, log=VLLM_LOG)
    (run,) = read_runs(tmp_path, WINDOW)

    phases = _phases(round_boundaries(WINDOW, [run], _shape(ROLE_WARMUP)))

    boot = [(name, seconds) for name, seconds in phases if name.startswith("server_boot")]
    assert boot == [("server_boot.weights_load", 20.0)], (
        "the weights took the whole boot; the later parts have none left"
    )


def test_a_run_whose_log_says_nothing_of_its_boot_is_one_boot_phase(tmp_path: Path) -> None:
    _write_run(_run_dir(tmp_path, "variant_00_warmup"), T0 + 90.0, 80.0, log="INFO: Application startup complete.\n")
    (run,) = read_runs(tmp_path, WINDOW)

    names = [name for name, _ in _phases(round_boundaries(WINDOW, [run], _shape(ROLE_WARMUP)))]

    assert names[:2] == ["round_setup", "server_boot"]


def test_every_phase_a_round_names_is_a_phase_the_contract_accepts() -> None:
    run = _run(boot_seconds=80.0, eval_at=T0 + 150.0, client_measured_at=T0 + 160.0, ended=T0 + 200.0)
    for shape in (_shape(), _shape(ROLE_WARMUP), _shape(agentx=True)):
        for boundary in round_boundaries(WINDOW, [run], shape):
            PhaseTiming(boundary.name, 0.0, 0.0)


#: SGLang's line for the graph capture in a newer release, with a word for what was captured and ``elapsed=`` in place of
#: ``Time elapsed:``; the load lines are the ones the release prints with a time on each.
SGLANG_NEW_LOG = """\
[2026-10-24 06:38:01 TP0] Load weight begin. avail mem=250.00 GB
[2026-10-24 06:38:03 TP0] Load weight end. elapsed=2.00 s, type=Qwen3ForCausalLM, avail mem=235.00 GB
[2026-10-24 06:38:08 TP0] Capture target decode CUDA graph begin. This can take up to several minutes. avail mem=52.72 GB
[2026-10-24 06:38:19 TP0] Capture target decode CUDA graph end. elapsed=11.34 s, mem usage=1.52 GB, avail mem=51.20 GB
The server is fired up and ready to roll!
"""


def _magpie_run(root: Path, *, log: str, launch: str = "benchmark_sglang_20261024_063800") -> Path:
    """A run as Magpie leaves it: the ready stamp in the run's directory, the server log one directory deeper."""
    directory = _run_dir(root, "variant_00_test", warmup=True)
    _write_run(directory, T0 + 90.0, 80.0)
    (directory / launch).mkdir()
    (directory / launch / "server.log").write_text(log, encoding="utf-8")
    return directory


def test_the_graph_capture_of_the_newer_sglang_wording_is_read_and_the_older_wording_still_is(tmp_path: Path) -> None:
    newer = tmp_path / "newer.log"
    newer.write_text(SGLANG_NEW_LOG, encoding="utf-8")

    assert boot_parts(newer) == BootParts(weights_load=2.0, compile=None, graph_capture=11.34)
    older = tmp_path / "older.log"
    older.write_text(SGLANG_LOG, encoding="utf-8")
    assert boot_parts(older).graph_capture == 12.40


def test_a_capture_that_only_began_or_failed_is_not_a_graph_capture_time(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "[t TP0] Capture target decode CUDA graph begin. elapsed=5.00 s\n"
        "Exception: Capture cuda graph failed: HIP error: an illegal memory access was encountered\n",
        encoding="utf-8",
    )

    assert boot_parts(log) == BootParts()


def test_a_server_log_one_directory_below_the_runs_is_found_and_splits_the_boot(tmp_path: Path) -> None:
    _magpie_run(tmp_path, log=SGLANG_NEW_LOG)
    (run,) = read_runs(tmp_path, WINDOW)

    phases = _phases(round_boundaries(WINDOW, [run], _shape(ROLE_WARMUP)))

    assert [name for name, _ in phases[:4]] == [
        "round_setup",
        "server_boot.other",
        "server_boot.weights_load",
        "server_boot.graph_capture",
    ]
    assert [seconds for _, seconds in phases[2:4]] == [2.0, 11.34]


def test_of_several_launches_below_a_run_the_last_by_name_is_read_and_a_log_in_the_directory_itself_comes_first(
    tmp_path: Path,
) -> None:
    directory = _magpie_run(tmp_path, log=SGLANG_LOG, launch="benchmark_sglang_20261024_063000")
    later = directory / "benchmark_sglang_20261024_063800"
    later.mkdir()
    (later / "server.log").write_text(SGLANG_NEW_LOG, encoding="utf-8")

    assert boot_parts(server_log_of(directory)).graph_capture == 11.34
    (directory / "server.log").write_text(VLLM_LOG, encoding="utf-8")
    assert server_log_of(directory) == directory / "server.log"
    assert server_log_of(tmp_path / "no_such_run") == tmp_path / "no_such_run" / "server.log"
