# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a server's log says aiter built: the hit counter and the compile time of a test (4.21)."""

from __future__ import annotations

from pathlib import Path

from hyperloom.measure.jit_builds import BuildLog, scan_builds, server_logs

# What aiter wrote in real server logs: the cpp_itfs form (a path) and the module builder's form (a bracketed md_name, in colour).
CPP = (
    "[aiter] start build /root/.aiter/build/pa_ragged_d3ad7c1d008d204c2ad5e06a04e4f5a4\n"
    "clang (LLVM option parsing): Unknown command line argument\n"
    "[aiter] finish build /root/.aiter/build/pa_ragged_d3ad7c1d008d204c2ad5e06a04e4f5a4, cost 12.48115537s\n"
)
MODULE = (
    "[aiter] [pid=11 pname=MainProcess] start build [module_gemm_a8w8] under /j/build/module_gemm_a8w8\n"
    "\x1b[32mfinish build [module_gemm_a8w8], cost 33.5s \x1b[0m\n"
)


def _log(directory: Path, text: str, name: str = "server.log") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_the_builds_a_log_says_are_counted_by_module_and_timed_by_the_cost_it_prints(tmp_path) -> None:
    log = _log(tmp_path, CPP + MODULE)

    built = scan_builds([log])

    assert built.seconds == {"pa_ragged_d3ad7c1d008d204c2ad5e06a04e4f5a4": 12.48115537, "module_gemm_a8w8": 33.5}
    assert built.builds_seen == 2 and round(built.compile_seconds, 3) == 45.981


def test_ranks_that_build_one_module_together_are_one_build_of_the_longest_cost(tmp_path) -> None:
    ranks = "".join(MODULE.replace("33.5", cost) for cost in ("20.0", "18.2", "19.9"))

    built = scan_builds([_log(tmp_path, ranks)])

    assert built.builds_seen == 1 and built.compile_seconds == 20.0


def test_a_log_with_no_build_line_is_a_test_that_found_everything_built(tmp_path) -> None:
    quiet = "[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so\nDocker build ok\n"

    built = scan_builds([_log(tmp_path, quiet)])

    assert built == BuildLog({}, frozenset()) and built.builds_seen == 0 and built.compile_seconds == 0.0


def test_a_module_that_started_but_did_not_finish_still_counts_as_a_build_seen(tmp_path) -> None:
    built = scan_builds([_log(tmp_path, MODULE.splitlines()[0] + "\n")])

    assert built.builds_seen == 1 and built.compile_seconds == 0.0


def test_the_logs_of_every_run_of_every_round_are_read_and_a_log_that_cannot_be_read_is_skipped(tmp_path) -> None:
    _log(tmp_path / "variant_0", MODULE)
    _log(tmp_path / "variant_0" / "warmup_round", CPP)
    (tmp_path / "variant_1").mkdir()
    (tmp_path / "variant_1" / "server.log").symlink_to(tmp_path / "nowhere.log")
    _log(tmp_path / "variant_2", MODULE, name="not-a-server.log")

    logs = server_logs(tmp_path)

    assert [p.relative_to(tmp_path).as_posix() for p in logs] == [
        "variant_0/server.log",
        "variant_0/warmup_round/server.log",
    ]
    assert scan_builds(logs + [tmp_path / "gone.log"]).builds_seen == 2
