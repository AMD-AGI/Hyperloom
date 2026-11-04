# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An sglang point whose tuned tables make a prebuilt aiter module stale gets a watchdog that waits for the rebuild.

The image is the aiter package of ``test_table_modules`` with its module config: a table set for
``AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE`` that differs from the image's makes ``module_gemm_a8w8_blockscale_
bpreshuffle`` (and ``module_deepgemm_opus``, which reads the same table) stale.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.measure.framework.tests.test_table_modules import BPRESHUFFLE, CONFIG, _package, _table
from hyperloom.measure.jit.modules import CKTILE_MODULE_BUILD_LIMIT_SEC, MODULE_BUILD_LIMIT_SEC
from hyperloom.measure.runner import rebuild_watchdog, run
from hyperloom.measure.runner.rebuild_watchdog import raise_rebuild_watchdog, rebuild_watchdog_sec
from hyperloom.measure.socket import TestCall

_UTF8 = "utf-8"
_FLAG = "--watchdog-timeout"


@pytest.fixture
def image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    package = _package(tmp_path / "image" / "aiter")
    (package / "jit" / "optCompilerConfig.json").write_text(json.dumps(CONFIG), encoding=_UTF8)
    monkeypatch.setattr(rebuild_watchdog, "locate_package", lambda name, python, cwd: package)
    return package


def _point(
    tmp_path: Path, name: str, *, framework: str = "sglang", agentx: bool = False, args: str = "", **envs
) -> Path:
    if agentx:
        launch = {"env": envs, "append_args": args.split()}
        bench = {"framework": framework, "agentx": {"enabled": True, "launch_overrides": launch}}
    else:
        bench = {"framework": framework, "envs": {**envs, **({"EXTRA_SGLANG_ARGS": args} if args else {})}}
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding=_UTF8)
    return path


def _call(tmp_path: Path, *points: Path) -> TestCall:
    return TestCall(points=points, output_dir=tmp_path, timeout_sec=1, cwd=str(tmp_path), magpie_python="", task_id="t")


def _bench(point: Path) -> dict:
    return yaml.safe_load(point.read_text(encoding=_UTF8))["benchmark"]


def _tuned(tmp_path: Path) -> dict[str, str]:
    return {BPRESHUFFLE: _table(tmp_path, "tuned.csv", "M,N,K,kernelName\n9,9,9,bp_new\n")}


def test_a_table_that_makes_modules_stale_raises_the_watchdog_of_both_carriers(image, tmp_path: Path) -> None:
    tuned = _tuned(tmp_path)
    plain = _point(tmp_path, "plain.yaml", args="--tp 4", **tuned)
    agentx = _point(tmp_path, "agentx.yaml", agentx=True, args="--tp 4", **tuned)

    note = raise_rebuild_watchdog(_call(tmp_path, plain, agentx), None, {}, tmp_path)

    seconds = rebuild_watchdog_sec({"module_gemm_a8w8_blockscale_bpreshuffle", "module_deepgemm_opus"})
    assert _bench(plain)["envs"]["EXTRA_SGLANG_ARGS"] == f"--tp 4 {_FLAG} {seconds}"
    assert _bench(agentx)["agentx"]["launch_overrides"]["append_args"] == ["--tp", "4", _FLAG, str(seconds)]
    assert note is not None and f"raised to {seconds}s" in note and "module_gemm_a8w8_blockscale_bpreshuffle" in note


def test_the_estimate_is_the_builds_plus_the_margin() -> None:
    assert rebuild_watchdog_sec(["m"]) == rebuild_watchdog.MODULE_BUILD_SEC + rebuild_watchdog.WATCHDOG_MARGIN_SEC
    assert rebuild_watchdog_sec(["m"]) > 300, "longer than sglang's default watchdog"


def test_a_cktile_module_is_given_the_cktile_build_limit_and_a_module_of_any_other_kind_the_common_one() -> None:
    """A CK-tile build is slowest when several tests build at once: 600 s each would give the cktile module and the CK
    one 1500 s with the margin, which the CK-tile build can overrun."""
    cktile, ck = "module_gemm_a8w8_bpreshuffle_cktile", "module_gemm_a8w8_bpreshuffle"
    margin = rebuild_watchdog.WATCHDOG_MARGIN_SEC

    assert rebuild_watchdog_sec([cktile, ck]) == CKTILE_MODULE_BUILD_LIMIT_SEC + MODULE_BUILD_LIMIT_SEC + margin > 1500
    assert rebuild_watchdog_sec([cktile]) == CKTILE_MODULE_BUILD_LIMIT_SEC + margin > rebuild_watchdog_sec([ck])
    assert rebuild_watchdog_sec([ck]) == 900, "a module of any other kind keeps 600 s, so a hang there ends as soon"
    assert rebuild_watchdog_sec(["module_moe_cktile2stages_b16_silu"]) == rebuild_watchdog_sec([cktile]), (
        "a name aiter makes at run time from a CK-tile entry is one"
    )


def test_a_stale_cktile_module_raises_the_watchdog_by_its_own_time(
    image, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = frozenset({"module_gemm_a8w8_bpreshuffle_cktile", "module_gemm_a8w8_bpreshuffle"})
    monkeypatch.setattr(rebuild_watchdog, "stale_table_modules", lambda *args: stale)
    point = _point(tmp_path, "point.yaml", args="--tp 1", **_tuned(tmp_path))

    note = raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path)

    assert _watchdog(point, False) == f"--tp 1 {_FLAG} 3600", (
        "2700 s for the CK-tile module, 600 s for the other, 300 s"
    )
    assert note is not None and "raised to 3600s" in note


def test_a_module_the_patch_touches_that_the_server_builds_raises_the_watchdog_too(image, tmp_path: Path) -> None:
    """A table the test does not change makes nothing stale; the patch's CK-tile module is still built at first call."""
    point = _point(tmp_path, "point.yaml", args="--tp 1")

    note = raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path, {"module_gemm_a8w8_bpreshuffle_cktile"})

    assert _watchdog(point, False) == f"--tp 1 {_FLAG} 3000"
    assert note is not None and "module_gemm_a8w8_bpreshuffle_cktile" in note


#: A module entry shaped like the image aiter's (e7d2453f) CK-tile GEMM, which reads its compiler from the environment.
_CKTILE_ENTRY = {
    "srcs": ["f'{AITER_CSRC_DIR}/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile.cu'"],
    "hip_clang_path": "os.environ.get('FLATMM_HIP_CLANG_PATH')",
    "blob_gen_cmd": "''",
}


@pytest.mark.parametrize(("inherited", "stale"), [({}, True), ({"FLATMM_HIP_CLANG_PATH": "/opt/clang"}, False)])
def test_a_build_variable_set_otherwise_than_the_runner_has_raises_the_watchdog_for_the_module_that_reads_it(
    image, tmp_path: Path, inherited: dict, stale: bool
) -> None:
    """The module is built at the first prefill under the server's value, as one a table makes stale; a value the runner
    has already is the one the image's module stands for, and changes nothing."""
    config = {**CONFIG, "module_gemm_a8w8_bpreshuffle_cktile": _CKTILE_ENTRY}
    (image / "jit" / "optCompilerConfig.json").write_text(json.dumps(config), encoding=_UTF8)
    point = _point(tmp_path, "point.yaml", args="--tp 1", FLATMM_HIP_CLANG_PATH="/opt/clang")

    note = raise_rebuild_watchdog(_call(tmp_path, point), None, inherited, tmp_path)

    if stale:
        assert _watchdog(point, False) == f"--tp 1 {_FLAG} 3000"
        assert note is not None and "module_gemm_a8w8_bpreshuffle_cktile" in note
    else:
        assert note is None and _FLAG not in point.read_text(encoding=_UTF8)


def _watchdog(point: Path, agentx: bool) -> str:
    bench = _bench(point)
    if agentx:
        return " ".join(bench["agentx"]["launch_overrides"]["append_args"])
    return bench["envs"]["EXTRA_SGLANG_ARGS"]


#: Every way a launch carries the flag: AgentX's argument list, the plain launch's argument string, and ``=``.
_CARRIERS = [(False, " "), (False, "="), (True, " "), (True, "=")]


@pytest.mark.parametrize(("agentx", "sep"), _CARRIERS)
def test_a_present_watchdog_below_the_estimate_is_raised_to_it(
    image, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agentx: bool, sep: str
) -> None:
    """The AgentX default (1800 s) is shorter than rebuilding three modules (2100 s): the default must not cut the
    rebuild."""
    monkeypatch.setattr(rebuild_watchdog, "stale_table_modules", lambda *args: frozenset({"m1", "m2", "m3"}))
    point = _point(tmp_path, "point.yaml", agentx=agentx, args=f"--tp 4 {_FLAG}{sep}1800 --mem 0.8", **_tuned(tmp_path))

    note = raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path)

    assert rebuild_watchdog_sec({"m1", "m2", "m3"}) == 2100
    assert _watchdog(point, agentx) == f"--tp 4 {_FLAG}{sep}2100 --mem 0.8"
    assert note is not None and "raised to 2100s" in note


@pytest.mark.parametrize(("agentx", "sep"), _CARRIERS)
@pytest.mark.parametrize("present", ["4000", "1500", "fast"])
def test_a_present_watchdog_at_or_above_the_estimate_is_kept(
    image, tmp_path: Path, agentx: bool, sep: str, present: str
) -> None:
    """A candidate's 4000 s and a value equal to the estimate stay; a value that is not a number is not rewritten."""
    assert rebuild_watchdog_sec(["m1", "m2"]) == 1500
    point = _point(tmp_path, "point.yaml", agentx=agentx, args=f"--tp 4 {_FLAG}{sep}{present}", **_tuned(tmp_path))
    before = point.read_text(encoding=_UTF8)

    assert raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path) is None
    assert point.read_text(encoding=_UTF8) == before


@pytest.mark.parametrize("agentx", [False, True])
def test_the_last_of_two_watchdogs_is_the_one_raised(image, tmp_path: Path, agentx: bool) -> None:
    """sglang's parser takes the last value, so that is the one compared and raised."""
    point = _point(tmp_path, "point.yaml", agentx=agentx, args=f"{_FLAG} 4000 {_FLAG} 120", **_tuned(tmp_path))

    raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path)

    assert _watchdog(point, agentx) == f"{_FLAG} 4000 {_FLAG} 1500"


#: A plain launch's argument string, and the same string once the watchdog is raised to 1500 s: in place for the flag
#: with ``=`` or a space, among other arguments too; kept for the flag last with no value (not a number); added when the
#: string only holds the flag's name inside another word.
_LINES = [
    (f"{_FLAG}=5", f"{_FLAG}=1500"),
    (f"{_FLAG} 5", f"{_FLAG} 1500"),
    (f"--tp 4 {_FLAG} 5 --mem 0.8", f"--tp 4 {_FLAG} 1500 --mem 0.8"),
    (f"--tp 4 {_FLAG}", f"--tp 4 {_FLAG}"),
    (f"--tp 4 {_FLAG}-x 5", f"--tp 4 {_FLAG}-x 5 {_FLAG} 1500"),
    (f"--tp 4 x{_FLAG} 5", f"--tp 4 x{_FLAG} 5 {_FLAG} 1500"),
]


@pytest.mark.parametrize(("line", "raised"), _LINES)
def test_a_plain_launch_names_the_flag_only_as_a_whole_argument(image, tmp_path: Path, line: str, raised: str) -> None:
    point = _point(tmp_path, "point.yaml", args=line, **_tuned(tmp_path))

    raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path)

    assert _watchdog(point, agentx=False) == raised


def test_a_test_whose_tables_are_the_images_keeps_sglangs_watchdog(image, tmp_path: Path) -> None:
    point = _point(tmp_path, "point.yaml", args="--tp 4")

    assert raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path) is None
    assert _FLAG not in point.read_text(encoding=_UTF8)


def test_a_vllm_point_is_left_as_it_is(image, tmp_path: Path) -> None:
    point = _point(tmp_path, "point.yaml", framework="vllm", **_tuned(tmp_path))

    assert raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path) is None
    assert _FLAG not in point.read_text(encoding=_UTF8)


class _Session:
    """What the runner reads of a compiled-code session: the modules its test's server builds itself."""

    def __init__(self, built: set[str]) -> None:
        self.built_by_server = frozenset(built)


def _own_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: Path, session, gfx_arch: str | None):
    """``run._own_directories`` for one point, with what it hands the other directories' owner recorded."""
    given: list[tuple] = []
    monkeypatch.setattr(run, "own_aiter_jit", lambda *args: given.append(args) or {})
    context = SimpleNamespace(environ={}, work=tmp_path, notes=[], compile=tmp_path / "compile")
    call = run._own_directories(context, _call(tmp_path, point), session, None, gfx_arch)
    return call, context.notes, given


def test_the_runner_gives_the_watchdog_the_modules_its_sessions_server_builds(
    image, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A patch to the CK-tile GEMM's sources makes no table or variable stale: only the session knows its server builds it."""
    point = _point(tmp_path, "point.yaml", args="--tp 1")
    session = _Session({"module_gemm_a8w8_bpreshuffle_cktile"})

    call, notes, _ = _own_directories(tmp_path, monkeypatch, point, session, "gfx950")

    assert _watchdog(point, False) == f"--tp 1 {_FLAG} 3000"
    assert len(notes) == 1 and "module_gemm_a8w8_bpreshuffle_cktile" in notes[0]
    assert call.jit is session


def test_a_runner_with_no_session_has_no_module_its_server_builds_for_the_watchdog(
    image, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    point = _point(tmp_path, "point.yaml", args="--tp 1")
    before = point.read_text(encoding=_UTF8)

    call, notes, _ = _own_directories(tmp_path, monkeypatch, point, None, "gfx950")

    assert (notes, call.jit) == ([], None) and point.read_text(encoding=_UTF8) == before


@pytest.mark.parametrize(("card", "raised"), [("gfx950", False), ("gfx942", True)])
def test_the_runner_tells_the_watchdog_and_the_jit_directory_which_card_it_is_on(
    image, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, card: str, raised: bool
) -> None:
    """``GPU_ARCHS`` naming the card is the image's own value for a module that reads it; naming another is not."""
    config = {**CONFIG, "module_opus": {"srcs": [], "flags_extra_hip": ["os.environ.get('GPU_ARCHS', '')"]}}
    (image / "jit" / "optCompilerConfig.json").write_text(json.dumps(config), encoding=_UTF8)
    point = _point(tmp_path, "point.yaml", args="--tp 1", GPU_ARCHS="gfx950")

    _, notes, given = _own_directories(tmp_path, monkeypatch, point, None, card)

    assert bool(notes) is raised
    assert [args[-1] for args in given] == [card], "the JIT directory's owner is told too"


@pytest.mark.parametrize(("envs", "raised"), [({}, False), ({"CU_NUM": "304"}, True)], ids=["untouched", "cu-num-304"])
def test_a_module_built_once_under_another_value_than_the_servers_raises_the_watchdog(
    image, tmp_path: Path, envs: dict, raised: bool
) -> None:
    """The launch preflight's prewarm built ``module_gemm_a8w8_bpreshuffle`` into the image for gfx950/256;
    a server whose ``CU_NUM`` differs is not given it and builds it at its first prefill."""
    module = "module_gemm_a8w8_bpreshuffle"
    (image / "jit" / f"{module}.so").write_bytes(b"\x7fELF prewarmed")
    entry = {"name": module, "gfx": "gfx950", "cu_num": 256, "build_variables": ["CU_NUM"], "build_environment": {}}
    document = {"schema": "hyperloom.aiter_prewarm/1", "modules": [entry]}
    (image / "jit" / "hyperloom-prewarm.json").write_text(json.dumps(document), encoding=_UTF8)
    point = _point(tmp_path, "point.yaml", args="--tp 1", **envs)

    note = raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path, gfx_arch="gfx950")

    if raised:
        assert _watchdog(point, False) == f"--tp 1 {_FLAG} {rebuild_watchdog_sec([module])}"
        assert note is not None and module in note
    else:
        assert note is None
