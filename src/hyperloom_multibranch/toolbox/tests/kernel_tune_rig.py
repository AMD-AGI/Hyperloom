# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the ``kernel_tune`` tests stand on: a landed test with a server log, a stand-in for the Forge command, and the
real local toolbox (pool, leases, tool jobs, notebook) built over them.

The server log lines are copied from the log of a real Qwen3-14B-FP8 test on gfx950 (SGLang, aiter): the ones that say
a dense blockscale GEMM missed its tuned table, a per-token one that Forge's reader does not take, and the lines between
them. The tuned rows are rows of the image's own ``a8w8_blockscale_bpreshuffle`` table (``kernel_rows_cc2bf0ef``).
``SERVER_ARGS`` is the line SGLang printed at that server's launch, cut to the arguments the readers use (the decode graph
batch sizes among them), and ``HIT_LOG`` the lines aiter logged for lookups that resolved in a tuned row, in the words of
the lines of that run.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import threading
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from hyperloom.contracts.measurement import Purpose, TestKind, TestResult, Validity
from hyperloom.measure.tests.hosted_fixtures import runner_template, spec_context
from hyperloom.orchestrator.actions.cancel_channel import current_cancel_scope
from hyperloom.sandbox.card_occupancy import CardReading, read_holders
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox import local_hosted
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox, LocalOptions, LocalSession
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.kernel_rows_cc2bf0ef import ENV, HEADER, SHARED, install
from hyperloom_multibranch.toolbox.tests.test_card_placement import _PointPerBranch
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup
from hyperloom_multibranch.toolbox.tools import kernel_tune

TUNER = "a8w8_blockscale_bpreshuffle"
TABLE = "a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
TEST_ID = "t1"
#: Lines 43-52 of the server log of a real test: a per-token a8w8 table that missed (Forge's reader takes no such line),
#: and the four (N, K) of the dense blockscale table at M=496, with aiter's import lines and a progress bar between.
SERVER_LOG = """\
[aiter] shape is M:512, N:7168, K:5120, q_dtype_w:torch.float8_e4m3fn, not found tuned config in /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv, will use default config!
[aiter] import [module_gemm_a8w8_bpreshuffle] under /sgl-workspace/aiter/aiter/jit/module_gemm_a8w8_bpreshuffle.so
[aiter] shape is M:512, N:34816, K:5120, q_dtype_w:torch.float8_e4m3fn, not found tuned config in /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv, will use default config!
[aiter] shape is M:512, N:5120, K:17408, q_dtype_w:torch.float8_e4m3fn, not found tuned config in /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv, will use default config!
Capturing batches (bs=512 avail_mem=54.88 GB):   2%|▏         | 1/52 [00:03<03:10,  3.73s/it]Capturing batches (bs=496 avail_mem=53.99 GB):   2%|▏         | 1/52 [00:03<03:10,  3.73s/it][aiter] merge tuned file under model_configs/ and configs/ /sgl-workspace/aiter/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv
[aiter] shape is M:496, N:7168, K:5120, not found tuned config in /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, will use default config!
[aiter] import [module_gemm_a8w8_blockscale_bpreshuffle] under /sgl-workspace/aiter/aiter/jit/module_gemm_a8w8_blockscale_bpreshuffle.so
[aiter] shape is M:496, N:5120, K:5120, not found tuned config in /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, will use default config!
[aiter] shape is M:496, N:34816, K:5120, not found tuned config in /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, will use default config!
[aiter] shape is M:496, N:5120, K:17408, not found tuned config in /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, will use default config!
"""
#: The ``server_args`` line of that server, cut to the keys of the graph config. Its decode graphs were captured for the
#: sizes of ``GRAPH_SIZES``; the prefill list, a flag that is unset and the key a reader must not take for the list are in it.
GRAPH_SIZES = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 256, 512)
SERVER_ARGS = (
    "[2026-10-22 04:34:21] server_args={'model_path': '/models/m', 'tp_size': 1, 'cuda_graph_bs_decode': None, "
    "'cuda_graph_bs_prefill': None, 'cuda_graph_config': {'decode': {'backend': 'full', 'max_bs': 512, "
    f"'bs': {list(GRAPH_SIZES)}, 'tc_compiler': 'eager', 'max_context_size': None}}, "
    "'prefill': {'backend': 'disabled', 'max_bs': 16384, 'bs': [4, 8, 12, 16, 20, 24, 28, 32, 48, 64]}}, "
    "'disable_cuda_graph': False}\n"
)
#: Lookups that resolved in a tuned row: M is the one asked for, ``padded_M`` the row found. Two of the blockscale table
#: (one at its merged view, one at the private copy of a table a test put), one of a per-token a8w8 table (its line does not
#: print ``q_dtype_w``, which keys it) and one of the dense bf16 table (which prints every column that keys it).
HIT_LOG = """\
[aiter] shape is M:12, N:7168, K:5120, found padded_M: 16, N:7168, K:5120 is tuned on cu_num = 256 in /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv , kernel name is a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256!
[aiter] shape is M:64, N:5120, K:17408, found padded_M: 64, N:5120, K:17408 is tuned on cu_num = 256 in /work/tables/AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE/5c34e83d432c56e2/k-5c34e83d432c56e2 , kernel name is a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256!
[aiter] shape is M:512, N:7168, K:5120 found padded_M: 512, N:7168, K:5120 is tuned on cu_num = 256 in /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv , kernel name is a8w8_bpreshuffle_x!
[aiter] shape is M:4, N:6144, K:4096 dtype='torch.bfloat16' otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False found padded_M: 4, N:6144, K:4096 is tuned on cu_num = 256 in /tmp/aiter_configs/bf16_tuned_gemm.csv, libtype is torch, kernel name is torch
"""
#: The shapes of ``HIT_LOG`` that become demand: (M of the row found, N, K) per table variable.
HIT_SHAPES = {
    ENV: {("16", "7168", "5120"), ("64", "5120", "17408")},
    "AITER_CONFIG_GEMM_BF16": {("4", "6144", "4096")},
}
#: The dense blockscale shapes of that log (at M=496): what Forge's demand holds for it, ranked by misses (all one).
LOG_SHAPES = {("496", "7168", "5120"), ("496", "5120", "5120"), ("496", "34816", "5120"), ("496", "5120", "17408")}
SECRET_ENV = "HF_TOKEN"


def table_text(header: str, rows: Iterable[str]) -> str:
    """A table of ``rows`` under ``header``, as the tuner writes one."""
    return "".join(f"{line}\n" for line in (header, *rows))


def tuned_rows(*, us: Sequence[float] = (7.9691, 8.1), err: Sequence[float] = (0.0029, 0.0)) -> tuple[str, ...]:
    """Rows of the image's table (the first two of ``SHARED``) with the microseconds and mismatch ratio given."""
    rows = []
    for line, microseconds, ratio in zip(SHARED, us, err, strict=False):
        cells = line.split(",")
        cells[HEADER.split(",").index("us")] = str(microseconds)
        cells[HEADER.split(",").index("errRatio")] = str(ratio)
        rows.append(",".join(cells))
    return tuple(rows)


def profile_row(
    m: int, n: int, k: int, *, lib: str, kernel_id: int, split_k: int, us: float, err: float, name: str
) -> str:
    """A line of the profile of every candidate the tuner measured: the columns of the table, whichever candidate won."""
    columns = HEADER.split(",")
    cells = dict.fromkeys(columns, "0") | {
        "gfx": "gfx950",
        "cu_num": "256",
        "M": str(m),
        "N": str(n),
        "K": str(k),
        "libtype": lib,
        "kernelId": str(kernel_id),
        "splitK": str(split_k),
        "us": str(us),
        "kernelName": name,
        "errRatio": str(err),
    }
    return ",".join(cells[column] for column in columns)


def forge_dropped(
    m: int, n: int, k: int, *, lib: str = "asm", split_k: int = 6, us: float = 24.4, err: float = 0.0125
) -> dict[str, Any]:
    """What Forge's report says of a row it took out of the table: its shape, libtype, splitK, microseconds and ratio."""
    return {
        "M": str(m),
        "N": str(n),
        "K": str(k),
        "libtype": lib,
        "splitK": str(split_k),
        "us": str(us),
        "err_ratio": err,
    }


def landed_test(session: Path, model: Path, *, log: str | None = SERVER_LOG, concs: Sequence[int] = (32, 128)) -> Path:
    """The landed output of a test of the session: a point configuration per concurrency, and the server log."""
    if not (model / "config.json").exists():
        model.mkdir(parents=True, exist_ok=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
    landed = session / "tests" / TEST_ID / "attempt-1"
    landed.mkdir(parents=True)
    for conc in concs:
        point = landed / f"variant_00_c{conc}"
        point.mkdir(parents=True)
        benchmark = {
            "framework": "sglang",
            "model": str(model),
            "precision": "fp8",
            "envs": {"TP": 1, "CONC": str(conc)},
        }
        (point / "config.yaml").write_text(yaml.safe_dump({"benchmark": benchmark}), encoding="utf-8")
    if log is not None:
        (landed / "server.log").write_text(log, encoding="utf-8")
    return landed


async def record_test(
    notebook: LocalNotebook, branch: str = "b1", output_dir: str = f"tests/{TEST_ID}/attempt-1"
) -> str:
    """The notebook's record of a test of ``branch`` whose output landed in ``output_dir`` of the session."""
    await notebook.write_request(replace(req(branch, purpose=Purpose.EXPLORE, kind=TestKind.TEST), request_id=TEST_ID))
    await notebook.write_result(TestResult(request_id=TEST_ID, validity=Validity.SUCCEEDED, output_dir=output_dir))
    return TEST_ID


def flags(cmd: Sequence[str]) -> dict[str, str]:
    """The ``--flag value`` pairs of a Forge command line; a flag without a value reads ``""``."""
    found: dict[str, str] = {}
    for index, word in enumerate(cmd):
        if word.startswith("--"):
            follows = cmd[index + 1] if index + 1 < len(cmd) else "--"
            found[word] = "" if follows.startswith("--") else follows
    return found


@dataclass
class FakeForge:
    """Answers a Forge command as the tuner does: its ``result.json`` and the table it names under ``tuners/<tuner>/``.

    ``rows`` are the lines of the table; ``dropped`` the rows Forge says it removed as wrong, and ``profile`` the lines of
    the profile of every candidate the tuner measured (written beside the table, as the tuner does, under ``profile_file``
    when the tuner names it its own way); ``hold`` makes the run wait,
    as a tuner on a card does, until ``release`` is set or the job's cancel scope fires (which ends a real child's process
    group); ``fail`` makes it leave no result.
    """

    rows: Sequence[str] = field(default_factory=tuned_rows)
    header: str = HEADER
    dropped: Sequence[Mapping[str, Any]] = ()
    profile: Sequence[str] = ()
    profile_file: str = ""
    hold: bool = False
    fail: bool = False
    release: threading.Event = field(default_factory=threading.Event)
    calls: list[tuple[list[str], dict[str, str]]] = field(default_factory=list)
    demands: list[dict[str, Any]] = field(default_factory=list)
    started: threading.Event = field(default_factory=threading.Event)

    def __call__(
        self, cmd: list[str], *, env: dict[str, str], timeout: float, text: bool
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(cmd), dict(env)))
        wanted = flags(cmd)
        self.demands.append(json.loads(Path(wanted["--demand"]).read_text(encoding="utf-8")))
        self.started.set()
        scope = current_cancel_scope()
        while self.hold and not self.release.is_set():
            if scope is not None and scope.cancelled:
                return subprocess.CompletedProcess(cmd, -15, "", "")
            self.release.wait(0.01)
        if self.fail:
            return subprocess.CompletedProcess(cmd, 2, "", "Traceback: the tuner crashed")
        out, tuner = Path(wanted["--output-dir"]), wanted["--tuner"]
        artifact = out / "tuners" / tuner / f"candidate_{tuner}.csv"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(table_text(self.header, self.rows), encoding="utf-8")
        if self.profile:
            name = self.profile_file or f"profile_{tuner}.csv"
            (artifact.parent / name).write_text(table_text(self.header, self.profile), encoding="utf-8")
        run = {"tuner": tuner, "status": "ok", "dropped_inaccurate": list(self.dropped), "shape_results": []}
        report = {
            "status": "ok",
            "micro_decision": "candidate",
            "tuners_run": [run],
            "artifacts": {tuner: str(artifact)},
            "total_elapsed_s": 1.0,
        }
        (out / "result.json").write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")


@dataclass
class Rig:
    box: LocalHostedToolbox
    notebook: LocalNotebook
    session: Path
    model: Path
    configs: Path
    forge: FakeForge

    def landed(self, **kw: Any) -> Path:
        return landed_test(self.session, self.model, **kw)

    async def submit(self, branch: str = "b1", **params: Any) -> str:
        return await self.box.submit_analysis(branch, kernel_tune.KERNEL_TUNE_TOOL, params)

    async def finished(self, job: str) -> Any:
        """The status of ``job`` once it ended and its pointer was pushed."""
        return await self.box._tools.wait(job)  # type: ignore[union-attr]

    def result(self, job: str) -> dict[str, Any]:
        """The ``result.json`` the job wrote for its branch's inbox."""
        return json.loads((self.session / "tool-results" / job / "result.json").read_text(encoding="utf-8"))


Make = Callable[..., Awaitable[Rig]]


def csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def build_rig(tmp_path: Path, monkeypatch: Any, made: list[LocalHostedToolbox], forge: FakeForge | None = None) -> Make:
    """``make(cards=2, tp=1, **branches)``: a local toolbox with the real tools on that many cards, Forge faked."""
    forge = forge or FakeForge()
    monkeypatch.setattr(kernel_tune, "run_with_session_kill", forge)

    def fake_runner(name: str, *, gpus: int, python: str, envs: Any) -> Any:
        return replace(runner_template(name, gpus=gpus), envs=dict(envs or {}))

    monkeypatch.setattr(local_hosted, "local_runner_template", fake_runner)

    async def make(
        cards: int = 2,
        tp: int = 1,
        points: Mapping[str, Path] | None = None,
        card_holders: Callable[[Sequence[int]], CardReading] = read_holders,
    ) -> Rig:
        session = tmp_path / f"session-{len(made)}"
        session.mkdir()
        configs = install(tmp_path / f"image-{len(made)}")
        notebook = LocalNotebook(session / "notebook", Writer(Role.TOOLBOX, "toolbox"))
        box = local_hosted.build_local_toolbox(
            LocalSession(session, f"run-{len(made)}", range(cards), tp=tp),
            notebook,
            AcceptanceSetup(FakeRepo(), _PointPerBranch(dict(points or {})), IMAGE, WorkloadKind.AGENTX),
            spec_context(session, benchmark_backend="bypass", blob_root=str(session / "hosted" / "blobs")),
            options=LocalOptions(poll_interval_sec=0.05, framework_configs=str(configs), card_holders=card_holders),
        )
        made.append(box)
        await box.start()
        return Rig(box, notebook, session, tmp_path / "model", configs, forge)

    return make


__all__ = [
    "ENV",
    "GRAPH_SIZES",
    "HIT_LOG",
    "HIT_SHAPES",
    "LOG_SHAPES",
    "SECRET_ENV",
    "SERVER_ARGS",
    "SERVER_LOG",
    "TABLE",
    "TEST_ID",
    "TUNER",
    "FakeForge",
    "Rig",
    "build_rig",
    "csv_rows",
    "flags",
    "forge_dropped",
    "landed_test",
    "profile_row",
    "record_test",
    "tuned_rows",
]
