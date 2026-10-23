# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the ``kernel_tune`` tests stand on: a landed test with a server log, a stand-in for the Forge command, and the
real local toolbox (pool, leases, tool jobs, notebook) built over them.

The server log lines are copied from the log of a real Qwen3-14B-FP8 test on gfx950 (SGLang, aiter): the ones that say
a dense blockscale GEMM missed its tuned table, a per-token one that Forge's reader does not take, and the lines between
them. The tuned rows are rows of the image's own ``a8w8_blockscale_bpreshuffle`` table (``kernel_rows_cc2bf0ef``).
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

    ``rows`` are the lines of the table; ``dropped`` the rows Forge says it removed as wrong; ``hold`` makes the run wait,
    as a tuner on a card does, until ``release`` is set or the job's cancel scope fires (which ends a real child's process
    group); ``fail`` makes it leave no result.
    """

    rows: Sequence[str] = field(default_factory=tuned_rows)
    header: str = HEADER
    dropped: Sequence[Mapping[str, Any]] = ()
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

    async def make(cards: int = 2, tp: int = 1, points: Mapping[str, Path] | None = None) -> Rig:
        session = tmp_path / f"session-{len(made)}"
        session.mkdir()
        configs = install(tmp_path / f"image-{len(made)}")
        notebook = LocalNotebook(session / "notebook", Writer(Role.TOOLBOX, "toolbox"))
        box = local_hosted.build_local_toolbox(
            LocalSession(session, f"run-{len(made)}", range(cards), tp=tp),
            notebook,
            AcceptanceSetup(FakeRepo(), _PointPerBranch(dict(points or {})), IMAGE, WorkloadKind.AGENTX),
            spec_context(session, benchmark_backend="bypass", blob_root=str(session / "hosted" / "blobs")),
            options=LocalOptions(poll_interval_sec=0.05, framework_configs=str(configs)),
        )
        made.append(box)
        await box.start()
        return Rig(box, notebook, session, tmp_path / "model", configs, forge)

    return make


__all__ = [
    "ENV",
    "LOG_SHAPES",
    "SECRET_ENV",
    "SERVER_LOG",
    "TABLE",
    "TEST_ID",
    "TUNER",
    "FakeForge",
    "Rig",
    "build_rig",
    "csv_rows",
    "flags",
    "landed_test",
    "record_test",
    "tuned_rows",
]
