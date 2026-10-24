# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``kernel_tune`` tool: Forge's GEMM tuner on one card of its own, for the shapes a test's server log reports untuned.

A specialist names a test (``test_request_id``) whose server log says which GEMM shapes aiter looked up in a tuned table
and did not find, or lists shapes itself (``shapes``); the test also says which model, framework and precision the
shapes are for. The tool runs the tuner Forge has for each table (``kernelforge gemm-tune run``, through
``hyperloom.orchestrator.kernel.forge_gemm_tuning.build_command``, the command the single-lineage flow runs) with those
shapes as its demand, on one card, and answers with the tuned rows of each table as a rows-only CSV that
``put_kernel_artifact(rows_only=true)`` takes (:mod:`.tune_rows`), each row checked against the reference GEMM (an asm row
at a splitK above 1, by its relative-norm error when its mismatch ratio is above the limit).

The tuner is pinned to the aiter the tests' servers run: ``AITER_ROOT_DIR`` of its process is the root of the aiter install
whose ``configs`` the tool reads the tables' shape keys from (``serving_aiter_root``), and the job's result says so
(``aiter``). Without a pin Forge tunes against whatever aiter it finds, which need not be the one that serves; a toolbox
whose install cannot be found fails the job before it takes a card.

The card is a lease from the pool the measurements and the branches' sandboxes take theirs from: the tool job counts in
``MeasurementCapacity.leased_cards`` from the moment it starts until it ends, whatever way it ends, and a job that
would leave fewer cards than one test needs is refused (``lease_leaves_no_measurement``), so a tool job never starves
measurement. The microseconds in the rows are the tuner's micro-benchmark: advice for the specialist, never an input of
the gate. A test of the rows is what says whether they help.

Three things a call may ask beyond the shapes. ``retune`` tunes the shapes the tables already serve (the server log's hit
lines) when it reports no miss. The decode CUDA-graph batch sizes the test's servers captured are added to the demand as
M, since a decode step replays a graph of exactly one of them (what Forge does with those M is its own: it rounds the M of
a demand up to a power of two). ``max_err_ratio`` (up to 0.05, default 0.01) also returns the rows with a mismatch ratio
above the default and within it, in a CSV of their own (``rows_csv_relaxed``), never in ``rows_csv``: whether they stand
is the specialist's explicit choice, and only an end-to-end accuracy check decides it.
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import GPU_MASK_ENV_NAMES, is_secret_shaped_env_name, scrub_benchmark_process_env
from hyperloom.common.outside_facts import AITER_META_MARKER_DIR, AITER_META_WHEEL_DIR, ROCR_VISIBLE_DEVICES_ENV
from hyperloom.orchestrator.actions.cancel_channel import current_cancel_scope
from hyperloom.orchestrator.actions.executors import COOPERATIVE_REAP_BUDGET_SEC, run_with_session_kill
from hyperloom.orchestrator.kernel import forge_gemm_tuning
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.toolbox import BranchLease, RequestRejected
from hyperloom_multibranch.toolbox.branch_leases import BranchLeases
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolFailure, ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.tune_inputs import (
    DENSE_TABLES,
    ServedWorkload,
    demand_from_logs,
    explicit_demand,
    served_workload,
    write_demand_of,
)
from hyperloom_multibranch.toolbox.tools.tune_rows import ERR_RATIO_LIMIT, MAX_ERR_RATIO_LIMIT, REL_NORM_ERR_LIMIT
from hyperloom_multibranch.toolbox.tools.tune_tables import (
    TableStatus,
    TuneTable,
    failed,
    read_run,
    skipped,
    tables_summary,
)

__all__ = [
    "DEFAULT_TIMEOUT_SEC",
    "HOLDER_PREFIX",
    "KERNEL_TUNE_TOOL",
    "LANDED_PARAM",
    "MAX_ERR_RATIO_PARAM",
    "MAX_TIMEOUT_SEC",
    "RETUNE_PARAM",
    "SHAPES_PARAM",
    "TEST_PARAM",
    "TIMEOUT_PARAM",
    "KernelTuneTool",
    "check_params",
    "kernel_tune_tool",
    "serving_aiter_root",
]

KERNEL_TUNE_TOOL = "kernel_tune"
#: What the caller names: the test whose landed output is read, optional shapes of its own, and the time to spend.
TEST_PARAM = "test_request_id"
SHAPES_PARAM = "shapes"
TIMEOUT_PARAM = "timeout_sec"
#: The mismatch ratio a call allows beyond the default, and whether it asks for the shapes the tables serve.
MAX_ERR_RATIO_PARAM = "max_err_ratio"
RETUNE_PARAM = "retune"
#: What the server adds: the landed output directory of the test, resolved inside the session; never the caller's.
LANDED_PARAM = "landed_dir"
#: A lease of a tool job names its holder ``tool:<job id>``, which no branch id is.
HOLDER_PREFIX = "tool:"
DEFAULT_TIMEOUT_SEC = 1800
MAX_TIMEOUT_SEC = 4 * 3600
#: The least a tuner is given: a run that starts with less ends before the JIT build of its first kernels does.
MIN_TUNER_SEC = 600
MAX_SHAPES = 64
#: What the subprocess is given beyond the budget Forge keeps itself to, to write its report.
_GRACE_SEC = 300
_CARDS = 1
#: The variable Forge reads the aiter it tunes against from.
_AITER_ROOT_ENV = "AITER_ROOT_DIR"
_CANCEL_REASON = "kernel_tune_cancelled"
_SHAPE_NAMES = frozenset({"env", "M", "N", "K", "dtype", "otype", "bias", "scaleAB", "bpreshuffle"})
_ADVICE = (
    "The microseconds are the tuner's micro-benchmark on one card: advice, not a measurement the gate reads. Put each "
    "table with put_kernel_artifact(kind=gemm, target, env, content=rows_csv, rows_only=true) and run a test: only its "
    "result says whether the rows help."
)
_VERIFICATION = (
    "Every row was compared by aiter's tuner with a reference GEMM of its shape, element by element within aiter's "
    "tolerance (rtol and atol of 1e-2 unless the tuner sets its own), and the mismatch fraction recorded; "
    f"a row is handed over only with a finite mismatch ratio of at most {ERR_RATIO_LIMIT:g}, or, for an asm kernel at "
    "a splitK above 1 (whose bf16 accumulation moves that ratio while the output stays accurate), a relative-norm "
    f"error of at most {REL_NORM_ERR_LIMIT:g} that Forge measured (verified_rows say which check decided each row: "
    "check). Rows that failed or could not be checked are listed as rejected with verified=false and the reason, "
    "and are not in rows_csv."
)
_RELAXED = (
    f" This call allowed a mismatch ratio up to {{limit:g}}: the rows above {ERR_RATIO_LIMIT:g} and within it that the "
    "default check removed are in each table's rows_csv_relaxed (listed in relaxed_rows, each verified=false with its "
    "err_ratio and the reason err_ratio_above_default). They are not checked rows and are never in rows_csv. Putting "
    "them is your explicit choice, and only an end-to-end accuracy check (GSM8K, and the long-context check for "
    "attention or KV paths) says whether they stand."
)


def _unknown(params: Mapping[str, Any]) -> list[str]:
    return sorted(
        set(params) - {TEST_PARAM, SHAPES_PARAM, TIMEOUT_PARAM, MAX_ERR_RATIO_PARAM, RETUNE_PARAM, LANDED_PARAM}
    )


def _shape_problem(shape: Any) -> str | None:
    if not isinstance(shape, Mapping):
        return "is not an object"
    if extra := sorted(set(shape) - _SHAPE_NAMES):
        return f"has unknown column(s) {extra}"
    if missing := sorted({"env", "M", "N", "K"} - set(shape)):
        return f"lacks {missing}"
    if shape["env"] not in DENSE_TABLES:
        return f"names {shape['env']!r}, not one of {', '.join(sorted(DENSE_TABLES))}"
    if any(not isinstance(shape[name], int) or isinstance(shape[name], bool) or shape[name] < 1 for name in "MNK"):
        return "has an M, N or K that is not a positive integer"
    return None


def check_params(params: Mapping[str, Any]) -> None:
    """Refuse the caller's parameters that no run can use; raises :class:`InvalidRequest`.

    ``shapes`` is a list of objects, each naming ``env`` (the variable of a dense aiter GEMM table), ``M``, ``N``, ``K``
    and, for a table keyed on more, its dtype and flag columns. ``max_err_ratio`` is a mismatch ratio from the default
    (0.01) to aiter's own filter (0.05), ``retune`` a boolean that does not go with ``shapes`` (those are tuned as named).
    """
    if unknown := _unknown(params):
        raise InvalidRequest(
            f"kernel_tune takes {TEST_PARAM}, {SHAPES_PARAM}, {TIMEOUT_PARAM}, {MAX_ERR_RATIO_PARAM} and "
            f"{RETUNE_PARAM}, not {unknown}"
        )
    shapes = params.get(SHAPES_PARAM)
    if shapes is not None:
        if not isinstance(shapes, Sequence) or isinstance(shapes, str) or not 0 < len(shapes) <= MAX_SHAPES:
            raise InvalidRequest(f"{SHAPES_PARAM} is a list of 1 to {MAX_SHAPES} shapes")
        for index, shape in enumerate(shapes):
            if (problem := _shape_problem(shape)) is not None:
                raise InvalidRequest(f"shape {index} {problem}")
    timeout = params.get(TIMEOUT_PARAM, DEFAULT_TIMEOUT_SEC)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or not MIN_TUNER_SEC <= timeout <= MAX_TIMEOUT_SEC:
        raise InvalidRequest(f"{TIMEOUT_PARAM} is whole seconds from {MIN_TUNER_SEC} to {MAX_TIMEOUT_SEC}")
    ratio = params.get(MAX_ERR_RATIO_PARAM, ERR_RATIO_LIMIT)
    if not _is_ratio(ratio) or not ERR_RATIO_LIMIT <= ratio <= MAX_ERR_RATIO_LIMIT:
        raise InvalidRequest(
            f"{MAX_ERR_RATIO_PARAM} is a mismatch ratio from {ERR_RATIO_LIMIT:g} (the default) to {MAX_ERR_RATIO_LIMIT:g}"
        )
    retune = params.get(RETUNE_PARAM, False)
    if not isinstance(retune, bool):
        raise InvalidRequest(f"{RETUNE_PARAM} is true or false")
    if retune and shapes is not None:
        raise InvalidRequest(
            f"{RETUNE_PARAM} tunes the shapes the test's tables served; the {SHAPES_PARAM} you name are tuned as named, "
            "covered by a table or not, so give one or the other"
        )


def _is_ratio(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class _Plan:
    """What a run tunes: the workload the shapes are for, the demand of each table (most missed first), and the root of the
    aiter it tunes against."""

    workload: ServedWorkload
    demands: tuple[Mapping[str, Any], ...]
    source: str
    aiter_root: Path
    max_err_ratio: float
    #: What was added to the demand beyond the shapes (``demand_notes``), and the mismatch ratio the call allowed.
    notes: Mapping[str, Any]


def _plan(landed: Path, params: Mapping[str, Any], configs: Path) -> _Plan:
    shapes = params.get(SHAPES_PARAM) or ()
    aiter_root = serving_aiter_root(configs)
    workload = served_workload(landed)
    retune = params.get(RETUNE_PARAM, False)
    demand = explicit_demand(shapes) if shapes else demand_from_logs(landed, retune=retune, conc=workload.conc)
    notes = {
        "retune": retune,
        "served_keys": demand.get("served_keys", 0),
        "served_lines_unread": demand.get("served_lines_unread", 0),
        "cuda_graph_batch_sizes": demand.get("cuda_graph_batch_sizes", []),
    }
    ratio = float(params.get(MAX_ERR_RATIO_PARAM, ERR_RATIO_LIMIT))
    return _Plan(workload, tuple(demand["demands"]), "shapes" if shapes else "server_log", aiter_root, ratio, notes)


def serving_aiter_root(configs: Path) -> Path:
    """The root of the aiter install whose ``configs`` directory is ``configs``: where its ``csrc`` (the tuner scripts) is.

    ``configs`` is the install's ``aiter/configs``, the tables every test reads and the server imports ``aiter`` from, with
    ``aiter/jit/core.py`` beside it (the file the shape keys are read from). The root is the directory around the
    package, or its ``aiter_meta`` for a packaged install, as Forge's own search takes it.

    Raises:
        ToolFailure: When ``configs`` is not the ``configs`` of such an install.
    """
    package = Path(os.path.abspath(configs)).parent
    if configs.name == "configs" and configs.is_dir() and (package / "jit" / "core.py").is_file():
        for root in (package.parent, package.parent / AITER_META_WHEEL_DIR):
            if (root / AITER_META_MARKER_DIR).is_dir():
                return root
    raise ToolFailure(
        f"the aiter the tests' servers run cannot be found from {str(configs)!r}: it is not the aiter/configs of an "
        f"install with aiter/jit/core.py beside it and {AITER_META_MARKER_DIR} around it, so the tuner cannot be pinned to "
        f"it ({_AITER_ROOT_ENV}) and is not run against whatever aiter it finds"
    )


def _tuner_env(cards: Sequence[int], aiter_root: Path) -> dict[str, str]:
    """This process's environment without credentials, hooks or a mask of the visible cards, masked to ``cards`` and
    with the tuner's aiter pinned to ``aiter_root``."""
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in GPU_MASK_ENV_NAMES and not is_secret_shaped_env_name(name)
    }
    env[ROCR_VISIBLE_DEVICES_ENV] = ",".join(map(str, cards))
    env[_AITER_ROOT_ENV] = str(aiter_root)
    return scrub_benchmark_process_env(env)


async def _reaped(worker: asyncio.Future[Any]) -> None:
    """Wait, up to the reap budget, for the thread that runs a child to end, so its card is not given back under it."""
    worker.add_done_callback(lambda done: done.cancelled() or done.exception())
    await asyncio.wait({worker}, timeout=COOPERATIVE_REAP_BUDGET_SEC)


async def _forge(cmd: list[str], env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run the Forge command in a session of its own, ended with the job: a cancelled job has the child's process group
    killed, and this returns only when the thread that runs it ended (or the reap budget did)."""
    worker = asyncio.ensure_future(asyncio.to_thread(run_with_session_kill, cmd, env=env, timeout=timeout, text=True))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        if (scope := current_cancel_scope()) is not None:
            scope.cancel(reason=_CANCEL_REASON)
        await _reaped(worker)
        raise


class KernelTuneTool:
    """``kernel_tune``: Forge's tuner for the shapes of a test on a card leased for the job.

    ``leases`` is the pool the branches' sandboxes lease from, ``configs`` the aiter install's ``configs`` directory the
    tables' shape keys are read from (the one the tests read, see ``LocalOptions.framework_configs``).
    """

    name = KERNEL_TUNE_TOOL

    def __init__(self, leases: BranchLeases, configs: Path) -> None:
        self._leases = leases
        self._configs = configs

    def validate(self, request: ToolRequest) -> None:
        """Refuse a request that cannot work: bad parameters (:class:`InvalidRequest`), or a lease that would leave fewer
        cards than one test needs (:class:`RequestRejected`, ``lease_leaves_no_measurement``)."""
        check_params(request.params)
        if not request.params.get(LANDED_PARAM):
            raise InvalidRequest(f"a kernel_tune request names the test whose output it reads with {TEST_PARAM}")
        self._leases.check(_CARDS)

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        params = request.params
        plan = await asyncio.to_thread(_plan, Path(params[LANDED_PARAM]), params, self._configs)
        lease = await self._lease(workdir)
        try:
            tables = await self._tune_all(plan, lease, workdir, int(params.get(TIMEOUT_PARAM, DEFAULT_TIMEOUT_SEC)))
        finally:
            self._leases.release(lease.lease_id)
        return _output(plan, lease, tables, str(params.get(TEST_PARAM, "")))

    async def _lease(self, workdir: Path) -> BranchLease:
        try:
            return await self._leases.lease(f"{HOLDER_PREFIX}{workdir.name}", _CARDS, workdir)
        except RequestRejected as exc:
            raise ToolFailure(f"{exc.code}: {exc.reason}") from None
        except SandboxUnavailable as exc:
            raise ToolFailure(str(exc)) from None

    async def _tune_all(self, plan: _Plan, lease: BranchLease, workdir: Path, timeout: int) -> list[TuneTable]:
        """One Forge run per demanded table, in turn on the one card, the time shared out equally; the tables the time
        does not reach are said to be skipped."""
        affordable = max(1, timeout // MIN_TUNER_SEC)
        share = timeout // min(len(plan.demands), affordable)
        tables = []
        for index, demand in enumerate(plan.demands):
            if index >= affordable:
                tables.append(skipped(demand, f"no time left in {TIMEOUT_PARAM} for it: {timeout} s"))
                continue
            tables.append(await self._tune(plan, demand, lease.gpu_ids, workdir / demand["tuner"], share))
        return tables

    async def _tune(
        self, plan: _Plan, demand: Mapping[str, Any], cards: Sequence[int], out: Path, budget: int
    ) -> TuneTable:
        workload = plan.workload
        out.mkdir(parents=True, exist_ok=True)
        demand_file = write_demand_of(demand, out / "demand.json")
        cmd = forge_gemm_tuning.build_command(
            {
                "model_path": workload.model_path,
                "framework": workload.framework,
                "precision": workload.precision,
                "tp": workload.tp,
                "conc": workload.conc,
                "mp": len(cards),
                "output_dir": str(out),
                "timeout": budget,
                "global_timeout": budget,
                "tuner": demand["tuner"],
                "demand_json": str(demand_file),
            }
        )
        try:
            done = await _forge(cmd, _tuner_env(cards, plan.aiter_root), budget + _GRACE_SEC)
        except subprocess.TimeoutExpired:
            return failed(demand, f"the tuner did not end within {budget + _GRACE_SEC} s and was stopped")
        return read_run(demand, out, done, self._configs, plan.max_err_ratio)


def _output(plan: _Plan, lease: BranchLease, tables: list[TuneTable], test_request_id: str) -> ToolOutput:
    workload = plan.workload
    result = {
        "source": {"shapes": plan.source, "test_request_id": test_request_id},
        "workload": {
            "model": Path(workload.model_path).name,
            "framework": workload.framework,
            "precision": workload.precision,
            "tp": workload.tp,
            "conc": workload.conc,
        },
        "cards": list(lease.gpu_ids),
        "aiter": {"pinned": True, "root_dir": str(plan.aiter_root)},
        "demand": plan.notes,
        "max_err_ratio": plan.max_err_ratio,
        "tables": [table.document() for table in tables],
        "verification": _VERIFICATION
        + (_RELAXED.format(limit=plan.max_err_ratio) if plan.max_err_ratio > ERR_RATIO_LIMIT else ""),
        "advice": _ADVICE,
    }
    ok = any(table.status in (TableStatus.TUNED, TableStatus.NO_ROWS) for table in tables)
    return ToolOutput(result=result, summary=f"{tables_summary(tables)}; microseconds are advice only", ok=ok)


def kernel_tune_tool(leases: BranchLeases, configs: Path) -> KernelTuneTool:
    """The tool a local toolbox serves: it leases its card from ``leases``."""
    return KernelTuneTool(leases, configs)
