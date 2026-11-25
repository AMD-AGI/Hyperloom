# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A sweep of a kernel knob (:mod:`hyperloom.common.outside_facts.kernel_knob`): one build per value, each key's
instance timed on the GPU against the build of the pinned commit as it is, and a value recommended for a key only
where it is faster beyond the noise of the measurement.

Every key's instance is compiled once per build, so a build forcing value ``v`` on every key of the sweep times ``v``
on all of them at once; the keys are independent instances, so the winner of each is chosen on its own, and the patch
handed back forces each key's winner and leaves the others to upstream. The builds are the session's
(:class:`~hyperloom.measure.runner.native.session.NativeBuildSession` through its caller), keyed by their patch: one
already built is reused, and they run ``parallel`` at a time on the build CPUs.

Each key is timed over the caller's weight shapes (``m`` x ``k``) by ``test-backend-ops``
(:mod:`hyperloom.measure.runner.native.backend_ops`), one process per arm per round. A case multiplies ``copies``
weights of its shape in one launch: a model's decode step reads far more weight than a GPU's last-level cache holds,
and one weight that fits there would time the cache instead. A round holds the GPU once
(``hold``: the caller's lock or lease, released between rounds) and runs every arm in an order rotated each round,
the default build twice, so slow drift spreads over all arms and the two default runs measure what no change makes
(the A/A gap). A key's time in a round is the sum over its shapes; an arm's time is the median over the rounds. The
noise of a comparison is the larger of the A/A gap and three standard errors of the difference of two medians (from
the median absolute deviation of each arm); a gain within it is refused as a winner. The winners' builds are then
checked against the CPU backend on the same cases, and a key whose case does not match is refused too.

The times are a microbenchmark: advice to the caller, never a measurement a gate reads. The patch goes through the
normal integration and test path, and only its test says whether it helps the model.
"""

from __future__ import annotations

import difflib
import itertools
import logging
import math
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ContextManager

from hyperloom.common.outside_facts import LLAMACPP_BACKEND_OPS, kernel_knobs
from hyperloom.common.outside_facts.kernel_knob import KernelKnobSpec, KnobKey, KnobSpecError
from hyperloom.measure.runner.native.backend_ops import CHECK_OK, BackendOpsError, MulMatCase, case_file
from hyperloom.orchestrator.actions.executors.candidate_builds import TreeBuild

__all__ = [
    "DEFAULT_COPIES",
    "SweepError",
    "SweepHooks",
    "SweepPlan",
    "SweepResult",
    "SweepTiming",
    "Verdict",
    "knob_patch",
    "plan_sweep",
    "run_sweep",
]

logger = logging.getLogger(__name__)

_DEFAULT_ARM = "default"
_REPLICA_ARM = "default#2"
_MAD_TO_SIGMA = 1.4826
#: The standard error of a median is this many times that of a mean of normal samples.
_MEDIAN_SE = 1.2533
_Z = 3.0
_CASES_FILE = "cases.txt"
#: Fewer rounds give no spread to judge the noise by; more only hold the GPU longer.
_MIN_ROUNDS, _MAX_ROUNDS = 3, 15
_MAX_COPIES = 64
#: Four copies put a 5120 x 17408 Q8_0 weight (95 MB) at 380 MB, past the last-level cache of the GPUs swept (96 MB on
#: a W7900).
DEFAULT_COPIES = 4
_MAX_SHAPES = 16
_BENCH_FAILED = "bench_failed"
_TIMED = "timed"
_ADVICE = (
    "The times are test-backend-ops microbenchmarks of the knob's kernel instances on one GPU: advice, not a "
    "measurement the gate reads. Hand the patch to the normal integration path and run a test: only its result says "
    "whether the model gets faster."
)


class SweepError(RuntimeError):
    """The sweep cannot be measured: the build of the pinned commit failed, or so did its timing, or no value was
    timed."""


class Verdict(str, Enum):
    """How a value compares with the default on a key, beyond the noise or within it; or refused by the check."""

    FASTER = "faster"
    SLOWER = "slower"
    WITHIN_NOISE = "within_noise"
    CHECK_FAILED = "check_failed"


@dataclass(frozen=True)
class SweepPlan:
    """What to sweep: the knob, the keys and values of it, the weight shapes (``m``, ``k``) each key is timed at, and
    the rounds."""

    name: str
    knob: KernelKnobSpec
    keys: tuple[KnobKey, ...]
    values: tuple[str, ...]
    shapes: tuple[tuple[int, int], ...]
    rounds: int
    copies: int


@dataclass(frozen=True)
class SweepTiming:
    """How each key is timed: the weight shapes (``m``, ``k``), the rounds, and the copies of a weight in a launch."""

    shapes: Sequence[tuple[int, int]]
    rounds: int
    copies: int


@dataclass(frozen=True)
class SweepHooks:
    """What a sweep runs through: ``build`` builds a git patch over the pinned commit (``b""`` for the commit as it is)
    and lays its executables; ``hold`` holds the GPU for one round and gives the environment its processes run in;
    ``time`` and ``check`` are :func:`~hyperloom.measure.runner.native.backend_ops.time_cases` and
    :func:`~hyperloom.measure.runner.native.backend_ops.check_cases` with their timeout bound."""

    build: Callable[[bytes], TreeBuild]
    hold: Callable[[], ContextManager[Mapping[str, str]]]
    time: Callable[..., dict[str, float]]
    check: Callable[..., dict[str, str]]


def _declared_knobs() -> str:
    """Every knob the rule pack declares, as a refusal lists them: its component, its axes with their values, and the
    values a key may be given."""
    return "; ".join(
        f"{name} of {knob.component}, axes "
        + ", ".join(f"{axis} {list(held)}" for axis, held in knob.axes)
        + f", values {list(knob.values)}"
        for name, knob in sorted(kernel_knobs().items())
    )


def plan_sweep(name: str, where: Mapping[str, Sequence[str]], values: Sequence[str], timing: SweepTiming) -> SweepPlan:
    """The sweep of the knob ``name`` over the keys whose axes hold the values ``where`` names (an axis it does not name
    takes all of its values), forcing ``values`` (all of the knob's when empty), timed as ``timing`` says.

    Raises:
        KnobSpecError: The knob, an axis, a value or a shape is not one the knob holds, or the rounds or the copies are
            out of range.
    """
    knob = kernel_knobs().get(name)
    if knob is None:
        raise KnobSpecError(f"no kernel knob {name!r}; the knobs are: {_declared_knobs()}")
    axes = dict(knob.axes)
    if unknown := sorted(set(where) - set(axes)):
        raise KnobSpecError(f"{name} has the axes {sorted(axes)}, not {unknown}")
    picked = {axis: tuple(str(value) for value in where.get(axis) or held) for axis, held in axes.items()}
    keys = tuple(knob.key_of(dict(zip(picked, combo, strict=True))) for combo in itertools.product(*picked.values()))
    chosen = tuple(str(value) for value in values) or knob.values
    if stray := [value for value in chosen if value not in knob.values]:
        raise KnobSpecError(f"{name} takes the values {list(knob.values)}, not {stray}")
    if not 0 < len(timing.shapes) <= _MAX_SHAPES or any(m < 1 or k < 1 for m, k in timing.shapes):
        raise KnobSpecError(f"a sweep is timed at 1 to {_MAX_SHAPES} weight shapes m x k of positive sizes")
    if not _MIN_ROUNDS <= timing.rounds <= _MAX_ROUNDS:
        raise KnobSpecError(f"a sweep runs {_MIN_ROUNDS} to {_MAX_ROUNDS} rounds, not {timing.rounds}")
    if not 1 <= timing.copies <= _MAX_COPIES:
        raise KnobSpecError(f"a case multiplies 1 to {_MAX_COPIES} copies of a weight, not {timing.copies}")
    return SweepPlan(name, knob, keys, chosen, tuple(timing.shapes), timing.rounds, timing.copies)


@dataclass(frozen=True)
class _Arm:
    name: str
    value: str | None
    patch: bytes


@dataclass
class _Timed:
    build: TreeBuild
    samples: list[dict[str, float]] = field(default_factory=list)
    detail: str = ""


@dataclass(frozen=True)
class _Row:
    key: KnobKey
    value: str
    median_us: float
    gain: float
    noise: float
    verdict: Verdict


@dataclass(frozen=True)
class SweepResult:
    """The sweep's answer: its JSON ``document`` and the patch forcing each key's winning value (empty when no value
    beat the default beyond the noise)."""

    document: Mapping[str, Any]
    patch: bytes


def knob_patch(knob: KernelKnobSpec, source: str, settings: Mapping[KnobKey, str]) -> bytes:
    """A git patch of ``knob``'s file over the pinned ``source`` that forces ``settings``; empty for none."""
    if not settings:
        return b""
    after = knob.apply(source, settings)
    path = knob.file
    lines = difflib.unified_diff(
        source.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{path}", f"b/{path}"
    )
    return (f"diff --git a/{path} b/{path}\n" + "".join(lines)).encode()


def _case_name(key: KnobKey, shape: tuple[int, int]) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", "_".join(key)) + f"_{shape[0]}x{shape[1]}"


def _cases(plan: SweepPlan) -> dict[KnobKey, list[MulMatCase]]:
    cases = {}
    for key in plan.keys:
        bench = plan.knob.bench_case(key)
        cases[key] = [
            MulMatCase(_case_name(key, (m, k)), bench["type_a"], m, k, int(bench["n"]), plan.copies)
            for m, k in plan.shapes
        ]
    return cases


def _arms(plan: SweepPlan, source: str) -> list[_Arm]:
    return [_Arm(_DEFAULT_ARM, None, b"")] + [
        _Arm(f"value={value}", value, knob_patch(plan.knob, source, dict.fromkeys(plan.keys, value)))
        for value in plan.values
    ]


def _totals(timed: _Timed, cases: Sequence[MulMatCase]) -> list[float]:
    return [sum(sample[case.name] for case in cases) for sample in timed.samples]


def _sigma(totals: Sequence[float]) -> float:
    """The relative spread of one arm's per-round totals: its median absolute deviation, as a standard deviation."""
    middle = statistics.median(totals)
    return _MAD_TO_SIGMA * statistics.median(abs(total - middle) for total in totals) / middle


class _Bench:
    """The rounds of the timed arms (see the module)."""

    def __init__(self, cases: Path, names: Sequence[str], time: Callable[..., dict[str, float]]) -> None:
        self.cases, self.names, self.time = cases, names, time

    def rounds(
        self, arms: dict[str, _Timed], rounds: int, hold: Callable[[], ContextManager[Mapping[str, str]]]
    ) -> None:
        order = [*arms]
        with hold() as env:
            for name in order:
                self._one(arms, name, env, keep=False)
        for index in range(rounds):
            shift = index % len(order)
            with hold() as env:
                for name in order[shift:] + order[:shift]:
                    self._one(arms, name, env, keep=True)

    def _one(self, arms: dict[str, _Timed], name: str, env: Mapping[str, str], *, keep: bool) -> None:
        timed = arms[name]
        if timed.detail or timed.build.bin is None:
            return
        executable = timed.build.bin / LLAMACPP_BACKEND_OPS["executable"]
        try:
            sample = self.time(executable, self.cases, self.names, env)
        except BackendOpsError as failed:
            if name in (_DEFAULT_ARM, _REPLICA_ARM):
                raise SweepError(f"the build of the pinned commit could not be timed: {failed}") from None
            logger.warning("the sweep drops %s, whose timing failed: %s", name, failed)
            timed.detail = str(failed)
            return
        if keep:
            timed.samples.append(sample)


def _built(arms: list[_Arm], build: Callable[[bytes], TreeBuild], parallel: int) -> dict[str, _Timed]:
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        builds = list(pool.map(build, [arm.patch for arm in arms]))
    if builds[0].bin is None:
        raise SweepError(f"the build of the pinned commit failed ({builds[0].status.value}): {builds[0].detail}")
    timed = {arm.name: _Timed(found) for arm, found in zip(arms, builds, strict=True)}
    timed[_REPLICA_ARM] = _Timed(builds[0])
    return timed


def _rows(
    plan: SweepPlan, arms: list[_Arm], timed: dict[str, _Timed], cases: dict[KnobKey, list[MulMatCase]]
) -> list[_Row]:
    rows = []
    root = math.sqrt(plan.rounds)
    for key in plan.keys:
        base = _totals(timed[_DEFAULT_ARM], cases[key])
        replica = _totals(timed[_REPLICA_ARM], cases[key])
        middle = statistics.median(base)
        gap = abs(statistics.median(replica) / middle - 1)
        for arm in arms[1:]:
            if timed[arm.name].detail or not timed[arm.name].samples:
                continue
            totals = _totals(timed[arm.name], cases[key])
            median = statistics.median(totals)
            gain = middle / median - 1
            noise = max(gap, _Z * _MEDIAN_SE * math.hypot(_sigma(base), _sigma(totals)) / root)
            verdict = Verdict.FASTER if gain > noise else Verdict.SLOWER if gain < -noise else Verdict.WITHIN_NOISE
            rows.append(_Row(key, str(arm.value), median, gain, noise, verdict))
    return rows


def _winners(rows: list[_Row]) -> dict[KnobKey, _Row]:
    best: dict[KnobKey, _Row] = {}
    for row in rows:
        if row.verdict is Verdict.FASTER and (row.key not in best or row.gain > best[row.key].gain):
            best[row.key] = row
    return best


def _checked(
    winners: dict[KnobKey, _Row],
    timed: dict[str, _Timed],
    cases: dict[KnobKey, list[MulMatCase]],
    cases_file: Path,
    hooks: SweepHooks,
) -> dict[KnobKey, str]:
    """Each winner's cases checked against the CPU backend on its own build: the reason of each key that failed."""
    failed: dict[KnobKey, str] = {}
    executable = LLAMACPP_BACKEND_OPS["executable"]
    for value in sorted({row.value for row in winners.values()}):
        keys = [key for key, row in winners.items() if row.value == value]
        names = [case.name for key in keys for case in cases[key]]
        built = timed[f"value={value}"].build.bin
        with hooks.hold() as env:
            try:
                verdicts = hooks.check(built / executable, cases_file, names, env)
            except BackendOpsError as unchecked:
                logger.warning("the check of value=%s against the CPU did not run: %s", value, unchecked)
                verdicts = dict.fromkeys(names, f"not checked: {unchecked}")
        for key in keys:
            bad = [f"{case.name}: {verdicts[case.name]}" for case in cases[key] if verdicts[case.name] != CHECK_OK]
            if bad:
                failed[key] = "; ".join(bad)
    return failed


def run_sweep(
    plan: SweepPlan, read_source: Callable[[str], str], workdir: Path, hooks: SweepHooks, *, parallel: int = 2
) -> SweepResult:
    """Sweep ``plan`` over the pinned tree whose files ``read_source`` reads, through ``hooks`` (see the module).

    Raises:
        SweepError: The pinned commit did not build or could not be timed, or no value was timed (the message names
            each value's failure).
        KnobSpecError: The knob's anchor is not in the pinned source exactly once.
        BackendOpsError: The pinned tree's enum header does not name a type or the operation of a case.
    """
    source = read_source(plan.knob.file)
    arms = _arms(plan, source)
    cases = _cases(plan)
    workdir.mkdir(parents=True, exist_ok=True)
    cases_file = workdir / _CASES_FILE
    cases_file.write_text(
        case_file(
            [case for listed in cases.values() for case in listed], read_source(LLAMACPP_BACKEND_OPS["enum_header"])
        ),
        encoding="utf-8",
    )
    timed = _built(arms, hooks.build, parallel)
    names = [case.name for listed in cases.values() for case in listed]
    _Bench(cases_file, names, hooks.time).rounds(timed, plan.rounds, hooks.hold)
    rows = _rows(plan, arms, timed, cases)
    if not rows:
        untimed = [_arm_document(arm, timed[arm.name]) for arm in arms[1:]]
        errors = "; ".join(f"{arm['arm']} {arm['status']}: {arm['detail']}" for arm in untimed)
        raise SweepError(f"no value was timed: {errors}")
    winners = _winners(rows)
    refused = _checked(winners, timed, cases, cases_file, hooks)
    rows = [
        _Row(row.key, row.value, row.median_us, row.gain, row.noise, Verdict.CHECK_FAILED)
        if row.key in refused and winners[row.key] is row
        else row
        for row in rows
    ]
    chosen = {key: row.value for key, row in winners.items() if key not in refused}
    patch = knob_patch(plan.knob, source, chosen)
    document = {**_measured(plan, arms, timed, cases, rows), **_recommended(plan, chosen, refused, patch)}
    return SweepResult(document, patch)


def _arm_document(arm: _Arm, timed: _Timed) -> dict[str, Any]:
    build = timed.build
    status = build.status.value if build.bin is None else _BENCH_FAILED if timed.detail else _TIMED
    return {
        "arm": arm.name,
        "value": arm.value,
        "status": status,
        "build_key": build.key,
        "rounds": len(timed.samples),
        "detail": timed.detail or build.detail,
    }


def _pct(share: float) -> float:
    return round(100 * share, 2)


def _measured(
    plan: SweepPlan,
    arms: list[_Arm],
    timed: dict[str, _Timed],
    cases: dict[KnobKey, list[MulMatCase]],
    rows: list[_Row],
) -> dict[str, Any]:
    knob = plan.knob
    defaults = {key: statistics.median(_totals(timed[_DEFAULT_ARM], cases[key])) for key in plan.keys}
    gaps = [abs(statistics.median(_totals(timed[_REPLICA_ARM], cases[key])) / defaults[key] - 1) for key in plan.keys]
    sigmas = [_sigma(_totals(timed[_DEFAULT_ARM], cases[key])) for key in plan.keys]
    ranked = sorted(rows, key=lambda row: (plan.keys.index(row.key), -row.gain))
    return {
        "knob": plan.name,
        "file": knob.file,
        "values": list(plan.values),
        "shapes": [{"m": m, "k": k} for m, k in plan.shapes],
        "rounds": plan.rounds,
        "copies": plan.copies,
        "arms": [_arm_document(arm, timed[arm.name]) for arm in arms],
        "noise": {
            "aa_gap_median_pct": _pct(statistics.median(gaps)),
            "aa_gap_max_pct": _pct(max(gaps)),
            "round_sigma_median_pct": _pct(statistics.median(sigmas)),
        },
        "table": [
            {
                "key": knob.label(row.key),
                "value": row.value,
                "default_us": round(defaults[row.key], 2),
                "median_us": round(row.median_us, 2),
                "gain_pct": _pct(row.gain),
                "noise_pct": _pct(row.noise),
                "verdict": row.verdict.value,
            }
            for row in ranked
        ],
    }


def _recommended(
    plan: SweepPlan, chosen: Mapping[KnobKey, str], refused: Mapping[KnobKey, str], patch: bytes
) -> dict[str, Any]:
    label = plan.knob.label
    return {
        "winners": {label(key): value for key, value in chosen.items()},
        "refused": {label(key): reason for key, reason in refused.items()},
        "no_winner": [label(key) for key in plan.keys if key not in chosen],
        "patch": patch.decode(),
        "advice": _ADVICE,
    }
