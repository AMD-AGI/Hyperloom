# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One multi-branch round as the multi-branch design describes it (6.1 one test, 7.3 MB1 acceptance), scripted.

The specialists' decisions are scripted (no model): which change to try, when to ask for a retest. Everything they
call is real: the MCP service and its tokens, the toolbox, the notebook, the experiment repository, the rule gate.
:func:`run_round` returns what the variant's test asserts on; the checks every variant must pass are made here.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from hyperloom.common.accuracy import ACCURACY_METRICS
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.contracts.measurement import AccuracyValue, Purpose, TestKind, TestRequest, TestResult, Validity
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta, apply
from hyperloom_multibranch.contracts.intent import IntentLayer, intent_fingerprint
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, Conclusion
from hyperloom_multibranch.rulegate.accuracy import baseline_accuracy
from hyperloom_multibranch.rulegate.decision import Action
from hyperloom_multibranch.rulegate.inferencex import (
    DispersionThreshold,
    InferenceXBaseline,
    InferenceXReading,
    compare,
    confirm,
)
from hyperloom_multibranch.toolbox.tests.integration_rig import Specialist, Stack

_TOOL_WAIT_SEC = 3300.0
_POLL_SEC = 0.25
_SOURCE, _CONFIG = IntentLayer.SOURCE.value, IntentLayer.CONFIG.value
_WITHDRAWN = "withdrawn"


def config_delta(max_running_requests: int) -> RecipeDelta:
    return RecipeDelta(config=ConfigDelta(server_args=f"--max-running-requests {max_running_requests}"))


@dataclass(frozen=True)
class Probe:
    """One thing a specialist tries: what it attacks, how, in which layer, and the change that does it."""

    target: str
    mechanism: str
    layer: str
    delta: RecipeDelta

    @property
    def fingerprint(self) -> str:
        return intent_fingerprint(self.target, self.mechanism, IntentLayer(self.layer))


@dataclass
class Outcome:
    """What a round did, for the variant's own assertions."""

    baseline: tuple[float, ...] = ()
    first: dict[str, str] = field(default_factory=dict)
    retest: dict[str, str] = field(default_factory=dict)
    explore2: str = ""
    cancelled: tuple[str, ...] = ()
    judgements: dict[str, Judgement] = field(default_factory=dict)
    actions: dict[str, Action] = field(default_factory=dict)
    max_running_s1: int = 0
    request_ids: list[str] = field(default_factory=list)


def throughput(result: TestResult) -> float:
    return next(r.metrics[GRADED_OUTPUT] for r in result.rounds if r.phase == "measure")


def accuracy_score(result: TestResult) -> float:
    values: tuple[AccuracyValue, ...] = result.accuracy
    return next(v.score for metric in ACCURACY_METRICS for v in values if v.metric == metric)


def reading(result: TestResult) -> InferenceXReading:
    return InferenceXReading(throughput(result), accuracy_score(result))


async def wait_until(check: Callable[[], Awaitable[Any]], what: str, timeout: float = _TOOL_WAIT_SEC) -> Any:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        found = await check()
        if found:
            return found
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(_POLL_SEC)


async def submit(sp: Specialist, parent: str, probe: Probe, **fields: Any) -> str:
    """Steps 3 and 4 of 6.1: claim the intent in the notebook, then submit the test with the specialist's own token."""
    await sp.call(
        "notebook_claim",
        parent_commit=parent,
        target=probe.target,
        mechanism=probe.mechanism,
        layer=probe.layer,
        change=probe.delta.to_document(),
    )
    arguments = {
        "kind": "test",
        "purpose": "explore",
        "parent_commit": parent,
        "delta": probe.delta.to_document(),
        "intent_fingerprint": probe.fingerprint,
        "accuracy": True,
        **fields,
    }
    return str(json.loads(await sp.call("submit_test", **arguments))["request_id"])


async def result_state(sp: Specialist, request_id: str) -> dict[str, Any]:
    return dict(json.loads(await sp.call("test_result", request_id=request_id)))


async def measured(sp: Specialist, request_id: str) -> TestResult:
    """Wait until the toolbox wrote the result, and return what the specialist reads back through its tool."""

    async def done() -> dict[str, Any] | None:
        state = await result_state(sp, request_id)
        assert state["state"] != _WITHDRAWN, state
        return state if state["state"] == "measured" else None

    state = await wait_until(done, f"the result of {request_id}")
    result = TestResult.from_document(state["result"])
    assert result.validity is Validity.SUCCEEDED, result
    return result  # type: ignore[no-any-return]


async def baselines(stack: Stack, count: int = 2) -> tuple[list[TestResult], list[str]]:
    """PRELUDE: the scheduler measures the baseline, with the accuracy round, before any specialist starts."""
    ids = [
        await stack.toolbox.submit_test(
            TestRequest(
                kind=TestKind.TEST,
                purpose=Purpose.BASELINE,
                branch_id=INTEGRATION_BRANCH,
                parent_commit=stack.baseline_commit,
                delta=RecipeDelta(),
                accuracy=True,
            )
        )
        for _ in range(count)
    ]
    await asyncio.wait_for(stack.toolbox.drain(), _TOOL_WAIT_SEC)
    records = [await stack.gate_notebook.record(request_id) for request_id in ids]
    assert all(r.result is not None and r.result.validity is Validity.SUCCEEDED for r in records), records
    return [r.result for r in records if r.result is not None], ids


async def parent_of(stack: Stack, commit: str) -> str:
    done = await asyncio.to_thread(
        subprocess.run,
        ["git", "--git-dir", str(stack.root / "experiments.git"), "rev-parse", f"{commit}^"],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


async def land(stack: Stack, record: TestRecord, judgement: Judgement, outcome: Outcome, branch: str) -> Action:
    """Settle through the real rule gate and check that the repository and the notebook say what the table says."""
    before = await stack.repo.head(branch)
    action = await stack.gate.settle(record.request, judgement)
    after = await stack.repo.head(branch)
    outcome.actions[record.request_id], outcome.judgements[record.request_id] = action, judgement
    settled = await stack.gate_notebook.record(record.request_id)
    if action is Action.COMMIT:
        assert await parent_of(stack, after) == before == record.request.parent_commit, "a commit on the wrong parent"
        recipe = apply(await stack.repo.read_recipe(before), record.request.delta)
        assert await stack.repo.read_recipe(after) == recipe
        assert (settled.conclusion, settled.commit_id) == (Conclusion.COMMITTED, after)
    else:
        assert after == before, f"{action.value} must leave {branch} where it was"
        assert settled.commit_id is None
    return action


def decided(baseline: InferenceXBaseline, parent: dict[str, float], record: TestRecord) -> Judgement:
    """First measurement against the baseline record, with its own accuracy round (6.1 step 7)."""
    assert record.result is not None
    return compare(reading(record.result), baseline, parent, record.request_id)


@dataclass(frozen=True)
class RoundProbes:
    """The changes the scripted specialists try: s1 a source patch and a second config change, s2 a config change and
    one more queued behind it, s3 two changes it never gets to run."""

    patch: Probe
    config: Probe
    explore2: Probe
    queued: Probe
    never: tuple[Probe, Probe]


async def retest(stack: Stack, sp: Specialist, parent: str, probe: Probe, first_id: str) -> str:
    """A confirm-retest of ``first_id``: same change, with the accuracy round; it holds no claim of its own.

    The specialist runtime submits it through the toolbox interface: the MCP service refuses a specialist's confirm.
    """
    arguments = {
        "kind": "test",
        "purpose": "confirm",
        "branch_id": sp.branch,
        "parent_commit": parent,
        "delta": probe.delta.to_document(),
        "retest_of": first_id,
        "accuracy": True,
    }
    return await stack.toolbox.submit_test(TestRequest.from_document(arguments))


async def cancel_what_waits(stack: Stack, s2: Specialist, s3: Specialist, queued: str, never: list[str]) -> None:
    """Withdraw a queued test with its own token, and every test of a stopped branch with the scheduler's call."""
    queue = json.loads(await s2.call("queue_state"))
    assert queue["queued_by_branch"] == {"s2": 1, "s3": 2} and queue["running_by_branch"] == {"s1": 1, "s2": 1}, queue
    assert json.loads(await s2.call("cancel_test", request_id=queued)) == {"outcome": _WITHDRAWN}
    assert await stack.toolbox.cancel_branch("s3") == 2
    for sp, request_id in [(s2, queued), *((s3, r) for r in never)]:
        state = await result_state(sp, request_id)
        assert state["state"] == _WITHDRAWN and state["result"] is None, state
        assert stack.toolbox.backend.journal.latest(request_id) is None, "a withdrawn test must never reach a sandbox"


async def run_round(stack: Stack, probes: RoundProbes) -> Outcome:
    out = Outcome()
    base = stack.baseline_commit
    for branch in ("s1", "s2", "s3"):
        await stack.repo.create_branch(branch, base)
    s1, s2, s3 = (Specialist(stack, branch) for branch in ("s1", "s2", "s3"))

    base_results, base_ids = await baselines(stack)
    values = [throughput(r) for r in base_results]
    bar_pct = DispersionThreshold(tuple(values)).threshold_pct
    baseline = InferenceXBaseline(bar_pct, baseline_accuracy([accuracy_score(r) for r in base_results]))
    parent = {GRADED_OUTPUT: fmean(values)}
    out.baseline = tuple(values)

    # Two branches measure at once on a full pool; the rest of what they and s3 submit has to wait.
    t1, t2 = await submit(s1, base, probes.patch), await submit(s2, base, probes.config)
    queued = await submit(s2, base, probes.queued)
    never = [await submit(s3, base, probe) for probe in probes.never]
    await cancel_what_waits(stack, s2, s3, queued, never)
    out.first, out.cancelled = {"s1": t1, "s2": t2}, (queued, *never)
    first = {"s1": await measured(s1, t1), "s2": await measured(s2, t2)}
    records = {b: await stack.gate_notebook.record(rid) for b, rid in out.first.items()}
    firsts = {b: decided(baseline, parent, records[b]) for b in records}

    # s1 tries a second change and asks for the retest of its first at once: with K = 1 the retest must not wait for it.
    out.explore2 = await submit(s1, base, probes.explore2)
    asks = {"s1": (s1, probes.patch)} | (
        {"s2": (s2, probes.config)} if firsts["s2"].verdict is Verdict.NEEDS_CONFIRM else {}
    )
    out.retest = {b: await retest(stack, sp, base, probe, out.first[b]) for b, (sp, probe) in asks.items()}

    async def k_exempt_seen() -> bool:
        state = json.loads(await s1.call("queue_state"))
        out.max_running_s1 = max(out.max_running_s1, state["running_by_branch"].get("s1", 0))
        return out.max_running_s1 >= 2 or (await result_state(s1, out.retest["s1"]))["state"] == "measured"

    await wait_until(k_exempt_seen, "the retest running beside the explore test of the same branch")
    retests = {b: await measured(sp, out.retest[b]) for b, (sp, _) in asks.items()}
    await settle_round(stack, out, (records, firsts, first, retests), (baseline, parent))
    out.request_ids = [*base_ids, t1, t2, out.explore2, *out.retest.values()]
    await check_audit(stack, out, [*base_results, *first.values(), *retests.values()])
    await check_specialist_view(stack, s1, s2, out, probes)
    return out


async def settle_round(
    stack: Stack,
    out: Outcome,
    measurements: tuple[Any, ...],
    judged_against: tuple[InferenceXBaseline, dict[str, float]],
) -> None:
    """Judge every candidate with the rule gate's comparison and land it through the rule gate (6.1 step 7)."""
    records, firsts, _, retests = measurements
    baseline, parent = judged_against
    for branch, first in firsts.items():
        out.judgements[f"{branch}-first"] = first
        if first.verdict is not Verdict.NEEDS_CONFIRM:
            await land(stack, records[branch], first, out, branch)
            continue
        assert await land(stack, records[branch], first, out, branch) is Action.RETEST
        assert (await stack.gate_notebook.record(records[branch].request_id)).judgement is None
        final = confirm(first, reading(retests[branch]), baseline, parent, out.retest[branch])
        await land(stack, records[branch], final, out, branch)
    explore = Specialist(stack, "s1")
    await measured(explore, out.explore2)
    record = await stack.gate_notebook.record(out.explore2)
    await land(stack, record, decided(baseline, parent, record), out, "s1")


async def check_audit(stack: Stack, out: Outcome, results: list[TestResult]) -> None:
    """Every test ran in a sandbox of its own, and its bundle verified against that sandbox's key."""
    results = [*results, *[r for rid in [out.explore2] if (r := (await stack.gate_notebook.record(rid)).result)]]
    sandboxes = [r.sandbox.sandbox_id for r in results if r.sandbox is not None]
    assert len(sandboxes) == len(results) == len(set(sandboxes)), "two tests shared a sandbox"
    for result in results:
        assert result.manifest_sha256 and result.signing_key_fingerprint, result
        assert stack.toolbox.backend.journal.latest(result.request_id).extra["verified"] is True  # type: ignore[union-attr]
    for request_id in out.request_ids:
        record = await stack.gate_notebook.record(request_id)
        assert record.result is not None and record.withdrawn_at is None, request_id
    for request_id in out.cancelled:
        assert (await stack.gate_notebook.record(request_id)).withdrawn_at is not None


async def check_specialist_view(
    stack: Stack, s1: Specialist, s2: Specialist, out: Outcome, probes: RoundProbes
) -> None:
    """What the tokens and the notebook let a specialist see: its own tests, the others' intents, its inbox."""
    other = await s2.raw("test_result", request_id=out.first["s1"])
    assert other["error"]["code"] == -32001, "a specialist read another branch's test"
    tried = await s2.refused(
        "notebook_claim",
        parent_commit=stack.baseline_commit,
        target=probes.patch.target,
        mechanism=probes.patch.mechanism,
        layer=probes.patch.layer,
        change=probes.patch.delta.to_document(),
    )
    assert tried.startswith("refused") and out.first["s1"] in tried, tried
    inbox = json.loads(await s1.call("poll_inbox"))
    assert out.first["s1"] in {item["pointer"] for item in inbox["items"]}, inbox
