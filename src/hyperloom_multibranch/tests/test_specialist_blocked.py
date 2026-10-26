# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hyperloom.contracts.recipe import LaunchConfig, Recipe
from hyperloom.hosted.mcp_server import ToolError, ToolForbidden
from hyperloom_multibranch.contracts.fork import AdmissionStatus
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import ClaimKey, Learning
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, Conclusion, LearningKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import delta, intent, request, result
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.specialist.blocked import BlockedCandidates, register_blocked_tools
from hyperloom_multibranch.tests.fakes import InMemoryRepo, RecordingScheduler, RecordingVerifier

BRANCH, ROOT = "s1", "c0"
KV = replace(intent(BRANCH, ROOT, "kv cache", "fp8 kv cache"), target_share_pct=30.0)
DIAGNOSIS = "per-tensor KV scales lose the outlier heads; scale per head"
LESSON = "fp8 KV cache on qwen3-8b drops the eval score by 0.08 whatever the scales: the model is sensitive to it"


def _blocked(*request_ids: str, accuracy: AccuracyVerdict = AccuracyVerdict.FAILED) -> Judgement:
    return Judgement(Verdict.BLOCKED, 3.0, 2.0, request_ids, {"output_throughput": 1030.0}, accuracy)


class _Scheduler(RecordingScheduler):
    """Also keeps the loop each repair request was admitted on."""

    def __init__(self, status: AdmissionStatus) -> None:
        super().__init__(status)
        self.loops: list[asyncio.AbstractEventLoop] = []

    async def admit_repair(self, request):
        self.loops.append(asyncio.get_running_loop())
        return await super().admit_repair(request)


class _Server:
    def __init__(self) -> None:
        self.handlers = {}

    def register(self, name, *, description, schema, handler, scopes):
        self.handlers[name] = handler

    async def call(self, name: str, branch: str = BRANCH, **arguments):
        return await self.handlers[name](arguments, SimpleNamespace(principal=SimpleNamespace(branch=branch)))


class _Rig:
    def __init__(self, root, status: AdmissionStatus, home: asyncio.AbstractEventLoop | None = None) -> None:
        self.root = root
        self.toolbox = LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox"))
        self.runtime = LocalNotebook(root, Writer(Role.SPECIALIST_RUNTIME, BRANCH, BRANCH))
        self.gate_notebook = LocalNotebook(root, Writer(Role.GATE, "gate"))
        self.reader = LocalNotebook(root, Writer(Role.OBSERVER, "observer"))
        self.repo = InMemoryRepo(Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 1")))
        self.scheduler = _Scheduler(status)
        self.gate = RuleGate(self.gate_notebook, self.repo, self.scheduler, RecordingVerifier())
        self.server = _Server()
        register_blocked_tools(self.server, blocked=BlockedCandidates(self.reader, self.gate), home=home)

    async def setup(self) -> None:
        await self.repo.create_branch(BRANCH, await self.repo.head(INTEGRATION_BRANCH))
        await self.runtime.acquire(KV, "fp8-kv", ttl_seconds=60.0)
        await self.measure("r1", claimed=True)

    async def measure(self, request_id: str, *, claimed: bool = False, of=KV, change=None) -> None:
        candidate = request(request_id, of, change or delta())
        await self.toolbox.write_request(candidate)
        if claimed:
            await self.toolbox.attach_request(ClaimKey(ROOT, of.fingerprint), request_id, ttl_seconds=60.0)
        await self.toolbox.write_result(result(request_id))

    async def block(self, request_id: str, **judged) -> None:
        candidate = (await self.reader.record(request_id)).request
        await self.gate.settle(candidate, _blocked(request_id, f"{request_id}b", **judged))

    async def learn(self, *request_ids: str, branch: str = BRANCH) -> None:
        learner = LocalNotebook(self.root, Writer(Role.SPECIALIST_RUNTIME, branch, branch))
        await learner.append_learning(Learning(branch, LearningKind.PITFALL, LESSON, request_ids=request_ids))


@pytest.fixture
async def rig(tmp_path):
    rig = _Rig(tmp_path / "notebook", AdmissionStatus.ACCEPTED)
    await rig.setup()
    await rig.block("r1")
    return rig


@pytest.mark.parametrize(
    ("status", "branch", "conclusion"),
    [
        (AdmissionStatus.ACCEPTED, "fork-1", Conclusion.REPAIR_REQUESTED),
        (AdmissionStatus.PENDING, "fork-1", Conclusion.REPAIR_REQUESTED),
        (AdmissionStatus.REJECTED, "", Conclusion.ACCURACY_FAILED),
    ],
)
async def test_a_repair_goes_to_the_scheduler_with_the_claimed_intent_and_the_diagnosis(
    tmp_path, status, branch, conclusion
):
    rig = _Rig(tmp_path / "notebook", status)
    await rig.setup()
    await rig.block("r1")

    answer = await rig.server.call("request_repair", request_id="r1", diagnosis=DIAGNOSIS)

    assert answer == {
        "request_id": "r1",
        "status": status.value,
        "reason": "",
        "branch": branch,
        "conclusion": conclusion.value,
    }
    (asked,) = rig.scheduler.repairs
    assert (asked.intent, asked.candidate_delta, asked.diagnosis) == (KV, delta(), DIAGNOSIS)
    assert asked.judgement == _blocked("r1", "r1b")
    assert (await rig.reader.record("r1")).conclusion is conclusion


async def test_an_intent_is_sent_for_repair_once_per_commit(rig):
    await rig.measure("r2", change=delta("--max-num-seqs 2"))
    await rig.block("r2")
    await rig.server.call("request_repair", request_id="r1", diagnosis=DIAGNOSIS)

    with pytest.raises(ToolError, match=f"r1 already asked to repair this intent on {ROOT}; decline r2"):
        await rig.server.call("request_repair", request_id="r2", diagnosis=DIAGNOSIS)

    assert len(rig.scheduler.repairs) == 1
    assert (await rig.reader.record("r2")).conclusion is Conclusion.DIAGNOSIS_PENDING


async def test_a_concluded_candidate_is_not_concluded_again(rig):
    await rig.learn("r1")
    await rig.server.call("request_repair", request_id="r1", diagnosis=DIAGNOSIS)

    with pytest.raises(ToolError, match="r1 is already concluded: repair_requested"):
        await rig.server.call("decline_repair", request_id="r1")
    with pytest.raises(ToolError, match="already concluded"):
        await rig.server.call("request_repair", request_id="r1", diagnosis=DIAGNOSIS)


async def test_a_decline_waits_for_a_learning_of_its_branch_that_cites_the_candidate(rig):
    await rig.learn()
    await rig.learn("r1", branch="s2")
    with pytest.raises(ToolError, match="notebook_learn"):
        await rig.server.call("decline_repair", request_id="r1")

    await rig.learn("r1")
    answer = await rig.server.call("decline_repair", request_id="r1")

    assert answer == {"request_id": "r1", "conclusion": Conclusion.ACCURACY_FAILED.value}
    assert rig.scheduler.repairs == []


async def test_a_blocked_gain_a_guard_held_back_is_declined_as_recorded_only(rig):
    await rig.measure("r2", change=delta("--max-num-seqs 2"))
    await rig.block("r2", accuracy=AccuracyVerdict.PASSED)
    await rig.learn("r2")

    answer = await rig.server.call("decline_repair", request_id="r2")

    assert answer["conclusion"] == Conclusion.RECORDED_ONLY.value


async def test_another_branch_s_candidate_is_not_the_caller_s_to_conclude(rig):
    with pytest.raises(ToolForbidden, match="r1 is a test of s1"):
        await rig.server.call("request_repair", branch="s2", request_id="r1", diagnosis=DIAGNOSIS)
    with pytest.raises(ToolForbidden):
        await rig.server.call("decline_repair", branch="s2", request_id="r1")


async def test_only_a_candidate_waiting_for_its_diagnosis_is_concluded(rig):
    await rig.measure("r2", change=delta("--max-num-seqs 2"))
    await rig.measure("r3", change=delta("--max-num-seqs 3"))
    flat = Judgement(Verdict.FLAT, 0.1, 2.0, ("r3",), {"output_throughput": 1001.0}, AccuracyVerdict.PASSED)
    await rig.gate.settle((await rig.reader.record("r3")).request, flat)

    for request_id, refusal in [("r9", "no test r9"), ("r2", "r2 is not judged"), ("r3", "r3 is flat, not blocked")]:
        with pytest.raises(ToolError, match=refusal):
            await rig.server.call("request_repair", request_id=request_id, diagnosis=DIAGNOSIS)
    with pytest.raises(ToolError, match="names the root cause"):
        await rig.server.call("request_repair", request_id="r1", diagnosis="  ")
    assert rig.scheduler.repairs == []


async def test_a_candidate_without_an_intent_is_declined_not_repaired(rig):
    await rig.toolbox.write_request(replace(request("r2", KV, delta("--max-num-seqs 2")), intent_fingerprint=None))
    await rig.toolbox.write_result(result("r2"))
    await rig.block("r2")

    with pytest.raises(ToolError, match="r2 was submitted without an intent"):
        await rig.server.call("request_repair", request_id="r2", diagnosis=DIAGNOSIS)
    await rig.learn("r2")
    assert (await rig.server.call("decline_repair", request_id="r2"))["conclusion"] == "accuracy_failed"


async def test_a_call_answered_on_another_loop_is_admitted_on_the_runtime_s_loop(tmp_path):
    home = asyncio.get_running_loop()
    rig = _Rig(tmp_path / "notebook", AdmissionStatus.ACCEPTED, home=home)
    await rig.setup()
    await rig.block("r1")

    def answer_on_a_loop_of_its_own():
        return asyncio.run(rig.server.call("request_repair", request_id="r1", diagnosis=DIAGNOSIS))

    answer = await asyncio.to_thread(answer_on_a_loop_of_its_own)

    assert answer["status"] == "accepted"
    assert rig.scheduler.loops == [home]
