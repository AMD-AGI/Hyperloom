# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fakes for the tests of an autonomous agent's specialist: the onboarding's sandbox and process, a local sandbox and
toolbox, the component agent and its plan, an image's aiter tables, and a fake Arbor checkout with a branch's spec.
It imports nothing of the autonomous session, so a test that only onboards an agent does not need it."""

import asyncio
import hashlib
import json
import os
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

from hyperloom.contracts.measurement import AttachmentRef, TestRequest
from hyperloom.contracts.recipe import ComponentPatches, KernelArtifact, KernelKind, LaunchConfig, Recipe
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.surface import SurfaceViolation, check_launch_envs
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.contracts.toolbox import BranchLease, LeaseCharge, RequestRejected
from hyperloom_multibranch.specialist.agent_onboarding import CommandResult, OnboardingSetup
from hyperloom_multibranch.specialist.agent_spec import AgentSpec
from hyperloom_multibranch.specialist.arbor_adapter import BRANCH_RUN
from hyperloom_multibranch.specialist.gate_client import FixedGateClient
from hyperloom_multibranch.toolbox.kernel_artifacts import KernelVariables
from inference_suite.contracts import FixedWorkload


class Process:
    def __init__(self, returncode: int | None, output: str = "") -> None:
        self.returncode, self._output, self.killed = returncode, output, False

    async def output(self) -> str:
        return self._output

    async def kill(self) -> None:
        self.killed = True


class Sandbox:
    """Runs nothing: each install exits with the next of ``installs`` (0 once they run out), each start returns the
    next of ``starts`` and, when ``keep`` is set, writes the agent's keep file the way a started agent would."""

    def __init__(self, starts: list[Process], installs: tuple[int, ...] = (), keep: str = "") -> None:
        self.starts, self.installs, self.keep = starts, list(installs), keep
        self.ran: list[tuple[str, Path, Mapping[str, str]]] = []

    async def run(self, command: str, cwd: Path, env: Mapping[str, str], timeout: float) -> CommandResult:
        self.ran.append((command, cwd, env))
        code = self.installs.pop(0) if self.installs else 0
        return CommandResult(code, "ERROR: no matching distribution\n" if code else "")

    async def start(self, command: str, cwd: Path, env: Mapping[str, str]) -> Process:
        self.ran.append((command, cwd, env))
        if self.keep:
            (cwd / self.keep).parent.mkdir(parents=True, exist_ok=True)
            (cwd / self.keep).write_text('{"keeps": 0}', encoding="utf-8")
        return self.starts.pop(0)


class Onboarder:
    def __init__(self, answers: list[str]) -> None:
        self.answers, self.prompts = answers, []

    async def __call__(self, checkout: Path, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)


def _setup(sandbox: Sandbox, onboard, waits: list[float] | None = None) -> OnboardingSetup:
    """Onboarding that skips the waits between attempts, recording them in ``waits``, and gives a started agent a
    tenth of a second to stay up."""

    async def no_wait(seconds: float) -> None:
        if waits is not None:
            waits.append(seconds)
        await asyncio.sleep(0)

    return OnboardingSetup(sandbox, onboard, start_seconds=0.1, poll_seconds=0.01, sleep=no_wait)


BRANCH = "a1"


ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)


#: An agent in Arbor's component mode, session in run/: every ``gap`` s it commits each of ``keeps`` (a file of its own
#: at the first) and snapshots it to run/snaps, the outbox MB's config names: ``change.patch`` against its parent,
#: ``served.patch`` against the root, the launch, ``table`` as AITER_CONFIG_GEMM_A8W8's table after an image default
#: (too large to copy with ``huge``, wrong digest with ``bad_sha``) and an AITER JIT link outside its tree; ``parts``
#: gives the variable's value with ``table`` appended or replacing its last part, in Arbor's multi-part shape. LATEST
#: names each keep, or the last only without ``latest_each``. ``skip`` gets no snapshot; ``stray`` is made on a commit
#: MB never gave it; ``orphan`` on the root (``orphan_notes`` say so). It removes the last exit file, ends with ``exit``
#: after ``tail`` s or 0 on SIGTERM, writing ``reason`` to run/component_exit.json before an exit 0 only.
COMPONENT_AGENT = """
import hashlib, json, os, pathlib, signal, subprocess, sys, time
LISTED = "/sgl-workspace/aiter/aiter/configs/a8w8_tuned_gemm.csv:/arbor/workspace/tuned.csv"
config = json.loads(pathlib.Path("campaign.json").read_text())
plan, workspace, root = config["plan"], pathlib.Path(config["workspace"]), config["component"]["root"]
session = pathlib.Path("run")
outbox = session / config["component"].get("outbox", "keeps")
outbox.mkdir(parents=True, exist_ok=True)
(session / "component_exit.json").unlink(missing_ok=True)


def leave(reason, code):
    if code == 0 and reason:
        (session / "component_exit.json").write_text(json.dumps({"reason": reason, "keeps": 0, "ts": time.time()}))
    if code == 2:
        print("component mode: docker_container must be @host", file=sys.stderr, flush=True)
    sys.exit(code)


def git(*args):
    argv = ["git", "-c", "user.name=a", "-c", "user.email=a@a", *args]
    return subprocess.run(argv, cwd=workspace, check=True, capture_output=True, text=True).stdout


signal.signal(signal.SIGTERM, lambda *_: leave("sigterm", 0))
parent, previous, commits = root, None, {}
for number, seqs in enumerate(plan["keeps"], start=1):
    time.sleep(plan["gap"])
    retracting = number == plan["retracting_keep"] and plan["retracts"]
    if retracting:
        git("revert", "--no-edit", commits[plan["retracts"]["keep"]])
    else:
        (workspace / "vllm" / "config.py").write_text(f"max_num_seqs = {seqs}\\n")
    if number == 1:
        (workspace / "vllm" / "first.py").write_text("FIRST = 1\\n")
        if plan["scratch"]:
            (workspace / "eval").mkdir()
            (workspace / "eval" / "server.log").write_bytes(b"Loading 10%\\r 20%\\r 30%\\n")
            (workspace / "eval" / "gpucore.1.gpu").write_bytes(bytes(range(256)) * 64)
    if not retracting:
        git("add", "-A")
        git("commit", "-qm", f"keep {number}")
    commit = commits[number] = git("rev-parse", "HEAD").strip()
    if number == plan["skip"]:
        continue
    made_on = root if number == plan["orphan"] else parent
    built = outbox / f".{number:04d}.tmp"
    built.mkdir()
    (built / "change.patch").write_text(git("diff", "--binary", made_on, commit))
    (built / "served.patch").write_text(git("diff", "--binary", root, commit))
    keep = {"number": number, "previous": previous, "root": root, "parent": made_on, "commit": commit,
            "change": f"seqs {seqs}", "intent": "", "outside": 1, "tables": {}, "tables_outside_workspace": []}
    if number == plan["stray"]:
        keep["parent"] = "f" * 40
    if number == plan["retracting_keep"] and plan["retracts"]:
        keep["retracts"] = plan["retracts"]
    if number == plan["orphan"] and plan["orphan_notes"]:
        keep["notes"] = f"the commit of keep {previous} is gone from the workspace; this change is against the root"
    envs = dict(plan["envs"])
    if plan["parts"]:
        name, own = "AITER_CONFIG_GEMM_A8W8", (session / "own-tuned.csv").resolve()
        own.write_text(plan["table"].format(seqs=seqs))
        given = envs[name].split(os.pathsep)
        parts = {"append": [*given, str(own)], "replace": [*given[:-1], str(own)], "given": given}[plan["parts"]]
        envs[name], files = os.pathsep.join(parts), []
        for k, part in enumerate(parts):
            if os.path.isfile(part):
                data, file = pathlib.Path(part).read_bytes(), f"tables/{name}/{k}-{os.path.basename(part)}"
                (built / file).parent.mkdir(parents=True, exist_ok=True)
                (built / file).write_bytes(data)
                files.append({"path": part, "file": file, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        keep["tables"][name] = {"path": envs[name], "files": files}
    elif plan["table"]:
        name, table = "AITER_CONFIG_GEMM_A8W8", plan["table"].format(seqs=seqs).encode()
        envs[name] = LISTED
        digest = "0" * 64 if plan["bad_sha"] else hashlib.sha256(table).hexdigest()
        file = None if plan["huge"] else "tuned.csv"
        keep["tables"][name] = {"path": envs[name], "file": file, "sha256": digest, "bytes": len(table)}
        if not plan["huge"]:
            (built / "tables" / name).mkdir(parents=True)
            (built / "tables" / name / "tuned.csv").write_bytes(table)
    (built / "keep.json").write_text(json.dumps(keep))
    (built / "config.json").write_text(json.dumps({"server_flags": plan["flags"].format(seqs=seqs), "server_env": envs}))
    outside = [{"link": "vllm/module.so", "target": "/nonexistent/aiter/jit/module.so", "sha256": "0" * 64, "size": 1}]
    (built / "outside.json").write_text(json.dumps(outside))
    built.rename(outbox / f"{number:04d}")
    if plan["latest_each"] or number == len(plan["keeps"]):
        (outbox / ".LATEST.tmp").write_text(f"{number:04d}")
        os.replace(outbox / ".LATEST.tmp", outbox / "LATEST")
    parent, previous = commit, number
    print(f"kept {seqs}", flush=True)
time.sleep(plan["tail"])
leave(plan["reason"], plan["exit"])
"""


@dataclass(frozen=True)
class Component:
    """The plan of :data:`COMPONENT_AGENT`."""

    keeps: tuple[int, ...] = (256, 512)
    gap: float = 0.3
    tail: float = 0.3
    exit: int = 0
    reason: str = "done"
    stray: int = 0
    skip: int = 0
    orphan: int = 0
    orphan_notes: bool = True
    latest_each: bool = True
    envs: dict[str, str] = field(default_factory=dict)
    flags: str = "--max-num-seqs {seqs}"
    #: The launch the config MB wrote starts the agent with, as Arbor's ``server_flags`` and ``server_env``.
    started: dict[str, object] | None = None
    table: str = ""
    huge: bool = False
    bad_sha: bool = False
    #: ``append``, ``replace`` or ``given``: what the agent does to the value of AITER_CONFIG_GEMM_A8W8 it was given.
    parts: str = ""
    #: Whether the first keep also commits the agent's scratch outside vLLM: a log with progress bars and a core dump.
    scratch: bool = False
    #: What the ``keep.json`` of keep ``retracting_keep`` says it retracts; that keep reverts the named keep's commit.
    retracts: dict[str, object] | None = None
    retracting_keep: int = 1


async def _until(condition, what: str, seconds: float = 10.0) -> None:
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


class LocalProcess:
    def __init__(self, process: asyncio.subprocess.Process, log: Path) -> None:
        self._process, self._log = process, log

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def output(self) -> str:
        return self._log.read_text(encoding="utf-8", errors="replace")

    async def kill(self) -> None:
        """SIGTERM the process group, SIGKILL it after 5 s; return once none of it is left, as the sandbox does."""
        if self._process.returncode is not None:
            return
        os.killpg(self._process.pid, signal.SIGTERM)
        for _ in range(100):
            await asyncio.sleep(0.05)
            try:
                os.killpg(self._process.pid, 0)
            except ProcessLookupError:
                break
        else:
            os.killpg(self._process.pid, signal.SIGKILL)
        await self._process.wait()


class LocalSandbox:
    """Runs commands as local processes, each in a process group of its own."""

    def __init__(self, logs: Path) -> None:
        self._logs, self.started, self.envs, self.leases, self.prompts = logs, [], [], [], []
        self.ran: list[str] = []
        #: The leases whose sandbox ended.
        self.gone: set[str] = set()
        self.usage = _Usage()
        self.watch_said: list[str] = []

    def watched(self) -> str:
        return self.watch_said.pop(0) if self.watch_said else ""

    async def run(self, command: str, cwd: Path, env: Mapping[str, str], timeout: float) -> CommandResult:
        self.ran.append(command)
        done = subprocess.run(command, shell=True, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True)
        return CommandResult(done.returncode, done.stdout + done.stderr)

    async def start(self, command: str, cwd: Path, env: Mapping[str, str]) -> LocalProcess:
        self._logs.mkdir(parents=True, exist_ok=True)
        log = self._logs / f"agent-{len(self.started)}.log"
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env={**os.environ, **env},
            stdout=log.open("w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.started.append(command)
        self.envs.append(dict(env))
        return LocalProcess(process, log)


class Toolbox:
    """Accepts each request and writes it to the blackboard, so it is in flight until the test settles it; ``refusals``
    are raised first, one a submit."""

    def __init__(self, blackboard: LocalBlackboard) -> None:
        self._blackboard, self.requests, self.refusals = blackboard, [], []
        self.refused_envs: set[str] = set()
        self.leases: list[tuple[str, int]] = []
        self.lifetimes: list[float | None] = []
        self.charges: list[LeaseCharge | None] = []
        self.released: list[str] = []
        self.cards_left = 1
        self.lease_refusals: list[Exception] = []
        self.lease_tries = 0
        self.tables: dict[str, bytes] = {}
        self.attachments: dict[str, bytes] = {}
        self._variables = KernelVariables()

    async def put_attachment(self, attachment_id: str, kind: Literal["file", "dir"], data: bytes) -> AttachmentRef:
        ref = AttachmentRef(attachment_id, kind, hashlib.sha256(data).hexdigest())
        self.attachments[ref.sha256] = data
        return ref

    async def put_kernel_artifact(
        self, kind: KernelKind, target: str | None, env: str, data: bytes, rows_only: bool = False
    ) -> KernelArtifact:
        artifact = self._variables.put(kind, target, env, data, rows_only)
        self.tables[artifact.sha256] = data
        return artifact

    async def lease_branch_sandbox(
        self,
        branch_id: str,
        cards: int,
        *,
        wait_sec: float | None = None,
        charge: LeaseCharge | None = None,
        lifetime_sec: float | None = None,
    ) -> BranchLease:
        self.lease_tries += 1
        if self.lease_refusals:
            raise self.lease_refusals.pop(0)
        if self.cards_left < cards:
            raise RequestRejected("leasing would leave no card to measure on", "lease_leaves_no_measurement")
        self.leases.append((branch_id, cards))
        self.lifetimes.append(lifetime_sec)
        self.charges.append(charge)
        root = self._blackboard.root.parent / "lease"
        return BranchLease(f"L{len(self.leases)}", branch_id, (3,), root, {"HIP_VISIBLE_DEVICES": "3"}, 1)

    async def release_branch_sandbox(self, lease_id: str) -> None:
        self.released.append(lease_id)

    async def check_test(self, request: TestRequest) -> None:
        try:
            check_launch_envs(request.delta.config.envs)
        except SurfaceViolation as error:
            raise RequestRejected(str(error), "surface_mismatch") from error
        if self.refused_envs & set(request.delta.config.envs):
            names = ", ".join(sorted(self.refused_envs))
            reason = f"the environment does not compile: the renderer dropped or changed launch variables {names}"
            raise RequestRejected(reason, "delta_not_applicable")

    async def submit_test(self, request: TestRequest) -> str:
        if self.refusals:
            raise self.refusals.pop(0)
        await self.check_test(request)
        accepted = TestRequest(**{**request.__dict__, "request_id": f"r{len(self.requests) + 1}"})
        self.requests.append(accepted)
        await self._blackboard.write_request(accepted)
        return accepted.request_id or ""


class _Usage:
    def __init__(self) -> None:
        self.recorded: list = []

    async def record_llm_usage(self, usage) -> None:
        self.recorded.append(usage)


#: A stand-in for the image's aiter install: one table variable, its default table, a per-model table and the untuned
#: table its rows' shape key is read from.
AITER_CORE = """
import os

AITER_ROOT_DIR = "/unused"
AITER_CONFIG_GEMM_A8W8 = os.getenv("AITER_CONFIG_GEMM_A8W8", f"{AITER_ROOT_DIR}/aiter/configs/a8w8_tuned_gemm.csv")


class AITER_CONFIG:
    @property
    def AITER_CONFIG_GEMM_A8W8_FILE(self):
        return self.get_config_file("AITER_CONFIG_GEMM_A8W8", AITER_CONFIG_GEMM_A8W8, "a8w8_tuned_gemm")
"""


TABLE_ENV = "AITER_CONFIG_GEMM_A8W8"


IMAGE_DEFAULT = "M,N,K,us\n1,5120,5120,9\n64,5120,5120,10\n"


IMAGE_MODEL = "M,N,K,us\n8192,5120,5120,30\n"


CHAMPION_ROWS = "M,N,K,us\n64,5120,5120,7\n"


def _aiter(tmp_path: Path) -> Path:
    configs = tmp_path / "image" / "aiter" / "configs"
    (configs / "model_configs").mkdir(parents=True)
    (configs.parent / "jit").mkdir()
    (configs.parent / "jit" / "core.py").write_text(AITER_CORE, encoding="utf-8")
    (configs / "a8w8_untuned_gemm.csv").write_text("M,N,K\n", encoding="utf-8")
    (configs / "a8w8_tuned_gemm.csv").write_text(IMAGE_DEFAULT, encoding="utf-8")
    (configs / "model_configs" / "a8w8_tuned_gemm_qwen.csv").write_text(IMAGE_MODEL, encoding="utf-8")
    return configs


def _replayed(blobs: LocalBlobStore) -> Recipe:
    """The root with a Recipe KB champion's tuned rows, kept in the session's blob store as a replay keeps them."""
    sha256 = blobs.put(CHAMPION_ROWS.encode())
    load = {"load": {"via": "env_file", "env": TABLE_ENV, "rows_only": True}}
    table = KernelArtifact(KernelKind.GEMM, "gemm_a8w8", f"kernel/gemm/gemm_a8w8/{sha256}.csv", sha256, load)
    return replace(ROOT, kernels=(table,))


MODEL = "/models/Qwen3-14B-FP8"


#: The gate's client as MB builds it; it names the model by its hub name, which the server must answer to.
CLIENT = FixedGateClient(
    "/opt/InferenceX/inferencex-e2e",
    "Qwen/Qwen3-14B-FP8",
    FixedWorkload(
        concurrency=8,
        isl=1024,
        osl=1024,
        random_range_ratio=0.8,
        num_prompts=80,
        num_warmups=16,
        seed=0,
        warmup_seed=1,
        tokenizer_mode="auto",
        trust_remote_code=False,
    ),
)


PORT = 31110


DEADLINE = 1793244000.7


CONFIG_FILE, SESSION_DIR = BRANCH_RUN.config_file, BRANCH_RUN.session_dir


#: A start table as MB gives its variable: the private copies of the image's default tables, then the start's rows.
START_TABLE_PARTS = ("/mb/agent/start-tables/AITER_CONFIG_GEMM_A8W8/a8w8_tuned_gemm.csv", "/mb/start-tables/r.csv")


#: Arbor's entry point in a fake checkout. It runs :data:`COMPONENT_AGENT` on the config ``--config`` names, which
#: reads campaign.json and keeps its session in run/ of its working directory: run/ is linked to the config's session
#: and the plan comes from fake-plan.json, its environment on top of the config's ``server_env``, as Arbor's snapshot
#: records the launch it measured. With ``refuse`` in the plan it refuses instead, as Arbor does, after ``tail``
#: seconds: the reasons on stderr and in the session's refusal file, then exit 2.
FAKE_MAIN = """
import json, os, pathlib, runpy, sys, time
config = json.loads(pathlib.Path(sys.argv[sys.argv.index("--config") + 1]).read_text())
plan = json.loads(pathlib.Path("fake-plan.json").read_text())
plan["envs"] = {**config["server_env"], **plan["envs"]}
session = pathlib.Path(config["session_root"])
if plan["refuse"]:
    time.sleep(plan["tail"])
    (session / "component_refused.json").write_text(json.dumps({"reasons": plan["refuse"], "ts": "2026-10-29T00:00:00Z"}))
    for reason in plan["refuse"]:
        print(f"[arbor] refusing to start: {reason}", file=sys.stderr, flush=True)
    sys.exit(2)
if not os.path.lexists("run"):
    os.symlink(session, "run")
pathlib.Path("campaign.json").write_text(json.dumps({**config, "workspace": config["workspace_root"], "plan": plan}))
runpy.run_path("component.py", run_name="__main__")
"""


#: Arbor's split in a fake checkout: the open directions of the map dealt in turn to the shares, or exit 1 when there
#: are fewer than shares. Its generate: one direction per op of the profile at 2 % or more, largest first, or exit 1
#: when there is none.
FAKE_DIRECTIONS = """
import json, pathlib, sys
args = sys.argv[1:]
if args[0] == "generate":
    shares = json.loads(pathlib.Path(args[args.index("--profile") + 1]).read_text())["shares"]
    ops = [op for op, share in sorted(shares.items(), key=lambda item: -item[1]) if share >= 2]
    if not ops:
        sys.exit("nothing above --min-share 2")
    directions = [{"id": f"m-{op}", "kind": "increment", "title": op} for op in ops]
    pathlib.Path(args[args.index("--out") + 1]).write_text(json.dumps({"directions": directions}))
    sys.exit(0)
directions = json.loads(pathlib.Path(args[1]).read_text())["directions"]
shares, out = int(args[args.index("--shares") + 1]), pathlib.Path(args[args.index("--out") + 1])
live = [entry for entry in directions if entry.get("status", "open") == "open"]
if len(live) < shares:
    sys.exit(f"{len(live)} independent groups for {shares} shares")
out.mkdir(parents=True, exist_ok=True)
for number in range(1, shares + 1):
    (out / f"share-{number}.json").write_text(json.dumps({"directions": live[number - 1 :: shares]}))
"""


#: What an Arbor checkout needs for the adapter: its markers and the serving pack's validator.
CHECKOUT_FILES = {
    "arbor/__init__.py": "",
    "arbor/__main__.py": FAKE_MAIN,
    "arbor/component.py": "# component mode\n",
    "arbor/directions.py": FAKE_DIRECTIONS,
    "docs/hosts.md": "# Running Arbor inside a host (component mode)\n",
    "packs/serving/pack.yaml": "name: serving\n",
    "packs/serving/validator/bench.sh": "#!/usr/bin/env bash\n",
    "packs/serving/validator/correctness.sh": "#!/usr/bin/env bash\n",
}


def _git(cwd: Path, *args: str) -> str:
    argv = ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args]
    return subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _write(root: Path, files: Mapping[str, str]) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def _committed(repo: Path) -> Path:
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _spec(tmp_path: Path, **changes: object) -> AgentSpec:
    """A branch's spec, its workspace a repository at the root it names and the gate's client aimed at :data:`PORT`."""

    workspace = _committed(_write(tmp_path / "workspace", {"vllm/config.py": "max_num_seqs = 128\n"}))
    spec = AgentSpec(
        model=MODEL,
        framework="vllm",
        tp=2,
        precision="fp8",
        gpu_type="mi355x",
        workload={"kind": "inferencex", "isl": 1024, "osl": 1024, "conc": 8},
        metric="output_throughput",
        server_args="--max-num-seqs 128",
        envs={"VLLM_ROCM_USE_AITER": "1", TABLE_ENV: os.pathsep.join(START_TABLE_PARTS)},
        workspace=str(workspace),
        cards=(4, 5),
        seconds=3600.0,
        port=PORT,
        client=CLIENT.stage(PORT, tmp_path / "agent" / "mb-bench"),
        knowledge=str(tmp_path / "agent" / "mb-integration.md"),
        deadline=DEADLINE,
        root=_git(workspace, "rev-parse", "HEAD"),
    )
    return replace(spec, **changes)


#: Arbor main 1fa2069's arbor/component.py, as far as the adapter reads it: its component keys, "review" among them.
GATED_COMPONENT = (
    'BLOCK_KEYS = ("host", "deadline", "seconds", "notes", "bench_command", "outbox", "phase_minutes",\n'
    '              "root", "plan", "stop_at", "profile", "review")\n'
    'REVIEW_BY = ("host",)\n'
)


MAP = {
    "directions": [
        {"id": "m-attn", "kind": "increment", "title": "attention decode path", "rank": 1},
        {"id": "m-moe", "kind": "increment", "title": "MoE GEMM tables", "rank": 2},
        {"id": "m-allreduce", "kind": "increment", "title": "custom all-reduce", "rank": 3},
        {"id": "m-sampler", "kind": "increment", "title": "sampler", "status": "done", "why": "iteration 4, KEEP 2"},
    ]
}


LAUNCH = f"python3 -m arbor run --config {CONFIG_FILE}"
REASONS = ["component.plan[0]: m-attn: no title", "component mode needs a deadline"]


def _arbor_repo(tmp_path: Path, agent: Component = Component(), refuse: tuple[str, ...] = ()) -> Path:
    """A fake Arbor's repository, run as :data:`FAKE_MAIN` says."""

    plan = json.dumps({**asdict(agent), "refuse": list(refuse)})
    files = {**CHECKOUT_FILES, "component.py": COMPONENT_AGENT, "fake-plan.json": plan}
    return _committed(_write(tmp_path / "arbor-repo", files))
