# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
import os
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ConfigDelta
from hyperloom.measure.compile import (
    ComponentLayout,
    LocalBlobStore,
    PackageBase,
    PackageLocator,
    UnsupportedComponentError,
)
from hyperloom_multibranch.blackboard.tests.factories import result
from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.specialist import arbor_adapter
from hyperloom_multibranch.specialist.agent_manifest import REFUSED_FILE, AgentManifest
from hyperloom_multibranch.specialist.agent_onboarding import PROFILE_FILE, AgentSpec, make_ready, root_profile
from hyperloom_multibranch.specialist.arbor_adapter import (
    BRANCH_RUN,
    DOSSIER_DIR,
    DOSSIER_PASS,
    INSTALL,
    LAUNCH_KEYS,
    MARKERS,
    PLAN_FILE,
    ArborCheckout,
    ArborRefused,
    SplitFailed,
    arbor_config,
    arbor_manifest,
    bench_command,
    generate_directions,
    is_arbor,
    link_built,
    mapped,
    read_launch,
    refusals,
    refused_reasons,
    split_directions,
)
from hyperloom_multibranch.specialist.autonomous_session import AGENT_DIR, MANIFEST_FILE, START_TABLES_FILE
from hyperloom_multibranch.specialist.gate_client import GateClient
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import PreflightFailed, SessionCrashed, SessionFinished, SessionRefused
from hyperloom_multibranch.specialist.worktree import SKIPPED
from hyperloom_multibranch.tests.test_specialist_agent_onboarding import Onboarder, Process, Sandbox, _setup
from hyperloom_multibranch.tests.test_specialist_autonomous_session import (
    BRANCH,
    CHAMPION_ROWS,
    COMPONENT_AGENT,
    TABLE_ENV,
    Component,
    _aiter,
    _imported,
    _replayed,
    _rig,
    _until,
)

MODEL = "/models/Qwen3-14B-FP8"
#: The gate's client as MB builds it; it names the model by its hub name, which the server must answer to.
CLIENT = GateClient("/opt/InferenceX", "Qwen/Qwen3-14B-FP8", 1024, 1024, 8, 0.8, None, False)
PORT = 31110
DEADLINE = 1793244000.7
CONFIG_FILE, SESSION_DIR = BRANCH_RUN.config_file, BRANCH_RUN.session_dir
LAUNCH = f"python3 -m arbor run --config {CONFIG_FILE}"
REASONS = ["component.plan[0]: m-attn: no title", "component mode needs a deadline"]
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


def _arbor_repo(tmp_path: Path, agent: Component = Component(), refuse: tuple[str, ...] = ()) -> Path:
    """A fake Arbor's repository, run as :data:`FAKE_MAIN` says."""

    plan = json.dumps({**asdict(agent), "refuse": list(refuse)})
    files = {**CHECKOUT_FILES, "component.py": COMPONENT_AGENT, "fake-plan.json": plan}
    return _committed(_write(tmp_path / "arbor-repo", files))


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
        client=CLIENT.command(PORT, tmp_path / "agent" / "mb-bench"),
        knowledge=str(tmp_path / "agent" / "mb-integration.md"),
        deadline=DEADLINE,
        root=_git(workspace, "rev-parse", "HEAD"),
    )
    return replace(spec, **changes)


def _filled(command: str, port: int, out: str) -> str:
    """``command`` as Arbor fills a host's bench command (arbor/component.py's ``fill_bench``)."""
    return command.replace("{port}", str(port)).replace("{out}", shlex.quote(out))


@pytest.mark.parametrize("missing", ["", *MARKERS])
def test_an_arbor_checkout_is_known_by_its_component_mode_its_entry_point_and_its_host_contract(
    tmp_path: Path, missing: str
) -> None:
    checkout = _write(tmp_path / "checkout", CHECKOUT_FILES)
    if missing:
        (checkout / missing).unlink()

    assert is_arbor(checkout) == (not missing)


def test_arbor_s_config_follows_from_the_spec_on_the_serving_pack_in_component_mode(tmp_path: Path) -> None:
    spec, checkout = _spec(tmp_path), tmp_path / "checkout"
    session = checkout / SESSION_DIR

    config = arbor_config(spec, checkout, BRANCH_RUN)

    validator = checkout / "packs" / "serving" / "validator"
    assert (config["pack"], config["mode"], config["docker_container"]) == ("serving", "component", "@host")
    assert (config["framework"], config["model_path"], config["tp"]) == ("vllm", MODEL, 2)
    assert config["model_name"] == "Qwen/Qwen3-14B-FP8"
    assert config["server_flags"] == "--max-num-seqs 128"
    assert config["server_env"] == {
        arbor_adapter.ACCURACY_FLOOR_ENV: "0.5",
        "VLLM_ROCM_USE_AITER": "1",
        TABLE_ENV: os.pathsep.join(START_TABLE_PARTS),
    }
    assert config["bench_script"] == str(validator / "bench.sh")
    assert config["correctness_script"] == str(validator / "correctness.sh")
    assert (config["workspace_root"], config["session_root"]) == (spec.workspace, str(session))
    assert (config["port"], config["gpu_ids"]) == (PORT, [0, 1])
    assert (config["concurrency"], config["isl"], config["osl"]) == (8, 1024, 1024)
    assert config["component"] == {
        "host": "hyperloom-mb",
        "deadline": 1793244000,
        "root": spec.root,
        "outbox": "keeps",
        "notes": spec.knowledge,
        "bench_command": bench_command(spec.client),
    }


def test_a_spec_with_no_client_runs_the_pack_s_bench_on_a_server_named_by_its_model_path(tmp_path: Path) -> None:
    config = arbor_config(_spec(tmp_path, client="", knowledge=""), tmp_path / "checkout", BRANCH_RUN)

    assert config["model_name"] == MODEL
    assert "bench_command" not in config["component"] and "notes" not in config["component"]


#: A recipe's own SGLang arguments for a GLM-like MoE model, served at TP4 and EP4.
GLM_ARGS = "--mem-fraction-static 0.85 --chunked-prefill-size 16384"


@pytest.mark.parametrize(
    ("args", "tp", "ep", "flags"),
    [
        ("", 2, 1, "--tp-size 2"),
        (GLM_ARGS, 4, 4, f"{GLM_ARGS} --tp-size 4 --ep-size 4"),
        ("", 4, 3, "--tp-size 4"),
    ],
    ids=["tp2", "glm-tp4-ep4", "ep-not-dividing-tp"],
)
def test_an_sglang_server_is_launched_at_the_workload_s_tp_and_ep(
    tmp_path: Path, args: str, tp: int, ep: int, flags: str
) -> None:
    spec = _spec(tmp_path, framework="sglang", server_args=args, tp=tp, ep=ep)

    config = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)

    assert (config["server_flags"], config["tp"]) == (flags, tp)


@pytest.mark.parametrize(
    "args",
    [
        "--tp 4",
        "--tp-size 4",
        "--tensor-parallel-size 4",
        "--tp=4",
        "--tp-size=4",
        "--tensor-parallel-size=4",
        "--tp-size 4 --ep 4",
        "--tp-size 4 --ep-size=4",
        "--tp-size 4 --expert-parallel-size 4",
    ],
)
def test_an_sglang_size_the_recipe_sets_in_any_spelling_is_not_set_again(tmp_path: Path, args: str) -> None:
    ep = 4 if "--e" in args else 1
    spec = _spec(tmp_path, framework="sglang", server_args=args, tp=4, ep=ep)

    assert arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)["server_flags"] == args


def test_a_vllm_server_takes_the_tp_from_arbor_and_its_arguments_as_the_recipe_gives_them(tmp_path: Path) -> None:
    spec = _spec(tmp_path, server_args="", tp=4, ep=4)

    config = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)

    assert (config["framework"], config["server_flags"], config["tp"]) == ("vllm", "", 4)


def test_the_gate_s_client_is_arbor_s_bench_command_with_only_its_port_and_result_dir_left_to_fill() -> None:
    client = CLIENT.command(PORT, Path("/mb/a1/agent/mb-bench"))

    command = bench_command(client)

    assert "--base-url http://0.0.0.0:{port} " in command and "--result-dir {out} " in command
    assert "--num-warmups 16" in command, "the client's warm-up requests are kept"
    words = shlex.split(client)
    words[words.index("--base-url") + 1] = "http://0.0.0.0:8123"
    words[words.index("--result-dir") + 1] = "/s/iter_3/host bench"
    assert shlex.split(_filled(command, 8123, "/s/iter_3/host bench")) == words


@pytest.mark.parametrize(
    ("client", "templated"),
    [
        (
            "python3 bench.py --base-url=http://127.0.0.1:31110/v1 --result-dir='/mb/a b' --num-warmups 4",
            "python3 bench.py --base-url=http://127.0.0.1:{port}/v1 --result-dir={out} --num-warmups 4",
        ),
        (
            "vllm bench serve --port 31110 --save-result --result-dir /mb/out",
            "vllm bench serve --port {port} --save-result --result-dir {out}",
        ),
        (
            "python3 b.py --base-url http://0.0.0.0:31110 --num-prompts 16 > /dev/null && "
            "python3 b.py --base-url http://0.0.0.0:31110 --result-dir /mb/out",
            "python3 b.py --base-url http://0.0.0.0:{port} --num-prompts 16 > /dev/null && "
            "python3 b.py --base-url http://0.0.0.0:{port} --result-dir {out}",
        ),
    ],
    ids=["joined", "port", "warm-up-first"],
)
def test_a_client_s_port_and_result_dir_are_templated_in_every_form_it_gives_them(client: str, templated: str) -> None:
    assert bench_command(client) == templated


@pytest.mark.parametrize(
    ("client", "said"),
    [
        ("python3 bench.py --base-url http://0.0.0.0:31110", "names no --result-dir"),
        ("python3 bench.py --result-dir /mb/out", "names no port"),
        ("python3 bench.py --base-url http://0.0.0.0 --result-dir /mb/out", "names no port"),
        ("python3 bench.py --base-url http://0.0.0.0:31110 --result-dir $OUT", "expands variables"),
        ("python3 bench.py --base-url 'http://0.0.0.0:31110 --result-dir /mb/out", "word by word"),
        ("python3 bench.py --base-url http://0.0.0.0:31110 --result-dir", "ends at --result-dir"),
    ],
)
def test_a_client_arbor_could_not_fill_is_refused_with_why(client: str, said: str) -> None:
    with pytest.raises(ArborRefused, match=said):
        bench_command(client)


def test_arbor_s_session_is_the_directory_its_manifest_s_snapshots_name(tmp_path: Path) -> None:
    spec, checkout = _spec(tmp_path), _write(tmp_path / "checkout", CHECKOUT_FILES)

    manifest = arbor_manifest(spec, ArborCheckout("git@host:org/arbor", "abc123", checkout, {}), BRANCH_RUN)

    assert manifest.snapshots is not None
    session = checkout / manifest.snapshots.session
    assert Path(manifest.config["session_root"]) == session.absolute() and session.is_dir()
    assert manifest.snapshots_outbox == f"{SESSION_DIR}/keeps"
    assert (manifest.config_file, manifest.launch, manifest.resume) == (CONFIG_FILE, LAUNCH, LAUNCH)
    assert manifest.install == INSTALL and manifest.keep is None and manifest.config_source is None
    assert manifest.ledger is not None and manifest.ledger.path == f"{SESSION_DIR}/ledger.jsonl"
    assert (manifest.ledger.kind, manifest.ledger.text) == ("type", "claim")
    assert AgentManifest.read(json.loads(json.dumps(manifest.to_document()))) == manifest


def _located(installed: Path) -> PackageLocator:
    def locate(layout: ComponentLayout) -> PackageBase:
        return PackageBase(layout.import_name, installed, "vllm", "0.27.1.dev5+gf46a9dfe2")

    return locate


def test_the_workspace_is_given_the_installed_framework_s_built_files_and_keeps_them_out_of_its_stage(
    tmp_path: Path,
) -> None:
    files = {"config.py": "max_num_seqs = 64\n", "_C.abi3.so": "elf", "attention/_moe_C.abi3.so": "elf"}
    installed = _write(tmp_path / "site-packages" / "vllm", {**files, "__pycache__/config.cpython-314.pyc": "pyc"})
    spec, checkout = _spec(tmp_path), _write(tmp_path / "checkout", CHECKOUT_FILES)
    workspace = Path(spec.workspace)
    (workspace / ".git" / "info" / "exclude").write_text("".join(f"{pattern}\n" for pattern in SKIPPED))

    arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout, {}), BRANCH_RUN, _located(installed))

    assert (workspace / "vllm" / "_C.abi3.so").resolve() == (installed / "_C.abi3.so").resolve()
    assert (workspace / "vllm" / "attention" / "_moe_C.abi3.so").is_symlink()
    assert (workspace / "vllm" / "config.py").read_text() == "max_num_seqs = 128\n"
    assert not (workspace / "vllm" / "__pycache__").exists()
    assert _git(workspace, "status", "--porcelain") == ""
    assert link_built(workspace, "vllm", _located(installed)) == 0


def test_a_branch_given_the_session_s_dossier_finds_it_in_its_checkout_where_arbor_looks_one_up(
    tmp_path: Path,
) -> None:
    dossier = _write(tmp_path / "pass" / "docs" / "dossier" / "qwen3-8b", {"README.md": "# the pass's\n"})
    spec, checkout = _spec(tmp_path, dossier=str(dossier)), _write(tmp_path / "checkout", CHECKOUT_FILES)

    arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout, {}), BRANCH_RUN, _located(tmp_path / "none"))
    (dossier / "README.md").write_text("# edited after\n")
    arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout, {}), BRANCH_RUN, _located(tmp_path / "none"))

    assert (checkout / DOSSIER_DIR / "qwen3-8b" / "README.md").read_text() == "# the pass's\n"  # a copy there stays


def test_a_framework_not_installed_or_with_no_layout_links_nothing(tmp_path: Path) -> None:
    def missing(layout: ComponentLayout) -> PackageBase:
        raise UnsupportedComponentError(f"package {layout.import_name} is not installed in this image")

    assert link_built(tmp_path, "vllm", missing) == 0
    assert link_built(tmp_path, "no-such-framework", _located(tmp_path)) == 0
    assert list(tmp_path.iterdir()) == []


def test_arbor_s_install_checks_each_import_it_needs_before_installing_it() -> None:
    assert INSTALL[1:3] == (
        "python3 -c 'import yaml' 2>/dev/null || python3 -m pip install --quiet pyyaml",
        "python3 -c 'import claude_agent_sdk' 2>/dev/null || python3 -m pip install --quiet claude-agent-sdk",
    )
    assert "(3, 11)" in INSTALL[0] and INSTALL[-1] == "python3 -c 'import yaml, claude_agent_sdk'"


async def test_the_root_s_op_shares_reach_arbor_s_profile_as_mb_writes_them_beside_the_adapter_s_config(
    tmp_path: Path,
) -> None:
    shares = {"gemm": 41.5, "attention": 22.0, "all_reduce": 9.0}
    plan = _write(tmp_path / "shares", {"share-1.json": json.dumps(MAP)}) / "share-1.json"
    spec = _spec(tmp_path, plan=str(plan))
    spec = replace(spec, root_shares=OpShares(spec.root, "prelude-104393", "p1", shares))
    sandbox = Sandbox([Process(None)])

    ready = await make_ready(str(_arbor_repo(tmp_path)), spec, tmp_path / "agent", _setup(sandbox, None))

    component = json.loads((ready.source_dir / CONFIG_FILE).read_text())["component"]
    profile = ready.source_dir / SESSION_DIR / PROFILE_FILE
    assert component["profile"] == str(profile)
    document = json.loads(profile.read_text())
    assert (document["commit"], document["job"], document["shares"]) == (spec.root, "prelude-104393", shares)
    assert component["plan"] == str(ready.source_dir.absolute() / PLAN_FILE) and component["host"] == "hyperloom-mb"
    assert component["bench_command"] == bench_command(spec.client) and component["notes"] == spec.knowledge


def test_the_branch_s_share_of_the_directions_is_arbor_s_plan_carried_into_its_checkout(tmp_path: Path) -> None:
    plan = tmp_path / "shares" / "share-2.json"
    _write(plan.parent, {plan.name: json.dumps({"directions": [{"id": "m-attn", "kind": "increment", "title": "a"}]})})
    checkout = _write(tmp_path / "checkout", CHECKOUT_FILES)

    manifest = arbor_manifest(
        _spec(tmp_path, plan=str(plan)), ArborCheckout("arbor", "abc123", checkout, {}), BRANCH_RUN
    )

    carried = checkout.absolute() / PLAN_FILE
    assert manifest.config["component"]["plan"] == str(carried)
    assert carried.read_text(encoding="utf-8") == plan.read_text(encoding="utf-8")
    assert "profile" not in manifest.config["component"]


def test_arbor_runs_the_pack_the_spec_names_and_one_its_checkout_lacks_is_refused(tmp_path: Path) -> None:
    own = {
        "packs/agentx-qwen38-1/pack.yaml": "name: agentx-qwen38-1\n",
        "packs/agentx-qwen38-1/validator/bench.sh": "#!/usr/bin/env bash\n",
        "packs/agentx-qwen38-1/validator/correctness.sh": "#!/usr/bin/env bash\n",
    }
    checkout = _write(tmp_path / "checkout", {**CHECKOUT_FILES, **own})
    spec = _spec(tmp_path)

    config = arbor_config(replace(spec, pack="agentx-qwen38-1"), checkout, BRANCH_RUN)
    missing = arbor_config(replace(spec, pack="agentx-gone"), checkout, BRANCH_RUN)

    assert config["pack"] == "agentx-qwen38-1"
    assert config["bench_script"] == str(checkout / "packs" / "agentx-qwen38-1" / "validator" / "bench.sh")
    assert not [reason for reason in refusals(config, {}) if "pack" in reason or "script" in reason]
    assert any("agentx-gone" in reason and "pack.yaml" in reason for reason in refusals(missing, {}))
    assert arbor_config(spec, checkout, BRANCH_RUN)["pack"] == "serving"


#: A launch Magpie resolves from an InferenceX AgentX recipe (boards/hyperloom-multibranch#157 c4405), as a launch file.
RESOLVED = {
    "server_flags": "--mem-fraction-static 0.8 --chunked-prefill-size 16384",
    "server_env": {"SGLANG_SIMULATE_ACC_LEN": "3", "VLLM_ROCM_USE_AITER": "0"},
    "model_name": "Qwen/Qwen3.8-Flash-Next",
}
AGENTX_WORKLOAD = {"kind": "agentx", "conc": 8}
#: The repository's template of a launch file.
TEMPLATE = Path(__file__).parents[3] / "examples" / "multibranch-arbor-launch" / "launch.template.json"


def _launch_file(tmp_path: Path, document: object) -> str:
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def test_a_launch_file_is_served_with_the_recipe_s_launch_added_to_it(tmp_path: Path) -> None:
    launch = _launch_file(tmp_path, RESOLVED)
    spec = _spec(tmp_path, framework="sglang", server_args="--max-running-requests 64", tp=1, client="", launch=launch)

    config = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)

    flags = f"{RESOLVED['server_flags']} --max-running-requests 64 --tp-size 1"
    envs = {
        arbor_adapter.ACCURACY_FLOOR_ENV: "0.5",
        "SGLANG_SIMULATE_ACC_LEN": "3",
        "VLLM_ROCM_USE_AITER": "1",
        TABLE_ENV: os.pathsep.join(START_TABLE_PARTS),
    }
    assert (config["server_flags"], config["server_env"]) == (flags, envs)
    assert config["model_name"] == "Qwen/Qwen3.8-Flash-Next"
    assert "bench_command" not in config["component"]


def test_arbor_is_held_to_mb_s_accuracy_floor_unless_the_launch_or_the_recipe_sets_its_own(tmp_path: Path) -> None:
    # Without a floor the serving pack books every iteration whose accuracy it must judge as invalid; MB's eval and the
    # pack's score one model points apart, so the floor is MB's absolute one, not its baseline less its tolerance.
    floor, spec = arbor_adapter.ACCURACY_FLOOR_ENV, _spec(tmp_path)
    given = replace(spec, launch=_launch_file(tmp_path, {**RESOLVED, "server_env": {floor: "0.85"}}))
    recipe = replace(spec, envs={**spec.envs, floor: "0.8"})

    configs = [arbor_config(each, tmp_path / "checkout", BRANCH_RUN) for each in (spec, given, recipe)]

    assert [config["server_env"][floor] for config in configs] == ["0.5", "0.85", "0.8"]


def test_arbor_is_given_mb_s_rules_as_directives_the_dossier_copies(tmp_path: Path) -> None:
    # Operator, 2026-11-01: the benchmark's rules are considered in validation, and are part of the dossier.
    spec = _spec(tmp_path)
    agentx = replace(spec, workload={"kind": "agentx", "conc": 8})

    rules = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)["directives"]
    agentx_rules = arbor_config(agentx, tmp_path / "checkout", BRANCH_RUN)["directives"]

    assert all(rule.startswith(arbor_adapter.HOST_RULE) for rule in (*rules, *agentx_rules))
    fixed, kept, accuracy = rules
    assert "inferencex at concurrency 8, isl 1024, osl 1024" in fixed and "ISL" in fixed and "CONC" in fixed
    assert "output_throughput" in kept
    assert "at least 0.5" in accuracy and "at most 0.05" in accuracy
    assert "the root it measures itself" in accuracy
    assert "error rate stays at or under 10%" in agentx_rules[-1]


def test_arbor_s_accuracy_tolerance_is_the_drop_mb_s_rule_allows(tmp_path: Path) -> None:
    # Arbor measures the root's full GSM8K itself and holds a numerics change to it less this tolerance, on its own
    # eval's scale: MB's rule allows ACCURACY_THRESHOLD under the root, so Arbor allows the same.
    config = arbor_config(_spec(tmp_path), tmp_path / "checkout", BRANCH_RUN)

    assert config["policy"] == {"accuracy_tolerance": 0.05}


def test_a_launch_file_s_client_measures_arbor_only_when_mb_gives_none(tmp_path: Path) -> None:
    client = "python3 bench.py --port 8000 --result-dir /tmp/out --model served"
    spec = _spec(tmp_path, launch=_launch_file(tmp_path, {**RESOLVED, "model_name": "", "client": client}))

    given = arbor_config(replace(spec, client=""), tmp_path / "checkout", BRANCH_RUN)
    own = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)

    assert (given["component"]["bench_command"], given["model_name"]) == (bench_command(client), "served")
    assert (own["component"]["bench_command"], own["model_name"]) == (bench_command(spec.client), "Qwen/Qwen3-14B-FP8")


def test_an_agentx_branch_is_refused_before_arbor_starts_unless_it_is_given_the_resolved_launch(tmp_path: Path) -> None:
    checkout = _write(tmp_path / "checkout", CHECKOUT_FILES)
    spec = _spec(tmp_path, framework="sglang", server_args="", workload=AGENTX_WORKLOAD)

    with pytest.raises(ArborRefused, match="--agent-launch"):
        arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout, {}), BRANCH_RUN)
    launched = replace(spec, launch=_launch_file(tmp_path, RESOLVED))
    manifest = arbor_manifest(launched, ArborCheckout("arbor", "abc123", checkout, {}), BRANCH_RUN)

    assert manifest.config["server_flags"] == f"{RESOLVED['server_flags']} --tp-size 2"


@pytest.mark.parametrize(
    ("document", "said"),
    [
        ("{", "cannot be read"),
        (["--tp 1"], "is not a JSON object"),
        ({"server_env": {}}, "no server_flags"),
        ({**RESOLVED, "server_env": {"N": 1}}, "server_env is not an object of strings"),
        ({**RESOLVED, "flags": "--tp 1", "model_name": 7}, "keys a launch does not: flags; its model_name is not"),
        ({**RESOLVED, "client": "python3 bench.py --port 8000"}, "names no --result-dir"),
    ],
    ids=["unreadable", "a-list", "no-flags", "env-not-strings", "unknown-key-and-name", "client-arbor-cannot-fill"],
)
def test_a_launch_file_the_adapter_cannot_use_is_refused_with_why(tmp_path: Path, document: object, said: str) -> None:
    path = Path(_launch_file(tmp_path, document))
    if isinstance(document, str):
        path.write_text(document, encoding="utf-8")

    with pytest.raises(ArborRefused) as refused:
        read_launch(path)

    assert said.split("; ")[0] in str(refused.value) and said.split("; ")[-1] in str(refused.value)


def test_the_repository_s_launch_template_is_a_launch_file_the_adapter_reads() -> None:
    launch = read_launch(TEMPLATE)

    assert set(json.loads(TEMPLATE.read_text(encoding="utf-8"))) == set(LAUNCH_KEYS)
    assert launch.server_flags and launch.server_env and launch.client == ""


@pytest.mark.parametrize(
    ("plan", "said"),
    [("shares/share-1.json", "not an absolute path"), ("/nonexistent/share-1.json", "cannot be read")],
)
def test_a_plan_arbor_could_not_read_is_refused(tmp_path: Path, plan: str, said: str) -> None:
    config = arbor_config(_spec(tmp_path, plan=plan), tmp_path / "checkout", BRANCH_RUN)

    assert any(said in reason for reason in refusals(config, {}))


def test_the_dossier_pass_stops_at_the_map_in_a_session_of_its_own_and_takes_no_plan(tmp_path: Path) -> None:
    spec = _spec(tmp_path, plan="/shares/share-1.json")
    checkout = _write(tmp_path / "checkout", CHECKOUT_FILES)

    manifest = arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout, {}), DOSSIER_PASS)

    component = manifest.config["component"]
    assert component["stop_at"] == "map" and "plan" not in component and "profile" not in component
    assert (checkout / DOSSIER_PASS.session_dir).is_dir()
    assert manifest.config_file == DOSSIER_PASS.config_file and manifest.snapshots is not None
    assert manifest.snapshots.session == DOSSIER_PASS.session_dir != SESSION_DIR


def test_what_arbor_refuses_at_its_start_is_refused_before_it_starts(tmp_path: Path) -> None:
    spec = _spec(tmp_path, deadline=0.0, root="f" * 40)
    config = arbor_config(spec, tmp_path / "empty-checkout", BRANCH_RUN)
    config = {**config, "docker_container": "arbor-c", "component": {**config["component"], "x": 1}}

    reasons = refusals(config, {"ARBOR_ROLL": "1"})

    expected = [
        "the component block has keys Arbor does not: x",
        "no deadline: component.deadline is 0",
        "docker_container is 'arbor-c', not @host",
        "ARBOR_ROLL=1 is set",
        "bench_script",
        "correctness_script",
        "the pack 'serving' has no pack.yaml",
        f"component.root {'f' * 40} is not a commit of the workspace {spec.workspace}",
    ]
    assert len(reasons) == len(expected) and all(said in reason for said, reason in zip(expected, reasons))


#: Arbor main 1fa2069's arbor/component.py, as far as the adapter reads it: its component keys, "review" among them.
GATED_COMPONENT = (
    'BLOCK_KEYS = ("host", "deadline", "seconds", "notes", "bench_command", "outbox", "phase_minutes",\n'
    '              "root", "plan", "stop_at", "profile", "review")\n'
    'REVIEW_BY = ("host",)\n'
)
#: Arbor 05d04d5's: no review gate.
UNGATED_COMPONENT = (
    'BLOCK_KEYS = ("host", "deadline", "seconds", "notes", "bench_command", "outbox", "phase_minutes",\n'
    '              "root", "plan", "stop_at", "profile")\n'
)


@pytest.mark.parametrize(
    ("component", "review", "block", "top"),
    [
        (GATED_COMPONENT, "host", "host", None),
        (GATED_COMPONENT, "off", None, False),
        (GATED_COMPONENT, "", None, None),
        (UNGATED_COMPONENT, "host", None, None),
        (UNGATED_COMPONENT, "off", None, None),
        ("# no component mode here\n", "host", None, None),
    ],
)
def test_the_review_gate_is_handed_over_only_to_an_arbor_that_has_one(
    tmp_path: Path, component: str, review: str, block: str | None, top: bool | None
) -> None:
    checkout = _write(tmp_path / "checkout", {**CHECKOUT_FILES, "arbor/component.py": component})

    config = arbor_config(_spec(tmp_path, review=review), checkout, BRANCH_RUN)

    assert arbor_adapter.has_review_gate(checkout) == ("review" in component)
    assert config["component"].get("review") == block and config.get("review_gate") == top
    assert not [reason for reason in refusals(config, {}) if "component" in reason and "review" in reason]


def test_a_review_by_anyone_but_the_host_is_refused(tmp_path: Path) -> None:
    config = arbor_config(_spec(tmp_path), tmp_path / "checkout", BRANCH_RUN)
    config = {**config, "component": {**config["component"], "review": "operator"}}

    assert "component.review is 'operator', not 'host'" in refusals(config, {})


async def test_an_arbor_checkout_is_made_ready_by_the_adapter_without_the_onboarding_model(tmp_path: Path) -> None:
    repo, process = _arbor_repo(tmp_path), Process(None)
    sandbox, onboard = Sandbox([process]), Onboarder([])

    ready = await make_ready(str(repo), _spec(tmp_path), tmp_path / "agent", _setup(sandbox, onboard))

    assert onboard.prompts == [] and ready.process is process
    config = json.loads((ready.source_dir / CONFIG_FILE).read_text())
    assert config == dict(ready.manifest.config) and config["component"]["host"] == "hyperloom-mb"
    assert "profile" not in config["component"], "the session's root has no profile"
    assert [command for command, _cwd, _env in sandbox.ran] == [*INSTALL, LAUNCH]
    assert all(cwd == ready.source_dir for _command, cwd, _env in sandbox.ran)


async def test_a_config_the_adapter_refuses_fails_the_preflight_in_one_attempt_with_every_reason(
    tmp_path: Path,
) -> None:
    sandbox, onboard, waits = Sandbox([]), Onboarder([]), []
    spec = _spec(tmp_path, deadline=0.0, root="f" * 40)

    with pytest.raises(PreflightFailed) as failed:
        await make_ready(str(_arbor_repo(tmp_path)), spec, tmp_path / "agent", _setup(sandbox, onboard, waits))

    [line] = failed.value.reasons
    assert line.startswith("attempt 1: Arbor would refuse the config written for it: no deadline")
    assert f"component.root {'f' * 40} is not a commit of the workspace" in line
    assert sandbox.ran == [] and onboard.prompts == [] and waits == []


async def test_a_client_arbor_could_not_fill_fails_the_preflight_in_one_attempt(tmp_path: Path) -> None:
    sandbox, waits = Sandbox([]), []
    spec = _spec(tmp_path, client="python3 bench.py --base-url http://0.0.0.0:31110")

    with pytest.raises(PreflightFailed, match="names no --result-dir") as failed:
        await make_ready(str(_arbor_repo(tmp_path)), spec, tmp_path / "agent", _setup(sandbox, Onboarder([]), waits))

    assert len(failed.value.reasons) == 1 and sandbox.ran == [] and waits == []


class _RefusingSandbox(Sandbox):
    """Starts Arbor as one that refuses its config: its session's refusal file says why."""

    async def start(self, command: str, cwd: Path, env: Mapping[str, str]) -> Process:
        refused = {"reasons": REASONS, "ts": "2026-10-29T00:00:00Z"}
        (cwd / SESSION_DIR / REFUSED_FILE).write_text(json.dumps(refused), encoding="utf-8")
        return await super().start(command, cwd, env)


async def test_arbor_refusing_its_config_as_it_starts_fails_the_preflight_with_its_reasons_in_one_attempt(
    tmp_path: Path,
) -> None:
    process = Process(2, "[arbor] refusing to start: component.plan[0]: m-attn: no title\n")
    sandbox, waits = _RefusingSandbox([process]), []

    with pytest.raises(PreflightFailed) as failed:
        await make_ready(str(_arbor_repo(tmp_path)), _spec(tmp_path), tmp_path / "agent", _setup(sandbox, None, waits))

    assert failed.value.reasons == (f"attempt 1: `{LAUNCH}` refused its config: {'; '.join(REASONS)}",)
    assert process.killed and waits == []


async def test_an_arbor_run_on_the_adapter_s_config_has_its_keeps_imported_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arbor_adapter, "INSTALL", ())
    agent = Component(keeps=(256, 512), gap=0.1, latest_each=False, tail=0.6)
    rig = await _rig(tmp_path, source=_arbor_repo(tmp_path, agent), start_seconds=0.2)

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    assert rig.sandbox.prompts == [] and rig.sandbox.started == [LAUNCH]
    [request] = rig.toolbox.requests
    assert "+max_num_seqs = 512" in rig.patch(request)
    checkout = tmp_path / BRANCH / "session" / AGENT_DIR / "src"
    assert (checkout / SESSION_DIR / "keeps" / "LATEST").read_text() == "0002"
    assert (await rig.notices())[-1] == f"{NOTICE} the agent finished its run: done"


async def test_arbor_starts_with_the_start_s_kernel_tables_and_a_keep_that_keeps_them_imports_no_table_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arbor_adapter, "INSTALL", ())
    agent = Component(keeps=(256,), gap=0.1, flags="--max-num-seqs 128", tail=0.6)
    start = _replayed(LocalBlobStore(tmp_path / "blobs"))
    source = _arbor_repo(tmp_path, agent)
    rig = await _rig(tmp_path, start=start, start_seconds=0.2, aiter=_aiter(tmp_path), source=source)

    request = await _imported(rig)

    config = json.loads((tmp_path / BRANCH / "session" / AGENT_DIR / "src" / CONFIG_FILE).read_text())
    given = json.loads((tmp_path / BRANCH / "session" / START_TABLES_FILE).read_text())[TABLE_ENV]["value"]
    assert config["server_env"] == {arbor_adapter.ACCURACY_FLOOR_ENV: "0.5", TABLE_ENV: given}
    assert Path(given.split(os.pathsep)[-1]).read_text() == CHAMPION_ROWS
    assert (request.delta.add_kernels, request.delta.remove_kernels) == ((), ())
    assert request.delta.config == ConfigDelta() and "+max_num_seqs = 256" in rig.patch(request)


async def test_an_arbor_started_with_too_little_time_ends_its_branch_as_finished_after_one_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arbor_adapter, "INSTALL", ())
    agent = Component(keeps=(), tail=0.0, reason="deadline")
    rig = await _rig(tmp_path, source=_arbor_repo(tmp_path, agent), start_seconds=5.0)

    await asyncio.wait_for(rig.session.preflight(), 20)
    with pytest.raises(SessionFinished, match="the agent finished its run: deadline"):
        await asyncio.wait_for(rig.session.run(), 10)

    assert rig.sandbox.started == [LAUNCH]
    assert (await rig.notices())[-1] == f"{NOTICE} the agent finished its run: deadline"


async def test_an_arbor_restarted_after_a_crash_is_resumed_by_its_stored_manifest_without_onboarding_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arbor_adapter, "INSTALL", ())
    rig = await _rig(tmp_path, source=_arbor_repo(tmp_path, Component(keeps=(), exit=3, tail=0.6)), start_seconds=0.2)
    config = tmp_path / BRANCH / "session" / AGENT_DIR / "src" / CONFIG_FILE

    await rig.session.preflight()
    written = config.read_text()
    with pytest.raises(SessionCrashed, match="the agent exited 3"):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    resumed = rig.restart()
    with pytest.raises(SessionCrashed, match="the agent exited 3"):
        await asyncio.wait_for(resumed.run(), 10)
    await resumed.stop()

    stored = json.loads((tmp_path / BRANCH / "session" / MANIFEST_FILE).read_text())
    assert (stored["launch"], stored["resume"]) == (LAUNCH, LAUNCH)
    assert rig.sandbox.started == [LAUNCH, LAUNCH] and rig.sandbox.prompts == []
    assert config.read_text() == written
    assert any(note == f"{NOTICE} the agent resumed with `{LAUNCH}`" for note in await rig.notices())


async def test_an_arbor_that_refuses_its_config_ends_the_branch_for_good_with_its_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arbor_adapter, "INSTALL", ())
    rig = await _rig(tmp_path, source=_arbor_repo(tmp_path, Component(tail=0.6), REASONS), start_seconds=0.2)

    await rig.session.preflight()
    with pytest.raises(SessionRefused) as refused:
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()

    assert str(refused.value) == f"the agent refused its config: {'; '.join(REASONS)}"
    assert isinstance(refused.value, SessionFinished) and not isinstance(refused.value, SessionCrashed)
    assert (await rig.notices())[-1] == f"{NOTICE} {refused.value}"


def test_a_refusal_file_s_reasons_are_read_in_order_and_none_is_empty(tmp_path: Path) -> None:
    path = tmp_path / REFUSED_FILE
    assert refused_reasons(path) == ""
    path.write_text(json.dumps({"reasons": REASONS, "ts": "2026-10-29T00:00:00Z"}))
    assert refused_reasons(path) == "; ".join(REASONS)
    path.write_text("{")
    assert refused_reasons(path) == ""


@pytest.mark.parametrize(("exit_file", "ended"), [(None, False), ("done", False), ("mapped", True)])
def test_a_dossier_pass_ended_mapped_only_when_its_exit_file_says_so(
    tmp_path: Path, exit_file: str | None, ended: bool
) -> None:
    if exit_file is not None:
        (tmp_path / "component_exit.json").write_text(json.dumps({"reason": exit_file, "detail": "m-attn, m-moe"}))

    assert mapped(tmp_path) is ended


MAP = {
    "directions": [
        {"id": "m-attn", "kind": "increment", "title": "attention decode path", "rank": 1},
        {"id": "m-moe", "kind": "increment", "title": "MoE GEMM tables", "rank": 2},
        {"id": "m-allreduce", "kind": "increment", "title": "custom all-reduce", "rank": 3},
        {"id": "m-sampler", "kind": "increment", "title": "sampler", "status": "done", "why": "iteration 4, KEEP 2"},
    ]
}


async def test_a_map_is_split_into_one_plan_per_share_by_arbor_s_command_line(tmp_path: Path) -> None:
    checkout = _write(tmp_path / "checkout", CHECKOUT_FILES)
    map_file = _write(tmp_path / "pass", {"directions.json": json.dumps(MAP)}) / "directions.json"

    shares = await split_directions(checkout, map_file, 2, tmp_path / "shares")

    assert shares == (tmp_path / "shares" / "share-1.json", tmp_path / "shares" / "share-2.json")
    dealt = [[entry["id"] for entry in json.loads(share.read_text())["directions"]] for share in shares]
    assert dealt == [["m-attn", "m-allreduce"], ["m-moe"]]
    with pytest.raises(SplitFailed, match="3 independent groups for 4 shares"):
        await split_directions(checkout, map_file, 4, tmp_path / "four")


#: A real Arbor checkout to split with, when the environment names one; its split runs offline, on the standard
#: library alone.
ARBOR_CHECKOUT = os.environ.get("HYPERLOOM_TEST_ARBOR_CHECKOUT", "")


@pytest.mark.skipif(not ARBOR_CHECKOUT, reason="HYPERLOOM_TEST_ARBOR_CHECKOUT names no Arbor checkout")
async def test_a_map_is_split_by_a_real_arbor_into_shares_each_a_plan_it_accepts(tmp_path: Path) -> None:
    checkout = Path(ARBOR_CHECKOUT)
    assert is_arbor(checkout)
    map_file = _write(tmp_path / "pass", {"directions.json": json.dumps(MAP)}) / "directions.json"

    shares = await split_directions(checkout, map_file, 2, tmp_path / "shares")

    dealt = [{entry["id"] for entry in json.loads(share.read_text())["directions"]} for share in shares]
    assert set().union(*dealt) == {"m-attn", "m-moe", "m-allreduce"} and not dealt[0] & dealt[1]
    with pytest.raises(SplitFailed, match="exited 1"):
        await split_directions(checkout, map_file, 4, tmp_path / "four")


@pytest.mark.skipif(not ARBOR_CHECKOUT, reason="HYPERLOOM_TEST_ARBOR_CHECKOUT names no Arbor checkout")
async def test_a_real_arbor_generates_from_the_root_s_profile_a_map_it_splits(tmp_path: Path) -> None:
    checkout = Path(ARBOR_CHECKOUT)
    ops = {"mla_decode_fwd": 38.0, "fused_moe_kernel": 31.0, "rccl_all_reduce": 12.0, "rmsnorm": 0.5}
    profile = root_profile(OpShares("c0ffee", "prelude-1", "p1", ops), tmp_path / "root")
    small = root_profile(OpShares("c0ffee", "prelude-1", "p1", {"rmsnorm": 0.5}), tmp_path / "small")
    assert profile is not None and small is not None

    generated = await generate_directions(checkout, profile, tmp_path / "map.json")
    shares = await split_directions(checkout, generated, 2, tmp_path / "shares")

    ids = [entry["id"] for entry in json.loads(generated.read_text())["directions"]]
    assert ids == ["m-attn", "m-moe", "m-comm"]
    dealt = [{entry["id"] for entry in json.loads(share.read_text())["directions"]} for share in shares]
    assert set().union(*dealt) == set(ids) and not dealt[0] & dealt[1]
    with pytest.raises(SplitFailed, match="exited 1"):
        await generate_directions(checkout, small, tmp_path / "none.json")


@pytest.mark.skipif(not ARBOR_CHECKOUT, reason="HYPERLOOM_TEST_ARBOR_CHECKOUT names no Arbor checkout")
def test_a_real_arbor_launches_sglang_at_the_workload_s_tp_and_ep(tmp_path: Path) -> None:
    spec = _spec(tmp_path, framework="sglang", server_args=GLM_ARGS, tp=4, ep=4)
    config = arbor_config(spec, Path(ARBOR_CHECKOUT), BRANCH_RUN)
    serve = "import json, sys; from arbor import frameworks; c = json.load(sys.stdin); print(frameworks.get(c).serve_cmd(c, c['server_flags']))"

    done = subprocess.run(
        ["python3", "-c", serve],
        cwd=ARBOR_CHECKOUT,
        input=json.dumps(config),
        capture_output=True,
        text=True,
        check=True,
    )

    words = shlex.split(done.stdout)
    assert words[:3] == ["python3", "-m", "sglang.launch_server"]
    assert (words[words.index("--tp-size") + 1], words[words.index("--ep-size") + 1]) == ("4", "4")
