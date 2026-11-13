# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ast
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from hyperloom.common.agentx_objective import AgentXObjective
from hyperloom.common.launch_credentials import redact_credential_env
from hyperloom.common.perf_metric import GRADED_INTVTY_P50, TOTAL_TOKEN_THROUGHPUT
from hyperloom.inference_optimizer.cli import base_launch as cli_base_launch
from hyperloom.orchestrator.state.base_launch import FrozenBase, server_argv, server_base
from hyperloom.measure.compile import (
    ComponentLayout,
    PackageBase,
    PackageLocator,
    UnsupportedComponentError,
)
from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.scheduling.session_base import freeze_session_base, served_launch
from hyperloom_multibranch.scheduling.settings import KernelOptimizer, Workload
from hyperloom_multibranch.specialist import arbor_adapter
from hyperloom_multibranch.specialist.agent_manifest import REFUSED_FILE, AgentManifest
from hyperloom_multibranch.specialist.agent_spec import AgentSpec
from hyperloom_multibranch.specialist.agent_onboarding import PROFILE_FILE, make_ready, root_profile
from hyperloom_multibranch.specialist.arbor_adapter import (
    ARBOR_PYTHON,
    OBJECTIVE_KEY,
    OBJECTIVE_MODULE,
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
    carry_dossier,
    dossier_notes,
    generate_directions,
    host_rules,
    is_arbor,
    link_built,
    exit_reason,
    mapped,
    read_launch,
    refusals,
    refused_reasons,
    split_directions,
)
from hyperloom_multibranch.specialist.runtime import PreflightFailed
from hyperloom_multibranch.specialist.worktree import SKIPPED
from hyperloom_multibranch.tests.agent_rig import (
    CHECKOUT_FILES,
    CLIENT,
    CONFIG_FILE,
    GATED_COMPONENT,
    LAUNCH,
    MAP,
    MODEL,
    PORT,
    REASONS,
    SESSION_DIR,
    START_TABLE_PARTS,
    TABLE_ENV,
    Onboarder,
    Process,
    Sandbox,
    _arbor_repo,
    _git,
    _setup,
    _spec,
    _write,
)
from hyperloom_multibranch.tests.base_records import MODEL_ID, base_record, write_record
from inference_suite.contracts import AgentxResult, Redacted, RunStatus
from inference_suite.contracts import Workload as LaunchWorkload
from inference_suite.contracts.report import AGENTX_COUNT_KEYS, AGENTX_LATENCY_KEYS, AGENTX_THROUGHPUT_KEYS


def _filled(command: str, port: int, out: str) -> str:
    """``command`` as Arbor fills a host's bench command (arbor/component.py's ``fill_bench``)."""
    return command.replace("{port}", str(port)).replace("{out}", shlex.quote(out))


def test_the_adapter_imports_nothing_from_the_onboarding_that_imports_it() -> None:
    tree = ast.parse(Path(arbor_adapter.__file__).read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}

    assert "hyperloom_multibranch.specialist.agent_onboarding" not in imported


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
    assert config["server_flags"] == "--max-num-seqs 128 --tensor-parallel-size 2"
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


@pytest.mark.parametrize(
    ("args", "flags"),
    [("", "--tensor-parallel-size 4"), ("-tp 4", "-tp 4"), ("--tensor-parallel-size=4", "--tensor-parallel-size=4")],
    ids=["plain-baseline-root", "short-spelling", "equals-spelling"],
)
def test_a_vllm_server_names_its_tp_so_a_plain_baseline_root_is_not_refused(
    tmp_path: Path, args: str, flags: str
) -> None:
    # A vLLM root with no launch of its own gave Arbor empty server_flags, which it refuses.
    spec = _spec(tmp_path, server_args=args, tp=4, ep=4)

    config = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)

    assert (config["framework"], config["server_flags"], config["tp"]) == ("vllm", flags, 4)


def test_empty_server_flags_are_refused_with_why(tmp_path: Path) -> None:
    config = {**arbor_config(_spec(tmp_path), tmp_path / "checkout", BRANCH_RUN), "server_flags": " "}

    assert any(reason.startswith("no server_flags:") for reason in refusals(config))


def test_the_gate_s_client_is_arbor_s_bench_command_with_only_its_port_and_result_dir_left_to_fill() -> None:
    client = CLIENT.stage(PORT, Path("/mb/a1/agent/mb-bench"))

    command = bench_command(client)

    warmup, measured = command.split(" && ")
    assert "--base-url http://0.0.0.0:{port} " in warmup and "--base-url http://0.0.0.0:{port} " in measured
    assert "--result-dir {out} " in measured
    assert "--num-prompts 16 " in warmup, "the client's warm-up requests are kept"
    words = shlex.split(client.replace("http://0.0.0.0:31110", "http://0.0.0.0:8123"))
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
        ("python3 -m inference_suite run --config /mb/c.yaml", "names no --result-dir or --run-dir"),
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


def _agentx_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """The gate's AgentX client staged for an agent on :data:`PORT`, as MB gives it to an AgentX session's Arbor."""

    from hyperloom.orchestrator.actions.executors import TOOL_CACHE_ENV
    from hyperloom_multibranch.specialist.gate_client import read_gate_client
    from hyperloom_multibranch.tests.test_specialist_gate_client import _agentx_session

    monkeypatch.setenv(TOOL_CACHE_ENV, str(tmp_path / "tools"))
    canonical, _ = _agentx_session(tmp_path, tmp_path / "tools")
    return read_gate_client(canonical, canonical.parent.parent, 4800).stage(PORT, tmp_path / "agent" / "mb-bench")


def test_the_gate_s_agentx_round_is_arbor_s_bench_command_with_only_its_run_dir_left_to_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _agentx_client(tmp_path, monkeypatch)

    command = bench_command(client)

    assert command.endswith(" --run-dir {out}") and "{port}" not in command
    words = shlex.split(client)
    words[words.index("--run-dir") + 1] = "/s/iter_3/host bench"
    assert shlex.split(_filled(command, PORT, "/s/iter_3/host bench")) == words


def _arbor_config_errors(config: Mapping[str, object]) -> list[str]:
    """What the vendored Arbor's own config check (arbor/component.py's ``config_errors``) refuses ``config`` for."""

    script = "import json, sys; from arbor import component; print(json.dumps(component.config_errors(json.load(sys.stdin))))"
    run = subprocess.run(
        [sys.executable, "-c", script], input=json.dumps(config), cwd=VENDORED_ARBOR, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def test_an_agentx_arbor_on_the_session_s_base_measures_with_the_gate_s_round_and_needs_no_bench_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = write_record(tmp_path / "record" / "base_launch.json")
    client = _agentx_client(tmp_path, monkeypatch)
    spec = replace(
        _spec(tmp_path, launch=str(record), workload=AGENTX_WORKLOAD, metric=GRADED_INTVTY_P50),
        server_args="",
        envs={},
        client=client,
        client_seconds=4800.0,
    )

    config = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)

    component = config["component"]
    assert component["bench_command"] == bench_command(client)
    assert component["phase_minutes"] == {"bench": 80}
    assert config["server_env"] == {arbor_adapter.ACCURACY_FLOOR_ENV: "0.5", **read_launch(record).server_env}
    assert _arbor_config_errors(config) == [] and _arbor_start_check(config) == []


def test_a_client_bounded_by_the_agent_s_own_cap_leaves_arbor_s_phase_minutes_alone(tmp_path: Path) -> None:
    assert "phase_minutes" not in arbor_config(_spec(tmp_path), tmp_path / "checkout", BRANCH_RUN)["component"]


def test_arbor_s_session_is_the_directory_its_manifest_s_snapshots_name(tmp_path: Path) -> None:
    spec, checkout = _spec(tmp_path), _write(tmp_path / "checkout", CHECKOUT_FILES)

    manifest = arbor_manifest(spec, ArborCheckout("git@host:org/arbor", "abc123", checkout), BRANCH_RUN)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {"config.py": "max_num_seqs = 64\n", "_C.abi3.so": "elf", "attention/_moe_C.abi3.so": "elf"}
    installed = _write(tmp_path / "site-packages" / "vllm", {**files, "__pycache__/config.cpython-314.pyc": "pyc"})
    spec, checkout = _spec(tmp_path), _write(tmp_path / "checkout", CHECKOUT_FILES)
    workspace = Path(spec.workspace)
    (workspace / ".git" / "info" / "exclude").write_text("".join(f"{pattern}\n" for pattern in SKIPPED))
    monkeypatch.setattr(arbor_adapter, "locate_installed_package", _located(installed))

    arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout), BRANCH_RUN)

    assert (workspace / "vllm" / "_C.abi3.so").resolve() == (installed / "_C.abi3.so").resolve()
    assert (workspace / "vllm" / "attention" / "_moe_C.abi3.so").is_symlink()
    assert (workspace / "vllm" / "config.py").read_text() == "max_num_seqs = 128\n"
    assert not (workspace / "vllm" / "__pycache__").exists()
    assert _git(workspace, "status", "--porcelain") == ""
    assert link_built(workspace, "vllm") == 0


def test_a_branch_given_the_session_s_dossier_finds_it_in_its_checkout_where_arbor_looks_one_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dossier = _write(tmp_path / "pass" / "docs" / "dossier" / "qwen3-8b", {"README.md": "# the pass's\n"})
    spec, checkout = _spec(tmp_path, dossier=str(dossier)), _write(tmp_path / "checkout", CHECKOUT_FILES)
    monkeypatch.setattr(arbor_adapter, "locate_installed_package", _located(tmp_path / "none"))

    arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout), BRANCH_RUN)
    (dossier / "README.md").write_text("# edited after\n")
    arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout), BRANCH_RUN)

    assert (checkout / DOSSIER_DIR / "qwen3-8b" / "README.md").read_text() == "# the pass's\n"  # a copy there stays


def test_a_dossier_that_is_gone_is_noted_unless_the_checkout_already_has_a_copy(tmp_path: Path) -> None:
    dossier = _write(tmp_path / "pass" / "docs" / "dossier" / "qwen3-8b", {"README.md": "# the pass's\n"})
    checkout, gone = _write(tmp_path / "checkout", CHECKOUT_FILES), str(tmp_path / "gone" / "qwen3-8b")
    said = f"the dossier pass's dossier {gone} is not there: started without a dossier"

    assert dossier_notes(checkout, str(dossier)) == dossier_notes(checkout, None) == dossier_notes(checkout, "") == ()
    assert dossier_notes(checkout, gone) == (said,)
    carry_dossier(checkout, dossier)
    assert dossier_notes(checkout, gone) == (), "the copy an earlier start made is the same dossier"


async def test_a_branch_whose_dossier_is_gone_starts_without_one_and_says_so(tmp_path: Path) -> None:
    gone = str(tmp_path / "pass" / "docs" / "dossier" / "qwen3-8b")
    plan = _write(tmp_path / "shares", {"share-1.json": json.dumps(MAP)}) / "share-1.json"

    ready = await make_ready(
        str(_arbor_repo(tmp_path)),
        _spec(tmp_path, plan=str(plan), dossier=gone),
        tmp_path / "agent",
        _setup(Sandbox([Process(None)]), None),
    )

    assert ready.notes == (f"the dossier pass's dossier {gone} is not there: started without a dossier",)
    assert not (ready.source_dir / DOSSIER_DIR / "qwen3-8b").exists()


def test_a_framework_not_installed_or_with_no_layout_links_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(layout: ComponentLayout) -> PackageBase:
        raise UnsupportedComponentError(f"package {layout.import_name} is not installed in this image")

    monkeypatch.setattr(arbor_adapter, "locate_installed_package", missing)
    assert link_built(tmp_path, "vllm") == 0
    monkeypatch.setattr(arbor_adapter, "locate_installed_package", _located(tmp_path))
    assert link_built(tmp_path, "no-such-framework") == 0
    assert list(tmp_path.iterdir()) == []


def test_arbor_s_install_checks_each_import_it_needs_as_arbor_before_installing_it() -> None:
    """Each import is checked after Arbor's own, with ``ARBOR_SITE`` set as Arbor's commands have it, so a module pip
    installed out of the interpreter's reach counts as there only for an Arbor that reads it."""
    assert ARBOR_PYTHON == 'env ARBOR_SETTINGS=/dev/null ${PIP_TARGET:+"ARBOR_SITE=$PIP_TARGET"} python3'
    assert INSTALL[1:3] == (
        f"{ARBOR_PYTHON} -c 'import arbor, yaml' 2>/dev/null || python3 -m pip install --quiet pyyaml",
        f"{ARBOR_PYTHON} -c 'import arbor, claude_agent_sdk' 2>/dev/null"
        " || python3 -m pip install --quiet claude-agent-sdk",
    )
    assert "(3, 11)" in INSTALL[0] and INSTALL[-1].startswith(
        f"{ARBOR_PYTHON} -c 'import arbor, yaml, claude_agent_sdk' ||"
    )


def test_arbor_s_commands_read_no_settings_file_of_the_user_running_them(tmp_path: Path) -> None:
    """A user's ``~/.arbor/settings.env`` would set ``ARBOR_*`` values MB's preflight refuses in the environment (a
    roll, a review gate, another agent backend); Arbor's commands point the file at an empty one, which Arbor reads as
    no settings, where an empty value would fall back to the user's file."""
    shown = subprocess.run(
        ["sh", "-c", ARBOR_PYTHON.replace(" python3", " sh -c 'echo \"$ARBOR_SETTINGS\"'")],
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "ARBOR_SETTINGS": str(tmp_path / "user.env")},
        capture_output=True,
        text=True,
        check=True,
    )

    assert shown.stdout.strip() == "/dev/null"


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

    manifest = arbor_manifest(_spec(tmp_path, plan=str(plan)), ArborCheckout("arbor", "abc123", checkout), BRANCH_RUN)

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
    assert not [reason for reason in refusals(config) if "pack" in reason or "script" in reason]
    assert any("agentx-gone" in reason and "pack.yaml" in reason for reason in refusals(missing))
    assert arbor_config(spec, checkout, BRANCH_RUN)["pack"] == "serving"


#: Arbor's arbor/objective.py as far as the adapter reads it: the forms opt_objective takes. 29c72f9 added the
#: structured one; df9b55a knew both AgentX metrics by name only.
OBJECTIVE_29C72F9 = 'FORMS = ("text", "structured")\n'
OBJECTIVE_DF9B55A = 'KNOWN = {"e2e_norm_intvty_p50": ("max", ""), "total_token_throughput": ("max", "")}\n'


def _structured(objective: AgentXObjective) -> dict[str, object]:
    return {
        "metric": objective.metric,
        "bar_pct": objective.bar_pct,
        "guards": [{"metric": guard.metric, "band_pct": objective.guard_band_pct} for guard in objective.guards],
    }


def _vendored_objective(asked: Mapping[str, object]) -> dict[str, object]:
    """The vendored Arbor's reading of a structured opt_objective (arbor/objective.py's ``structured``)."""

    script = (
        "import json, sys; from arbor import objective; print(json.dumps(objective.structured(json.load(sys.stdin))))"
    )
    run = subprocess.run(
        [sys.executable, "-c", script], input=json.dumps(asked), cwd=VENDORED_ARBOR, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@pytest.mark.parametrize("objective", list(AgentXObjective))
def test_arbor_keeps_on_the_agentx_session_s_rule_its_objective_bar_and_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, objective: AgentXObjective
) -> None:
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "4")
    spec = _spec(tmp_path, workload=AGENTX_WORKLOAD, metric=objective.metric)

    asked = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)[OBJECTIVE_KEY]

    assert asked == _structured(objective)
    assert objective.bar_pct == 2.0 and objective.guard_band_pct == 4.0
    read = _vendored_objective(asked)
    assert (read["metric"], read["direction"], read["bar_pct"]) == (objective.metric, "max", objective.bar_pct)
    assert [(guard["metric"], guard["band_pct"]) for guard in read["guards"]] == [
        (guard.metric, 4.0) for guard in objective.guards
    ]


def test_a_session_on_output_throughput_keeps_its_pack_s_objective(tmp_path: Path) -> None:
    assert OBJECTIVE_KEY not in arbor_config(_spec(tmp_path), tmp_path / "checkout", BRANCH_RUN)


@pytest.mark.parametrize(
    ("objective", "refused"),
    [(OBJECTIVE_29C72F9, False), (OBJECTIVE_DF9B55A, True), (None, True)],
    ids=["29c72f9", "df9b55a", "older"],
)
@pytest.mark.parametrize("metric", [GRADED_INTVTY_P50, TOTAL_TOKEN_THROUGHPUT])
def test_an_arbor_without_the_structured_objective_is_refused_naming_29c72f9(
    tmp_path: Path, objective: str | None, refused: bool, metric: str
) -> None:
    files = {**CHECKOUT_FILES, **({OBJECTIVE_MODULE: objective} if objective else {})}
    checkout = _write(tmp_path / "checkout", files)
    launch = _launch_file(tmp_path, RESOLVED)
    run = partial(
        arbor_manifest,
        replace(_spec(tmp_path), workload=AGENTX_WORKLOAD, metric=metric, client="", launch=launch),
        ArborCheckout("arbor", "abc", checkout),
        BRANCH_RUN,
    )

    if refused:
        with pytest.raises(ArborRefused, match="has no structured opt_objective: pin Arbor 29c72f9 or later"):
            run()
    else:
        assert run().config[OBJECTIVE_KEY]["metric"] == metric


def test_arbor_runs_the_validator_scripts_its_pack_names(tmp_path: Path) -> None:
    """Arbor's agentx pack names validator/bench_agentx.sh; a config naming validator/bench.sh was refused as no file."""
    own = {
        "packs/agentx/pack.yaml": "name: agentx\nvalidator:\n  bench: validator/bench_agentx.sh\n"
        "  correctness: validator/correctness.sh\n",
        "packs/agentx/validator/bench_agentx.sh": "#!/usr/bin/env bash\n",
        "packs/agentx/validator/correctness.sh": "#!/usr/bin/env bash\n",
    }
    checkout = _write(tmp_path / "checkout", {**CHECKOUT_FILES, **own})

    config = arbor_config(replace(_spec(tmp_path), pack="agentx"), checkout, BRANCH_RUN)

    validator = checkout / "packs" / "agentx" / "validator"
    assert config["bench_script"] == str(validator / "bench_agentx.sh")
    assert config["correctness_script"] == str(validator / "correctness.sh")
    assert not [reason for reason in refusals(config) if "pack" in reason or "script" in reason]
    unread = _write(tmp_path / "other", {"packs/x/pack.yaml": "validator: [\n"})
    assert arbor_adapter.validator_script(unread / "packs" / "x", "bench", "validator/bench.sh") == "validator/bench.sh"


def test_the_python_arbor_needs_is_the_one_its_checkout_names(tmp_path: Path) -> None:
    """A checkout from Arbor 6126676 names 3.10 (serving images ship it); one without the name needs 3.11."""
    import subprocess
    import sys

    check = next(command for command in arbor_adapter.INSTALL if "PYTHON_MIN" in command)
    code = check.split(" -c ", 1)[1].strip("'")
    named = _write(tmp_path / "named", {"arbor/version.py": "PYTHON_MIN: tuple[int, int] = (3, 10)\n"})
    later = _write(tmp_path / "later", {"arbor/version.py": "PYTHON_MIN: tuple[int, int] = (9, 0)\n"})
    older = _write(tmp_path / "older", {"arbor/__init__.py": ""})

    def run(cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True)

    assert run(named).returncode == 0
    assert "Arbor needs Python 9.0 or newer" in run(later).stderr
    assert (run(older).returncode == 0) == (sys.version_info[:2] >= (3, 11))


#: A launch resolved from an InferenceX AgentX recipe, as a launch file.
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


def test_a_launch_file_flag_the_recipe_removes_is_not_served(tmp_path: Path) -> None:
    launch = _launch_file(tmp_path, RESOLVED)
    spec = _spec(tmp_path, framework="sglang", server_args="", tp=1, client="", launch=launch)
    spec = replace(spec, remove_args=("--chunked-prefill-size",))

    config = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)

    assert config["server_flags"] == "--mem-fraction-static 0.8 --tp-size 1"


def test_arbor_is_held_to_mb_s_accuracy_floor_unless_the_launch_or_the_recipe_sets_its_own(tmp_path: Path) -> None:
    # Without a floor the serving pack books every iteration whose accuracy it must judge as invalid; MB's eval and the
    # pack's score one model points apart, so the floor is MB's absolute one, not its baseline less its tolerance.
    floor, spec = arbor_adapter.ACCURACY_FLOOR_ENV, _spec(tmp_path)
    given = replace(spec, launch=_launch_file(tmp_path, {**RESOLVED, "server_env": {floor: "0.85"}}))
    recipe = replace(spec, envs={**spec.envs, floor: "0.8"})

    configs = [arbor_config(each, tmp_path / "checkout", BRANCH_RUN) for each in (spec, given, recipe)]

    assert [config["server_env"][floor] for config in configs] == ["0.5", "0.85", "0.8"]


def test_arbor_is_given_mb_s_rules_as_directives_the_dossier_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The benchmark's rules are considered in validation, and are part of the dossier.
    monkeypatch.delenv("HYPERLOOM_PERF_NOISE_PCT", raising=False)
    spec = _spec(tmp_path)
    agentx = replace(spec, workload={"kind": "agentx", "conc": 8}, metric=GRADED_INTVTY_P50)

    rules = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)["directives"]
    agentx_rules = arbor_config(agentx, tmp_path / "checkout", BRANCH_RUN)["directives"]

    assert all(rule.startswith(arbor_adapter.HOST_RULE) for rule in (*rules, *agentx_rules))
    fixed, kept, accuracy = rules
    assert "inferencex at concurrency 8, isl 1024, osl 1024" in fixed and "ISL" in fixed and "CONC" in fixed
    assert "output_throughput" in kept and "a second measurement confirms" in kept
    agentx_kept = agentx_rules[1]
    assert "in one measurement, which is final unless a guard" in agentx_kept and "second" not in agentx_kept
    assert "bar of +2%" in agentx_kept and "output" not in agentx_kept
    assert "e2e_norm_intvty_p90 and total_token_throughput within 5%" in agentx_kept
    assert "at least 0.5" in accuracy and "at most 0.05" in accuracy
    assert "the root it measures itself" in accuracy
    assert len(agentx_rules) == 3 and "error rate stays at or under 10%" in agentx_rules[2]
    assert "long-context check drop below the root's is never merged" in agentx_rules[2]
    assert not [rule for rule in agentx_rules if "agentx." in rule or "run_status" in rule], "bench_read holds both"
    total = host_rules(replace(agentx, metric=TOTAL_TOKEN_THROUGHPUT))
    assert "e2e_norm_intvty_p90 and e2e_norm_intvty_p50 within 5%" in total[1] and "output" not in total[1]
    assert not [rule for rule in total if "agentx." in rule]
    unmeasured = host_rules(replace(agentx, client=""))
    assert len(unmeasured) == 3 and "error rate stays at or under 10%" in unmeasured[-1]


#: MB's AgentX round as the gate stages it, enough of it for the adapter.
AGENTX_CLIENT = "python -m inference_suite run --config /s/mb-bench.yaml --run-dir /s/mb-bench"


@pytest.mark.parametrize("metric", [GRADED_INTVTY_P50, TOTAL_TOKEN_THROUGHPUT])
def test_an_agentx_arbor_given_mb_s_round_has_its_harness_read_the_round_s_report_by_path(
    tmp_path: Path, metric: str
) -> None:
    spec = _spec(tmp_path, workload=AGENTX_WORKLOAD, metric=metric, client=AGENTX_CLIENT)

    config = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)

    read = config["component"]["bench_read"]
    assert read == {
        "file": "report.json",
        "keys": {
            "e2e_norm_intvty_p50": "agentx.e2e_norm_intvty.p50",
            "total_token_throughput": "agentx.throughput.total_tokens_per_second",
            "e2e_norm_intvty_p90": "agentx.e2e_norm_intvty.p90",
            "output_throughput": "agentx.throughput.output_tokens_per_second",
        },
        "require": {"run_status": "completed", "agentx.stopped_by_threshold": False, "agentx.fast_mode": False},
    }
    asked = config[OBJECTIVE_KEY]
    assert {asked["metric"], *(guard["metric"] for guard in asked["guards"])} <= set(read["keys"])
    assert _arbor_config_errors(config) == []


def _agentx_report(status: RunStatus, stopped_by_threshold: bool = False) -> dict[str, object]:
    """A round's report.json, its AgentX section as the round writes it, with the numbers a stopped round keeps."""

    agentx = AgentxResult(
        counts={**dict.fromkeys(AGENTX_COUNT_KEYS, 10), "dropped_total": 0},
        throughput={**dict.fromkeys(AGENTX_THROUGHPUT_KEYS, 100.0), "total_tokens_per_second": 9000.0},
        latency_seconds=dict.fromkeys(AGENTX_LATENCY_KEYS, 0.5),
        e2e_norm_intvty={"p50": 30.0, "p90": 20.0},
        stopped_by_threshold=stopped_by_threshold,
        fast_mode=False,
    )
    return {"run_status": status.value, "agentx": agentx.to_document()}


def _read_report(read: Mapping[str, object], metric: str, report: Mapping[str, object], out: Path) -> dict:
    """What the vendored Arbor's bench_read block (arbor/policy/claude_loop.py) books for one replay's ``report``."""

    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report))
    dest = out / "bench.json"
    script = (
        "import json, sys; from arbor.policy import claude_loop; a = json.load(sys.stdin);"
        " print(claude_loop._read_block(a['out'], a['dest'], a['read'], [a['metric']]))"
    )
    asked = {"out": str(out), "dest": str(dest), "read": read, "metric": metric}
    block = subprocess.run(
        [sys.executable, "-c", script], input=json.dumps(asked), cwd=VENDORED_ARBOR, capture_output=True, text=True
    )
    assert block.returncode == 0, block.stderr
    subprocess.run(["bash", "-c", block.stdout], cwd=out, check=True, capture_output=True)
    return json.loads(dest.read_text())


@pytest.mark.parametrize("metric", [GRADED_INTVTY_P50, TOTAL_TOKEN_THROUGHPUT])
def test_arbor_s_harness_books_a_completed_round_and_not_one_mb_s_gate_refuses(tmp_path: Path, metric: str) -> None:
    config = arbor_config(
        _spec(tmp_path, workload=AGENTX_WORKLOAD, metric=metric, client=AGENTX_CLIENT), tmp_path / "c", BRANCH_RUN
    )
    read = config["component"]["bench_read"]

    completed = _read_report(read, metric, _agentx_report(RunStatus.COMPLETED), tmp_path / "completed")
    partial = _read_report(read, metric, _agentx_report(RunStatus.PARTIAL), tmp_path / "partial")
    stopped = _read_report(read, metric, _agentx_report(RunStatus.COMPLETED, True), tmp_path / "stopped")

    assert completed == {
        "status": "ok",
        "source": "bench_command",
        "e2e_norm_intvty_p50": 30.0,
        "e2e_norm_intvty_p90": 20.0,
        "total_token_throughput": 9000.0,
        "output_throughput": 100.0,
    }
    assert partial == {"status": "failed", "error": 'bench_read: run_status is "partial", not "completed"'}
    assert stopped["status"] == "failed" and "agentx.stopped_by_threshold is true, not false" in stopped["error"]


def test_only_an_agentx_arbor_measured_by_mb_s_round_has_its_report_read_by_path(tmp_path: Path) -> None:
    inferencex = _spec(tmp_path, client=AGENTX_CLIENT)
    agentx = replace(inferencex, workload=AGENTX_WORKLOAD, metric=GRADED_INTVTY_P50)
    specs = [inferencex, replace(agentx, client="")]

    components = [arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)["component"] for spec in specs]

    assert [("bench_command" in each, "bench_read" in each) for each in components] == [
        (True, False),
        (False, False),
    ]


def test_arbor_is_told_its_source_changes_cannot_be_kept_when_the_spec_says_why(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    unkept = replace(spec, source_unkept="vllm is installed from a wheel")

    rules = arbor_config(unkept, tmp_path / "checkout", BRANCH_RUN)["directives"]

    assert rules[:-1] == host_rules(spec)
    assert rules[-1].startswith(arbor_adapter.HOST_RULE)
    assert "vllm is installed from a wheel" in rules[-1] and "launch flags" in rules[-1]


def test_arbor_s_accuracy_tolerance_is_the_drop_mb_s_rule_allows(tmp_path: Path) -> None:
    # Arbor measures the root's full GSM8K itself and holds a numerics change to it less this tolerance, on its own
    # eval's scale: MB's rule allows ACCURACY_THRESHOLD under the root, so Arbor allows the same.
    config = arbor_config(_spec(tmp_path), tmp_path / "checkout", BRANCH_RUN)

    assert config["policy"] == {"accuracy_tolerance": 0.05}


def test_an_agentx_branch_served_no_launch_is_refused_before_arbor_starts(tmp_path: Path) -> None:
    checkout = _write(tmp_path / "checkout", {**CHECKOUT_FILES, OBJECTIVE_MODULE: OBJECTIVE_29C72F9})
    spec = _spec(tmp_path, framework="sglang", server_args="", workload=AGENTX_WORKLOAD, metric=GRADED_INTVTY_P50)

    with pytest.raises(ArborRefused, match="served no launch to add it to"):
        arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout), BRANCH_RUN)


@pytest.fixture
def agentx_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FrozenBase:
    """An AgentX session's frozen base, its resolution from the recipe sources standing in as :func:`base_record`."""

    monkeypatch.setattr(cli_base_launch, "resolve_server_base", lambda _: base_record())
    workload = Workload(MODEL_ID, "vllm", 2, 8, "fp8", "mi355x", KernelOptimizer.FORGE, None)
    return freeze_session_base(tmp_path / "toolbox", workload, "")


def test_an_agentx_arbor_given_no_launch_is_served_the_session_s_frozen_base_and_its_start_check_passes(
    tmp_path: Path, agentx_session: FrozenBase
) -> None:
    checkout = _write(tmp_path / "checkout", {**CHECKOUT_FILES, OBJECTIVE_MODULE: OBJECTIVE_29C72F9})
    served = served_launch(agentx_session, "")
    spec = replace(
        _spec(tmp_path, launch=served, workload=AGENTX_WORKLOAD, metric=GRADED_INTVTY_P50), server_args="", envs={}
    )

    arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout), BRANCH_RUN)
    config = arbor_config(spec, checkout, BRANCH_RUN)

    base = server_base(agentx_session.launch)
    assert _shell_argv(config["server_flags"]) == list(server_argv(agentx_session.launch))
    assert config["server_env"] == {arbor_adapter.ACCURACY_FLOOR_ENV: "0.5", **base.env}
    assert config["component"]["base_launch"] == {
        "path": str(agentx_session.path),
        "digest": agentx_session.launch.digest(),
    }
    assert _arbor_start_check(config) == []


#: A credential a reference script exports, which the session freezes as its sha256.
_TOKEN = "hf_frozen_token_value"


@pytest.fixture
def credential_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FrozenBase:
    """An AgentX session whose base exports ``HF_TOKEN``, frozen as the session freezes a resolved base."""

    exported = base_record(env={**base_record().env, "HF_TOKEN": _TOKEN})
    monkeypatch.setattr(cli_base_launch, "resolve_server_base", lambda _: redact_credential_env(exported))
    workload = Workload(MODEL_ID, "vllm", 2, 8, "fp8", "mi355x", KernelOptimizer.FORGE, None)
    return freeze_session_base(tmp_path / "toolbox", workload, "")


def _credential_spec(tmp_path: Path, frozen: FrozenBase) -> AgentSpec:
    served = served_launch(frozen, "")
    return replace(
        _spec(tmp_path, launch=served, workload=AGENTX_WORKLOAD, metric=GRADED_INTVTY_P50), server_args="", envs={}
    )


def test_a_frozen_base_s_credential_is_served_to_arbor_from_the_host_and_its_start_check_passes(
    tmp_path: Path, credential_session: FrozenBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", _TOKEN)

    config = arbor_config(_credential_spec(tmp_path, credential_session), tmp_path / "checkout", BRANCH_RUN)

    assert isinstance(credential_session.launch.env["HF_TOKEN"], Redacted)
    assert config["server_env"]["HF_TOKEN"] == _TOKEN
    assert config["component"]["base_launch"]["path"] == str(credential_session.path)
    assert _arbor_start_check(config) == []


@pytest.mark.parametrize(
    ("held", "said"),
    [
        (None, "the host holds no value of it"),
        ("hf_another_token", "the host's value is not the one the base was resolved with"),
    ],
    ids=["unset", "another-value"],
)
def test_a_frozen_base_s_credential_the_host_does_not_hold_is_refused_before_arbor_starts(
    tmp_path: Path, credential_session: FrozenBase, monkeypatch: pytest.MonkeyPatch, held: str | None, said: str
) -> None:
    if held is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", held)

    with pytest.raises(ArborRefused, match=f"HF_TOKEN is redacted in the base and {said}"):
        arbor_config(_credential_spec(tmp_path, credential_session), tmp_path / "checkout", BRANCH_RUN)


def test_an_agentx_arbor_given_a_launch_file_is_served_it_over_the_frozen_base(
    tmp_path: Path, agentx_session: FrozenBase
) -> None:
    checkout = _write(tmp_path / "checkout", {**CHECKOUT_FILES, OBJECTIVE_MODULE: OBJECTIVE_29C72F9})
    served = served_launch(agentx_session, _launch_file(tmp_path, RESOLVED))
    spec = _spec(
        tmp_path, framework="sglang", server_args="", launch=served, workload=AGENTX_WORKLOAD, metric=GRADED_INTVTY_P50
    )

    manifest = arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout), BRANCH_RUN)

    assert manifest.config["server_flags"] == f"{RESOLVED['server_flags']} --tp-size 2"
    assert "base_launch" not in manifest.config["component"]


@pytest.mark.parametrize(
    ("document", "said"),
    [
        ("{", "cannot be read"),
        (["--tp 1"], "is not a JSON object"),
        ({"server_env": {}}, "no server_flags"),
        ({**RESOLVED, "server_flags": "--chat-template 'a b"}, "a quote is left open"),
        ({**RESOLVED, "server_env": {"N": 1}}, "server_env is not an object of strings"),
        ({**RESOLVED, "flags": "--tp 1", "model_name": 7}, "keys a launch does not: flags; its model_name is not"),
        ({**RESOLVED, "client": ""}, "keys a launch does not: client"),
    ],
    ids=[
        "unreadable",
        "a-list",
        "no-flags",
        "open-quote",
        "env-not-strings",
        "unknown-key-and-name",
        "a-client-the-gate-gives",
    ],
)
def test_a_launch_file_the_adapter_cannot_use_is_refused_with_why(tmp_path: Path, document: object, said: str) -> None:
    path = Path(_launch_file(tmp_path, document))
    if isinstance(document, str):
        path.write_text(document, encoding="utf-8")

    with pytest.raises(ArborRefused) as refused:
        read_launch(path)

    assert said.split("; ")[0] in str(refused.value) and said.split("; ")[-1] in str(refused.value)


def test_the_repository_s_launch_template_has_a_launch_file_s_keys_and_is_refused_until_filled(tmp_path: Path) -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    filled = {**RESOLVED, "server_flags": f"{RESOLVED['server_flags']} --reasoning-start-str <think>"}

    with pytest.raises(ArborRefused) as refused:
        read_launch(TEMPLATE)

    assert set(template) == set(LAUNCH_KEYS)
    assert "server_flags, server_env, model_name still hold the template's placeholders" in str(refused.value)
    assert read_launch(Path(_launch_file(tmp_path, filled))).server_flags == filled["server_flags"]


#: Where the repository vendors Arbor, whose start check a record's config must pass.
VENDORED_ARBOR = Path(__file__).parents[2] / "hyperloom_arbor"


def _arbor_start_check(config: Mapping[str, object]) -> list[str]:
    """What the vendored Arbor's own start check (arbor/base_launch.py) refuses ``config`` for."""

    script = (
        "import json, sys; from arbor import base_launch; print(json.dumps(base_launch.check(json.load(sys.stdin))[1]))"
    )
    run = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(config),
        cwd=VENDORED_ARBOR,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(run.stdout)


def test_a_base_launch_record_is_served_as_the_gate_seeds_its_launch_with_its_golden_layer(tmp_path: Path) -> None:
    record = write_record(tmp_path / "record" / "base_launch.json")

    launch = read_launch(record)

    assert launch.server_flags == (
        "--tensor-parallel-size 2 --max-num-seqs 128 --chat-template 'a b'"
        """ --speculative-config '{"method":"mtp","num_speculative_tokens":3}'"""
    )
    assert launch.server_env == {"VLLM_ROCM_USE_AITER": "1", "VLLM_SIMULATE_ACC": "3"}
    assert (launch.record, launch.digest, launch.entrypoint) == (str(record), base_record().digest(), ("vllm", "serve"))
    assert launch.model_name == ""


def _shell_argv(flags: str) -> list[str]:
    """The words a POSIX shell passes a server launched with ``flags``, as Arbor launches one through bash."""

    script = f"{shlex.quote(sys.executable)} -c 'import json, sys; print(json.dumps(sys.argv[1:]))' {flags}"
    return json.loads(subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout)


def _arbor_split(flags: str) -> list[str]:
    """``flags`` as the vendored Arbor's ``split_flags`` (arbor/base_launch.py), the split its start check reads."""

    script = "import json, sys; from arbor import base_launch; print(json.dumps(base_launch.split_flags(sys.argv[1])))"
    run = subprocess.run(
        [sys.executable, "-c", script, flags], cwd=VENDORED_ARBOR, capture_output=True, text=True, check=True
    )
    return json.loads(run.stdout)


#: The record's server argv, its JSON value compact, as the server is to receive it.
RECORD_ARGV = [
    *("--tensor-parallel-size", "2", "--max-num-seqs", "128", "--chat-template", "a b"),
    *("--speculative-config", '{"method":"mtp","num_speculative_tokens":3}'),
]


@pytest.mark.parametrize(
    ("recipe", "argv"),
    [
        ("", []),
        ('--compilation-config {"level": 3}', ["--compilation-config", '{"level":3}']),
        ('--override-generation-config={"t":1}', ['--override-generation-config={"t":1}']),
        ('--served-model-name="m 1"', ["--served-model-name=m 1"]),
        ('--override-generation-config {"stop": ["a b"]}', ["--override-generation-config", '{"stop":["a b"]}']),
    ],
)
def test_a_flag_reaches_the_server_arbor_launches_through_a_shell_as_its_words(
    tmp_path: Path, recipe: str, argv: list[str]
) -> None:
    record = write_record(tmp_path / "record" / "base_launch.json")
    spec = replace(_spec(tmp_path, launch=str(record)), server_args=recipe, envs={})

    flags = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)["server_flags"]

    assert _shell_argv(flags) == [*RECORD_ARGV, *argv]
    assert _arbor_split(flags) == _shell_argv(flags)


def test_arbor_s_own_start_check_passes_a_branch_served_a_record_on_the_base(tmp_path: Path) -> None:
    # The record names no served model and no client: the gate's client gives both.
    record = write_record(tmp_path / "record" / "base_launch.json")
    spec = replace(_spec(tmp_path, launch=str(record)), server_args="", envs={})

    config = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)

    assert config["model_name"] == "Qwen/Qwen3-14B-FP8"
    assert config["component"]["bench_command"] == bench_command(spec.client)
    assert config["component"]["base_launch"]["path"] == str(record)
    assert _arbor_start_check(config) == []


@pytest.mark.parametrize(
    "layer",
    [
        {"server_args": "--enable-chunked-prefill"},
        {"server_args": "--max-num-seqs 256"},
        {"envs": {"VLLM_ROCM_USE_AITER": "0"}},
        {"remove_args": ("--max-num-seqs",)},
    ],
    ids=["new-flag", "changed-flag", "changed-env", "removed-flag"],
)
def test_a_branch_whose_root_recipe_adds_to_the_record_starts_without_the_base_check(
    tmp_path: Path, layer: dict[str, object]
) -> None:
    # A root a Recipe KB champion was replayed onto serves more than the base; Arbor's check compares the whole launch.
    record = write_record(tmp_path / "record" / "base_launch.json")
    spec = replace(_spec(tmp_path, launch=str(record)), **{"server_args": "", "envs": {}, **layer})

    config = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)

    assert "base_launch" not in config["component"]
    assert _arbor_start_check(config) == []


def test_a_golden_layer_s_token_mode_is_served_to_arbor_as_the_root_runs_it(tmp_path: Path) -> None:
    simulated = {
        "SGLANG_SIMULATE_ACC_LEN": "3.48",
        "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
    }
    base = base_record()
    base = base_record(golden_al=replace(base.golden_al, env=simulated))
    record = write_record(tmp_path / "record" / "base_launch.json", base)
    spec = replace(_spec(tmp_path, launch=str(record)), server_args="", envs={})

    config = arbor_config(spec, tmp_path / "checkout", BRANCH_RUN)

    assert read_launch(record).server_env == {**base.env, **simulated}
    assert _arbor_start_check(config) == []


def test_a_base_launch_record_that_cannot_be_used_is_refused_with_why(tmp_path: Path) -> None:
    tampered = write_record(tmp_path / "tampered.json", args=["--max-num-seqs", "64"])
    scriptable = base_record(
        query=replace(base_record().query, workload=LaunchWorkload.SCRIPTABLE, concurrencies=(8,), tp=None),
        entrypoint=("bash", "bench.sh"),
        source=replace(base_record().source, files={"bench.sh": "c" * 64}),
        args=(),
        env={},
        golden_al=None,
        client=None,
    )
    script = write_record(tmp_path / "script.json", scriptable)

    with pytest.raises(ArborRefused, match="base_digest_mismatch"):
        read_launch(tampered)
    with pytest.raises(ArborRefused, match="runs a script and launches no server"):
        read_launch(script)


@pytest.mark.parametrize(
    ("plan", "said"),
    [("shares/share-1.json", "not an absolute path"), ("/nonexistent/share-1.json", "cannot be read")],
)
def test_a_plan_arbor_could_not_read_is_refused(tmp_path: Path, plan: str, said: str) -> None:
    config = arbor_config(_spec(tmp_path, plan=plan), tmp_path / "checkout", BRANCH_RUN)

    assert any(said in reason for reason in refusals(config))


def test_the_dossier_pass_stops_at_the_map_in_a_session_of_its_own_and_takes_no_plan(tmp_path: Path) -> None:
    spec = _spec(tmp_path, plan="/shares/share-1.json")
    checkout = _write(tmp_path / "checkout", CHECKOUT_FILES)

    manifest = arbor_manifest(spec, ArborCheckout("arbor", "abc123", checkout), DOSSIER_PASS)

    component = manifest.config["component"]
    assert component["stop_at"] == "map" and "plan" not in component and "profile" not in component
    assert (checkout / DOSSIER_PASS.session_dir).is_dir()
    assert manifest.config_file == DOSSIER_PASS.config_file and manifest.snapshots is not None
    assert manifest.snapshots.session == DOSSIER_PASS.session_dir != SESSION_DIR


def test_what_arbor_refuses_at_its_start_is_refused_before_it_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, deadline=0.0, root="f" * 40)
    config = arbor_config(spec, tmp_path / "empty-checkout", BRANCH_RUN)
    config = {**config, "docker_container": "arbor-c", "component": {**config["component"], "x": 1}}

    monkeypatch.setenv("ARBOR_ROLL", "1")
    reasons = refusals(config)

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
    assert not [reason for reason in refusals(config) if "component" in reason and "review" in reason]


def test_a_review_by_anyone_but_the_host_is_refused(tmp_path: Path) -> None:
    config = arbor_config(_spec(tmp_path), tmp_path / "checkout", BRANCH_RUN)
    config = {**config, "component": {**config["component"], "review": "operator"}}

    assert "component.review is 'operator', not 'host'" in refusals(config)


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


async def test_an_arbor_given_why_no_source_change_can_be_kept_is_told_so_and_made_ready_as_told(
    tmp_path: Path,
) -> None:
    spec = replace(_spec(tmp_path), source_unkept="the image's vllm names no upstream commit")

    ready = await make_ready(
        str(_arbor_repo(tmp_path)), spec, tmp_path / "agent", _setup(Sandbox([Process(None)]), None)
    )

    directives = json.loads((ready.source_dir / CONFIG_FILE).read_text())["directives"]
    assert any(spec.source_unkept in rule for rule in directives)
    assert ready.source_unkept == spec.source_unkept


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


@pytest.mark.parametrize(
    ("content", "said"),
    [
        (None, "it wrote no exit reason"),
        ("{", "its exit reason cannot be read: "),
        (json.dumps({"reason": ""}), "its exit file names no reason: {'reason': ''}"),
        (json.dumps({"reason": "deadline"}), "deadline"),
    ],
)
def test_arbor_s_exit_reason_is_its_exit_file_s_or_what_is_wrong_with_the_file(
    tmp_path: Path, content: str | None, said: str
) -> None:
    path = tmp_path / "component_exit.json"
    if content is not None:
        path.write_text(content)

    assert exit_reason(path).startswith(said)


async def test_a_map_is_split_into_one_plan_per_share_by_arbor_s_command_line(tmp_path: Path) -> None:
    checkout = _write(tmp_path / "checkout", CHECKOUT_FILES)
    map_file = _write(tmp_path / "pass", {"directions.json": json.dumps(MAP)}) / "directions.json"

    shares = await split_directions(checkout, map_file, 2, tmp_path / "shares")

    assert shares == (tmp_path / "shares" / "share-1.json", tmp_path / "shares" / "share-2.json")
    dealt = [[entry["id"] for entry in json.loads(share.read_text())["directions"]] for share in shares]
    assert dealt == [["m-attn", "m-allreduce"], ["m-moe"]]
    with pytest.raises(SplitFailed, match="3 independent groups for 4 shares"):
        await split_directions(checkout, map_file, 4, tmp_path / "four")


async def test_arbor_s_directions_command_reads_no_settings_file_of_the_user_running_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arbor reads its settings file on import (``arbor/__init__.py``); this one refuses any but the empty one.
    reads = (
        'import os, sys\nif os.environ.get("ARBOR_SETTINGS") != "/dev/null":\n    sys.exit("read a settings file")\n'
    )
    checkout = _write(tmp_path / "checkout", {**CHECKOUT_FILES, "arbor/__init__.py": reads})
    map_file = _write(tmp_path / "pass", {"directions.json": json.dumps(MAP)}) / "directions.json"
    monkeypatch.setenv("ARBOR_SETTINGS", str(tmp_path / "user.env"))

    shares = await split_directions(checkout, map_file, 2, tmp_path / "shares")

    assert len(shares) == 2


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


def test_a_workspace_its_agent_made_a_partial_clone_fetches_nothing_as_its_root_is_checked(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    workspace, ran, transport = Path(spec.workspace), tmp_path / "ran", tmp_path / "transport"
    transport.write_text(f"#!/bin/sh\necho ran >> {ran}\nexit 1\n")
    transport.chmod(0o755)
    with (workspace / ".git" / "config").open("a") as config:
        config.write(
            f"[core]\n\trepositoryformatversion = 1\n\tsshCommand = {transport}\n"
            '[extensions]\n\tpartialClone = promisor\n[remote "promisor"]\n\turl = ssh://example.invalid/r\n'
            "\tpromisor = true\n"
        )
    missing = "f" * 40

    assert arbor_adapter._has_commit(workspace, spec.root)
    assert not arbor_adapter._has_commit(workspace, missing)
    assert arbor_adapter._shipped(workspace) == frozenset()

    assert not ran.exists(), ran.read_text()
    subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{missing}^{{commit}}"], cwd=workspace, check=False)
    assert ran.exists(), "plain git fetches the missing object through the configured transport, so the plant is live"
