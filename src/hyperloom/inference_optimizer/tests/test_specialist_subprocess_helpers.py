# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for ``specialists.subprocess_`` helpers: worktree pick/setup, claude argv assembly, patch discovery, and
done-file parse/unwrap.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.specialists import patch_safety as ps
from hyperloom.orchestrator.specialists import subprocess_ as ss
from hyperloom.orchestrator.specialists.subprocess_ import (
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
    _setup_worktree,
)
from hyperloom.orchestrator.specialists.tree_snapshot import snapshot_tree


class _CP:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# -- _setup_worktree -------------------------------------------------------
def test_setup_worktree_reuses_existing(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    wt.mkdir()
    out, err = _setup_worktree(tmp_path, wt, "branch-x")
    assert out == wt and err == ""


def test_setup_worktree_spawn_failure(tmp_path: Path, monkeypatch) -> None:
    def _boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(ss.subprocess, "run", _boom)
    out, err = _setup_worktree(tmp_path, tmp_path / "wt2", "b")
    assert out is None and "failed to spawn" in err


def test_setup_worktree_nonzero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ss.subprocess, "run", lambda *a, **k: _CP(1, "", "fatal: oops"))
    out, err = _setup_worktree(tmp_path, tmp_path / "wt3", "b")
    assert out is None and "rc=1" in err


def test_setup_worktree_success(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ss.subprocess, "run", lambda *a, **k: _CP(0))
    target = tmp_path / "wt4"
    out, err = _setup_worktree(tmp_path, target, "b")
    assert out == target and err == ""


# -- _build_claude_cmd -----------------------------------------------------
def _dispatcher(**cfg_over: Any) -> SpecialistSubprocessDispatcher:
    cfg = SpecialistSubprocessConfig(**cfg_over)
    return SpecialistSubprocessDispatcher(cfg)


def test_build_claude_cmd_full(tmp_path: Path) -> None:
    fw = tmp_path / "fw"
    fw.mkdir()
    d = _dispatcher(
        model="claude-opus-4-7",
        mcp_config_path="/cfg/mcp.json",
        framework_source_roots=(str(fw),),
        extra_claude_args=("--foo", "bar"),
    )
    wt = tmp_path / "wt"
    wt.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    sys_file = ws / "system_prompt.md"
    cmd = d._build_claude_cmd(
        system_prompt_file=sys_file,
        system_prompt="SYS",
        workspace=ws,
        worktree=wt,
        disallowed_tools=frozenset({"KillShell", "SlashCommand"}),
    )
    assert "--model" in cmd and "claude-opus-4-7" in cmd
    assert cmd[cmd.index("--system-prompt-file") + 1] == str(sys_file)
    assert "--mcp-config" in cmd and "/cfg/mcp.json" in cmd
    assert "--allowedTools" not in cmd
    assert "-p" not in cmd
    deny_idx = cmd.index("--disallowedTools") + 1
    denied = set(cmd[deny_idx].split(","))
    assert "KillShell" in denied and "SlashCommand" in denied
    # --add-dir grants writes; integrate_patch is the only writer of source.
    assert str(wt) in cmd and str(ws) in cmd
    assert str(fw) not in cmd
    assert cmd[-2:] == ["--foo", "bar"]


def test_build_claude_cmd_minimal_no_model_no_mcp(tmp_path: Path) -> None:
    d = _dispatcher()
    ws = tmp_path / "ws"
    ws.mkdir()
    sys_file = ws / "system_prompt.md"
    cmd = d._build_claude_cmd(
        system_prompt_file=sys_file,
        system_prompt="SYS",
        workspace=ws,
        worktree=None,
    )
    assert "--model" not in cmd
    assert "--mcp-config" not in cmd
    assert "--allowedTools" not in cmd
    assert "-p" not in cmd
    assert "--agents" in cmd


def test_build_claude_cmd_injects_leaf_agents_when_task_allowed(tmp_path: Path) -> None:

    from hyperloom.orchestrator.specialists.leaf import LEAF_AGENT_NAME

    d = _dispatcher()
    ws = tmp_path / "ws"
    ws.mkdir()
    sys_file = ws / "system_prompt.md"
    cmd = d._build_claude_cmd(
        system_prompt_file=sys_file,
        system_prompt="SYS",
        workspace=ws,
        worktree=None,
    )
    agents_idx = cmd.index("--agents") + 1
    agents = json.loads(cmd[agents_idx])
    assert LEAF_AGENT_NAME in agents
    assert "Task" not in agents[LEAF_AGENT_NAME]["tools"]


# -- _collect_patches ------------------------------------------------------
def test_collect_patches(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    (wt / "patches").mkdir(parents=True)
    (wt / "patches" / "a.patch").write_text("p", encoding="utf-8")
    (wt / "patches" / "b.diff").write_text("d", encoding="utf-8")
    (wt / "patches" / "ignore.txt").write_text("x", encoding="utf-8")
    ws = tmp_path / "ws"
    out, roots = SpecialistSubprocessDispatcher._collect_patches(wt, ws)
    names = sorted(Path(p).name for p in out)
    assert names == ["a.patch", "b.diff"]
    assert roots == {}


def test_collect_patches_none_worktree(tmp_path: Path) -> None:
    assert SpecialistSubprocessDispatcher._collect_patches(None, tmp_path) == ([], {})


def _make_harvest_worktree(tmp_path: Path, *, with_source: bool = True) -> tuple[Path, Path]:
    base = tmp_path / "base"
    base.mkdir()
    subprocess.run(["git", "init", "-q", str(base)], check=True)
    subprocess.run(["git", "-C", str(base), "config", "core.autocrlf", "false"], check=True)
    if with_source:
        (base / "runtime.py").write_text("# Runtime selection.\nBLOCK_SIZE = 64\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(base), "add", "runtime.py"], check=True)
    subprocess.run(
        ["git", "-C", str(base), "-c", "user.email=a@b", "-c", "user.name=t", "commit", "--allow-empty", "-qm", "base"],
        check=True,
    )
    wt = tmp_path / "worktree"
    subprocess.run(["git", "-C", str(base), "worktree", "add", "--detach", str(wt)], check=True)
    return base, wt


def _write_harvest_work_artifacts(wt: Path) -> dict[str, bytes]:
    # The dispatcher consumes done files here; the rebench prompt names scratch/rebench.
    artifacts = {
        "specialist_done.json": b'{"empty": true, "summary": "probe complete"}\n',
        "specialist_done.partial.json": b'{"summary": "probe in progress"}\n',
        "scratch/rebench/notes.md": b"# Probe notes\nNo installable change here.\n",
        "scratch/rebench/probe.py": b"print('one-off probe')\n",
        "scratch/rebench/specialist_rebench.with_envs.yaml": b"server_port: 31000\n",
        "scratch/rebench/process.log": b"probe completed\n",
        "artifacts/runtime.yaml": b"block_size: 128\n",
        ".hyperloom/state.json": b'{"status": "probing"}\n',
        "__pycache__/runtime.cpython-312.pyc": b"\x00\x00cached bytecode\n",
    }
    for rel, content in artifacts.items():
        target = wt / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return artifacts


def test_collect_patches_excludes_work_artifacts_from_mixed_source_harvest(tmp_path: Path) -> None:
    base, wt = _make_harvest_worktree(tmp_path)
    artifacts = _write_harvest_work_artifacts(wt)
    installable = {
        "runtime.py": "# Runtime selection.\nBLOCK_SIZE = 128\n",
        "kernels/new_kernel.py": "def block_size():\n    return 128\n",
        "configs/runtime.json": '{"block_size": 128}\n',
        "scratch/runtime.json": '{"block_size": 128}\n',
        "docs/runtime.md": "# Runtime configuration\nThe tuned block size is 128.\n",
    }
    for rel, content in installable.items():
        target = wt / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    patches, roots = SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base)

    assert len(patches) == 1
    patch_text = Path(patches[0]).read_text(encoding="utf-8")
    assert roots == {patches[0]: str(base)}
    assert set(ps.parse_patch_targets(patch_text).all) == set(installable)
    assert "__pycache__/" not in patch_text
    assert ps.ground_patch_text(patch_text, base_checkout=base, explicit_root=base).verdict == ps.GROUND_APPLIES
    for rel, content in artifacts.items():
        assert (wt / rel).read_bytes() == content


def test_collect_patches_work_artifacts_only_produces_no_delivery(tmp_path: Path) -> None:
    base, wt = _make_harvest_worktree(tmp_path, with_source=False)
    artifacts = _write_harvest_work_artifacts(wt)

    assert SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base) == ([], {})
    assert not (wt / "patches" / "_worktree_diff.patch").exists()
    for rel, content in artifacts.items():
        assert (wt / rel).read_bytes() == content


def test_collect_patches_heartbeat_in_worktree_produces_no_delivery(tmp_path: Path) -> None:
    """A heartbeat the agent writes at the worktree top is its liveness signal, not a change to the tree."""
    base, wt = _make_harvest_worktree(tmp_path)
    heartbeat = wt / "heartbeat.json"
    heartbeat.write_text('{"ts": "2026-09-28T19:12:00Z", "status": "running"}\n', encoding="utf-8")

    assert SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base) == ([], {})
    assert not (wt / "patches" / "_worktree_diff.patch").exists()
    subprocess.run(["git", "-C", str(wt), "add", "-N", "heartbeat.json"], check=True)
    diff = subprocess.run(["git", "-C", str(wt), "diff", "HEAD"], check=True, capture_output=True, text=True).stdout
    assert ps.patch_work_artifact_targets(diff) == ("heartbeat.json",)


def test_collect_patches_annotation_only_produces_no_delivery(tmp_path: Path) -> None:
    base, wt = _make_harvest_worktree(tmp_path)
    (wt / "runtime.py").write_text("# Runtime selection, unchanged.\nBLOCK_SIZE = 64\n", encoding="utf-8")

    assert SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base) == ([], {})
    assert not (wt / "patches" / "_worktree_diff.patch").exists()


def test_collect_patches_fallback_cannot_deliver_work_artifact_patch(tmp_path: Path) -> None:
    base, wt = _make_harvest_worktree(tmp_path)
    probe = wt / "scratch" / "rebench" / "newprobe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("print('one-off probe')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(wt), "add", "-N", "scratch/rebench/newprobe.py"], check=True)
    diff = subprocess.run(["git", "-C", str(wt), "diff", "HEAD"], check=True, capture_output=True, text=True).stdout
    manual = wt / "patches" / "probe.patch"
    manual.parent.mkdir()
    manual.write_text(diff, encoding="utf-8")
    assert ps.ground_patch_text(diff, base_checkout=base, explicit_root=base).verdict == ps.GROUND_APPLIES

    patches, roots = SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base)
    kept, _dropped, _grounding, _spans_roots = ps.vet_patches(patches, base_checkout=base, explicit_root=base)

    assert kept == []
    assert not roots
    assert probe.read_text(encoding="utf-8") == "print('one-off probe')\n"
    assert manual.read_text(encoding="utf-8") == diff


def _installed_package(tmp_path: Path) -> Path:
    """A pip-installed package: source, a shared object and a bytecode cache, no git anywhere."""
    package = tmp_path / "site-packages" / "vllm"
    (package / "model_executor" / "models").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "envs.py").write_text("VLLM_USE_X = 0\n", encoding="utf-8")
    (package / "model_executor" / "models" / "qwen3.py").write_text("BLOCK = 64\n", encoding="utf-8")
    (package / "_C.abi3.so").write_bytes(b"\x7fELF shared object")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "envs.cpython-314.pyc").write_bytes(b"\x00bytecode")
    return package


def test_harvest_off_an_installed_package_snapshot_applies_to_the_package(tmp_path: Path) -> None:
    """The worktree of a pip-installed framework holds the package itself, so its diff lands where the server imports."""
    package = _installed_package(tmp_path)
    repo, err = snapshot_tree(package, tmp_path / "session" / "tree_snapshots" / "vllm.git", refresh=True)
    assert err == "" and repo is not None
    wt, err = _setup_worktree(repo, tmp_path / "ws" / "worktree", "specialist-t1")
    assert err == "" and wt is not None
    assert (wt / "model_executor" / "models" / "qwen3.py").read_text(encoding="utf-8") == "BLOCK = 64\n"
    assert not (wt / "_C.abi3.so").exists()
    assert not (wt / "__pycache__").exists()

    (wt / "model_executor" / "models" / "qwen3.py").write_text("BLOCK = 128\n", encoding="utf-8")
    (wt / "envs.py").write_text("VLLM_USE_X = 1\n", encoding="utf-8")
    patches, roots = SpecialistSubprocessDispatcher._collect_patches(
        wt, tmp_path / "ws", worktree_base=package, worktree_base_commit=ss.head_commit(wt)
    )

    assert len(patches) == 1
    assert roots == {patches[0]: str(package)}
    patch_text = Path(patches[0]).read_text(encoding="utf-8")
    assert set(ps.parse_patch_targets(patch_text).all) == {"envs.py", "model_executor/models/qwen3.py"}
    assert ps.ground_patch_text(patch_text, base_checkout=None, explicit_root=package).verdict == ps.GROUND_APPLIES
    assert not (package / ".git").exists()
    assert (package / "model_executor" / "models" / "qwen3.py").read_text(encoding="utf-8") == "BLOCK = 64\n"


def test_snapshot_refresh_follows_the_package_and_a_held_snapshot_does_not(tmp_path: Path) -> None:
    """A KEEP lands in the package between dispatches; an integrate still deciding on a candidate must not."""
    package = _installed_package(tmp_path)
    repo = tmp_path / "session" / "tree_snapshots" / "vllm.git"
    snapshot_tree(package, repo, refresh=True)
    first = ss.head_commit(repo)

    (package / "envs.py").write_text("VLLM_USE_X = 2\n", encoding="utf-8")
    held, err = snapshot_tree(package, repo, refresh=False)
    assert err == "" and held == repo
    assert ss.head_commit(repo) == first

    snapshot_tree(package, repo, refresh=True)
    assert ss.head_commit(repo) != first
    wt, _ = _setup_worktree(repo, tmp_path / "ws" / "worktree", "specialist-t2")
    assert wt is not None and (wt / "envs.py").read_text(encoding="utf-8") == "VLLM_USE_X = 2\n"


def test_snapshot_is_not_invented_while_the_package_is_mid_integrate(tmp_path: Path) -> None:
    package = _installed_package(tmp_path)
    repo, err = snapshot_tree(package, tmp_path / "session" / "tree_snapshots" / "vllm.git", refresh=False)
    assert repo is None
    assert "being integrated into" in err


def test_collect_patches_does_not_rediscover_an_obsolete_harvest(tmp_path: Path) -> None:
    base, wt = _make_harvest_worktree(tmp_path)
    before = (wt / "runtime.py").read_text(encoding="utf-8")
    (wt / "runtime.py").write_text("BLOCK_SIZE = 128\n", encoding="utf-8")
    patches, _roots = SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base)
    assert len(patches) == 1
    harvested = Path(patches[0])
    evidence = harvested.read_bytes()
    (wt / "runtime.py").write_text(before, encoding="utf-8")

    assert SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base) == ([], {})
    assert harvested.read_bytes() == evidence


@pytest.mark.parametrize(
    ("target", "content"),
    [
        ("kernels/new_kernel.py", "def block_size():\n    return 128\n"),
        ("configs/runtime.json", '{"block_size": 128}\n'),
        ("configs/runtime.yaml", "block_size: 128\n"),
        ("scratch/runtime.json", '{"block_size": 128}\n'),
        ("configs/specialist_done.json", '{"block_size": 128}\n'),
    ],
)
def test_collect_patches_keeps_create_only_runtime_changes(tmp_path: Path, target: str, content: str) -> None:
    base, wt = _make_harvest_worktree(tmp_path)
    created = wt / target
    created.parent.mkdir(parents=True, exist_ok=True)
    created.write_text(content, encoding="utf-8")

    patches, roots = SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base)

    assert len(patches) == 1
    patch_text = Path(patches[0]).read_text(encoding="utf-8")
    targets = ps.parse_patch_targets(patch_text)
    assert targets.existing == ()
    assert targets.created == (target,)
    assert roots == {patches[0]: str(base)}
    assert ps.ground_patch_text(patch_text, base_checkout=base, explicit_root=base).verdict == ps.GROUND_APPLIES
    kept, dropped, grounding, _spans_roots = ps.vet_patches(patches, base_checkout=base, explicit_root=base)
    assert kept == patches
    assert dropped == []
    assert grounding == {patches[0]: ps.GROUND_APPLIES}


# -- _read_done ------------------------------------------------------------
def test_read_done_missing(tmp_path: Path) -> None:
    assert SpecialistSubprocessDispatcher._read_done(tmp_path / "absent.json") is None


def test_read_done_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "done.json"
    p.write_text("{bad", encoding="utf-8")
    assert SpecialistSubprocessDispatcher._read_done(p) is None


def test_read_done_non_dict(tmp_path: Path) -> None:
    p = tmp_path / "done.json"
    p.write_text("[1,2,3]", encoding="utf-8")
    assert SpecialistSubprocessDispatcher._read_done(p) is None


def test_read_done_flat_dict(tmp_path: Path) -> None:
    p = tmp_path / "done.json"
    p.write_text(json.dumps({"proposal_set": []}), encoding="utf-8")
    assert SpecialistSubprocessDispatcher._read_done(p) == {"proposal_set": []}


def test_read_done_unwraps_intent_envelope(tmp_path: Path) -> None:
    p = tmp_path / "done.json"
    p.write_text(
        json.dumps(
            {
                "intent_type": "specialist_done",
                "domain": "kernel_switch_specialist",
                "payload": {"proposal_set": [{"name": "v1"}]},
            }
        ),
        encoding="utf-8",
    )
    out = SpecialistSubprocessDispatcher._read_done(p)
    assert out["domain"] == "kernel_switch_specialist"
    assert out["proposal_set"] == [{"name": "v1"}]
    assert "intent_type" not in out and "payload" not in out
