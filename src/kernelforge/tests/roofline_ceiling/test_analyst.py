# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""The analyst session: what it is asked, where it may write, and how it is read."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from kernelforge.agent_backends.workspace_guard import WorkspaceGuard, WorkspaceSafetyError
from kernelforge.llm.workspace_policy import TOOL_OWNED_UNTRACKED_GLOBS
from kernelforge.roofline_ceiling import analyst as analyst_module
from kernelforge.roofline_ceiling.analyst import (
    CeilingAnalysisError,
    build_request,
    load_role,
    run_ceiling_analysis,
)
from kernelforge.roofline_ceiling.device_profile import DeviceIdentity
from kernelforge.roofline_ceiling.evidence import EvidenceBundle
from kernelforge.roofline_ceiling.report import DOCUMENT_FILENAME, REPORT_FILENAME

_GOOD = {"cases": {"c0": 12.8}, "mean_ideal_ms": 12.8}


_DERIVATION = "# Performance ceiling analysis\n\n## Conclusion\n\nc0 is HBM-bound at 12.8 ms.\n"


class _Backend:
    """A backend that writes canned files and records the specs it was given.

    Each attempt writes the next of ``payloads`` as the ceiling file and the
    next of ``documents`` as the derivation; ``None`` writes nothing, and with
    no ``documents`` given every attempt writes a derivation.
    """

    name = "fake"

    def __init__(self, *payloads, documents=None, leavings: dict[str, str] | None = None):
        self._payloads = list(payloads)
        self._documents = None if documents is None else list(documents)
        self._leavings = dict(leavings or {})
        self.specs: list = []

    async def run(self, spec, usage=None):
        self.specs.append(spec)
        scratch = Path(self._output_dir(spec))
        for name, body in self._leavings.items():
            leaving = scratch / name
            leaving.parent.mkdir(parents=True, exist_ok=True)
            leaving.write_text(body, encoding="utf-8")
        if self._payloads:
            payload = self._payloads.pop(0)
            if payload is not None:
                target = scratch / REPORT_FILENAME
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
        document = _DERIVATION if self._documents is None else (self._documents.pop(0) if self._documents else None)
        if document is not None:
            (scratch / DOCUMENT_FILENAME).write_text(document, encoding="utf-8")
        return type("Result", (), {"text": "done", "end_reason": "agent_stopped"})()

    @staticmethod
    def _output_dir(spec) -> str:
        return spec.additional_directories[0]


def _bundle(tmp_path) -> EvidenceBundle:
    artifacts = tmp_path / "evidence"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "trace").mkdir(exist_ok=True)
    (artifacts / "trace" / "kernel_stats.csv").write_text("name,count\n", encoding="utf-8")
    return EvidenceBundle(
        identity=DeviceIdentity(
            arch="gfx950",
            device_name="AMD Instinct MI355X",
            compute_partition="SPX",
            memory_partition="NPS1",
        ),
        artifacts_dir=artifacts,
        observed_ms={"c0": 40.0},
        notes=("kernel trace unavailable: nothing",),
    )


def _analyse(backend, tmp_path, **overrides):
    kwargs = {
        "workdir": str(tmp_path),
        "output_dir": tmp_path / "out",
        "kernel_files": ["kernel.py"],
        "driver_script": "driver.py",
        "performance_command": ["bash", "-c", "python3 driver.py"],
        "case_ids": ["c0"],
        "case_params": {},
        "evidence": _bundle(tmp_path),
    }
    kwargs.update(overrides)
    return asyncio.run(run_ceiling_analysis(backend, **kwargs))


# --- the role document ---------------------------------------------------------


def test_the_role_document_ships_with_the_package():
    role = load_role()

    assert "Performance Ceiling Analyst" in role
    assert "Step 0 — establish this machine's roofs" in role


def test_the_role_document_names_both_files_as_the_deliverable():
    role = load_role()

    assert "performance_ceiling.json" in role
    assert "performance_ceiling_analysis.md" in role


def test_the_role_document_hands_the_composition_to_the_analyst():
    """It offers a default rule and tells the analyst when to leave it."""
    role = load_role()

    assert "You own the whole estimate" in role
    assert "Depart from the default" in role
    assert "occupancy" in role.lower()


def test_the_role_document_keeps_installs_out_of_the_measured_environment():
    """A package changed under the kernel after the baseline skews every later measurement."""
    role = load_role()

    assert "python3 -m venv <tools_dir>/" in role
    assert "Do not activate that environment" in role
    assert "Never install" in role
    assert "`pip install -r" not in role


# --- the request ---------------------------------------------------------------


def test_the_request_names_the_machine_rather_than_supplying_its_roofs(tmp_path):
    """The analyst measures the peaks itself; handing it any would pre-empt that."""
    request = json.loads(
        build_request(
            kernel_files=["kernel.py"],
            driver_script="driver.py",
            performance_command=["bash", "-c", "run"],
            case_ids=["c0"],
            case_params={"tokens": 1},
            evidence=_bundle(tmp_path),
            output_dir=str(tmp_path / "out"),
            tools_dir=str(tmp_path / "tools"),
        )
    )

    assert request["machine"]["arch"] == "gfx950"
    assert request["machine"]["compute_partition"] == "SPX"
    assert "peak_flops" not in request["machine"]
    assert request["scored_case_ids"] == ["c0"]


def test_the_request_spells_out_both_files_and_where_they_go(tmp_path):
    request = json.loads(
        build_request(
            kernel_files=[],
            driver_script="",
            performance_command=[],
            case_ids=["c0"],
            case_params={},
            evidence=_bundle(tmp_path),
            output_dir=str(tmp_path / "out"),
            tools_dir=str(tmp_path / "tools"),
        )
    )

    assert request["output_dir"] == str(tmp_path / "out")
    assert request["tools_dir"] == str(tmp_path / "tools")
    assert set(request["output_files"]) == {"performance_ceiling.json", "performance_ceiling_analysis.md"}
    assert "equal-weight" in request["output_files"]["performance_ceiling.json"]


def test_the_request_hands_over_the_evidence_it_collected(tmp_path):
    request = json.loads(
        build_request(
            kernel_files=[],
            driver_script="",
            performance_command=[],
            case_ids=["c0"],
            case_params={},
            evidence=_bundle(tmp_path),
            output_dir=str(tmp_path / "out"),
            tools_dir=str(tmp_path / "tools"),
        )
    )

    assert "trace/kernel_stats.csv" in request["evidence_files"]
    assert request["observed_ms"] == {"c0": 40.0}
    assert "back-solved" in request["observed_ms_meaning"]
    assert "none may exceed it" in request["observed_ms_meaning"]
    # Told it is inflated, the analyst would loosen the one bound it has, toward a kernel that reads as done.
    assert "no profiler attached" in request["observed_ms_meaning"]
    assert "inflated by" not in request["observed_ms_meaning"]


# --- the session ---------------------------------------------------------------


def test_the_session_can_run_the_profiler_it_needs(tmp_path):
    """Measuring the roofs takes a shell, and installing the tool takes more."""
    backend = _Backend(_GOOD)

    _analyse(backend, tmp_path)

    policy = backend.specs[0].tool_policy
    assert backend.specs[0].writable is True
    assert (policy.read, policy.search, policy.write, policy.shell) == (True, True, True, True)


def test_the_session_writes_outside_the_workspace_so_the_guard_keeps_it(tmp_path):
    """The guard rolls back new files under the workspace, answer included."""
    backend = _Backend(_GOOD)

    _analyse(backend, tmp_path)

    scratch, tools = (Path(directory) for directory in backend.specs[0].additional_directories)
    assert scratch != tools
    assert tmp_path not in scratch.parents
    assert tmp_path not in tools.parents
    assert json.loads(backend.specs[0].user_prompt)["tools_dir"] == str(tools)


def test_the_kernel_under_optimization_stays_out_of_reach(tmp_path):
    """Two lines: the hook refuses the edit, the guard protects the kernel and driver by path."""
    backend = _Backend(_GOOD)

    _analyse(backend, tmp_path)

    spec = backend.specs[0]
    assert spec.protected_paths == ["kernel.py", "driver.py"]
    assert spec.protected_globs == []
    assert spec.allow_tracked_changes is False
    assert spec.ignored_untracked_globs == list(TOOL_OWNED_UNTRACKED_GLOBS)
    assert not spec.allow_untracked
    assert spec.hooks.pre_tool_use


def _git_workspace(root: Path) -> Path:
    root.mkdir()
    (root / "kernel.py").write_text("def kernel():\n    return 1\n", encoding="utf-8")
    (root / "driver.py").write_text("print('case_ms: c0 40.0')\n", encoding="utf-8")
    (root / "helpers.py").write_text("SCALE = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("forge_experiments/\n", encoding="utf-8")
    cache = root / "forge_experiments" / "aiter_cache" / "build"
    cache.mkdir(parents=True)
    (cache / "module.so").write_bytes(b"\0" * 4096)
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "base"],
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True)
    return root


def _guarded(workspace: Path, scratch: Path) -> WorkspaceGuard:
    spec = analyst_module._spec(
        system_prompt="",
        user_prompt="",
        workdir=str(workspace),
        model="",
        timeout_sec=60,
        writable_dirs=[str(scratch)],
        protected_paths=["kernel.py", "driver.py"],
        turns=1,
    )
    guard = WorkspaceGuard(spec)
    guard.prepare()
    return guard


def test_the_guard_leaves_the_campaigns_jit_cache_unread_and_uncounted(tmp_path):
    """Running the kernel compiles into the ignored tree; that is neither snapshotted nor a violation."""
    workspace = _git_workspace(tmp_path / "ws")

    guard = _guarded(workspace, tmp_path / "scratch")
    cache = workspace / "forge_experiments" / "aiter_cache" / "build"
    (cache / "module.so").write_bytes(b"\1" * 4096)
    (cache / "new_shard.so").write_bytes(b"\2" * 16)
    (workspace / ".rocprofv3").mkdir()
    (workspace / ".rocprofv3" / "trace.db").write_bytes(b"\3")

    assert not any("aiter_cache" in str(path) for path in guard.snapshots)
    assert guard.verify() == []


def test_the_guard_rolls_back_and_rejects_an_edit_to_the_kernel(tmp_path):
    workspace = _git_workspace(tmp_path / "ws")

    guard = _guarded(workspace, tmp_path / "scratch")
    (workspace / "kernel.py").write_text("def kernel():\n    return 2\n", encoding="utf-8")

    with pytest.raises(WorkspaceSafetyError, match="kernel.py"):
        guard.verify()
    assert (workspace / "kernel.py").read_text(encoding="utf-8") == "def kernel():\n    return 1\n"


def test_the_guard_rolls_back_and_rejects_an_edit_to_any_other_tracked_file(tmp_path):
    """The analyst may only read: a tracked file outside the measurement surface is restored too, not left changed."""
    workspace = _git_workspace(tmp_path / "ws")

    guard = _guarded(workspace, tmp_path / "scratch")
    (workspace / "helpers.py").write_text("SCALE = 2\n", encoding="utf-8")

    with pytest.raises(WorkspaceSafetyError, match="helpers.py"):
        guard.verify()
    assert (workspace / "helpers.py").read_text(encoding="utf-8") == "SCALE = 1\n"


def test_a_file_already_dirty_mid_campaign_is_restored_to_what_the_session_found(tmp_path):
    """Rollback restores the inherited dirty state, not HEAD, so the campaign's own work survives the rejection."""
    workspace = _git_workspace(tmp_path / "ws")
    (workspace / "helpers.py").write_text("SCALE = 3\n", encoding="utf-8")

    guard = _guarded(workspace, tmp_path / "scratch")
    (workspace / "helpers.py").write_text("SCALE = 4\n", encoding="utf-8")

    with pytest.raises(WorkspaceSafetyError, match="helpers.py"):
        guard.verify()
    assert (workspace / "helpers.py").read_text(encoding="utf-8") == "SCALE = 3\n"


def test_a_session_that_leaves_tracked_files_alone_is_accepted(tmp_path):
    workspace = _git_workspace(tmp_path / "ws")
    (workspace / "helpers.py").write_text("SCALE = 3\n", encoding="utf-8")

    guard = _guarded(workspace, tmp_path / "scratch")

    assert guard.verify() == []


def _deny_reason(hooks, tool_name: str, file_path: str):
    hook = hooks.pre_tool_use[0]
    verdict = asyncio.run(hook.callback({"tool_name": tool_name, "tool_input": {"file_path": file_path}}, None, None))
    return (verdict.get("hookSpecificOutput") or {}).get("permissionDecisionReason")


def test_the_hook_refuses_an_edit_outside_the_output_directories(tmp_path):
    backend = _Backend(_GOOD)
    _analyse(backend, tmp_path)

    reason = _deny_reason(backend.specs[0].hooks, "Write", str(tmp_path / "kernel.py"))

    assert reason is not None
    assert "read-only" in reason


def test_the_hook_allows_the_scratch_directory_it_named(tmp_path):
    backend = _Backend(_GOOD)
    _analyse(backend, tmp_path)

    scratch = backend.specs[0].additional_directories[0]
    assert _deny_reason(backend.specs[0].hooks, "Write", f"{scratch}/{REPORT_FILENAME}") is None
    assert _deny_reason(backend.specs[0].hooks, "Write", f"{scratch}/roofs/roofline.csv") is None


def test_the_hook_allows_the_tools_directory_it_named(tmp_path):
    backend = _Backend(_GOOD)
    _analyse(backend, tmp_path)

    tools = backend.specs[0].additional_directories[1]
    assert _deny_reason(backend.specs[0].hooks, "Write", f"{tools}/rocprof-compute/pyvenv.cfg") is None


def test_the_hook_leaves_tools_that_do_not_write_alone(tmp_path):
    backend = _Backend(_GOOD)
    _analyse(backend, tmp_path)

    assert _deny_reason(backend.specs[0].hooks, "Read", str(tmp_path / "kernel.py")) is None


# --- reading the answer back ---------------------------------------------------


def test_a_file_the_analyst_wrote_becomes_the_report(tmp_path):
    report = _analyse(_Backend(_GOOD), tmp_path)

    assert report.ideal_ms() == {"c0": 12.8}


def test_both_deliverables_are_moved_into_the_output_directory(tmp_path):
    backend = _Backend(_GOOD)

    _analyse(backend, tmp_path)

    assert (tmp_path / "out" / REPORT_FILENAME).is_file()
    assert (tmp_path / "out" / DOCUMENT_FILENAME).read_text() == _DERIVATION


def test_what_the_session_left_behind_is_kept_as_the_record(tmp_path):
    """How the roofs were established has to survive the scratch directory."""
    backend = _Backend(_GOOD, leavings={"roofs/roofline.csv": "device,HBMBw\n0,6230\n"})

    _analyse(backend, tmp_path)

    kept = tmp_path / "out" / "evidence" / "analyst" / "roofs" / "roofline.csv"
    assert kept.is_file()
    assert "HBMBw" in kept.read_text()


def test_what_the_session_installed_is_deleted_rather_than_published(tmp_path):
    """The profiler's environment is toolchain, not evidence of how the roofs were reached."""

    class _Installs(_Backend):
        async def run(self, spec, usage=None):
            installed = Path(spec.additional_directories[1]) / "rocprof-compute" / "pyvenv.cfg"
            installed.parent.mkdir(parents=True, exist_ok=True)
            installed.write_text("home = /usr/bin\n", encoding="utf-8")
            return await super().run(spec, usage)

    backend = _Installs(_GOOD)

    _analyse(backend, tmp_path)

    tools = Path(backend.specs[0].additional_directories[1])
    assert not tools.exists()
    assert not list((tmp_path / "out").rglob("pyvenv.cfg"))


def test_an_unreadable_file_is_handed_back_once_with_the_reason(tmp_path):
    backend = _Backend({"cases": {"c0": "fast"}}, _GOOD)

    report = _analyse(backend, tmp_path)

    assert report.ideal_ms() == {"c0": 12.8}
    assert len(backend.specs) == 2
    assert "finite positive number" in backend.specs[1].user_prompt
    # The derivation was already there, so the repair asks for it to be kept rather than rewritten.
    assert f"Leave {DOCUMENT_FILENAME} in place" in backend.specs[1].user_prompt


def test_a_session_that_never_writes_the_file_is_told_so(tmp_path):
    backend = _Backend(None, _GOOD)

    report = _analyse(backend, tmp_path)

    assert report.ideal_ms() == {"c0": 12.8}
    assert "not written" in backend.specs[1].user_prompt


def test_two_unreadable_attempts_end_the_run_rather_than_a_third(tmp_path):
    backend = _Backend({"cases": {}}, {"cases": {}})

    with pytest.raises(CeilingAnalysisError, match="no usable ceiling after 2 attempts: performance_ceiling.json"):
        _analyse(backend, tmp_path)

    assert len(backend.specs) == 2


def test_malformed_json_on_disk_is_a_repairable_failure(tmp_path):
    backend = _Backend("{not json", _GOOD)

    report = _analyse(backend, tmp_path)

    assert report.ideal_ms() == {"c0": 12.8}


# --- the derivation is part of the deliverable ----------------------------------


def test_a_readable_ceiling_without_its_derivation_is_sent_back_for_it(tmp_path):
    """The derivation is the only check on the roofs, so a ceiling is not published without it."""
    backend = _Backend(_GOOD, _GOOD, documents=[None, _DERIVATION])

    report = _analyse(backend, tmp_path)

    assert report.ideal_ms() == {"c0": 12.8}
    assert len(backend.specs) == 2
    repair = backend.specs[1].user_prompt
    assert f"{DOCUMENT_FILENAME} was not written" in repair
    assert "Leave" not in repair
    # The JSON was fine, so the repair does not ask for it again.
    assert "could not be read as a ceiling" not in repair
    assert (tmp_path / "out" / DOCUMENT_FILENAME).read_text() == _DERIVATION


def test_an_empty_derivation_counts_as_none(tmp_path):
    backend = _Backend(_GOOD, _GOOD, documents=["  \n", _DERIVATION])

    _analyse(backend, tmp_path)

    assert f"{DOCUMENT_FILENAME} is empty" in backend.specs[1].user_prompt


def test_an_undecodable_derivation_is_sent_back_rather_than_ending_the_run(tmp_path):
    """A session cut mid-write can leave half a multi-byte character; that is repairable, not fatal."""

    class _TruncatedFirst(_Backend):
        async def run(self, spec, usage=None):
            result = await super().run(spec, usage)
            if len(self.specs) == 1:
                (Path(self._output_dir(spec)) / DOCUMENT_FILENAME).write_bytes(b"# Roofs\n\nHBM 6.2 TB/s \xe2\x80")
            return result

    backend = _TruncatedFirst(_GOOD, _GOOD)

    report = _analyse(backend, tmp_path)

    assert report.ideal_ms() == {"c0": 12.8}
    assert f"{DOCUMENT_FILENAME} is not valid UTF-8 text" in backend.specs[1].user_prompt
    assert (tmp_path / "out" / DOCUMENT_FILENAME).read_text() == _DERIVATION


def test_a_ceiling_never_given_a_derivation_is_not_published(tmp_path):
    backend = _Backend(_GOOD, _GOOD, documents=[None, None])

    with pytest.raises(CeilingAnalysisError, match=f"{DOCUMENT_FILENAME} was not written"):
        _analyse(backend, tmp_path)

    assert len(backend.specs) == 2
    assert not (tmp_path / "out" / REPORT_FILENAME).exists()


def test_a_repair_asks_for_both_files_when_neither_can_be_used(tmp_path):
    backend = _Backend(None, _GOOD, documents=[None, _DERIVATION])

    _analyse(backend, tmp_path)

    repair = backend.specs[1].user_prompt
    assert "could not be read as a ceiling" in repair
    assert f"{DOCUMENT_FILENAME} was not written" in repair
