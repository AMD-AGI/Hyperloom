# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The enablement setup script: per-root final state, version check, no launch."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.orchestrator.enablement.artifacts import write_setting_script
from hyperloom.orchestrator.enablement.recipe.keep_probe import probe_environment_closure
from hyperloom.orchestrator.enablement.recipe.keep_records import build_root_records, capture_root_snapshots
from hyperloom.orchestrator.enablement.recipe.section import collect_enablement
from hyperloom.orchestrator.enablement.setup_script import (
    REFUSAL_MARKER,
    changed_versions,
    setup_script_record,
)
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

_PYTEST_VERSION = importlib.metadata.version("pytest")
_LAUNCH_MARKERS = ("sglang.launch_server", "vllm serve", "atom.entrypoints")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _closures(en: EnablementRound, *, after: dict[str, str], before: dict[str, str] | None = None) -> None:
    en.environment_closure_baseline = {"interpreter": sys.executable, "distributions": before or {"pytest": "0"}}
    en.environment_closure = {"interpreter": sys.executable, "distributions": after}


class _Stack:
    """Two roots edited in a session, captured the way an enablement KEEP captures them."""

    def __init__(self, tmp_path: Path) -> None:
        self.session = tmp_path / "session"
        self.checkout = tmp_path / "workspace" / "sglang"
        self.site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "aiter"
        self.originals = {
            self.checkout / "python" / "sglang" / "srt" / "server_args.py": "ORIGINAL = 1\n",
            self.checkout / "python" / "sglang" / "old.py": "stale\n",
            self.site / "ops" / "moe.py": "ORIGINAL = 2\n",
        }
        self.restore()
        self.final = {
            self.checkout / "python" / "sglang" / "srt" / "server_args.py": "FIXED = 1\n",
            self.site / "ops" / "moe.py": "FIXED = 2\n",
            self.site / "ops" / "launch.sh": "#!/bin/sh\necho fixed\n",
        }
        for path, text in self.final.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (self.site / "ops" / "launch.sh").chmod(0o751)
        (self.checkout / "python" / "sglang" / "old.py").unlink()
        targets = {
            str(self.checkout): {"python/sglang/srt/server_args.py": "upsert", "python/sglang/old.py": "delete"},
            str(self.site): {"ops/moe.py": "upsert", "ops/launch.sh": "upsert"},
        }
        records = build_root_records(
            contributions={str(self.checkout): {"patch_apply"}, str(self.site): {"patch_apply"}},
            base_sha_by_root={},
            git_roots=(),
            session_framework_root=str(self.checkout),
        )
        self.en = EnablementRound(
            framework_root=str(self.checkout),
            roots=records,
            source_snapshots=capture_root_snapshots(
                records=records,
                targets=targets,
                dest_root=self.session / "optimization_stack" / "enablement",
                session_dir=self.session,
            ),
            kept_patches=["/ws/p1.patch", "/ws/p2.patch"],
            patch_targets={"/ws/p1.patch": targets[str(self.checkout)], "/ws/p2.patch": targets[str(self.site)]},
            accepted_stack_targets={r["id"]: targets[r["path"]] for r in records},
            setup_commands=["echo first-setup", "echo second-setup"],
        )
        _closures(self.en, after={"pytest": _PYTEST_VERSION})

    def restore(self) -> None:
        """Put both roots back to the state a clean environment starts from."""
        for path, text in self.originals.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (self.site / "ops" / "launch.sh").unlink(missing_ok=True)

    def write(self) -> Path:
        write_setting_script(self.session, self.en, "sglang", model="/models/M", tp=8)
        return self.session / "reports" / "enablement"


def _run(script: Path, **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, check=False, **kwargs)


def test_each_root_gets_its_final_state_at_its_own_absolute_path(tmp_path):
    stack = _Stack(tmp_path)
    out = stack.write()
    stack.restore()

    proc = _run(out / "enablement_setup.sh", cwd="/")

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[:2] == ["first-setup", "second-setup"]
    for path, text in stack.final.items():
        assert path.read_text(encoding="utf-8") == text
    assert not (stack.checkout / "python" / "sglang" / "old.py").exists()
    assert stat.S_IMODE((stack.site / "ops" / "launch.sh").stat().st_mode) == 0o751
    text = (out / "enablement_setup.sh").read_text(encoding="utf-8")
    assert f"{_sha(stack.site / 'ops' / 'moe.py')} 0644" in text


def test_a_copy_whose_sha256_changed_is_not_installed(tmp_path):
    stack = _Stack(tmp_path)
    out = stack.write()
    stack.restore()
    copies = sorted((out / "setup_files").rglob("moe.py"))
    assert len(copies) == 1
    copies[0].write_text("TAMPERED = 1\n", encoding="utf-8")

    proc = _run(out / "enablement_setup.sh")

    assert proc.returncode != 0
    assert "sha256" in proc.stderr
    assert (stack.site / "ops" / "moe.py").read_text(encoding="utf-8") == "ORIGINAL = 2\n"


def test_a_file_that_lands_with_other_bytes_fails_the_script(tmp_path):
    stack = _Stack(tmp_path)
    out = stack.write()
    stack.restore()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # An ``install`` that writes something other than its source, as a full disk or a racing writer would.
    (fake_bin / "install").write_text('#!/bin/sh\nfor last; do :; done\necho corrupt > "$last"\n', encoding="utf-8")
    (fake_bin / "install").chmod(0o755)

    proc = _run(out / "enablement_setup.sh", env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"})

    assert proc.returncode != 0
    assert "sha256 of /" in proc.stderr


def test_a_directory_redirected_out_of_its_root_is_not_written_through(tmp_path):
    stack = _Stack(tmp_path)
    out = stack.write()
    stack.restore()
    outside = tmp_path / "outside"
    shutil.copytree(stack.checkout / "python", outside)
    shutil.rmtree(stack.checkout / "python")
    (stack.checkout / "python").symlink_to(outside)

    proc = _run(out / "enablement_setup.sh")

    assert proc.returncode != 0
    assert "resolves outside its root" in proc.stderr
    assert (outside / "sglang" / "old.py").read_text(encoding="utf-8") == "stale\n"
    assert (outside / "sglang" / "srt" / "server_args.py").read_text(encoding="utf-8") == "ORIGINAL = 1\n"


def test_a_deletion_redirected_out_of_its_root_is_not_followed(tmp_path):
    stack = _Stack(tmp_path)
    root_id = next(r["id"] for r in stack.en.roots if r["path"] == str(stack.checkout))
    for capture in stack.en.source_snapshots:
        if capture["root_id"] == root_id:
            capture["files"] = [f for f in capture["files"] if f["op"] == "delete"]
    stack.en.accepted_stack_targets[root_id] = {"python/sglang/old.py": "delete"}
    out = stack.write()
    stack.restore()
    outside = tmp_path / "outside"
    (stack.checkout / "python" / "sglang").rename(outside)
    (stack.checkout / "python" / "sglang").symlink_to(outside)

    proc = _run(out / "enablement_setup.sh")

    assert proc.returncode != 0
    assert "resolves outside its root" in proc.stderr
    assert (outside / "old.py").read_text(encoding="utf-8") == "stale\n"


def test_a_destination_symlinked_out_of_its_root_is_replaced_not_followed(tmp_path):
    stack = _Stack(tmp_path)
    out = stack.write()
    stack.restore()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = stack.checkout / "python" / "sglang" / "srt" / "server_args.py"
    target.unlink()
    target.symlink_to(outside)

    proc = _run(out / "enablement_setup.sh")

    assert proc.returncode == 0, proc.stderr
    assert list(outside.iterdir()) == []
    assert not target.is_symlink()
    assert target.read_text(encoding="utf-8") == "FIXED = 1\n"


def test_the_setup_script_never_launches_and_the_setting_script_launches_after_it(tmp_path):
    stack = _Stack(tmp_path)
    out = stack.write()
    stack.restore()
    setup = (out / "enablement_setup.sh").read_text(encoding="utf-8")
    assert not any(marker in setup for marker in _LAUNCH_MARKERS)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python3").write_text('#!/bin/sh\necho "LAUNCHED $*"\n', encoding="utf-8")
    (fake_bin / "python3").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    assert "LAUNCHED" not in _run(out / "enablement_setup.sh", env=env).stdout
    proc = _run(out / "enablement_setting.sh", env=env)

    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[-1] == "LAUNCHED -m sglang.launch_server --model-path=/models/M"
    assert lines.index("first-setup") < len(lines) - 1
    assert stack.final[stack.site / "ops" / "moe.py"] == (stack.site / "ops" / "moe.py").read_text()


def test_a_failed_setup_stops_the_launch(tmp_path):
    stack = _Stack(tmp_path)
    stack.en.setup_commands = ["false"]
    out = stack.write()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python3").write_text("#!/bin/sh\necho LAUNCHED\n", encoding="utf-8")
    (fake_bin / "python3").chmod(0o755)

    proc = _run(
        out / "enablement_setting.sh", env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}
    )

    assert proc.returncode != 0
    assert "LAUNCHED" not in proc.stdout


@pytest.mark.parametrize(
    ("after", "reason"),
    [
        ({"pytest": "0.0.1"}, "pytest: expected 0.0.1"),
        ({"pytest": _PYTEST_VERSION, "no-such-dist-hl": "1.0"}, "no-such-dist-hl: expected 1.0, found absent"),
    ],
)
def test_a_changed_version_that_does_not_come_back_fails_the_script(tmp_path, after, reason):
    stack = _Stack(tmp_path)
    _closures(stack.en, after=after)
    out = stack.write()
    stack.restore()

    proc = _run(out / "enablement_setup.sh")

    assert proc.returncode != 0
    assert reason in proc.stderr


def test_a_version_the_keep_asserted_is_checked_even_when_unchanged(tmp_path):
    stack = _Stack(tmp_path)
    _closures(stack.en, after={"pytest": "0"}, before={"pytest": "0"})
    stack.en.installed_versions_at_keep = {"pytest": "0"}

    proc = _run(stack.write() / "enablement_setup.sh")

    assert proc.returncode != 0
    assert "pytest: expected 0" in proc.stderr


def test_changed_versions_is_the_closure_difference_through_one_interpreter():
    en = EnablementRound()
    _closures(en, after={"a": "2", "b": "1", "new": "1"}, before={"a": "1", "b": "1", "gone": "3"})
    assert changed_versions(en) == (sys.executable, {"a": "2", "new": "1", "gone": None})

    en.environment_closure_baseline = {**en.environment_closure_baseline, "interpreter": "/other/python"}
    assert changed_versions(en) is None
    en.environment_closure_baseline = {}
    assert changed_versions(en) is None


def test_the_probe_names_the_interpreter_it_ran():
    closure, _assertions = probe_environment_closure(sys.executable, env=os.environ)
    assert closure["interpreter"] == sys.executable


def test_the_baseline_is_recorded_once_and_never_after_a_setup_ran():
    en = EnablementRound()
    assert en.record_closure_baseline({"interpreter": "/p", "distributions": {"a": "1"}})
    assert not en.record_closure_baseline({"interpreter": "/p", "distributions": {"a": "2"}})
    assert en.environment_closure_baseline["distributions"] == {"a": "1"}

    late = EnablementRound(setup_executions=[{"seq": 1}])
    assert not late.record_closure_baseline({"interpreter": "/p", "distributions": {"a": "1"}})


def _refusal_of(stack: _Stack) -> tuple[subprocess.CompletedProcess[str], str]:
    out = stack.write()
    stack.restore()
    text = (out / "enablement_setup.sh").read_text(encoding="utf-8")
    return _run(out / "enablement_setup.sh"), text


@pytest.mark.parametrize(
    ("arrange", "reason"),
    [
        (lambda en: en.kept_patches.append("/ws/advanced.patch"), "kept patch advanced.patch"),
        (lambda en: en.source_snapshots.pop(), "was declared but not captured"),
        (
            lambda en: setattr(en, "kept_artifacts", [{"target": "/x/cfg.json", "rel_target": "cfg.json"}]),
            "installed file cfg.json was not captured",
        ),
        (lambda en: setattr(en, "environment_closure_baseline", {}), "package versions"),
    ],
)
def test_a_stack_the_records_do_not_cover_is_refused_before_anything_runs(tmp_path, arrange, reason):
    stack = _Stack(tmp_path)
    arrange(stack.en)

    proc, text = _refusal_of(stack)

    assert proc.returncode != 0
    assert reason in proc.stderr
    assert "first-setup" not in proc.stdout
    assert (stack.site / "ops" / "moe.py").read_text(encoding="utf-8") == "ORIGINAL = 2\n"
    assert REFUSAL_MARKER in text


def test_a_missing_capture_payload_is_refused(tmp_path):
    stack = _Stack(tmp_path)
    for payload in (stack.session / "optimization_stack").rglob("moe.py"):
        payload.unlink()

    proc, _text = _refusal_of(stack)

    assert proc.returncode != 0
    assert "is not in the session" in proc.stderr


def test_standalone_is_claimed_only_for_a_sufficient_recipe_whose_script_does_not_refuse(tmp_path):
    stack = _Stack(tmp_path)
    out = stack.write()
    record = setup_script_record(stack.session, sufficient=True)
    assert record == {
        "path": "reports/enablement/enablement_setup.sh",
        "sha256": _sha(out / "enablement_setup.sh"),
        "standalone": True,
    }
    assert setup_script_record(stack.session, sufficient=False)["standalone"] is False

    stack.en.kept_patches.append("/ws/advanced.patch")
    stack.write()
    assert setup_script_record(stack.session, sufficient=True)["standalone"] is False


def test_the_section_records_the_setup_script_beside_its_verdict(tmp_path):
    stack = _Stack(tmp_path)
    out = stack.write()
    state = {"enablement": {"last_specialist_task_id": "spec-1", "environment_closure": stack.en.environment_closure}}

    section = collect_enablement(stack.session, state, [])

    assert section["replay_sufficiency"]["status"] == "insufficient"
    assert section["setup_script"] == {
        "path": "reports/enablement/enablement_setup.sh",
        "sha256": _sha(out / "enablement_setup.sh"),
        "standalone": False,
    }
    assert "interpreter" not in section["environment_closure"]


def test_both_scripts_are_owner_only(tmp_path):
    out = _Stack(tmp_path).write()
    for name in ("enablement_setting.sh", "enablement_setup.sh"):
        assert stat.S_IMODE((out / name).stat().st_mode) == 0o700


def test_the_setting_script_demands_a_model_when_none_is_known(tmp_path):
    write_setting_script(tmp_path, EnablementRound(), "sglang")

    proc = _run(tmp_path / "reports" / "enablement" / "enablement_setting.sh")

    assert proc.returncode != 0
    assert "set MODEL to the model path" in proc.stderr


def test_the_setting_script_exports_the_accepted_launch(tmp_path):
    en = EnablementRound(accepted_config={"extra_envs": {"VLLM_ROCM_USE_AITER": "1"}, "extra_server_args": "--tp 4"})
    rel = write_setting_script(tmp_path, en, "vllm", model="/models/M", tp=8)
    text = (tmp_path / rel).read_text(encoding="utf-8")
    assert "export VLLM_ROCM_USE_AITER=1" in text
    assert text.index('bash "$SCRIPT_DIR"/enablement_setup.sh') < text.index("export VLLM_ROCM_USE_AITER=1")
    assert text.rstrip().endswith("vllm serve $MODEL --tp 4")


def _attempt_runtime(tmp_path: Path, task_id: str) -> dict[str, str]:
    """An attempt runtime as a round provisions it, under its own attempt directory."""
    venv = tmp_path / "stacks" / "sglang" / task_id / "venv"
    return {"venv_root": str(venv), "python_path": str(venv / "bin" / "python"), "bin_path": str(venv / "bin")}


def _assert_runtime_refused(stack: _Stack, out: Path) -> None:
    proc = _run(out / "enablement_setting.sh", env={**os.environ, "PATH": f"{out}:{os.environ['PATH']}"})
    assert proc.returncode != 0
    assert "isolated attempt runtime" in proc.stderr
    assert "first-setup" not in proc.stdout
    assert (stack.site / "ops" / "moe.py").read_text(encoding="utf-8") == "ORIGINAL = 2\n"
    assert "LAUNCHED" not in proc.stdout
    assert REFUSAL_MARKER in (out / "enablement_setup.sh").read_text(encoding="utf-8")
    # A sufficient verdict assumes a consumer rebuilds the runtime; this script does not.
    assert setup_script_record(stack.session, sufficient=True)["standalone"] is False


def _fake_launcher(out: Path) -> None:
    """A ``python3`` on PATH that reports a launch, so a launch on the base interpreter is seen."""
    fake = out / "python3"
    fake.write_text("#!/bin/sh\necho LAUNCHED\n", encoding="utf-8")
    fake.chmod(0o755)


def test_a_later_round_keep_on_its_own_runtime_is_refused_without_setup_commands(tmp_path):
    # The baseline was read through the first round's runtime, the KEEP through a
    # later round's: no diff, so no version check would be emitted at all.
    stack = _Stack(tmp_path)
    stack.en.setup_commands = []
    first, later = _attempt_runtime(tmp_path, "spec-1"), _attempt_runtime(tmp_path, "spec-2")
    stack.en.environment_closure_baseline = {"interpreter": first["python_path"], "distributions": {"pytest": "0"}}
    stack.en.environment_closure = {"interpreter": later["python_path"], "distributions": {"pytest": "1"}}
    stack.en.active_runtime = later
    assert changed_versions(stack.en) is None
    out = stack.write()
    stack.restore()
    _fake_launcher(out)

    _assert_runtime_refused(stack, out)


def test_a_baseline_round_keep_on_its_runtime_is_refused(tmp_path):
    stack = _Stack(tmp_path)
    runtime = _attempt_runtime(tmp_path, "spec-1")
    stack.en.environment_closure_baseline = {"interpreter": runtime["python_path"], "distributions": {"pytest": "0"}}
    stack.en.environment_closure = {"interpreter": runtime["python_path"], "distributions": {"pytest": "1"}}
    stack.en.active_runtime = runtime
    assert changed_versions(stack.en) is not None
    out = stack.write()
    stack.restore()
    _fake_launcher(out)

    _assert_runtime_refused(stack, out)


def test_the_base_environment_is_not_refused_as_a_runtime(tmp_path):
    stack = _Stack(tmp_path)
    stack.en.active_runtime = {"venv_root": "", "python_path": "", "envs": {}}
    stack.write()
    assert setup_script_record(stack.session, sufficient=True)["standalone"] is True
