import asyncio
import hashlib
from pathlib import Path

import kernelforge.loop.insession_gate as gate_module
import pytest
from kernelforge.loop.insession_gate import InSessionGate
from kernelforge.llm.git import GitError


def _gate(tmp_path: Path) -> tuple[InSessionGate, Path]:
    workspace = tmp_path / "ws"
    scripts = workspace / "scripts"
    source = workspace / "aiter" / "csrc"
    scripts.mkdir(parents=True)
    source.mkdir(parents=True)

    (workspace / "config.yaml").write_text("task_type: image_kernel\n")
    (workspace / "forge_driver.py").write_text("print('driver')\n")
    (scripts / "task_runner.py").write_text("print('runner')\n")
    kernel = source / "kernel.cu"
    kernel.write_text("__global__ void kernel() {}\n")

    gate = InSessionGate(
        driver_script=str(workspace / "forge_driver.py"),
        snr_threshold=30.0,
        baseline_case_times={"case": 1.0},
        best_mean_case_speedup=1.0,
        kernel_file=str(kernel),
        target_files=[str(kernel)],
    )
    return gate, workspace


def test_gate_protects_task_harness_and_config(tmp_path: Path):
    gate, workspace = _gate(tmp_path)

    assert gate._is_protected(str(workspace / "config.yaml"))
    assert gate._is_protected(str(workspace / "scripts" / "task_runner.py"))
    assert not gate._is_protected(str(workspace / "aiter" / "csrc" / "kernel.cu"))


def test_declared_source_hint_cannot_override_protection(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    driver = workspace / "forge_driver.py"
    config = workspace / "config.yaml"
    driver.write_text("print('driver')\n")
    config.write_text("task: protected\n")
    gate = InSessionGate(
        driver_script=str(driver),
        snr_threshold=30.0,
        baseline_case_times={"case": 1.0},
        best_mean_case_speedup=1.0,
        kernel_file=str(workspace / "kernel.py"),
        target_files=[str(config)],
    )

    assert gate._is_protected(str(config))


def test_non_target_implementation_edits_are_counted(tmp_path: Path):
    gate, workspace = _gate(tmp_path)

    assert gate.count_target_edits(str(workspace), ["src/helper.py"]) == 1
    assert gate.count_target_edits(str(workspace), ["scripts/task_runner.py"]) == 0


def test_bash_allows_readonly_diagnostics_with_dev_null(tmp_path: Path):
    gate, _workspace = _gate(tmp_path)

    assert not gate._bash_may_modify_protected('find / -name "*.hsaco" -newermt "-20 min" 2>/dev/null | head')
    assert not gate._bash_may_modify_protected('grep -R "task_runner.py" aiter/csrc 2>/dev/null | head')


def test_bash_allows_tmp_outputs_but_blocks_protected_writes(tmp_path: Path):
    gate, workspace = _gate(tmp_path)

    assert not gate._bash_may_modify_protected("python probe.py > /tmp/probe.log")
    assert gate._bash_may_modify_protected("echo hacked > config.yaml")
    assert gate._bash_may_modify_protected("echo hacked 2>>forge_driver.py")
    assert gate._bash_may_modify_protected("echo hacked &>> forge_driver.py")
    assert gate._bash_may_modify_protected("sed -i s/pass/fail/ scripts/task_runner.py")
    assert gate._bash_may_modify_protected(f"python - <<'PY'\nopen('{workspace / 'config.yaml'}', 'w').write('x')\nPY")


def test_bash_blocks_a_write_hidden_behind_a_wrapper_option(tmp_path: Path):
    """The verb that acts is what a rule has to be matched against.

    Reaching it means stepping over leading assignments and wrappers, and doing
    that by counting words needs the option grammar of every wrapper: ``env -u
    FOO`` and ``timeout --signal=KILL 60`` each take an argument that is not
    itself an option, so counting landed on ``FOO`` and ``60`` and let the write
    through. ``env`` was also named in the docstring as a wrapper and missing
    from the set that lists them.
    """
    gate, _workspace = _gate(tmp_path)

    assert gate._bash_may_modify_protected("env FOO=bar tee forge_driver.py")
    assert gate._bash_may_modify_protected("env -u FOO tee forge_driver.py")
    assert gate._bash_may_modify_protected("timeout --signal=KILL 60 tee forge_driver.py")
    assert gate._bash_may_modify_protected("env FOO=1 timeout 60 sudo tee forge_driver.py")


def test_bash_still_allows_running_the_driver_under_a_wrapper(tmp_path: Path):
    """Reading every word of a wrapped command as a verb must not deny the run.

    Running the driver under ``timeout`` is the ordinary way a session measures
    itself, so the extra verb positions may not turn its own name into a write.
    """
    gate, _workspace = _gate(tmp_path)

    assert not gate._bash_may_modify_protected("timeout 300 python3 forge_driver.py --warmup 3 --bench-mode")
    assert not gate._bash_may_modify_protected("env FOO=1 python3 forge_driver.py")
    assert not gate._bash_may_modify_protected("./configure --prefix=/usr")


def test_bash_allows_kernel_heredoc_followed_by_driver_read(tmp_path: Path):
    gate, workspace = _gate(tmp_path)
    kernel = workspace / "aiter" / "csrc" / "kernel.cu"
    command = f"""python3 - <<'PY'
p = {str(kernel)!r}
s = open(p).read()
open(p, 'w').write(s + '\\n')
PY
python3 forge_driver.py
"""

    assert not gate._bash_may_modify_protected(command)


def test_bash_allows_dynamic_csv_write_followed_by_driver_benchmark(
    tmp_path: Path,
):
    gate, _workspace = _gate(tmp_path)
    command = """python3 - <<'PY'
from pathlib import Path

output = Path.cwd() / "kimik3_fp4_tuned_fmoe.csv"
with open(output, "w") as stream:
    stream.write("kernel,latency\\n")
PY
python3 forge_driver.py --warmup 10 --iters 30 --bench-mode
"""

    assert not gate._bash_may_modify_protected(command)


def test_bash_blocks_protected_heredoc_write_with_resolved_variable(
    tmp_path: Path,
):
    gate, workspace = _gate(tmp_path)
    command = f"""python3 - <<'PY'
p = {str(workspace / "forge_driver.py")!r}
open(p, 'w').write('hacked')
PY
"""

    assert gate._bash_may_modify_protected(command)


def test_bash_keeps_ambiguous_inline_protected_write_conservative(
    tmp_path: Path,
):
    gate, _workspace = _gate(tmp_path)
    command = """python3 - <<'PY'
name = get_target()
open(name, 'w').write('x')
print('forge_driver.py')
PY
"""

    assert gate._bash_may_modify_protected(command)


def test_safe_heredoc_does_not_allow_a_later_python_payload(tmp_path: Path):
    gate, workspace = _gate(tmp_path)
    command = f"""python3 - <<'PY'
open('scratch.txt', 'w').write('safe')
PY
python3 -c "open({str(workspace / "forge_driver.py")!r}, mode='wb').write(b'x')"
"""

    assert gate._bash_may_modify_protected(command)


def test_safe_python_payload_does_not_allow_a_later_shell_write(tmp_path: Path):
    gate, _workspace = _gate(tmp_path)
    command = """python3 - <<'PY'
from pathlib import Path
Path('scratch.txt').write_text('safe')
PY
mv scratch.txt forge_driver.py
"""

    assert gate._bash_may_modify_protected(command)


def test_python_c_supports_path_open_mode_variants(tmp_path: Path):
    gate, workspace = _gate(tmp_path)
    driver = workspace / "forge_driver.py"

    assert gate._bash_may_modify_protected(
        f"python -c \"from pathlib import Path; Path({str(driver)!r}).open(mode='a+').write('x')\""
    )


def test_python_c_payload_scanner_rejects_long_unterminated_quotes():
    commands = (
        'python -c "' + "\\!" * 10_000,
        "python -c '" + "\\&" * 10_000,
    )

    assert all(not gate_module._python_command_payloads(command) for command in commands)


def test_candidate_diff_fingerprint_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch,
):
    gate, _workspace = _gate(tmp_path)

    def _unreadable_index(*_args, **_kwargs):
        raise GitError(128, ["git", "diff"], "", "fatal: unable to read index")

    monkeypatch.setattr(gate_module, "git", _unreadable_index)

    with pytest.raises(GitError, match="unable to read index"):
        gate._candidate_diff_sha256()


def test_python_rename_and_replace_apis_protect_both_paths(tmp_path: Path):
    gate, workspace = _gate(tmp_path)
    driver = workspace / "forge_driver.py"
    commands = [
        f"python -c \"import os; os.rename({str(driver)!r}, 'saved.py')\"",
        f"python -c \"import os; os.replace('scratch.py', {str(driver)!r})\"",
        (f"python -c \"from pathlib import Path; Path({str(driver)!r}).rename('saved.py')\""),
        (f"python -c \"from pathlib import Path; Path('scratch.py').replace({str(driver)!r})\""),
    ]

    assert all(gate._bash_may_modify_protected(command) for command in commands)


def test_stop_detects_protected_snapshot_changes(tmp_path: Path):
    gate, workspace = _gate(tmp_path)

    (workspace / "config.yaml").write_text("task_type: image_kernel\nagent: hacked\n")
    assert "modified" in gate._protected_changes()


def test_safe_stop_runs_canonical_validation_and_converges(
    tmp_path: Path,
    monkeypatch,
):
    # Harness intact: the gate runs its canonical correctness+bench self-check and,
    # on a correct + faster candidate, ALLOWS the stop as a real convergence
    # (best_ms=1.0 in the helper; 0.5ms beats it by > noise floor).
    gate, _workspace = _gate(tmp_path)

    async def _corr(**_kwargs):
        return {"passed": True, "message": ""}

    async def _bench(**_kwargs):
        return {
            "success": True,
            "median_ms": 0.5,
            "case_times": {"case": 0.5},
            "measurements": [
                {
                    "success": True,
                    "case_times": {"case": 0.5},
                    "unscored_cases": [],
                }
                for _ in range(3)
            ],
        }

    monkeypatch.setattr(gate_module, "test_correctness", _corr, raising=False)
    monkeypatch.setattr(gate_module, "measure_wallclock", _bench, raising=False)
    monkeypatch.setattr(
        gate,
        "_candidate_diff_sha256",
        lambda: hashlib.sha256(b"").hexdigest(),
    )

    result = asyncio.run(gate._on_stop({}, None, None))

    assert result == {}
    assert gate.end_reason == "converged"
    assert gate.passed is True
    assert gate.last_wall_ms == 0.5


def test_snapshot_covers_driver_and_glob_only_harness(tmp_path: Path):
    workspace = tmp_path / "ws"
    source = workspace / "aiter" / "csrc"
    source.mkdir(parents=True)
    kernel = source / "kernel.cu"
    kernel.write_text("__global__ void kernel() {}\n")

    # Driver with a NON-default name: protected only via its exact abspath.
    driver = workspace / "custom_driver.py"
    driver.write_text("print('drive')\n")
    # Harness caught only by a basename glob (*harness*.py) at the root.
    harness = workspace / "test_kernel_harness.py"
    harness.write_text("print('harness')\n")

    gate = InSessionGate(
        driver_script=str(driver),
        snr_threshold=30.0,
        baseline_case_times={"case": 1.0},
        best_mean_case_speedup=1.0,
        kernel_file=str(kernel),
        target_files=[str(kernel)],
    )

    snapshot = gate._protected_snapshot
    assert "custom_driver.py" in snapshot
    assert "test_kernel_harness.py" in snapshot

    driver.write_text("print('drive')\nhacked = 1\n")
    assert "custom_driver.py" in gate._protected_changes()


def test_snapshot_recurses_nested_globs_and_protected_directories(tmp_path: Path):
    gate, workspace = _gate(tmp_path)
    nested_glob = workspace / "src" / "deep" / "test_oracle.py"
    nested_dir = workspace / "pkg" / "deep" / "benchmarks" / "oracle.bin"
    for path in (nested_glob, nested_dir):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")

    # Recreate the gate after protected files exist so they become the baseline.
    kernel = workspace / "aiter" / "csrc" / "kernel.cu"
    gate = InSessionGate(
        driver_script=str(workspace / "forge_driver.py"),
        snr_threshold=30.0,
        kernel_file=str(kernel),
    )
    nested_glob.write_text("changed\n")
    nested_dir.write_text("changed\n")

    changes = gate._protected_changes()
    assert "src/deep/test_oracle.py" in changes
    assert "pkg/deep/benchmarks/oracle.bin" in changes


def test_snapshot_read_failure_is_an_integrity_violation(
    tmp_path: Path,
    monkeypatch,
):
    gate, workspace = _gate(tmp_path)
    driver = workspace / "forge_driver.py"
    original_read_bytes = Path.read_bytes

    def fail_driver_read(path: Path) -> bytes:
        if path == driver:
            raise OSError("permission denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_driver_read)

    reason = gate.finalize_integrity()
    assert gate.integrity_verdict == "violation"
    assert gate.integrity_violation is True
    assert "could not read protected path" in reason


def test_stop_blocks_before_validation_when_snapshot_read_fails(
    tmp_path: Path,
    monkeypatch,
):
    gate, workspace = _gate(tmp_path)
    driver = workspace / "forge_driver.py"
    original_read_bytes = Path.read_bytes
    validation_calls: list[int] = []

    def fail_driver_read(path: Path) -> bytes:
        if path == driver:
            raise OSError("permission denied")
        return original_read_bytes(path)

    async def unexpected_validation(**_kwargs):
        validation_calls.append(1)
        return {"passed": True}

    monkeypatch.setattr(Path, "read_bytes", fail_driver_read)
    monkeypatch.setattr(
        gate_module,
        "test_correctness",
        unexpected_validation,
    )

    result = asyncio.run(gate._on_stop({}, None, None))
    assert result["decision"] == "block"
    assert "errors=" in result["reason"]
    assert gate.integrity_violation is True
    assert validation_calls == []


def test_restore_protected_files_restores_nested_and_removes_added(
    tmp_path: Path,
):
    gate, workspace = _gate(tmp_path)
    nested = workspace / "pkg" / "tests" / "oracle.bin"
    nested.parent.mkdir(parents=True)
    nested.write_text("original\n")
    kernel = workspace / "aiter" / "csrc" / "kernel.cu"
    gate = InSessionGate(
        driver_script=str(workspace / "forge_driver.py"),
        snr_threshold=30.0,
        kernel_file=str(kernel),
    )
    nested.write_text("changed\n")
    added = workspace / "other" / "tests" / "new_oracle.bin"
    added.parent.mkdir(parents=True)
    added.write_text("new\n")

    assert gate.finalize_integrity()
    gate.restore_protected_files()

    assert nested.read_text() == "original\n"
    assert not added.exists()
    assert gate.integrity_verdict == "clean"


def test_driver_outside_the_workspace_does_not_move_the_measured_root(tmp_path: Path):
    """The declared workspace wins over the driver's own directory.

    ``forge-fuse`` writes its driver into the run's ``--output-dir``, which sits
    outside the framework tree and is not a repository. Inferring the root from
    the driver put ``git diff HEAD -- .`` in a non-repo directory, where git
    switches to its implicit ``--no-index`` mode, reads ``HEAD`` as a filename
    and exits 1 with ``Could not access 'HEAD'`` -- so the stop-time fingerprint
    raised on every session -- and pointed the protected-file inventory at a
    directory containing none of the protected files.
    """
    import subprocess

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "config.yaml").write_text("task_type: repository\n")
    kernel = workspace / "kernel.py"
    kernel.write_text("VALUE = 1\n")
    for cmd in (
        ["git", "init", "-q", "."],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "base"],
    ):
        subprocess.run(cmd, cwd=workspace, check=True, capture_output=True)

    outside = tmp_path / "run_output"
    outside.mkdir()
    driver = outside / "driver_fusion.py"
    driver.write_text("print('driver')\n")

    def build(**extra):
        return InSessionGate(
            driver_script=str(driver),
            snr_threshold=30.0,
            baseline_case_times={"case": 1.0},
            best_mean_case_speedup=1.0,
            kernel_file=str(kernel),
            target_files=[str(kernel)],
            **extra,
        )

    gate = build(workspace=workspace)
    assert gate.workspace_root == workspace.resolve()
    # The whole point: this is what raised GitError in production.
    assert isinstance(gate._candidate_diff_sha256(), str)
    # And the harness next to the kernel is protected again -- matched relative
    # to the tree the agent actually edits.
    assert gate._is_protected("config.yaml")

    # Without a declared workspace the driver's directory is still the fallback,
    # which is correct for every task that keeps its driver inside the tree.
    assert build().workspace_root == outside.resolve()
