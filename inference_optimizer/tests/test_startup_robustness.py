"""Tests for the startup-robustness preflight + launch-info wire format.

Covers five independent surfaces that all live in ``cli.py``:

* ``_validate_credentials`` — fail-fast gate for missing
  ``SAFE_API_KEY`` / ``OPENAI_BASE_URL``. Strict by design: no bypass
  flag or env var; specialist agents and the catalog probe both need
  live credentials so a run without them cannot finish anyway.
* ``_resolve_gpu_type`` — probe-first gpu_type resolution. The probe
  always wins on disagreement; the user-supplied value is only used
  when the probe fails (CPU sandbox / no rocm-smi).
* ``_emit_launch_info`` — machine-readable stdout sentinel +
  optional JSON dump for launcher scripts.
* ``_clean_stale_aiter_locks`` — sweep stale aiter JIT lock files
  left behind by killed runs (root cause of the three failed
  Qwen3-30B-A3B mi355x launches).
* ``_spawn_robustness_monitor`` — fork the robustness monitor as a
  setsid'd background daemon so a long run survives unexpected
  crashes without the operator remembering the cookbook.

These are pure-function entry points; the heavier integration of these
helpers into ``_run_optimize`` is exercised by the broader cli + auth
override tests.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from inference_optimizer import cli


# ---------------------------------------------------------------------------
# _validate_credentials
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_creds_env(monkeypatch):
    for var in ("SAFE_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_validate_credentials_passes_when_both_present(clean_creds_env):
    clean_creds_env.setenv("SAFE_API_KEY", "sk-fake")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    cli._validate_credentials()


def test_validate_credentials_exits_2_when_safe_api_key_missing(
    clean_creds_env, capsys,
):
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    with pytest.raises(SystemExit) as exc_info:
        cli._validate_credentials()
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    missing_line = err.split("Missing required credential(s):")[1].split("\n")[0]
    assert "SAFE_API_KEY" in missing_line
    assert "OPENAI_BASE_URL" not in missing_line


def test_validate_credentials_exits_2_when_openai_base_url_missing(
    clean_creds_env, capsys,
):
    clean_creds_env.setenv("SAFE_API_KEY", "sk-fake")
    with pytest.raises(SystemExit) as exc_info:
        cli._validate_credentials()
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "OPENAI_BASE_URL" in err


def test_validate_credentials_lists_both_missing(clean_creds_env, capsys):
    with pytest.raises(SystemExit):
        cli._validate_credentials()
    err = capsys.readouterr().err
    missing_line = err.split("Missing required credential(s):")[1].split("\n")[0]
    assert "SAFE_API_KEY" in missing_line
    assert "OPENAI_BASE_URL" in missing_line


def test_validate_credentials_no_bypass_paths(clean_creds_env):
    """Sanity check: setting HYPERLOOM_SKIP_CREDS_CHECK does NOT bypass.

    The bypass path was removed deliberately — credentials are required
    for any real run, so a "skip" knob would only ever mask the same
    failure showing up further into the pipeline.
    """
    clean_creds_env.setenv("HYPERLOOM_SKIP_CREDS_CHECK", "1")
    with pytest.raises(SystemExit) as exc_info:
        cli._validate_credentials()
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# _resolve_gpu_type
# ---------------------------------------------------------------------------
def test_resolve_gpu_type_probe_only():
    """No --gpu-type passed; probe wins."""
    gpu, warns = cli._resolve_gpu_type(user_specified="", probed="mi355x")
    assert gpu == "mi355x"
    assert warns == []


def test_resolve_gpu_type_user_only():
    """Probe failed (CPU sandbox); user value is used as-is, no warn."""
    gpu, warns = cli._resolve_gpu_type(user_specified="mi300x", probed="")
    assert gpu == "mi300x"
    assert warns == []


def test_resolve_gpu_type_agreement_silent():
    gpu, warns = cli._resolve_gpu_type(user_specified="mi355x", probed="mi355x")
    assert gpu == "mi355x"
    assert warns == []


def test_resolve_gpu_type_disagreement_probe_always_wins():
    """The historical footgun: --gpu-type mi300x on an MI355X host. New
    behaviour: probe wins unconditionally, warn loudly. No override
    knob; corrupting baseline + KB rows by trusting a wrong hint is
    worse than any plausible escape-hatch use case.
    """
    gpu, warns = cli._resolve_gpu_type(
        user_specified="mi300x", probed="mi355x",
    )
    assert gpu == "mi355x"
    assert len(warns) == 1
    assert "mi300x" in warns[0]
    assert "mi355x" in warns[0]


def test_resolve_gpu_type_no_inputs_returns_empty():
    """No probe, no user value — caller will fall back to Magpie's own
    detection or treat gpu_type as unset.
    """
    gpu, warns = cli._resolve_gpu_type(user_specified="", probed="")
    assert gpu == ""
    assert warns == []


# ---------------------------------------------------------------------------
# _emit_launch_info
# ---------------------------------------------------------------------------
def test_emit_launch_info_prints_kv_sentinel(tmp_path, capsys):
    session_dir = tmp_path / "model" / "20260101T000000Z"
    session_dir.mkdir(parents=True)
    info = cli._emit_launch_info(
        pid=12345,
        session_dir=session_dir,
        session_id="sess-xyz",
        run_log="/tmp/run.log",
        gpu_type="mi355x",
        framework="sglang",
        model="/models/qwen3",
        launch_info_file=None,
    )
    out = capsys.readouterr().out
    assert "HYPERLOOM_LAUNCH " in out
    line = [ln for ln in out.splitlines() if ln.startswith("HYPERLOOM_LAUNCH")][0]
    body = line[len("HYPERLOOM_LAUNCH "):]
    parsed = dict(token.split("=", 1) for token in body.split(" "))
    assert parsed["pid"] == "12345"
    assert parsed["session_dir"] == str(session_dir)
    assert parsed["session_id"] == "sess-xyz"
    assert parsed["gpu_type"] == "mi355x"
    assert parsed["framework"] == "sglang"
    assert parsed["model"] == "/models/qwen3"
    assert info["event"] == "launch"


def test_emit_launch_info_writes_json_file(tmp_path, capsys):
    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    out_file = tmp_path / "subdir" / "launch.json"
    cli._emit_launch_info(
        pid=7777,
        session_dir=session_dir,
        session_id="sid",
        run_log="",
        gpu_type="mi300x",
        framework="vllm",
        model="m",
        launch_info_file=str(out_file),
    )
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["pid"] == 7777
    assert data["session_dir"] == str(session_dir)
    assert data["session_id"] == "sid"
    assert data["framework"] == "vllm"
    assert data["manifest"] == str(session_dir / "manifest.json")
    out = capsys.readouterr().out
    assert "Launch info file" in out
    assert str(out_file) in out


def test_emit_launch_info_no_file_no_extra_print(tmp_path, capsys):
    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    cli._emit_launch_info(
        pid=1,
        session_dir=session_dir,
        session_id="s",
        run_log="",
        gpu_type="",
        framework="",
        model="",
        launch_info_file=None,
    )
    out = capsys.readouterr().out
    assert "Launch info file" not in out


# ---------------------------------------------------------------------------
# CLI flag wiring (parser end-to-end)
# ---------------------------------------------------------------------------
def test_parser_accepts_launch_info_file():
    parser = cli._build_parser()
    ns = parser.parse_args([
        "optimize",
        "--model", "/models/test",
        "--launch-info-file", "/tmp/launch.json",
    ])
    assert ns.launch_info_file == "/tmp/launch.json"


def test_parser_default_launch_info_file_is_none():
    parser = cli._build_parser()
    ns = parser.parse_args(["optimize", "--model", "/models/test"])
    assert ns.launch_info_file is None


def test_parser_does_not_expose_removed_bypass_flags():
    """Regression guard: --no-creds-check and --gpu-type-force were
    removed when the bypass design was rejected. Re-adding them would
    re-open the silent-wrong-baseline footgun. Lock the absence.
    """
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "optimize", "--model", "/m", "--no-creds-check",
        ])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "optimize", "--model", "/m", "--gpu-type-force",
        ])


def test_parser_robustness_monitor_default_on():
    parser = cli._build_parser()
    ns = parser.parse_args(["optimize", "--model", "/m"])
    assert ns.robustness_monitor is True


def test_parser_robustness_monitor_can_be_disabled():
    parser = cli._build_parser()
    ns = parser.parse_args([
        "optimize", "--model", "/m", "--no-robustness-monitor",
    ])
    assert ns.robustness_monitor is False


# ---------------------------------------------------------------------------
# _clean_stale_aiter_locks
# ---------------------------------------------------------------------------
def _make_aiter_tree(root):
    """Build a minimal aiter jit/build/ layout with mixed lock ages.

    Layout::

        <root>/
          lock_module_moe_stale            (stale, top-level)
          lock_module_moe_fresh            (fresh, top-level)
          module_moe/
            build/
              lock                         (stale)
              .ninja_lock                  (stale)
              compile_commands.json        (not a lock)
          module_other/
            build/
              lock                         (fresh)
          some_random.so                   (not a lock)
    """
    stale_mtime = time.time() - 30 * 60  # 30 min ago

    (root / "module_moe" / "build").mkdir(parents=True)
    (root / "module_other" / "build").mkdir(parents=True)

    top_stale = root / "lock_module_moe_stale"
    top_fresh = root / "lock_module_moe_fresh"
    inner_stale_lock = root / "module_moe" / "build" / "lock"
    inner_stale_ninja = root / "module_moe" / "build" / ".ninja_lock"
    inner_non_lock = root / "module_moe" / "build" / "compile_commands.json"
    inner_fresh = root / "module_other" / "build" / "lock"
    bare_so = root / "some_random.so"

    for p, content in (
        (top_stale, "stale"),
        (top_fresh, "fresh"),
        (inner_stale_lock, "stale"),
        (inner_stale_ninja, "stale"),
        (inner_non_lock, "{}"),
        (inner_fresh, "fresh"),
        (bare_so, "fake binary"),
    ):
        p.write_text(content)

    for p in (top_stale, inner_stale_lock, inner_stale_ninja):
        os.utime(p, (stale_mtime, stale_mtime))

    return {
        "stale_top": top_stale,
        "fresh_top": top_fresh,
        "stale_inner_lock": inner_stale_lock,
        "stale_inner_ninja": inner_stale_ninja,
        "non_lock": inner_non_lock,
        "fresh_inner": inner_fresh,
        "bare_so": bare_so,
    }


def test_clean_stale_aiter_locks_deletes_stale_keeps_fresh(tmp_path):
    layout = _make_aiter_tree(tmp_path)
    stats = cli._clean_stale_aiter_locks(
        aiter_jit_dir=tmp_path, stale_minutes=5,
    )
    assert stats["deleted"] == 3
    assert stats["skipped_fresh"] == 2
    assert stats["errors"] == 0
    assert not layout["stale_top"].exists()
    assert not layout["stale_inner_lock"].exists()
    assert not layout["stale_inner_ninja"].exists()
    assert layout["fresh_top"].exists()
    assert layout["fresh_inner"].exists()
    assert layout["non_lock"].exists()
    assert layout["bare_so"].exists()


def test_clean_stale_aiter_locks_handles_missing_dir():
    """When aiter cannot be located, return empty stats — never raise."""
    stats = cli._clean_stale_aiter_locks(
        aiter_jit_dir=type("X", (), {"is_dir": lambda self: False})(),  # noqa: E731
    )
    assert stats["scanned"] == 0
    assert stats["deleted"] == 0


def test_clean_stale_aiter_locks_respects_stale_minutes(tmp_path):
    """Bumping the threshold up keeps moderately-old locks alive."""
    (tmp_path / "lock_module_x").write_text("x")
    moderately_old = time.time() - 4 * 60  # 4 min ago
    os.utime(tmp_path / "lock_module_x", (moderately_old, moderately_old))
    stats = cli._clean_stale_aiter_locks(
        aiter_jit_dir=tmp_path, stale_minutes=10,
    )
    assert stats["deleted"] == 0
    assert stats["skipped_fresh"] == 1
    assert (tmp_path / "lock_module_x").exists()


def test_clean_stale_aiter_locks_auto_discovers_via_env_override(
    tmp_path, monkeypatch,
):
    """``$INFERENCE_OPTIMIZER_AITER_JIT_DIR`` should resolve when no
    explicit ``aiter_jit_dir`` argument is passed.
    """
    (tmp_path / "build").mkdir()
    stale_lock = tmp_path / "build" / "lock_module_z"
    stale_lock.write_text("x")
    stale_mtime = time.time() - 30 * 60
    os.utime(stale_lock, (stale_mtime, stale_mtime))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", str(tmp_path))
    stats = cli._clean_stale_aiter_locks(stale_minutes=5)
    assert stats["dir"] in {str(tmp_path), str(tmp_path / "build")}
    assert stats["deleted"] == 1
    assert not stale_lock.exists()


# ---------------------------------------------------------------------------
# _spawn_robustness_monitor
# ---------------------------------------------------------------------------
def test_spawn_robustness_monitor_missing_template_returns_none(
    tmp_path, capsys,
):
    """No template -> warn-only, return None (optimize must still run)."""
    repo_root = tmp_path / "fake_repo"
    repo_root.mkdir()
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    info = cli._spawn_robustness_monitor(
        session_dir=session_dir,
        optimizer_pid=99999,
        repo_root=repo_root,
        max_hours=1.0,
        target_gain=10,
        user_data_path=str(tmp_path),
    )
    assert info is None
    err = capsys.readouterr().err
    assert "robustness_monitor.sh.example not found" in err


def test_spawn_robustness_monitor_happy_path(tmp_path, capsys):
    """Template present -> copy + chmod + fork bash; monitor exits
    immediately because the optimizer pid is already dead (the session
    has no state.json so ``is_terminal_session`` exits non-zero, then
    /proc/<dead-pid> check forces a resume attempt; we kill the spawned
    subprocess before it actually re-execs anything).
    """
    repo_root = tmp_path / "repo"
    optimizer_runs = repo_root / "optimizer_runs"
    optimizer_runs.mkdir(parents=True)
    template = optimizer_runs / "robustness_monitor.sh.example"
    template.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"PID_FILE=$PID_FILE\"\n"
        "echo \"REPO_ROOT=$REPO_ROOT\"\n"
        "echo \"INFERENCE_OPTIMIZER_SESSION_DIR=$INFERENCE_OPTIMIZER_SESSION_DIR\"\n"
        "echo \"MAX_HOURS=$MAX_HOURS\"\n"
        "echo \"TARGET_GAIN=$TARGET_GAIN\"\n"
        "exit 0\n"
    )
    os.chmod(template, 0o755)

    session_dir = tmp_path / "session_xyz"
    session_dir.mkdir()

    info = cli._spawn_robustness_monitor(
        session_dir=session_dir,
        optimizer_pid=99999,
        repo_root=repo_root,
        max_hours=2.5,
        target_gain=20,
        user_data_path=str(tmp_path),
    )
    assert info is not None
    assert info["pid"] > 0
    monitor_script = session_dir / "optimizer_runs" / "robustness_monitor.sh"
    assert monitor_script.is_file()
    assert os.access(monitor_script, os.X_OK)
    pid_file = session_dir / "optimizer_runs" / "optimizer.pid"
    assert pid_file.read_text() == "99999"

    # The child is a short-lived stub; wait for it so its log is flushed.
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(info["pid"], 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    log_text = ""
    if (log_path := info.get("monitor_log")):
        # Log may have been written by the now-dead bash; capture and
        # verify it received the env we passed.
        try:
            log_text = open(log_path).read()
        except FileNotFoundError:
            log_text = ""
    assert f"PID_FILE={pid_file}" in log_text
    assert f"REPO_ROOT={repo_root}" in log_text
    assert f"INFERENCE_OPTIMIZER_SESSION_DIR={session_dir}" in log_text
    assert "MAX_HOURS=2.5" in log_text
    assert "TARGET_GAIN=20" in log_text


def test_spawn_robustness_monitor_skips_target_gain_when_zero(tmp_path):
    """target_gain=0 / None means the operator did not set a percent
    target (likely using --target-tput instead); the monitor should
    NOT carry over a meaningless TARGET_GAIN=0 to the resume launch
    where it would override the legitimate per-throughput target.
    """
    repo_root = tmp_path / "repo"
    optimizer_runs = repo_root / "optimizer_runs"
    optimizer_runs.mkdir(parents=True)
    template = optimizer_runs / "robustness_monitor.sh.example"
    template.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"TARGET_GAIN=${TARGET_GAIN:-<unset>}\"\n"
        "exit 0\n"
    )
    os.chmod(template, 0o755)
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    info = cli._spawn_robustness_monitor(
        session_dir=session_dir,
        optimizer_pid=99999,
        repo_root=repo_root,
        max_hours=1.0,
        target_gain=0,
        user_data_path=str(tmp_path),
    )
    assert info is not None
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(info["pid"], 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    log_text = open(info["monitor_log"]).read()
    assert "TARGET_GAIN=<unset>" in log_text
