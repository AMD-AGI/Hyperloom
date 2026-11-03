"""Run validator scripts inside a Docker container and read their JSON output.

The pack's validator defines what to measure. This module is the generic mechanism:
docker exec → run script → read JSON. No config globals, no domain knowledge.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path



def _docker_exec(
    container: str, script: str, env: dict[str, str], timeout: int,
) -> subprocess.CompletedProcess:
    from arbor.execute.workspace import HOST
    if container == HOST:
        # A host workspace: the script runs here, with the same variables.
        return subprocess.run(
            ["bash", script], env={**os.environ, **env},
            capture_output=True, text=True, timeout=timeout,
        )
    env_args = []
    for k, v in env.items():
        env_args += ["-e", f"{k}={v}"]
    return subprocess.run(
        ["docker", "exec"] + env_args + [container, "bash", script],
        capture_output=True, text=True, timeout=timeout,
    )


def _read_output(path: Path, container: str, container_path: str) -> tuple[bool, str]:
    """Read JSON from host path, falling back to docker cp if not bind-mounted."""
    from arbor.execute.workspace import HOST
    if path.exists():
        return True, path.read_text()
    if container == HOST:
        return False, ""
    try:
        r = subprocess.run(
            ["docker", "cp", f"{container}:{container_path}", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and path.exists():
            return True, path.read_text()
    except Exception:
        pass
    return False, ""


def _live_flags(config: dict) -> str:
    """Return the effective server_flags: workspace arbor/config.json wins.

    THE ASYMMETRY THIS FIXES, AND IT COST A CAMPAIGN A FICTITIOUS KEEP.
    ``workspace.py`` seeds arbor/config.json with BOTH ``server_env`` and
    ``server_flags``, and the packs tell the agent that file is how you change
    the operating point. But only the env half was ever merged: ``_live_env``
    below applies "workspace values win", while ``_start_server`` read
    ``config["server_flags"]`` straight off the STATIC campaign config. So an
    agent editing server_env changed the server, and an agent editing
    server_flags changed nothing — silently, with the file on disk showing the
    edit it believed had landed.

    Found by the cuda-graph campaign, which booked a class-A KEEP for landing
    ``--profiler-config`` and only discovered eight iterations later that
    /proc/<pid>/cmdline contained no such flag and POST /start_profile returned
    404, while the workspace config held the correct string and shell-quoting
    had been ruled out by expanding it inside the container.

    The consequence is larger than one KEEP: every flag-valued lever was
    unreachable. That includes ``--max-cudagraph-capture-size``, which is the
    named fix for graph eviction at the capture ceiling — measured at +3.0 to
    +3.4% on p90_interactivity. A campaign cannot land a flag fix through a
    path that discards flags.
    """
    from arbor.execute.workspace import read_workspace_config
    static = str(config.get("server_flags", "") or "")
    container = config.get("docker_container", "")
    workspace = config.get("workspace_root", "")
    if container and workspace:
        ws_cfg = read_workspace_config(container, workspace)
        ws_flags = ws_cfg.get("server_flags")
        # An ABSENT key means "no opinion" and keeps the static flags. An
        # explicit empty string means the agent deliberately cleared them, and
        # is honoured — the same distinction read_workspace_config already
        # makes for env.
        if ws_flags is not None:
            return str(ws_flags)
    return static


def _live_env(config: dict) -> dict[str, str]:
    """Return the effective server_env: static config merged with workspace arbor/config.json.

    KernelForge pattern: arbor/config.json lives inside workspace_root and is
    git-tracked alongside source patches. Whatever Claude last committed there
    is what the server actually runs with — both are rolled back together on REVERT.
    Static config values are the baseline; workspace values override them.
    """
    from arbor.execute.workspace import read_workspace_config
    static = {k: str(v) for k, v in config.get("server_env", {}).items()}
    container = config.get("docker_container", "")
    workspace = config.get("workspace_root", "")
    if container and workspace:
        ws_cfg = read_workspace_config(container, workspace)
        ws_env = {k: str(v) for k, v in ws_cfg.get("server_env", {}).items()}
        static.update(ws_env)  # workspace values win
    return static


def run_bench(
    config: dict,
    session_dir: str | Path,
    extra_env: dict[str, str] | None = None,
) -> dict:
    """Run the pack's bench script. Returns the JSON it writes."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    out = session_dir / "bench_output.json"

    container = config.get("docker_container") or ""
    script = config.get("bench_script") or ""
    if not container or not script:
        return {"status": "failed", "error": "config missing docker_container or bench_script"}
    timeout = int(config.get("bench_timeout", 600))

    env = {
        **_live_env(config),
        "PORT": str(config.get("port", 8888)),
        "MODEL_PATH": config.get("model_path", ""),
        "MODEL_NAME": config.get("model_name", ""),
        "ISL": str(config.get("isl", 1024)),
        "OSL": str(config.get("osl", 1024)),
        "CONCURRENCY": str(config.get("concurrency", 64)),
        "NUM_PROMPTS": str(config.get("num_prompts", 640)),
        "BENCH_OUTPUT": str(out),
        "BENCH_SERVING": config.get("bench_serving_script", ""),
        **(extra_env or {}),
    }

    try:
        r = _docker_exec(container, script, env, timeout)
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "bench script timed out"}
    except FileNotFoundError:
        return {"status": "failed", "error": "docker not found on PATH"}
    if r.returncode != 0:
        return {"status": "failed", "error": r.stderr or r.stdout}

    ok, text = _read_output(out, container, str(out))
    if not ok:
        return {"status": "failed", "error": "bench script did not write output"}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        return {"status": "failed", "error": f"invalid JSON: {e}"}

    raw.setdefault("status", "ok")
    return raw


def run_correctness(
    config: dict,
    session_dir: str | Path,
    extra_env: dict[str, str] | None = None,
    cap: float | None = None,
) -> dict:
    """Run the pack's correctness script. Returns {passed, score}.

    ``cap`` (seconds) bounds it below the pack's ``bench_timeout``: component mode's
    phase cap or the time left before the host's deadline. A run that cap cuts off says
    ``capped``, so the caller books nothing instead of a failed check.
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    out = session_dir / "correctness_output.json"

    container = config.get("docker_container") or ""
    script = config.get("correctness_script") or ""
    if not container or not script:
        return {"passed": False, "score": 0.0,
                "error": "config missing docker_container or correctness_script"}
    timeout = int(config.get("bench_timeout", 600))
    capped = cap is not None and cap < timeout
    if capped:
        timeout = max(1, int(cap))

    # The pack's eval size and floor (GSM8K_LIMIT, GSM8K_MIN) are campaign settings
    # and were dropped here, so the script ran its own defaults whatever the config
    # said. Read from the campaign config, never the workspace: the agent does not
    # get to move its own floor.
    from arbor.measure.accuracy import EVAL_KEYS
    static_env = config.get("server_env") or {}
    env = {
        "PORT": str(config.get("port", 8888)),
        "MODEL_PATH": config.get("model_path", ""),
        "MODEL_NAME": config.get("model_name", ""),
        "CORRECTNESS_OUTPUT": str(out),
        **{k: str(static_env[k]) for k in EVAL_KEYS if k in static_env},
        **(extra_env or {}),
    }

    try:
        r = _docker_exec(container, script, env, timeout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "score": 0.0, "error": "correctness script timed out",
                **({"capped": timeout} if capped else {})}
    except FileNotFoundError:
        return {"passed": False, "score": 0.0, "error": "docker not found on PATH"}
    if r.returncode != 0:
        return {"passed": False, "score": 0.0, "error": r.stderr or r.stdout}

    ok, text = _read_output(out, container, str(out))
    if not ok:
        return {"passed": False, "score": 0.0, "error": "correctness script did not write output"}
    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        return {"passed": False, "score": 0.0, "error": f"invalid JSON: {e}"}

    result.setdefault("passed", False)
    result.setdefault("score", 0.0)
    return result
