# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builders of a baseline-prelude child's session for the scheduling tests: a framework git checkout, the
``state.json`` of a completed child with one kept patch on it, and a roofline profile."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hyperloom_multibranch.scheduling.prelude_reader import MB_LATEST_STATE_SCHEMA_VERSION

REMOTE_URL = "https://example.invalid/sglang.git"
COMPLETED_STOP_REASON = "time_exhausted"
BASELINE_TPUT = 1234.5
ROOFLINE_PEAK = 4000.0
PATCH_TEXT = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+b\n"
BASE_SHA = "ab" * 20
SETUP_SCRIPT = "#!/bin/bash\npip install x\n"
ROOFLINE_RUN_ID = "5f3e583beda4475e841f4bb7bacc9e4b"
TRACE_FILE = "1785743528.38-TP-0.trace.json.gz"
POINT_CONFIG = "benchmark:\n  envs:\n    CONC: 16\n    OSL: 1024\n"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(("git", "-C", str(root), *args), capture_output=True, text=True, check=True).stdout.strip()


def framework_checkout(parent: Path, remote: str = REMOTE_URL) -> Path:
    """A git checkout with one commit and, unless ``remote`` is empty, an ``origin`` remote."""

    root = parent / "framework"
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "f.py").write_text("a\n")
    _git(root, "add", "f.py")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "init")
    if remote:
        _git(root, "remote", "add", "origin", remote)
    return root


def head_of(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD")


def state_document(framework: Path, patch_file: Path) -> dict[str, object]:
    """A completed child session's ``state.json``, with one kept patch on ``framework``."""

    return {
        "schema_version": MB_LATEST_STATE_SCHEMA_VERSION,
        "stop_reason": COMPLETED_STOP_REASON,
        "baseline_tput": BASELINE_TPUT,
        "framework_version": "0.4.1",
        "gpu_type": "mi355x",
        "enablement": {
            "accepted_config": {"extra_server_args": " --mem-fraction-static 0.8 ", "extra_envs": {"FOO": "1"}},
            "kept_patches": [str(patch_file)],
            "patch_roots": {str(patch_file): str(framework)},
            "framework_root": str(framework),
            "base_sha_by_root": {str(framework): BASE_SHA},
            "setup_commands": ["pip install x"],
        },
        "roofline_snapshots": [
            {"theoretical_peak_tok_per_sec": ROOFLINE_PEAK},
            {"theoretical_peak_tok_per_sec": None},
        ],
    }


def write_child_session(session_dir: Path, state: dict[str, object]) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "state.json").write_text(json.dumps(state))
    return session_dir


def child_with_patch(tmp_path: Path) -> Path:
    """A completed child session whose one kept patch applies to a fresh framework checkout, with its setting script."""

    framework = framework_checkout(tmp_path)
    session = tmp_path / "child"
    patch_file = session / "enablement" / "1.patch"
    patch_file.parent.mkdir(parents=True)
    patch_file.write_text(PATCH_TEXT)
    script = session / "reports" / "enablement" / "enablement_setting.sh"
    script.parent.mkdir(parents=True)
    script.write_text(SETUP_SCRIPT)
    return write_child_session(session, state_document(framework, patch_file))


def with_roofline_profile(session: Path, run_id: str = ROOFLINE_RUN_ID) -> Path:
    """Add to the child ``session`` a roofline run ``run_id`` with a trace and its point config; return the trace
    directory."""

    run = session / "runs" / "roofline" / run_id
    trace_dir = run / "benchmark_sglang_20260803_074937" / "torch_trace"
    (trace_dir / "capture_traces").mkdir(parents=True)
    (trace_dir / TRACE_FILE).write_bytes(b"trace")
    (trace_dir / "capture_traces" / "bs_8_rank0.json.gz").write_bytes(b"capture")
    (run / "baseline_config.with_envs.yaml").write_text(POINT_CONFIG)
    return trace_dir
