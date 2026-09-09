# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a generated script under conditions we chose, not conditions it chose."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils import cap_build_parallelism

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 1800
_TAIL_CHARS = 4000


@dataclass
class SandboxResult:
    """What running a generated script produced."""

    ok: bool
    returncode: int | None
    elapsed_s: float
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    produced: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "elapsed_s": round(self.elapsed_s, 1),
            "timed_out": self.timed_out,
            "produced": list(self.produced),
            "stderr_tail": self.stderr_tail[-1200:],
        }


def run_generated_tuner(
    script: Path,
    work_dir: Path,
    *,
    expect: list[Path] | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    gpu_id: str = "0",
    env_overrides: dict[str, str] | None = None,
) -> SandboxResult:
    """Execute ``script`` and report what survived."""
    work_dir.mkdir(parents=True, exist_ok=True)
    expect = expect or []
    # Anything left from an earlier attempt would be read as this run's output.
    for path in expect:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not clear %s before the sandbox run: %s", path, exc)

    env = dict(os.environ)
    env.update(
        {
            "HIP_VISIBLE_DEVICES": gpu_id,
            "CUDA_VISIBLE_DEVICES": gpu_id,
            # A generated script has no business reaching the network, and saying so costs nothing even though it is
            # not enforcement.
            "no_proxy": "*",
        }
    )
    env.update(env_overrides or {})
    # After the overrides, so a caller that named a number keeps it. A generated
    # tuner reaches aiter the same way every other tuner does, and it is the one
    # least able to survive an OOM-killed build: it gets a single sandbox run and
    # a dead build reads as "the script it wrote does not work".
    cap_build_parallelism(env)

    log_path = work_dir / "sandbox.log"
    started = time.perf_counter()
    timed_out = False
    rc: int | None = None
    try:
        with log_path.open("w", encoding="utf-8") as sink:
            proc = subprocess.run(
                [sys.executable or "python3", str(script)],
                cwd=str(work_dir),
                env=env,
                stdout=sink,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        log.warning("tier3: generated tuner exceeded %ds; keeping what it wrote", timeout_s)
    except OSError as exc:
        return SandboxResult(
            False, None, time.perf_counter() - started, stderr_tail=f"could not start the script: {exc}"
        )

    elapsed = time.perf_counter() - started
    tail = ""
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-_TAIL_CHARS:]
    except OSError as exc:
        # The log is for diagnosis only.
        tail = f"(the run's log at {log_path} could not be read: {exc})"

    produced = [str(p) for p in expect if p.is_file()]
    ok = bool(expect) and len(produced) == len(expect)
    if not ok:
        missing = [p.name for p in expect if not p.is_file()]
        log.warning(
            "tier3: generated tuner produced %d of %d expected files (rc=%s, timed_out=%s); missing %s",
            len(produced),
            len(expect),
            rc,
            timed_out,
            missing,
        )
    return SandboxResult(ok, rc, elapsed, timed_out, tail, tail, produced)
