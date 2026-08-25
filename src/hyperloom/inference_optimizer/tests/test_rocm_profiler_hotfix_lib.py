"""Unit tests for rocm_profiler_hotfix_lib.sh sync helper."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
HOTFIX_LIB = REPO_ROOT / "src/hyperloom/inference_optimizer/assets/rocm_profiler_hotfix_lib.sh"


def _run_sync(tmp_path: Path, *, rocm_lib: Path, torch_lib: Path, hip_name: str, tracer_name: str) -> subprocess.CompletedProcess:
    rocm_lib.mkdir(parents=True, exist_ok=True)
    torch_lib.mkdir(parents=True, exist_ok=True)
    (rocm_lib / hip_name).write_bytes(b"hip-hotfix-bytes")
    (rocm_lib / tracer_name).write_bytes(b"tracer-hotfix-bytes")
    (rocm_lib / "libamdhip64.so").symlink_to(hip_name)
    (rocm_lib / "libroctracer64.so").symlink_to(tracer_name)

    fake_py = tmp_path / "python"
    fake_py.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f'printf "%s\\n" "{torch_lib}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_py.chmod(0o755)

    runner = tmp_path / "runner.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'source "{HOTFIX_LIB}"',
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                f'ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR="{rocm_lib}"',
                f'resolve_python() {{ printf "%s" "{fake_py}"; }}',
                f'sync_rocm_profiler_libs_to_torch_lib "{rocm_lib}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return subprocess.run(["bash", str(runner)], capture_output=True, text=True, check=False)


def test_sync_rocm_profiler_libs_copies_into_torch_lib(tmp_path: Path):
    rocm_lib = tmp_path / "rocm" / "lib"
    torch_lib = tmp_path / "torch" / "lib"
    result = _run_sync(
        tmp_path,
        rocm_lib=rocm_lib,
        torch_lib=torch_lib,
        hip_name="libamdhip64.so.7.2.53211-hotfix",
        tracer_name="libroctracer64.so.4.1.70202",
    )
    assert result.returncode == 0, result.stderr
    assert (torch_lib / "libamdhip64.so").read_bytes() == b"hip-hotfix-bytes"
    assert (torch_lib / "libroctracer64.so").read_bytes() == b"tracer-hotfix-bytes"
