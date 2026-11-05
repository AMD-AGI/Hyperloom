"""A tuner's spawned workers start without the profiler registration their parent exported, and the sitecustomize
they would otherwise load still runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from kernelforge.workflows.gemm_tune.profiler_env import without_inherited_profiler
from kernelforge.workflows.gemm_tune.utils import run_subprocess

_CHILD = """\
import multiprocessing as mp, os

def seen(_):
    return os.environ.get("ROCPROFILER_REGISTER_LIBRARY", ""), os.environ.get("IMAGE_SITE_RAN", "")

if __name__ == "__main__":
    os.environ["ROCPROFILER_REGISTER_LIBRARY"] = "/opt/rocm/lib/librocprofiler-sdk.so"
    with mp.get_context("spawn").Pool(1) as pool:
        print(pool.map(seen, [0])[0])
"""


def test_a_worker_spawned_after_the_parent_profiled_does_not_inherit_the_registration(tmp_path: Path) -> None:
    image_site = tmp_path / "image-site"
    image_site.mkdir()
    (image_site / "sitecustomize.py").write_text("import os\nos.environ['IMAGE_SITE_RAN'] = '1'\n")
    script = tmp_path / "tuner.py"
    script.write_text(_CHILD)
    env = without_inherited_profiler({**os.environ, "PYTHONPATH": str(image_site)})

    done = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, check=True)

    assert done.stdout.strip() == "('', '1')"


def test_every_tuner_command_gets_the_guard(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROCPROFILER_REGISTER_LIBRARY", "/opt/rocm/lib/librocprofiler-sdk.so")
    code = "import os; print(os.environ.get('ROCPROFILER_REGISTER_LIBRARY', 'unset'))"

    rc, out, _err = run_subprocess([sys.executable, "-c", code], cwd=tmp_path, timeout_s=60)

    assert (rc, out.strip()) == (0, "unset")
