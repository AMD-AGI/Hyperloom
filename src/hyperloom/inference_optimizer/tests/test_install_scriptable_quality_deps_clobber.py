# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guards for ensure_xdit_quality_deps()'s torch-clobber protection.

The bug: ensure_xdit_quality_deps() ran a bare `pip install scikit-image lpips`.
lpips declares `torch>=0.4.0`, so pip's resolver pulled a PyPI (CUDA) torch and
REPLACED the vendor ROCm torch (and triton) already installed in the shared
venv. The command exited 0; the only signal was a cosmetic "not importable"
warn. Every framework co-tenant in that venv (atom/vllm/sglang) then failed at
`torch.cuda.is_available()` with "Found no NVIDIA driver".

The fix makes the optional install structurally unable to move the load-bearing
core:
  1. pin torch/torchvision/triton to their installed versions via `pip -c`;
  2. a post-install tripwire that aborts HARD (not a silent warn) if torch's
     ROCm build vanished anyway.

These tests extract the real shell functions from install.sh and run them with a
fake `$PYTHON` whose fake `pip` models the resolver: an UNCONSTRAINED install
swaps the ROCm torch for a CUDA build, a properly constrained one does not.
"""

from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
INSTALL_SH = (
    REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"
)

_ROCM_HIP = "7.2.53211"


def _extract_func(name: str) -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert m, f"could not locate {name}() in install.sh"
    return m.group(0)


def _extract_array(name: str) -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(name)}=\([^)]*\)", text, re.S | re.M)
    assert m, f"could not locate array {name} in install.sh"
    return m.group(0)


_FAKE_PYTHON = textwrap.dedent(
    '''\
    #!/usr/bin/env python3
    import os, re, sys

    args = sys.argv[1:]
    state = os.environ["XDIT_STATE"]
    piplog = os.environ["XDIT_PIPLOG"]
    mode = os.environ.get("CLOBBER_MODE", "respect")

    def read_hip():
        with open(state) as f:
            return f.read().strip()

    def write_hip(v):
        with open(state, "w") as f:
            f.write(v)

    # `python -c "<code>"`
    if args[:1] == ["-c"]:
        code = args[1]
        if "importlib.metadata" in code:
            pkg = re.search(r"version\\('([^']+)'\\)", code).group(1)
            versions = {
                "torch": "2.10.0+rocm7.2.4.test",
                "torchvision": "0.25.0+rocm7.2.4.test",
                "triton": "3.6.0+rocm7.2.4.test",
            }  # torchaudio intentionally absent -> exercises the skip path
            if pkg in versions:
                print(versions[pkg]); sys.exit(0)
            sys.stderr.write("no metadata\\n"); sys.exit(1)
        if "torch.version.hip" in code:
            print(read_hip()); sys.exit(0)
        if "import skimage" in code or "import lpips" in code:
            sys.exit(1)  # not importable -> triggers the install path
        sys.exit(0)

    # `python -m pip install ...`
    if args[:2] == ["-m", "pip"] and "install" in args:
        with open(piplog, "a") as f:
            f.write(" ".join(args) + "\\n")
        torch_pinned = False
        for i, a in enumerate(args):
            if a == "-c" and i + 1 < len(args):
                try:
                    with open(args[i + 1]) as cf:
                        if any(ln.strip().startswith("torch==") for ln in cf):
                            torch_pinned = True
                except OSError:
                    pass
        is_rollback = "-r" in args
        # Model the resolver: an unconstrained install pulls a CUDA torch and
        # drops the ROCm build (hip -> empty). A constrained one leaves it.
        if not is_rollback and (mode == "always" or not torch_pinned):
            write_hip("")
        sys.exit(0)

    sys.exit(0)
    '''
)


def _harness() -> str:
    parts = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'log()  { echo "[log] $*"; }',
        'warn() { echo "[warn] $*" >&2; }',
        'die()  { echo "[die] $*" >&2; exit 1; }',
        "CHECK_ONLY=0",
        "DRY_RUN=0",
        "PIP_EXTRA=()",
        'PYTHON="$FAKE_PYTHON"',
        _extract_array("_XDIT_QUALITY_DEPS"),
        _extract_array("_XDIT_CORE_PINS"),
        _extract_func("_write_core_constraints"),
        _extract_func("_torch_hip_version"),
        _extract_func("_guard_torch_not_clobbered"),
        _extract_func("ensure_xdit_quality_deps"),
        "ensure_xdit_quality_deps",
    ]
    return "\n\n".join(parts) + "\n"


def _run(tmp_path: Path, mode: str):
    fake_py = tmp_path / "fakepy"
    fake_py.write_text(_FAKE_PYTHON, encoding="utf-8")
    fake_py.chmod(0o755)
    state = tmp_path / "hip_state"
    state.write_text(_ROCM_HIP, encoding="utf-8")  # torch starts as a ROCm build
    piplog = tmp_path / "pip.log"
    piplog.write_text("", encoding="utf-8")
    script = tmp_path / "harness.sh"
    script.write_text(_harness(), encoding="utf-8")

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "FAKE_PYTHON": str(fake_py),
            "XDIT_STATE": str(state),
            "XDIT_PIPLOG": str(piplog),
            "CLOBBER_MODE": mode,
        },
    )
    return proc, state.read_text(encoding="utf-8").strip(), piplog.read_text(encoding="utf-8")


def test_constraints_prevent_torch_clobber(tmp_path: Path) -> None:
    # Realistic resolver ("respect"): a pinned install must NOT move torch.
    proc, hip_after, piplog = _run(tmp_path, mode="respect")
    assert proc.returncode == 0, proc.stdout
    # The dep install passed a constraints file (-c) to pip.
    install_lines = [
        ln for ln in piplog.splitlines() if "install" in ln and "-r" not in ln
    ]
    assert install_lines, f"no dep-install pip call recorded:\n{piplog}"
    assert any(" -c " in f" {ln} " for ln in install_lines), (
        f"dep install ran WITHOUT a constraints file:\n{piplog}"
    )
    # And torch is still the ROCm build afterwards.
    assert hip_after == _ROCM_HIP, (
        f"ROCm torch was clobbered despite the constraint (hip={hip_after!r})"
    )
    assert "[die]" not in proc.stdout, proc.stdout


def test_tripwire_aborts_when_torch_clobbered(tmp_path: Path) -> None:
    # If a clobber slips through anyway ("always"), the guard must abort HARD
    # (not silently warn) and attempt a rollback.
    proc, hip_after, piplog = _run(tmp_path, mode="always")
    assert proc.returncode != 0, f"expected hard abort, got rc=0:\n{proc.stdout}"
    assert "clobbered the load-bearing ROCm torch" in proc.stdout, proc.stdout
    assert hip_after == "", "sanity: clobber simulation should have emptied hip"
    # Rollback attempted: a force-reinstall --no-deps from the pinned file.
    assert any(
        "--force-reinstall" in ln and "--no-deps" in ln and "-r " in f" {ln} "
        for ln in piplog.splitlines()
    ), f"no rollback reinstall recorded:\n{piplog}"


def test_static_no_unconstrained_install_remains() -> None:
    body = _extract_func("ensure_xdit_quality_deps")
    # The dep install must carry the constraints file...
    assert '-c "$constraints"' in body, "dep install must pass the -c constraints file"
    # ...and the tripwire must run.
    assert "_guard_torch_not_clobbered" in body, "post-install tripwire must be invoked"
    # The old bare install (no constraints) must be gone.
    assert 'pip install --quiet --no-cache-dir \\\n    "${PIP_EXTRA[@]}"' not in body, (
        "the unconstrained pip install must not survive"
    )
