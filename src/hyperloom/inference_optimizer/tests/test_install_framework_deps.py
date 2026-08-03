# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behaviour of install.sh:ensure_framework_deps().

Scriptable frameworks run the model author's own code, which imports packages
no serving image ships. Rather than a special case per framework, a framework
declares its needs in assets/framework_deps/<framework>.txt and the installer
stays generic. These tests extract the real shell functions and run them
against a fake $PYTHON that models import probing and pip.

The load-bearing guard matters most: upstream manifests routinely pin
torch==2.7.1, and honouring that on a ROCm pod swaps the vendor torch for a
CUDA wheel and kills GPU access for every framework in the shared venv.
"""

from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ASSETS = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets"
INSTALL_SH = ASSETS / "install.sh"

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
    state = os.environ["FD_STATE"]          # newline-separated importable modules
    piplog = os.environ["FD_PIPLOG"]
    broken = set(filter(None, os.environ.get("FD_UNINSTALLABLE", "").split(",")))

    def installed():
        with open(state) as f:
            return set(filter(None, (ln.strip() for ln in f)))

    def add(mod):
        with open(state, "a") as f:
            f.write(mod + "\\n")

    if args[:1] == ["-c"]:
        script = args[1]
        if "importlib.metadata" in script:
            print("1.0.0")                   # every core pin has a version
            sys.exit(0)
        if "torch.version.hip" in script:
            print(os.environ.get("FD_HIP", ""))
            sys.exit(0)
        m = re.match(r"import ([A-Za-z_][A-Za-z0-9_]*)$", script.strip())
        sys.exit(0 if m and m.group(1) in installed() else 1)

    if args[:3] == ["-m", "pip", "install"]:
        spec = [a for a in args[3:] if not a.startswith("-")]
        # drop the constraints-file argument that follows -c
        if "-c" in args:
            cfile = args[args.index("-c") + 1]
            spec = [s for s in spec if s != cfile]
        spec = spec[-1] if spec else ""
        with open(piplog, "a") as f:
            f.write(spec + "\\n")
        if spec in broken:
            sys.exit(1)
        # import name: strip version specifier, '-' -> '_'
        add(re.split(r"[<>=!~;\\[]", spec)[0].strip().replace("-", "_"))
        sys.exit(0)

    sys.exit(0)
    '''
)


def _harness() -> str:
    return "\n\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'log()  { echo "[log] $*"; }',
            'warn() { echo "[warn] $*" >&2; }',
            'die()  { echo "[die] $*" >&2; exit 1; }',
            "CHECK_ONLY=0",
            "DRY_RUN=0",
            "PIP_EXTRA=()",
            'PYTHON="$FAKE_PYTHON"',
            '_script_dir="$FD_ASSET_DIR"',
            _extract_array("_XDIT_CORE_PINS"),
            _extract_array("_FRAMEWORK_DEPS_CORE_SKIP"),
            _extract_func("_write_core_constraints"),
            _extract_func("_torch_hip_version"),
            _extract_func("_guard_torch_not_clobbered"),
            _extract_func("_framework_dep_is_core"),
            _extract_func("ensure_framework_deps"),
            "ensure_framework_deps",
        ]
    ) + "\n"


def _run(tmp_path: Path, *, framework: str, manifest: str | None,
         preinstalled: tuple[str, ...] = (), uninstallable: tuple[str, ...] = (),
         asset_dir: Path | None = None):
    """Run ensure_framework_deps() and return (proc, pip_invocations)."""
    fake_py = tmp_path / "fakepy"
    fake_py.write_text(_FAKE_PYTHON, encoding="utf-8")
    fake_py.chmod(0o755)

    state = tmp_path / "installed"
    state.write_text("".join(f"{m}\n" for m in preinstalled), encoding="utf-8")
    piplog = tmp_path / "pip.log"
    piplog.write_text("", encoding="utf-8")

    if asset_dir is None:
        asset_dir = tmp_path / "assets"
        (asset_dir / "framework_deps").mkdir(parents=True)
        if manifest is not None:
            (asset_dir / "framework_deps" / f"{framework}.txt").write_text(
                manifest, encoding="utf-8"
            )

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
            "FD_STATE": str(state),
            "FD_PIPLOG": str(piplog),
            "FD_ASSET_DIR": str(asset_dir),
            "FD_HIP": _ROCM_HIP,
            "FD_UNINSTALLABLE": ",".join(uninstallable),
            "FRAMEWORK": framework,
        },
    )
    installs = [ln for ln in piplog.read_text(encoding="utf-8").splitlines() if ln]
    return proc, installs


def test_no_manifest_is_a_noop(tmp_path):
    """sglang/vllm ship their deps in the image and must not be touched."""
    proc, installs = _run(tmp_path, framework="sglang", manifest=None)
    assert proc.returncode == 0, proc.stdout
    assert installs == []
    assert "no manifest" in proc.stdout


def test_unset_framework_is_a_noop(tmp_path):
    proc, installs = _run(tmp_path, framework="", manifest=None)
    assert proc.returncode == 0, proc.stdout
    assert installs == []


def test_installs_only_what_is_missing(tmp_path):
    proc, installs = _run(
        tmp_path,
        framework="demo",
        manifest="einops\nplyfile\n",
        preinstalled=("plyfile",),
    )
    assert proc.returncode == 0, proc.stdout
    assert installs == ["einops"]


def test_import_name_override_is_honoured(tmp_path):
    """opencv-python imports as cv2; probing the pip name would reinstall it."""
    proc, installs = _run(
        tmp_path,
        framework="demo",
        manifest="opencv-python:cv2\n",
        preinstalled=("cv2",),
    )
    assert proc.returncode == 0, proc.stdout
    assert installs == []


def test_comments_blank_lines_and_inline_comments_are_stripped(tmp_path):
    manifest = "# header\n\n  einops   # why we need it\n\n# trailing note\n"
    proc, installs = _run(tmp_path, framework="demo", manifest=manifest)
    assert proc.returncode == 0, proc.stdout
    assert installs == ["einops"]


def test_load_bearing_packages_are_refused(tmp_path):
    """A manifest pinning torch must never reach pip on a ROCm pod."""
    proc, installs = _run(
        tmp_path,
        framework="demo",
        manifest="torch==2.7.1\nnumpy==1.26.4\neinops\n",
    )
    assert proc.returncode == 0, proc.stdout
    assert installs == ["einops"]
    assert "refusing load-bearing 'torch'" in proc.stdout
    assert "refusing load-bearing 'numpy'" in proc.stdout


def test_one_failing_package_does_not_block_the_others(tmp_path):
    """gsplat builds from sdist; its failure must not strand the cheap wheels."""
    proc, installs = _run(
        tmp_path,
        framework="demo",
        manifest="gsplat\neinops\n",
        uninstallable=("gsplat",),
    )
    assert proc.returncode == 0, proc.stdout
    assert installs == ["gsplat", "einops"]
    assert "unresolved: gsplat" in proc.stdout


def test_check_only_and_dry_run_short_circuit_before_pip(tmp_path):
    """Both diagnostic modes must return before the install loop."""
    body = _extract_func("ensure_framework_deps")
    for guard in ('if [ "$CHECK_ONLY" -eq 1 ]; then', 'if [ "$DRY_RUN" -eq 1 ]; then'):
        assert guard in body, f"missing {guard}"
        assert body.index(guard) < body.index("pip install"), (
            f"{guard} must precede the pip install loop"
        )


def test_shipped_worldmirror_manifest_parses_and_excludes_flash_attn(tmp_path):
    """The real manifest must stay parseable and keep flash_attn out.

    flash_attn has no ROCm wheel and is served by the SDPA shim; letting it into
    the manifest would make every install attempt fail.
    """
    manifest = (ASSETS / "framework_deps" / "worldmirror.txt").read_text(encoding="utf-8")
    proc, installs = _run(tmp_path, framework="worldmirror", manifest=manifest)
    assert proc.returncode == 0, proc.stdout
    assert "flash_attn" not in installs
    assert set(installs) == {
        "einops",
        "plyfile",
        "trimesh",
        "opencv-python",
        "gsplat",
        "onnxruntime",
        "moviepy",
        "pycolmap",
    }


def test_ensure_framework_deps_runs_in_install_sh(tmp_path):
    """The function is useless if the installer never calls it."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert re.search(r"^ensure_framework_deps$", text, re.M), (
        "ensure_framework_deps is defined but never invoked by install.sh"
    )
