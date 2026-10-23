# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the bootstrap script runs against in a test: a stand-in for the framework image, a source tree and a wheelhouse.

The framework image's ``python3`` is a venv (``/opt/venv``) that holds the stack (torch, sglang, aiter). The rig's
base is such a venv, with a module only the base has (``base_only_pkg``), a package the lock also pins and the base
already has (``already_pkg``), and an older ``lock_pkg`` that the lock replaces. The source tree is a Hyperloom
checkout in miniature: a ``runner`` that records what it can import, and a ``requirements/runner.lock`` over a local
wheelhouse. Nothing is fetched from an index.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PROBE_FILE",
    "Rig",
    "build_wheel",
    "copy_source",
    "lock_text",
    "make_rig",
    "pip_env",
    "read_probe",
    "run_script",
]

PROBE_FILE = "probe.json"

_RUNNER_MAIN = """\
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hyperloom.measure.runner")
    command = parser.add_subparsers(dest="command", required=True).add_parser("run")
    command.add_argument("--in", dest="in_dir", type=Path)
    command.add_argument("--out", dest="out_dir", type=Path)
    command.add_argument("--scratch", type=Path)
    command.add_argument("--spec-wait-sec", type=float, default=600.0)
    args = parser.parse_args(argv)
    import already_pkg
    import base_only_pkg
    import lock_pkg

    probe = {
        "argv": sys.argv[1:],
        "executable": sys.executable,
        "base_only": base_only_pkg.__file__,
        "already": already_pkg.__file__,
        "lock_file": lock_pkg.__file__,
        "lock_origin": lock_pkg.ORIGIN,
        "revision": os.environ.get("HYPERLOOM_CODE_REVISION"),
        "bootstrap_started": os.environ.get("HYPERLOOM_BOOTSTRAP_STARTED_UNIX"),
        "bootstrap_ended": os.environ.get("HYPERLOOM_BOOTSTRAP_ENDED_UNIX"),
        "signing_key_env": os.environ.get("HYPERLOOM_SIGNING_KEY_FILE"),
        "runner_file": __file__,
    }
    (args.out_dir / "%(probe)s").write_text(json.dumps(probe))
    spec = args.in_dir / "spec.json"
    deadline = time.monotonic() + args.spec_wait_sec
    while not spec.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.05)
    shutil.copy(spec, args.out_dir / "status.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
""" % {"probe": PROBE_FILE}


@dataclass(frozen=True)
class Rig:
    """The pieces a bootstrap run needs; every path is below ``root``, so one read-only bind shows it all to a sandbox.

    Attributes:
        root: The directory that holds everything.
        base_python: The base interpreter, a venv with ``base_only_pkg``, ``already_pkg`` and an old ``lock_pkg``.
        base_site: Its site-packages directory.
        source: A fake checkout whose ``requirements/runner.lock`` pins ``lock_pkg`` and ``already_pkg``.
        wheelhouse: The directory the wheels are in (``PIP_FIND_LINKS``).
        lock_pkg_sha256: The hash of the ``lock_pkg`` wheel, for a lock of a test's own.
    """

    root: Path
    base_python: Path
    base_site: Path
    source: Path
    wheelhouse: Path
    lock_pkg_sha256: str
    already_pkg_sha256: str

    def sandbox_env(self, source: Path | None = None, scratch: str = "/work/scratch") -> dict[str, str]:
        """The environment of a sandbox template that starts the script on this rig, below ``root`` and ``scratch``."""
        return {
            "HYPERLOOM_SCRATCH_DIR": scratch,
            "HYPERLOOM_BASE_PYTHON": str(self.base_python),
            "HYPERLOOM_SOURCE_DIR": str(source or self.source),
            **pip_env(self.wheelhouse),
        }


def build_wheel(
    directory: Path, name: str, version: str, module_source: str, requires: str | None = None
) -> tuple[Path, str]:
    """A pure-Python wheel of one module ``name`` and its SHA-256 (what ``--hash`` names).

    ``requires`` is a ``Requires-Dist`` the wheel declares, one the lock does not pin.
    """
    dist_info = f"{name}-{version}.dist-info"
    declared = f"Requires-Dist: {requires}\n" if requires else ""
    files = {
        f"{name}.py": module_source,
        f"{dist_info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{declared}",
        f"{dist_info}/WHEEL": "Wheel-Version: 1.0\nGenerator: rig\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(text.encode())}")
    files[f"{dist_info}/RECORD"] = "\n".join([*record, f"{dist_info}/RECORD,,"]) + "\n"
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
    return wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()


def lock_text(**pins: tuple[str, str]) -> str:
    """A hash lock of ``name=(version, sha256)`` entries, the shape ``pip-compile --generate-hashes`` writes."""
    return "".join(f"{name}=={version} \\\n    --hash=sha256:{digest}\n" for name, (version, digest) in pins.items())


def pip_env(wheelhouse: Path) -> dict[str, str]:
    """The environment a template sets so that pip reads the wheelhouse and no index."""
    return {"PIP_NO_INDEX": "1", "PIP_FIND_LINKS": str(wheelhouse)}


def _pip_install(python: Path, wheelhouse: Path, *requirements: str) -> None:
    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", "--no-deps", "--disable-pip-version-check", *requirements],
        env={**os.environ, **pip_env(wheelhouse)},
        check=True,
        capture_output=True,
    )


def copy_source(rig: Rig, destination: Path, lock: str | None = None) -> Path:
    """A copy of the rig's source tree at ``destination``, with ``lock`` as its ``requirements/runner.lock`` if given."""
    shutil.copytree(rig.source, destination)
    if lock is not None:
        (destination / "requirements" / "runner.lock").write_text(lock)
    return destination


def make_rig(root: Path) -> Rig:
    """Build the rig below the new directory ``root``."""
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    _, lock_sha = build_wheel(wheelhouse, "lock_pkg", "1.0", "ORIGIN = 'lock'\n", requires="unlisted_pkg>=1")
    _, already_sha = build_wheel(wheelhouse, "already_pkg", "1.0", "ORIGIN = 'wheelhouse'\n")
    old, _ = build_wheel(root, "lock_pkg", "0.1", "ORIGIN = 'base'\n")
    base = root / "base"
    subprocess.run([sys.executable, "-m", "venv", str(base)], check=True, capture_output=True)
    python = base / "bin" / "python"
    site = Path(
        subprocess.run(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    (site / "base_only_pkg.py").write_text("ORIGIN = 'base'\n")
    _pip_install(python, wheelhouse, "already_pkg==1.0")
    _pip_install(python, root, str(old))
    source = root / "source"
    runner = source / "src" / "hyperloom" / "measure" / "runner"
    runner.mkdir(parents=True)
    for package in (runner.parent.parent, runner.parent, runner):
        (package / "__init__.py").write_text("")
    (runner / "__main__.py").write_text(_RUNNER_MAIN)
    (source / "requirements").mkdir()
    (source / "requirements" / "runner.lock").write_text(
        lock_text(lock_pkg=("1.0", lock_sha), already_pkg=("1.0", already_sha))
    )
    return Rig(root, python, site, source, wheelhouse, lock_sha, already_sha)


def run_script(
    command: tuple[str, ...], rig: Rig, work: Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a start ``command`` with the rig's environment and ``work`` as the sandbox's ``/in``, ``/out``, scratch.

    The run's environment holds only what a template would set: no variable of this process reaches it except ``PATH``.
    """
    for name in ("in", "out"):
        (work / name).mkdir(exist_ok=True)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(work),
        "TMPDIR": str(work),
        "HYPERLOOM_IN_DIR": str(work / "in"),
        "HYPERLOOM_OUT_DIR": str(work / "out"),
        **rig.sandbox_env(scratch=str(work / "scratch")),
        **(extra_env or {}),
    }
    return subprocess.run(command, env=env, capture_output=True, text=True, timeout=180, check=False)


def read_probe(out: Path) -> dict[str, object]:
    """What the fake runner recorded in ``out``."""
    return json.loads((out / PROBE_FILE).read_text())
