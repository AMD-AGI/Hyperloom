# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the bootstrap script runs against in a test: a stand-in for the framework image and a runner cache.

The framework image's ``python3`` is a venv (``/opt/venv``) that holds the stack (torch, sglang, aiter). The rig's
base is such a venv, with a module only the base has (``base_only_pkg``), a package the lock also pins and the base
already has (``already_pkg``), and an older ``lock_pkg`` that the lock replaces. The source is a Hyperloom checkout in
miniature, a git repository whose ``runner`` records what it can import and whose ``requirements/runner.lock`` pins
wheels of a local wheelhouse; the runner cache is what :mod:`hyperloom.measure.runner.cache_build` makes of it, the
way a host does. Nothing is fetched from an index, and the script's environment points every network setting at a
listener that must never be reached (:class:`Tripwire`).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from hyperloom.measure.runner.cache_build import BuildOptions, WheelSource, ensure_runner_cache
from hyperloom.measure.runner.cache_layout import CACHE_ENV

__all__ = [
    "PROBE_FILE",
    "UNFETCHABLE",
    "CacheRef",
    "Commit",
    "Rig",
    "Tripwire",
    "build_sdist",
    "build_wheel",
    "commit_edit",
    "commit_variant",
    "copy_cache",
    "lock_text",
    "make_rig",
    "read_probe",
    "run_script",
    "tripwire",
]

PROBE_FILE = "probe.json"
#: A build requirement that the rig's source distribution declares and that no wheelhouse or index serves.
UNFETCHABLE = "unfetchable_build_dep>=1"
_GIT = ("git", "-c", "user.name=t", "-c", "user.email=t@example.invalid")

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
    import sdist_pkg

    probe = {
        "argv": sys.argv[1:],
        "executable": sys.executable,
        "base_only": base_only_pkg.__file__,
        "already": already_pkg.__file__,
        "lock_file": lock_pkg.__file__,
        "lock_origin": lock_pkg.ORIGIN,
        "sdist_origin": sdist_pkg.ORIGIN,
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


def _wheelhouse_only(wheelhouse: Path) -> BuildOptions:
    return BuildOptions(source=WheelSource(no_index=True, find_links=(str(wheelhouse),)))


@dataclass(frozen=True)
class CacheRef:
    """A runner cache directory and the commit whose entry a sandbox is told to use."""

    root: Path
    sha: str


@dataclass(frozen=True)
class Commit:
    """A commit of a repository on disk."""

    repo: Path
    sha: str


@dataclass(frozen=True)
class Rig:
    """The pieces a bootstrap run needs; every path is below ``root``, so one read-only bind shows it all to a sandbox.

    Attributes:
        root: The directory that holds everything.
        base_python: The base interpreter, a venv with ``base_only_pkg``, ``already_pkg`` and an old ``lock_pkg``.
        base_site: Its site-packages directory.
        repo: A git repository with the miniature checkout, whose ``requirements/runner.lock`` pins ``lock_pkg``,
            ``already_pkg`` and ``sdist_pkg``.
        wheelhouse: The directory the builder takes the wheels from; the script never sees it.
        cache: The runner cache built from ``repo`` and the commit it was built for.
        lock_pkg_sha256: The hash of the ``lock_pkg`` wheel, for a lock of a test's own.
        already_pkg_sha256: The hash of the ``already_pkg`` wheel.
        sdist_pkg_sha256: The hash of the ``sdist_pkg`` source distribution, the only file the wheelhouse has of it.
        tools_sha256: The hash of the ``setuptools`` wheel the lock pins as the build tool of source distributions.
    """

    root: Path
    base_python: Path
    base_site: Path
    repo: Path
    wheelhouse: Path
    cache: CacheRef
    lock_pkg_sha256: str
    already_pkg_sha256: str
    sdist_pkg_sha256: str
    tools_sha256: str

    @property
    def source(self) -> Path:
        """The source the script runs the runner from: ``source`` of the cache entry."""
        return self.cache.root / self.cache.sha / "source"

    def sandbox_env(self, cache: CacheRef | None = None, scratch: str = "/work/scratch") -> dict[str, str]:
        """The environment of a sandbox template that starts the script on this rig, below ``root`` and ``scratch``."""
        use = cache or self.cache
        return {
            "HYPERLOOM_SCRATCH_DIR": scratch,
            "HYPERLOOM_BASE_PYTHON": str(self.base_python),
            CACHE_ENV: str(use.root),
            "HYPERLOOM_SOURCE_SHA": use.sha,
        }

    def build_options(self) -> BuildOptions:
        """Build from the wheelhouse alone."""
        return _wheelhouse_only(self.wheelhouse)


class Tripwire:
    """A local listener that counts the connections made to it, for every network setting to point at.

    ``env`` names it as the package index, the extra index, the find-links page and the proxy; a script that obeyed any
    of them would make ``connections`` non-zero. It accepts and drops, so a client that does connect gives up fast.
    """

    def __init__(self) -> None:
        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(0.05)
        self._stop = threading.Event()
        self.connections = 0
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        url = f"http://127.0.0.1:{self._server.getsockname()[1]}"
        self.env = {
            "PIP_INDEX_URL": f"{url}/simple",
            "PIP_EXTRA_INDEX_URL": f"{url}/extra",
            "PIP_FIND_LINKS": f"{url}/links",
            "PIP_PROXY": url,
            "HTTP_PROXY": url,
            "HTTPS_PROXY": url,
            "http_proxy": url,
            "https_proxy": url,
        }

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except (TimeoutError, OSError):
                continue
            self.connections += 1
            client.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        self._server.close()


@contextmanager
def tripwire() -> Iterator[Tripwire]:
    """A :class:`Tripwire` for the block."""
    wire = Tripwire()
    try:
        yield wire
    finally:
        wire.close()


def build_wheel(
    directory: Path,
    name: str,
    version: str,
    module_source: str,
    requires: str | None = None,
    tag: str = "py3-none-any",
    requires_python: str | None = None,
) -> tuple[Path, str]:
    """A wheel of one module ``name`` tagged ``tag`` (pure Python unless told) and its SHA-256 (what ``--hash`` names).

    ``requires`` is a ``Requires-Dist`` the wheel declares, one the lock does not pin; ``requires_python`` its
    ``Requires-Python``.
    """
    dist_info = f"{name}-{version}.dist-info"
    declared = f"Requires-Dist: {requires}\n" if requires else ""
    declared += f"Requires-Python: {requires_python}\n" if requires_python else ""
    files = {
        f"{name}.py": module_source,
        f"{dist_info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{declared}",
        f"{dist_info}/WHEEL": "Wheel-Version: 1.0\nGenerator: rig\nRoot-Is-Purelib: true\nTag: {tag}\n",
    }
    record = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(text.encode())}")
    files[f"{dist_info}/RECORD"] = "\n".join([*record, f"{dist_info}/RECORD,,"]) + "\n"
    wheel = directory / f"{name}-{version}-{tag}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
    return wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()


_SDIST_BACKEND = """\
import pathlib
import zipfile


def get_requires_for_build_wheel(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name, version, tag = %(name)r, %(version)r, %(tag)r
    dist_info = f"{name}-{version}.dist-info"
    wheel = f"{name}-{version}-{tag}.whl"
    with zipfile.ZipFile(pathlib.Path(wheel_directory) / wheel, "w") as archive:
        archive.writestr(f"{name}.py", pathlib.Path(f"{name}.py").read_text())
        archive.writestr(f"{dist_info}/METADATA", f"Metadata-Version: 2.1\\nName: {name}\\nVersion: {version}\\n")
        archive.writestr(f"{dist_info}/WHEEL", f"Wheel-Version: 1.0\\nGenerator: rig\\nRoot-Is-Purelib: true\\nTag: {tag}\\n")
        archive.writestr(f"{dist_info}/RECORD", "")
    return wheel
"""


def build_sdist(
    directory: Path,
    name: str,
    version: str,
    module_source: str,
    wheel_tag: str = "py3-none-any",
    requires: tuple[str, ...] = (),
) -> tuple[Path, str]:
    """A source distribution of one module ``name`` and its SHA-256, which builds the wheel tagged ``wheel_tag``.

    It builds with an in-tree PEP 517 backend that needs nothing from an index, as an index-less builder must be able to;
    ``requires`` are build requirements it declares, which the backend never uses and no index has.
    """
    top = f"{name}-{version}"
    files = {
        "pyproject.toml": f"[build-system]\nrequires = {list(requires)!r}\n"
        'build-backend = "backend"\nbackend-path = ["."]\n',
        "backend.py": _SDIST_BACKEND % {"name": name, "version": version, "tag": wheel_tag},
        f"{name}.py": module_source,
        "PKG-INFO": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
    }
    sdist = directory / f"{top}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for path, text in files.items():
            member = tarfile.TarInfo(f"{top}/{path}")
            member.size = len(text.encode())
            archive.addfile(member, io.BytesIO(text.encode()))
    return sdist, hashlib.sha256(sdist.read_bytes()).hexdigest()


def lock_text(**pins: tuple[str, str]) -> str:
    """A hash lock of ``name=(version, sha256)`` entries, the shape ``pip-compile --generate-hashes`` writes."""
    return "".join(f"{name}=={version} \\\n    --hash=sha256:{digest}\n" for name, (version, digest) in pins.items())


def _pip_install(python: Path, wheelhouse: Path, *requirements: str) -> None:
    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", "--no-deps", "--disable-pip-version-check", *requirements],
        env={**os.environ, "PIP_NO_INDEX": "1", "PIP_FIND_LINKS": str(wheelhouse)},
        check=True,
        capture_output=True,
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run([*_GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _commit_all(repo: Path) -> str:
    """Commit everything in ``repo`` and return the commit hash."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "runner")
    return _git(repo, "rev-parse", "HEAD")


def commit_edit(rig: Rig, work: Path, edit: Callable[[Path], None]) -> Commit:
    """A commit of a copy of the rig's repository that ``edit`` has changed.

    ``work`` is a directory that exists; the copy is ``work/repo``. Nothing is built.
    """
    repo = work / "repo"
    shutil.copytree(rig.repo, repo)
    edit(repo)
    return Commit(repo, _commit_all(repo))


def commit_variant(rig: Rig, work: Path, edit: Callable[[Path], None]) -> CacheRef:
    """A cache of its own, ``work/cache``, built from a commit that ``edit`` changed (see :func:`commit_edit`)."""
    edited = commit_edit(rig, work, edit)
    ensure_runner_cache(edited.repo, edited.sha, work / "cache", rig.build_options())
    return CacheRef(work / "cache", edited.sha)


def copy_cache(rig: Rig, destination: Path) -> CacheRef:
    """A copy of the rig's cache at ``destination``, for a test that damages it."""
    shutil.copytree(rig.cache.root, destination)
    return CacheRef(destination, rig.cache.sha)


def make_rig(root: Path) -> Rig:
    """Build the rig below the new directory ``root``."""
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    _, lock_sha = build_wheel(wheelhouse, "lock_pkg", "1.0", "ORIGIN = 'lock'\n", requires="unlisted_pkg>=1")
    _, already_sha = build_wheel(wheelhouse, "already_pkg", "1.0", "ORIGIN = 'wheelhouse'\n")
    _, sdist_sha = build_sdist(wheelhouse, "sdist_pkg", "1.0", "ORIGIN = 'sdist'\n", requires=(UNFETCHABLE,))
    _, tools_sha = build_wheel(wheelhouse, "setuptools", "79.0.1", "ORIGIN = 'build tool'\n")
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
    repo = root / "repo"
    runner = repo / "src" / "hyperloom" / "measure" / "runner"
    runner.mkdir(parents=True)
    for package in (runner.parent.parent, runner.parent, runner):
        (package / "__init__.py").write_text("")
    (runner / "__main__.py").write_text(_RUNNER_MAIN)
    (repo / "requirements").mkdir()
    (repo / "requirements" / "runner.lock").write_text(
        lock_text(
            lock_pkg=("1.0", lock_sha),
            already_pkg=("1.0", already_sha),
            sdist_pkg=("1.0", sdist_sha),
            setuptools=("79.0.1", tools_sha),
        )
    )
    _git(repo, "init", "-q")
    sha = _commit_all(repo)
    ensure_runner_cache(repo, sha, root / "cache", _wheelhouse_only(wheelhouse))
    return Rig(
        root, python, site, repo, wheelhouse, CacheRef(root / "cache", sha), lock_sha, already_sha, sdist_sha, tools_sha
    )


def run_script(
    command: tuple[str, ...],
    rig: Rig,
    work: Path,
    extra_env: dict[str, str] | None = None,
    cache: CacheRef | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a start ``command`` with the rig's environment and ``work`` as the sandbox's ``/in``, ``/out``, scratch.

    ``cache`` is the runner cache the sandbox is told to use, the rig's own when omitted.

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
        **rig.sandbox_env(cache, scratch=str(work / "scratch")),
        **(extra_env or {}),
    }
    return subprocess.run(command, env=env, capture_output=True, text=True, timeout=180, check=False)


def read_probe(out: Path) -> dict[str, object]:
    """What the fake runner recorded in ``out``."""
    return json.loads((out / PROBE_FILE).read_text())
