# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A fake interpreter build and wheel index for the AgentX client environment: the build's ``python3`` is a script
that answers the version query and does pip's ``download``, ``install`` and ``install --dry-run --report`` against a
directory of fake wheels, so staging runs its real subprocesses offline. Installing the fake agentx-harness lays down
the two aiperf files staging patches, recorded in its ``RECORD`` as pip records them."""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
from pathlib import Path

import pytest

from inference_suite import client_env
from inference_suite.pins import PinnedArchive
from inference_suite.tests import aiperf_source

VERSION = "3.12.15"
WHEELS = {"agentx-harness": "1.0.13", "aiohttp": "3.13.5", "datasets": "4.7.0"}
INDEX_ENV = "FAKE_WHEEL_INDEX"

_FAKE_PYTHON = r"""#!{python}
import base64, hashlib, json, os, pathlib, re, shutil, sys
args = sys.argv[1:]
if args[:2] == ["-I", "-c"]:
    print("{version}")
    sys.exit(0)
assert args[:3] == ["-I", "-m", "pip"], args
command, rest = args[3], args[4:]
index = pathlib.Path(os.environ["{index_env}"])
prefix = pathlib.Path(__file__).resolve().parents[1]
site = prefix / "lib" / "python3.12" / "site-packages"
def value(flag):
    return rest[rest.index(flag) + 1]
def pins():
    for line in pathlib.Path(value("-r")).read_text().splitlines():
        if line and not line.startswith("#"):
            match = re.fullmatch(r"(\S+)==(\S+) --hash=sha256:(\w+)", line)
            yield match[1], match[2], match[3]
def wheel(directory, name, version):
    return pathlib.Path(directory) / f"{{name.replace('-', '_')}}-{{version}}-py3-none-any.whl"
if command == "download":
    dest = pathlib.Path(value("--dest"))
    dest.mkdir(parents=True, exist_ok=True)
    for name, version, sha in pins():
        source = wheel(index, name, version)
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != sha:
            sys.exit(f"no wheel {{name}}=={{version}} at sha256 {{sha}}")
        shutil.copy(source, dest)
elif "--dry-run" in rest:
    install = []
    for path in sorted(index.glob("*.whl")):
        name, version = path.name.split("-")[:2]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        install.append({{"metadata": {{"name": name, "version": version}},
                        "download_info": {{"archive_info": {{"hashes": {{"sha256": digest}}}}}}}})
    pathlib.Path(value("--report")).write_text(json.dumps({{"install": install}}))
else:
    links = value("--find-links")
    for name, version, sha in pins():
        source = wheel(links, name, version)
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != sha:
            sys.exit(f"THESE PACKAGES DO NOT MATCH THE HASHES: {{name}}")
        info = site / f"{{name.replace('-', '_')}}-{{version}}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Name: {{name}}\nVersion: {{version}}\n")
        rows = [f"../../../bin/{{name}},sha256={{prefix}},1"]
        tree = index / f"{{name}}.tree"
        for path in sorted(tree.rglob("*") if tree.is_dir() else ()):
            if path.is_file():
                relative = path.relative_to(tree).as_posix()
                (site / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, site / relative)
                data = path.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                rows.append(f"{{relative}},sha256={{digest}},{{len(data)}}")
        (info / "RECORD").write_bytes("".join(row + "\r\n" for row in rows).encode())
"""


def _archive(version: str) -> bytes:
    script = _FAKE_PYTHON.format(python=sys.executable, version=version, index_env=INDEX_ENV).encode()
    members = {
        "python/bin/python3": (script, 0o755),
        "python/lib/python3.12/site-packages/pip-26.2.2.dist-info/METADATA": (b"Name: pip\n", 0o644),
        "python/lib/python3.12/os.py": (b"", 0o644),
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, (data, mode) in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeClient:
    """The fake build installed as the pin, the wheel index the fake pip reads, and every URL downloaded."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str = VERSION) -> None:
        self.archive = _archive(version)
        self.index = tmp_path / "index"
        self.index.mkdir()
        #: What installing the fake agentx-harness lays down in site-packages, by path.
        self.harness = self.index / "agentx-harness.tree"
        for relative, text in (
            (aiperf_source.PARSER_FILE, aiperf_source.PARSER),
            (aiperf_source.METRIC_FILE, aiperf_source.OUTPUT_LENGTH_METRIC),
        ):
            (self.harness / relative).parent.mkdir(parents=True, exist_ok=True)
            (self.harness / relative).write_text(text, encoding="utf-8")
        lines = []
        for name, wheel_version in WHEELS.items():
            path = self.index / f"{name.replace('-', '_')}-{wheel_version}-py3-none-any.whl"
            path.write_bytes(f"{name} {wheel_version}".encode())
            lines.append(f"{name}=={wheel_version} --hash=sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}\n")
        self.lock = tmp_path / "agentx-client.lock"
        self.lock.write_text("# fake lock\n" + "".join(lines))
        self.build = PinnedArchive(
            "https://example.invalid/python.tar.gz", hashlib.sha256(self.archive).hexdigest(), VERSION
        )
        self.downloads: list[str] = []
        monkeypatch.setenv(INDEX_ENV, str(self.index))
        monkeypatch.setattr(client_env, "AGENTX_PYTHON_BUILD", self.build)
        monkeypatch.setattr(client_env, "AGENTX_LOCK", self.lock)
        monkeypatch.setattr(client_env, "_download", self._download)

    def _download(self, url: str, destination: Path) -> None:
        self.downloads.append(url)
        destination.write_bytes(self.archive)

    def lock_shas(self) -> dict[str, str]:
        return {pin.name: pin.sha256 for pin in client_env.read_lock(self.lock.read_text())}
