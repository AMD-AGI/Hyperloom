# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A stand-in ruppell tree, staged into a runner cache, for the tests of :mod:`hyperloom.measure.ruppell`.

ruppell is a private repository, so its tree is not in this one. The stand-in has the files the backend calls, each
doing what ruppell's does at ``0ef9b0d`` for the part the backend reads:

* ``runtime/build.sh`` writes the libraries into ``RUPPELL_BUILD`` and records its environment (``FAKE_BUILD_RC``
  makes it fail, ``FAKE_BUILD_SLEEP`` makes it slow);
* ``python -m ruppell run`` sets ``ruppell run``'s variables (``cli.py``'s ``cmd_run`` on ROCm, early load on) and
  runs the command;
* ``python -m ruppell census`` writes the census files from ``data/ruppell`` (a real W243 summary and a census that
  ruppell's own ``census`` made of its offline records), ``export`` writes ``report.json``, and
  ``tests/m2/check_m2.py`` prints a check (``FAKE_CHECK_OK=0`` fails it, ``FAKE_CENSUS_RC`` fails the census,
  ``FAKE_SUMMARY`` names another census summary).

The real tree was run on CPU through the same calls (the PR's evidence); these pin the backend's side.
"""

from __future__ import annotations

import io
import tarfile
import textwrap
from pathlib import Path

DATA = Path(__file__).parent / "data" / "ruppell"
COMMIT = "1" * 40

_BUILD = """\
#!/bin/bash
set -e
[ -n "${FAKE_BUILD_SLEEP:-}" ] && sleep "$FAKE_BUILD_SLEEP"
[ -n "${FAKE_BUILD_RC:-}" ] && { echo "hipcc: error: no ROCm" >&2; exit "$FAKE_BUILD_RC"; }
mkdir -p "$RUPPELL_BUILD"
for lib in libruppell_tool.so libruppell_marker.so libruppell_nccl_profiler.so; do echo x > "$RUPPELL_BUILD/$lib"; done
echo rocm > "$RUPPELL_BUILD/PLATFORM"
env | grep -E '^(RUPPELL_|ROCM_PATH=)' | sort > "$RUPPELL_BUILD/BUILD_ENV"
"""

_MAIN = """\
import json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.environ.get("FAKE_RUPPELL_DATA", "")
cmd, args = sys.argv[1], sys.argv[2:]
if cmd == "run":
    build = args[args.index("--build") + 1]
    out = args[args.index("--out") + 1]
    command = args[args.index("--") + 1:]
    env = dict(os.environ)
    tool = os.path.join(build, "libruppell_tool.so")
    env["RUPPELL_MARKER_LIB"] = os.path.join(build, "libruppell_marker.so")
    env["RUPPELL_TOOL_LIB"] = tool
    env["RUPPELL_OUT"] = out
    env["ROCP_TOOL_LIBRARIES"] = ":".join([p for p in env.get("ROCP_TOOL_LIBRARIES", "").split(":") if p] + [tool])
    env.setdefault("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "0")
    env["NCCL_PROFILER_PLUGIN"] = os.path.join(build, "libruppell_nccl_profiler.so")
    env["RUPPELL_NCCL_KERNEL_CH"] = "0"
    early = os.path.join(REPO, "python", "ruppell", "_rocm_early")
    env["PYTHONPATH"] = early + os.pathsep + os.path.join(REPO, "python") + os.pathsep + env.get("PYTHONPATH", "")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump({"command": command, "commit": open(os.path.join(REPO, ".commit")).read().strip()}, f)
    sys.exit(subprocess.run(command, env=env).returncode)
if cmd == "census":
    if os.environ.get("FAKE_CENSUS_RC"):
        print("census: no records", file=sys.stderr)
        sys.exit(int(os.environ["FAKE_CENSUS_RC"]))
    run = args[0]
    shutil.copyfile(os.path.join(DATA, "offline_census.jsonl"), os.path.join(run, "census.jsonl"))
    summary = os.environ.get("FAKE_SUMMARY") or os.path.join(DATA, "w243_census_summary.json")
    shutil.copyfile(summary, os.path.join(run, "census_summary.json"))
    open(os.path.join(run, "census_scopes.json"), "w").write("{}")
    open(os.path.join(run, "census_scopes.npz"), "wb").write(b"npz")
    sys.exit(0)
if cmd == "export":
    run, out = args[0], args[args.index("-o") + 1]
    open(out, "w").write("{}")
    open(os.path.join(run, "report.json"), "w").write("{}")
    sys.exit(0)
sys.exit(f"unknown {cmd}")
"""

_CHECK = """\
import json, os, sys
if os.environ.get("FAKE_CHECK_OK") == "0":
    print(json.dumps({"ok": False, "criteria": {}}))
    sys.exit(1)
print(open(os.path.join(os.environ["FAKE_RUPPELL_DATA"], "w243_check.json")).read())
"""


def tree_files() -> dict[str, tuple[str, int]]:
    """The stand-in tree: path -> (text, mode)."""
    return {
        "runtime/build.sh": (_BUILD, 0o755),
        "python/ruppell/__init__.py": ("", 0o644),
        "python/ruppell/__main__.py": (_MAIN, 0o644),
        "python/ruppell/_rocm_early/sitecustomize.py": ("# ruppell's start-up hook\n", 0o644),
        "tests/m2/check_m2.py": (textwrap.dedent(_CHECK), 0o644),
    }


def archive(commit: str = COMMIT, files: dict[str, tuple[str, int]] | None = None, *, top: str = "ruppell") -> bytes:
    """The tree as ``git archive`` writes it: one top directory and the commit in the pax header's comment."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT, pax_headers={"comment": commit}) as tar:
        directory = tarfile.TarInfo(f"{top}/")
        directory.type, directory.mode = tarfile.DIRTYPE, 0o775
        tar.addfile(directory)
        for path, (text, mode) in (files or tree_files()).items():
            raw = text.encode()
            info = tarfile.TarInfo(f"{top}/{path}")
            info.size, info.mode = len(raw), mode
            tar.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()
