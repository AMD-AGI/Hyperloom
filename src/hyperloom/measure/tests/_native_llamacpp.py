# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A stand-in llama.cpp for the single-branch native build tests: a checkout with llama.cpp's layout whose "compile" is
a Python step, so no machine needs CMake or a GPU toolchain.

The step makes ``build/bin/llama-server``: the ELF magic, then the hash of ``ggml/a.c``; the source failing with
``#error`` fails the compile as a compiler would.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

from hyperloom.measure.runner.native.preflight import Preflight
from hyperloom.measure.runner.native.recipes import NativeRecipe, llamacpp_recipe
from hyperloom.measure.tests._recipes import write_tree

#: The active image, as ``HYPERLOOM_ACTIVE_IMAGE_REF`` and ``HYPERLOOM_ACTIVE_IMAGE_DIGEST`` name it.
IMAGE = {"HYPERLOOM_ACTIVE_IMAGE_REF": "registry.invalid/hl", "HYPERLOOM_ACTIVE_IMAGE_DIGEST": "sha256:" + "a" * 64}
SOURCE = "ggml/a.c"

_COMPILE = """\
import hashlib, os, pathlib, sys
src = pathlib.Path("ggml/a.c").read_text()
if "#error" in src:
    sys.stderr.write("ggml/a.c:1: error: stop\\n")
    sys.exit(1)
out = pathlib.Path("build/bin/llama-server")
out.parent.mkdir(parents=True)
out.write_bytes(b"\\x7fELF" + hashlib.sha256(src.encode()).digest())
out.chmod(0o755)
"""


def _preflight(scratch: Path, gfx_arch: str) -> Preflight:
    return Preflight(None, {"compiler": "stand-in"})


def stand_in_recipe() -> NativeRecipe:
    """llama.cpp's recipe with the stand-in compile: what the build runner's child imports by name."""
    return replace(
        llamacpp_recipe(),
        steps=(("{python}", "-c", _COMPILE),),
        budget_sec=120,
        preflight=_preflight,
        variables=dict,
        executables=("bin/llama-server",),
    )


def git(checkout: Path, *args: str) -> str:
    command = ["git", "-c", "safe.directory=*", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    return subprocess.run(
        [*command, "-C", str(checkout), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def llama_checkout(path: Path, source: str = "int a;\n") -> Path:
    """A git checkout with llama.cpp's sections, ``ggml/a.c`` holding ``source``, at one commit."""
    write_tree(
        path,
        {
            SOURCE: source,
            "src/b.cpp": "",
            "include/c.h": "",
            "common/d.cpp": "",
            "tools/server/e.cpp": "",
            "CMakeLists.txt": "",
            ".gitignore": "build*/\n",
        },
    )
    git(path, "init", "-q")
    git(path, "add", "-A")
    git(path, "commit", "-qm", "base")
    return path
