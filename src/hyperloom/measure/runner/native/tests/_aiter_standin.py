# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A stand-in aiter for the tests of #125 stage 2b: a develop install whose ``aiter/jit/core.py`` builds a module the way
aiter's does, without a compiler or a GPU.

Its ``get_args_of_build`` evaluates the module's entry of ``optCompilerConfig.json`` as aiter does, and its
``build_module`` refuses ``GPU_ARCHS=native`` (it has no GPU to ask), logs ``start build`` and ``finish build``, fails like
a compiler on an ``#error`` line of a source (naming the file and line), and installs ``<name>.so`` into ``AITER_JIT_DIR``
as aiter does: written beside it, then ``os.replace``-d into place. The ``.so`` is the ELF magic, the offload target, the
``ENABLE_CK`` it was built with and the sources it was built from, so a test can tell which build it holds.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from hyperloom.measure.compile import PackageBase
from hyperloom.measure.runner.native.aiter_modules import aiter_module_recipe
from hyperloom.measure.runner.native.recipes import NativeRecipe, Preflight

MODULE = "module_gemm"
OTHER = "module_other"
KERNEL = "csrc/kernels/k.cu"
TABLE_VARIABLE = "AITER_CONFIG_GEMM_A8W8"
TABLE = "aiter/configs/a8w8_tuned_gemm.csv"
#: ``module_gemm`` is generated from the variable's tuned table, as aiter's GEMM modules are.
GENERATOR = f"f'{{AITER_CSRC_DIR}}/kernels/gen.py {{{{}}}} --tune_file {{AITER_CONFIGS.{TABLE_VARIABLE}_FILE}}'"
IMAGE_KERNEL = "// image k.cu\n"
FACTORY = f"{__name__}:recipe"
CORE = """\
import json, os
this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_ROOT_DIR = os.path.abspath(f"{this_dir}/../../")
AITER_META_DIR = os.path.abspath(os.environ.get("AITER_META_DIR") or AITER_ROOT_DIR)
AITER_CSRC_DIR = f"{AITER_META_DIR}/csrc"
ENABLE_CK = os.environ.get("ENABLE_CK", "1")
AITER_CONFIG_GEMM_A8W8 = os.getenv("AITER_CONFIG_GEMM_A8W8", f"{AITER_ROOT_DIR}/aiter/configs/a8w8_tuned_gemm.csv")


class AITER_CONFIG:
    def get_config_file(self, env_name, default_file, tuned_file_name):
        return os.getenv(env_name) or default_file

    @property
    def AITER_CONFIG_GEMM_A8W8_FILE(self):
        return self.get_config_file("AITER_CONFIG_GEMM_A8W8", AITER_CONFIG_GEMM_A8W8, "a8w8_tuned_gemm")


AITER_CONFIGS = AITER_CONFIG()
DEFAULTS = {"srcs": [], "flags_extra_cc": [], "flags_extra_hip": [], "blob_gen_cmd": "", "extra_include": [],
            "extra_ldflags": None, "verbose": False, "is_python_module": True, "is_standalone": False,
            "torch_exclude": False, "hip_clang_path": None, "third_party": []}


def get_args_of_build(name):
    with open(f"{this_dir}/optCompilerConfig.json", encoding="utf-8") as stream:
        entry = json.load(stream)[name]
    args = dict(DEFAULTS)
    for key, value in entry.items():
        args[key] = [eval(item) for item in value] if isinstance(value, list) else eval(value)
    return args


def build_module(md_name, srcs, flags_extra_cc, flags_extra_hip, blob_gen_cmd, extra_include, extra_ldflags, verbose,
                 is_python_module, is_standalone, torch_exclude, third_party, hipify=False,
                 flags_extra_hip_per_source=None):
    arch = os.environ.get("GPU_ARCHS", "native")
    if arch == "native":
        raise RuntimeError("no GPU to detect the architecture from")
    jit = os.environ["AITER_JIT_DIR"]
    print(f"[aiter] start build [{md_name}] under {jit}/build/{md_name}", flush=True)
    text = ""
    for src in srcs:
        with open(src, encoding="utf-8") as stream:
            source = stream.read()
        for number, line in enumerate(source.splitlines(), 1):
            if line.startswith("#error"):
                print(f"{src}:{number}:2: error: {line[7:]}", flush=True)
                raise RuntimeError(f"[aiter] build [{md_name}] failed !!!!!!")
        text += source
    os.makedirs(f"{jit}/build/{md_name}", exist_ok=True)
    pending = f"{jit}/build/{md_name}/{md_name}.so.tmp"
    with open(pending, "wb") as out:
        out.write(built(arch, ENABLE_CK, text))
    os.replace(pending, f"{jit}/{md_name}.so")
    print(f"finish build [{md_name}], cost 0.1s", flush=True)


def built(arch, enable_ck, text):
    return b"\\x7fELF" + f"amdhsa--{arch} ENABLE_CK={enable_ck}\\n{text}".encode()
"""


def built(arch: str, text: str, enable_ck: str = "1") -> bytes:
    """What the stand-in's build of a module makes from the sources ``text`` (the function of ``CORE``)."""
    return b"\x7fELF" + f"amdhsa--{arch} ENABLE_CK={enable_ck}\n{text}".encode()


def entry(source: str) -> dict:
    return {"srcs": [f"f'{{AITER_CSRC_DIR}}/kernels/{source}'"], "extra_include": [], "blob_gen_cmd": "''"}


def aiter_image(root: Path, extra: dict[str, str] | None = None) -> Path:
    """A develop-install checkout at ``root`` with two prebuilt modules, their config and their sources; the package.
    ``module_gemm`` is generated from the tuned table of ``AITER_CONFIG_GEMM_A8W8``."""
    config = {MODULE: {**entry("k.cu"), "blob_gen_cmd": GENERATOR}, OTHER: entry("other.cu")}
    files = {
        "aiter/__init__.py": "",
        "aiter/install_mode": "develop",
        "aiter/jit/__init__.py": "",
        "aiter/jit/core.py": CORE,
        "aiter/jit/optCompilerConfig.json": json.dumps(config),
        f"aiter/jit/{MODULE}.so": "image build",
        f"aiter/jit/{OTHER}.so": "image build",
        KERNEL: IMAGE_KERNEL,
        "csrc/kernels/other.cu": "// other.cu\n",
        TABLE: "M,N,K\n1,2,3\n",
        **(extra or {}),
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root / "aiter"


def package_base(package: Path) -> PackageBase:
    return PackageBase("aiter", package, "amd-aiter", "0.1.23")


def _preflight(scratch: Path, gfx_arch: str) -> Preflight:
    return Preflight(None, {"compiler": "stand-in"})


def _forging_preflight(scratch: Path, gfx_arch: str) -> Preflight:
    return Preflight(None, {"compiler": "stand-in", "env_ENABLE_CK": "0"})


def recipe() -> NativeRecipe:
    """aiter's module recipe with a preflight that asks nothing of the machine (the child makes it by :data:`FACTORY`)."""
    return replace(aiter_module_recipe(), preflight=_preflight, budget_sec=120)


def forging_recipe() -> NativeRecipe:
    """:func:`recipe` whose compile claims a variable it was not given (:data:`FORGING_FACTORY`)."""
    return replace(recipe(), preflight=_forging_preflight)


FORGING_FACTORY = f"{__name__}:forging_recipe"


PYTHON = sys.executable
