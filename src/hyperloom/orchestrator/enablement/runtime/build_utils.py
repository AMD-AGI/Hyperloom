# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reusable, injectable build utilities for off-loop targeted builds."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

# Injectable subprocess runner

_RunCallable = Callable[..., Any]


@dataclass
class RunResult:
    """Outcome of a single subprocess invocation."""

    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""


def run_argv(
    argv: list[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
    timeout_sec: int = 1800,
    run: _RunCallable = subprocess.run,
) -> RunResult:
    """Run *argv* in *cwd* and capture up to 4000 chars of each stream."""
    try:
        completed = run(
            argv,
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return RunResult(
            returncode=int(completed.returncode),
            stdout_tail=(completed.stdout or "")[-4000:],
            stderr_tail=(completed.stderr or "")[-4000:],
        )
    except subprocess.TimeoutExpired:
        return RunResult(returncode=-1)


# ROCm constraint file

ROCM_CONSTRAINTS_FILE = "rocm_constraints.txt"

_TORCH_HIP_PROBE = (
    "import sys, importlib.metadata; "
    "import torch; "
    "hip = getattr(torch.version, 'hip', None); "
    "sys.exit(0 if hip else 2); "
)
_ROCM_STACK_PROBE = (
    "import re,sys\n"
    "import importlib.metadata as m\n"
    "norm=lambda n: re.sub(r'[-_.]+','-',n).lower()\n"
    "free=norm(sys.argv[1])\n"
    "named={'torch','torchvision','torchaudio','triton'}\n"
    "pins={}\n"
    # Distributions come in sys.path order, so the first of a name is the one imported.
    "for d in m.distributions():\n"
    "    name=norm(d.metadata['Name'] or '')\n"
    "    if name and name!=free and (name in named or 'rocm' in (d.version or '').lower()):\n"
    "        pins.setdefault(name, d.version)\n"
    "print('\\n'.join(f'{n}=={v}' for n,v in sorted(pins.items())))\n"
)


class AbiMismatchError(RuntimeError):
    """Raised when the detected torch is not a ROCm build."""


def write_rocm_constraints(
    python_exe: str,
    constraint_path: str | Path,
    *,
    replacing: str,
    run: _RunCallable = subprocess.run,
) -> None:
    """Write a pip constraint file pinning the ROCm stack ``python_exe`` imports.

    torch and its companions by name, and every other distribution whose
    version carries a ROCm build label: PyPI serves no ROCm build of any of
    them, so a resolver free to move one swaps the GPU stack. ``replacing`` is
    the distribution the install is there to change. Distributions built for
    ROCm without saying so in their version (amd-aiter, sglang-kernel) cannot
    be told apart and stay unpinned.
    """
    hip_res = run(
        [python_exe, "-c", _TORCH_HIP_PROBE],
        capture_output=True,
        text=True,
        timeout=30,
    )
    rc = int(getattr(hip_res, "returncode", -1))
    if rc != 0:
        raise AbiMismatchError(f"installed torch at {python_exe!r} is not a ROCm build (probe rc={rc})")
    pins_res = run(
        [python_exe, "-c", _ROCM_STACK_PROBE, replacing],
        capture_output=True,
        text=True,
        timeout=30,
    )
    pins = (getattr(pins_res, "stdout", "") or "").split()
    if not any(p.startswith("torch==") for p in pins):
        raise RuntimeError(f"could not determine torch version from {python_exe!r}")
    Path(constraint_path).write_text("\n".join(pins) + "\n", encoding="utf-8")


# SGLang's ROCm dependency set

_SGLANG_ROCM_PYPROJECT = "pyproject_other.toml"
_SGLANG_ROCM_EXTRA_PREFIX = "srt_hip"


def _sglang_rocm_extra(pyproject: Path, torch_version: str) -> str:
    """The ``srt_hip*`` extra whose torch requirement admits ``torch_version``; an exact pin beats none."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    extras = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {}).get("optional-dependencies", {})
    torch = Version(torch_version)
    fitting: list[tuple[bool, str]] = []
    for name, reqs in extras.items():
        if not name.startswith(_SGLANG_ROCM_EXTRA_PREFIX):
            continue
        spec = next((r.specifier for r in map(Requirement, reqs) if r.name == "torch"), SpecifierSet())
        if spec.contains(torch, prereleases=True):
            fitting.append((bool(spec), name))
    if not fitting:
        raise ValueError(f"no {_SGLANG_ROCM_EXTRA_PREFIX}* extra in {pyproject} admits torch {torch_version}")
    return max(fitting)[1]


@contextmanager
def sglang_rocm_install_target(python_dir: Path, torch_version: str) -> Iterator[str]:
    """Yield the pip target that installs the SGLang checkout at ``python_dir`` with its ROCm dependencies.

    The checkout's ``pyproject.toml`` declares the CUDA stack; ROCm installs put
    ``pyproject_other.toml`` in its place. The swap lasts only as long as the
    install, because the checkout is also the tree patches are cut against.
    """
    pyproject = python_dir / "pyproject.toml"
    original = pyproject.read_bytes()
    rocm = python_dir / _SGLANG_ROCM_PYPROJECT
    if rocm.is_file():
        pyproject.write_bytes(rocm.read_bytes())
    try:
        yield f"{python_dir}[{_sglang_rocm_extra(pyproject, torch_version)}]"
    finally:
        pyproject.write_bytes(original)


# ROCm toolchain alignment check


def check_rocm_toolchain_alignment(
    *,
    env: Mapping[str, str] | None = None,
    run: _RunCallable = subprocess.run,
) -> tuple[bool, str]:
    """Advisory ROCm toolchain probe."""
    import os

    effective_env = dict(env) if env is not None else dict(os.environ)

    hipcc_res = run(
        ["which", "hipcc"],
        capture_output=True,
        text=True,
        timeout=10,
        env=effective_env,
    )
    hipcc_path = (getattr(hipcc_res, "stdout", "") or "").strip()
    if not hipcc_path or getattr(hipcc_res, "returncode", 1) != 0:
        return True, "hipcc not found; AITER/source builds need a ROCm compiler toolchain (warning only)"

    hipcc_root_res = run(
        ["sh", "-c", f"cd $(dirname {hipcc_path!r})/.. && pwd"],
        capture_output=True,
        text=True,
        timeout=10,
        env=effective_env,
    )
    hipcc_root = (getattr(hipcc_root_res, "stdout", "") or "").strip()

    # hip version from hipcc_root
    hip_header = Path(hipcc_root) / "include" / "hip" / "hip_runtime_api.h" if hipcc_root else None
    if hip_header and hip_header.is_file():
        content = hip_header.read_text(errors="replace")
        if "hipDeviceAttributePciChipId" not in content:
            return False, (
                f"hipcc headers at {hipcc_root} do not look compatible with the installed torch hip version; "
                "set ROCM_PATH/HIP_PATH/PATH to a ROCm 7.x toolchain before building AITER"
            )
    return True, "ok"


# vLLM ROCm platform probe

#: Exits 0 and prints ``vllm_rocm_ok`` only when torch is a ROCm build and vLLM
#: reports a ROCm platform.
VLLM_ROCM_PROBE = """\
import sys, torch
if not getattr(torch.version, "hip", None):
    print("torch is not a ROCm build", file=sys.stderr); raise SystemExit(1)
import vllm
try:
    from vllm.platforms import current_platform
except Exception as exc:
    print(f"cannot import vllm platform: {exc}", file=sys.stderr); raise SystemExit(1)
is_rocm = False
checker = getattr(current_platform, "is_rocm", None)
if callable(checker):
    is_rocm = bool(checker())
if "rocm" in f"{current_platform!r} {current_platform.__class__.__name__}".lower():
    is_rocm = True
if not is_rocm:
    print("vLLM did not report ROCm platform", file=sys.stderr); raise SystemExit(1)
print("vllm_rocm_ok")
"""


# Torch ABI probe

#: The prefix of the one stdout line the ABI probe's facts are read from. Anything else on stdout (a banner, a
#: warning a package prints while torch imports) is ignored.
ABI_PROBE_MARKER = "HYPERLOOM_TORCH_ABI="
_ABI_PROBE = (
    "import sys, json, torch; "
    "hip=getattr(getattr(torch, 'version', None), 'hip', None); "
    "print(" + repr(ABI_PROBE_MARKER) + " + json.dumps({"
    "'torch_version': str(getattr(torch, '__version__', '') or ''), "
    "'hip_version': str(hip or ''), "
    "'python_version': sys.version.split()[0], "
    "'is_rocm': bool(hip)"
    "}), flush=True)"
)
_ABI_PROBE_TIMEOUT_SEC = 30
_ABI_PROBE_TAIL = 800
#: A ROCm build label in a torch version (``2.11.0+rocm10.0.0``, ``2.9.0+rocm7.0``).
_ROCM_LOCAL_VERSION = re.compile(r"\+rocm\d", re.IGNORECASE)


def _empty_abi(error: str) -> dict[str, Any]:
    return {"torch_version": "", "hip_version": "", "python_version": "", "is_rocm": False, "error": error}


def _tail(text: str) -> str:
    text = (text or "").strip()
    return text[-_ABI_PROBE_TAIL:]


def _abi_line(stdout: str) -> dict[str, Any] | None:
    """The facts of the last marked line of ``stdout``; without one, the last line that is a JSON object."""
    import json

    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    marked = [line[len(ABI_PROBE_MARKER) :] for line in lines if line.startswith(ABI_PROBE_MARKER)]
    for candidate in reversed(marked or [line for line in lines if line.startswith("{")]):
        try:
            facts = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(facts, dict):
            return facts
    return None


def probe_torch_abi(
    python_exe: str,
    *,
    run: _RunCallable = subprocess.run,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return torch/Python ABI facts from the given interpreter.

    The facts are read from the probe's marked stdout line only, so other output cannot break the parse. torch is a
    ROCm build when ``torch.version.hip`` is set or its version carries a ``+rocmN`` label. When the probe fails the
    result's ``error`` names the return code and the tail of its stderr; ``torch_version`` is then empty.
    """
    kwargs: dict[str, Any] = {"capture_output": True, "text": True, "timeout": _ABI_PROBE_TIMEOUT_SEC}
    if env is not None:
        kwargs["env"] = dict(env)
    try:
        res = run([python_exe, "-c", _ABI_PROBE], **kwargs)
    except subprocess.TimeoutExpired:
        return _empty_abi(f"the torch probe of {python_exe} timed out after {_ABI_PROBE_TIMEOUT_SEC}s")
    except OSError as exc:
        return _empty_abi(f"the torch probe of {python_exe} could not start: {exc}")
    rc = int(getattr(res, "returncode", 1))
    stdout = getattr(res, "stdout", "") or ""
    stderr = getattr(res, "stderr", "") or ""
    facts = _abi_line(stdout) if rc == 0 else None
    if facts is None:
        detail = _tail(stderr) or _tail(stdout) or "no output"
        return _empty_abi(f"the torch probe of {python_exe} failed (rc={rc}): {detail}")
    torch_version = str(facts.get("torch_version") or "")
    hip_version = str(facts.get("hip_version") or "")
    return {
        "torch_version": torch_version,
        "hip_version": hip_version,
        "python_version": str(facts.get("python_version") or ""),
        "is_rocm": bool(facts.get("is_rocm", hip_version)) or bool(_ROCM_LOCAL_VERSION.search(torch_version)),
    }


# Version-sorted tag list (sort -V -r equivalent for autoselect)


def sort_tags_desc(tags: list[str] | tuple[str, ...]) -> list[str]:
    """Return *tags* in descending version order (newest first)."""
    import packaging.version  # available via pip; already a transitive dep

    def _key(t: str):
        try:
            return (1, packaging.version.Version(t.lstrip("v")))
        except packaging.version.InvalidVersion:
            return (0, packaging.version.Version("0"))

    return sorted(tags, key=_key, reverse=True)


__all__ = [
    "ROCM_CONSTRAINTS_FILE",
    "AbiMismatchError",
    "VLLM_ROCM_PROBE",
    "RunResult",
    "check_rocm_toolchain_alignment",
    "probe_torch_abi",
    "run_argv",
    "sglang_rocm_install_target",
    "sort_tags_desc",
    "write_rocm_constraints",
]
