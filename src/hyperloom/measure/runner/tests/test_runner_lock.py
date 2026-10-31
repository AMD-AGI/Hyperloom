# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``requirements/runner.lock`` is the ``runner`` extra compiled with hashes, which is all the sealed runner installs (4.8.1).

A Python with a lock of its own (``requirements/runner-cpXY.lock``, :func:`hyperloom.measure.runner.cache_layout.lock_file`)
is held to the same, and to the shared lock's pins but for those its header says it moved.

CI has no framework image to resolve in, so these tests hold the file to what the install relies on: every entry is an
exact pin that carries hashes (so ``pip install --require-hashes`` has nothing to resolve), the entries that
``pip-compile`` marked as asked for by ``pyproject.toml`` are exactly the ``runner`` extra (so the lock and the extra
cannot drift apart), and nothing of the framework stack, which the image provides, is in it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:  # tomllib is stdlib from 3.11; the ``ci`` extra pins tomli for 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on py3.10
    import tomli as tomllib  # type: ignore[no-redef]


def _find_repo_root() -> Path | None:
    """Walk up for the checkout's ``pyproject.toml``; ``None`` when the tests run from an installed package."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return None


_ROOT = _find_repo_root()
pytestmark = pytest.mark.skipif(_ROOT is None, reason="the lock tests need the source checkout")

#: Torch, its Triton and the serving engines come from the framework image; PyPI has no ROCm build of them.
_FRAMEWORK_NAMES = frozenset(
    {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm", "aiter", "amd-aiter"}
)
_FRAMEWORK_PREFIXES = ("sglang", "sgl-", "vllm", "triton-", "nvidia-")

_ENTRY = re.compile(r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s\\]+)(?: \\)?")
_HASH = re.compile(r"\s+--hash=sha256:(?P<digest>[0-9a-f]{64})(?: \\)?")
_VIA = re.compile(r"\s+#(?: via)?\s*(?P<source>\S.*)?")
_FROM_PYPROJECT = "(pyproject.toml)"


@dataclass
class _Pin:
    name: str
    version: str
    hashes: list[str]
    via: list[str]


def _parse(text: str) -> tuple[list[_Pin], list[str]]:
    """The pins of a ``pip-compile --generate-hashes`` file, and the lines that are neither pin, hash nor comment."""
    pins: list[_Pin] = []
    stray: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if entry := _ENTRY.fullmatch(line):
            pins.append(_Pin(canonicalize_name(entry["name"]), entry["version"], [], []))
        elif pins and (digest := _HASH.fullmatch(line)):
            pins[-1].hashes.append(digest["digest"])
        elif pins and (via := _VIA.fullmatch(line)):
            pins[-1].via.append(via["source"] or "")
        else:
            stray.append(line)
    return pins, stray


def _is_framework(name: str) -> bool:
    return name in _FRAMEWORK_NAMES or name.startswith(_FRAMEWORK_PREFIXES)


def _problems(lock_text: str, group: list[str]) -> list[str]:
    """Everything that stops ``pip install --require-hashes --no-deps -r`` of ``lock_text`` from being the ``group``."""
    pins, stray = _parse(lock_text)
    problems = [f"not a pin with hashes: {line.strip()!r}" for line in stray]
    problems += [f"{pin.name}=={pin.version} carries no hash" for pin in pins if not pin.hashes]
    wanted = {canonicalize_name(Requirement(spec).name): Requirement(spec) for spec in group}
    asked = {pin.name for pin in pins if any(source.endswith(_FROM_PYPROJECT) for source in pin.via)}
    problems += [f"{name} is in the runner extra but not asked for by the lock" for name in sorted(set(wanted) - asked)]
    problems += [f"{name} is asked for by the lock but not in the runner extra" for name in sorted(asked - set(wanted))]
    for pin in pins:
        wants = wanted.get(pin.name)
        if wants is not None and not wants.specifier.contains(pin.version, prereleases=True):
            problems.append(f"{pin.name}=={pin.version} does not satisfy the extra's {wants.specifier}")
    problems += [f"{pin.name} is the framework image's, not the runner's" for pin in pins if _is_framework(pin.name)]
    return problems


def _runner_extra() -> list[str]:
    assert _ROOT is not None
    document = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return document["project"]["optional-dependencies"]["runner"]


def _locks() -> list[str]:
    return sorted(path.name for path in (_ROOT / "requirements").glob("runner*.lock")) if _ROOT else ["runner.lock"]


@pytest.mark.parametrize("name", _locks())
def test_the_lock_is_the_runner_extra_pinned_with_hashes_and_holds_no_framework_package(name):
    assert _ROOT is not None
    lock = (_ROOT / "requirements" / name).read_text(encoding="utf-8")

    assert _problems(lock, _runner_extra()) == []


_OWN_LOCK = re.compile(r"runner-cp3[0-9]+\.lock")
_MOVED = re.compile(r"grep -viE '\^\((?P<names>[a-z0-9_|.-]+)\)=='")


def test_a_pythons_own_lock_is_the_shared_lock_but_for_the_pins_its_header_says_it_moved():
    assert _ROOT is not None
    own_locks = [name for name in _locks() if _OWN_LOCK.fullmatch(name)]
    assert own_locks, "no runner-cpXY.lock: the CPython 3.14 images need one"
    shared = {pin.name: pin.version for pin in _parse((_ROOT / "requirements" / "runner.lock").read_text())[0]}
    for name in own_locks:
        text = (_ROOT / "requirements" / name).read_text(encoding="utf-8")
        moved = _MOVED.search("\n".join(text.splitlines()[:12]))
        assert moved, f"{name}'s header does not say which pins it moved"
        allowed = {canonicalize_name(moved_name) for moved_name in moved["names"].split("|")}
        pins = {pin.name: pin.version for pin in _parse(text)[0]}
        assert sorted(set(pins) - set(shared)) == [], f"{name} pins what runner.lock does not"
        differing = {pin for pin in set(pins) & set(shared) if pins[pin] != shared[pin]}
        assert differing <= allowed, f"{name} moved {sorted(differing - allowed)} without saying so"


def test_the_extra_itself_leaves_the_framework_to_the_image():
    assert [spec for spec in _runner_extra() if _is_framework(canonicalize_name(Requirement(spec).name))] == []


@pytest.mark.parametrize("name", _locks())
def test_the_header_says_how_the_lock_is_regenerated(name):
    assert _ROOT is not None
    header = (_ROOT / "requirements" / name).read_text(encoding="utf-8").splitlines()[:12]

    assert any("pip-compile" in line and "--generate-hashes" in line for line in header)


_PIN = (
    "tenacity==9.2.1 \\\n    --hash=sha256:"
    + "a" * 64
    + "\n    # via\n    #   hyperloom-inference_optimizer (pyproject.toml)\n"
)


@pytest.mark.parametrize(
    ("lock", "group", "refusal"),
    [
        (_PIN, ["tenacity"], None),
        (
            _PIN.replace("tenacity==9.2.1 \\\n    --hash=sha256:" + "a" * 64, "tenacity==9.2.1"),
            ["tenacity"],
            "carries no hash",
        ),
        (_PIN.replace("==9.2.1", ">=9"), ["tenacity"], "not a pin with hashes"),
        (_PIN + "--index-url https://example.invalid/simple\n", ["tenacity"], "not a pin with hashes"),
        (_PIN + "-e .\n", ["tenacity"], "not a pin with hashes"),
        (_PIN.replace("--hash=sha256:" + "a" * 64, "--hash=md5:" + "a" * 32), ["tenacity"], "not a pin with hashes"),
        (_PIN, ["tenacity", "pyyaml"], "pyyaml is in the runner extra but not asked for"),
        (_PIN, [], "tenacity is asked for by the lock but not in the runner extra"),
        (_PIN, ["tenacity>=10"], "does not satisfy"),
        (_PIN.replace("tenacity", "torch").replace("9.2.1", "2.9.1"), ["torch"], "torch is the framework image's"),
        (_PIN.replace("tenacity", "nvidia-cublas-cu12"), ["nvidia-cublas-cu12"], "is the framework image's"),
    ],
)
def test_the_checks_name_what_they_refuse(lock, group, refusal):
    problems = _problems(lock, group)

    if refusal is None:
        assert problems == []
    else:
        assert any(refusal in problem for problem in problems), problems
