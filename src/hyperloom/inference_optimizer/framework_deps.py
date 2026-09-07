# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-framework runtime dependency installation."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CORE_PACKAGES",
    "Requirement",
    "Outcome",
    "TorchClobberedError",
    "manifest_path",
    "parse_manifest",
    "ensure",
]

# Never installable from a framework manifest.
CORE_PACKAGES = frozenset({"torch", "torchvision", "torchaudio", "triton", "numpy"})

# A '#' only starts a comment at line start or after whitespace, so a VCS spec such as
# git+https://host/repo.git#egg=name survives intact.
_COMMENT_RE = re.compile(r"(?:^|\s)#.*$")
_VERSION_SPEC_RE = re.compile(r"[<>=!~;\[]")
_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

_PROBE_MISSING = """\
import importlib.util, sys
missing = []
for name in sys.argv[1:]:
    try:
        found = importlib.util.find_spec(name) is not None
    except Exception:
        found = False
    if not found:
        missing.append(name)
print("\\n".join(missing))
"""

_PROBE_VERSIONS = """\
import importlib.metadata as md, sys
for name in sys.argv[1:]:
    try:
        print(f"{name}=={md.version(name)}")
    except Exception:
        pass
"""

_PROBE_TORCH_HIP = "import torch; print(torch.version.hip or '')"


class TorchClobberedError(RuntimeError):
    """Raised when an install replaced the ROCm torch with a non-ROCm build."""


@dataclass(frozen=True)
class Requirement:
    """One manifest entry: what to install and how to detect it."""

    spec: str
    import_name: str


@dataclass
class Outcome:
    """What ``ensure`` did, for the caller to report."""

    framework: str
    manifest: Path | None = None
    skipped_reason: str = ""
    installed: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    would_install: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing the manifest asked for is still missing."""
        return not self.failed and not self.invalid


def _framework_key(framework: str | None) -> str:
    return str(framework or "").strip().lower()


def manifest_path(framework: str | None, root: Path | None = None) -> Path:
    """Return the manifest location for ``framework`` (may not exist)."""
    if root is None:
        from hyperloom.inference_optimizer.session.paths import asset_root

        root = asset_root() / "assets" / "framework_deps"
    return Path(root) / f"{_framework_key(framework)}.txt"


def _split_import_name(entry: str) -> tuple[str, str]:
    """Split ``<pip-spec>[:<import-name>]`` without mangling URL/VCS specs."""
    head, sep, tail = entry.rpartition(":")
    if sep and head and _MODULE_NAME_RE.match(tail):
        return head.strip(), tail.strip()
    return entry, ""


def _package_base(spec: str) -> str:
    """Return the bare distribution name, version specifiers stripped."""
    return _VERSION_SPEC_RE.split(spec, 1)[0].strip().lower()


def parse_manifest(text: str) -> tuple[list[Requirement], list[str], list[str]]:
    """Parse manifest text."""
    requirements: list[Requirement] = []
    refused: list[str] = []
    invalid: list[str] = []
    for raw in text.splitlines():
        entry = _COMMENT_RE.sub("", raw).strip()
        if not entry:
            continue
        spec, import_name = _split_import_name(entry)
        base = _package_base(spec)
        if base in CORE_PACKAGES:
            refused.append(base)
            continue
        if not import_name:
            import_name = base.replace("-", "_")
        if not _MODULE_NAME_RE.match(import_name):
            invalid.append(entry)
            continue
        requirements.append(Requirement(spec, import_name))
    return requirements, refused, invalid


def _run(python_exe: str, script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([python_exe, "-c", script, *args], capture_output=True, text=True, check=False)


def _missing(python_exe: str, requirements: list[Requirement]) -> list[Requirement]:
    """Return the subset whose import name is not resolvable in python_exe."""
    if not requirements:
        return []
    names = [r.import_name for r in requirements]
    result = _run(python_exe, _PROBE_MISSING, *names)
    if result.returncode != 0:
        # Probe itself broke; attempt everything rather than silently skip.
        return list(requirements)
    absent = {ln.strip() for ln in (result.stdout or "").splitlines() if ln.strip()}
    return [r for r in requirements if r.import_name in absent]


def _write_core_constraints(python_exe: str, dest: Path) -> None:
    """Pin every installed core package to its exact current version."""
    result = _run(python_exe, _PROBE_VERSIONS, *sorted(CORE_PACKAGES))
    dest.write_text(result.stdout or "", encoding="utf-8")


def _torch_hip_version(python_exe: str) -> str:
    result = _run(python_exe, _PROBE_TORCH_HIP)
    return (result.stdout or "").strip() if result.returncode == 0 else ""


def ensure(
    framework: str | None,
    *,
    python_exe: str | None = None,
    pip_extra: tuple[str, ...] | list[str] = (),
    check_only: bool = False,
    dry_run: bool = False,
    root: Path | None = None,
) -> Outcome:
    """Install whatever ``framework``'s manifest declares and is not present."""
    key = _framework_key(framework)
    outcome = Outcome(framework=key)
    if not key:
        outcome.skipped_reason = "no framework selected"
        return outcome

    path = manifest_path(key, root)
    outcome.manifest = path
    if not path.is_file():
        outcome.skipped_reason = f"no manifest at {path}"
        return outcome

    requirements, refused, invalid = parse_manifest(path.read_text(encoding="utf-8"))
    outcome.refused = refused
    outcome.invalid = invalid
    if not requirements:
        outcome.skipped_reason = "manifest declares nothing installable"
        return outcome

    python_exe = python_exe or sys.executable
    missing = _missing(python_exe, requirements)
    outcome.already_present = [r.spec for r in requirements if r not in missing]
    if not missing:
        return outcome
    if check_only or dry_run:
        outcome.would_install = [r.spec for r in missing]
        return outcome

    with tempfile.TemporaryDirectory() as tmp:
        constraints = Path(tmp) / "core-constraints.txt"
        _write_core_constraints(python_exe, constraints)
        hip_before = _torch_hip_version(python_exe)

        for req in missing:
            completed = subprocess.run(
                [
                    python_exe,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--no-cache-dir",
                    "-c",
                    str(constraints),
                    *pip_extra,
                    req.spec,
                ],
                check=False,
            )
            if completed.returncode != 0:
                outcome.failed.append(req.spec)
                continue
            if _missing(python_exe, [req]):
                outcome.failed.append(f"{req.spec} (installed but not importable)")
            else:
                outcome.installed.append(req.spec)

        if hip_before and not _torch_hip_version(python_exe):
            raise TorchClobberedError(
                f"installing {key} deps replaced the ROCm torch (was hip="
                f"{hip_before}) with a non-ROCm build, which breaks GPU access "
                f"for every framework in this venv. Pin the offending package "
                f"in {path} or preinstall it in the image."
            )
    return outcome


def report(outcome: Outcome, *, prefix: str = "framework deps") -> None:
    """Print a one-or-two line summary of ``outcome``."""
    for name in outcome.refused:
        print(
            f"{prefix}: refusing load-bearing '{name}' from the "
            f"{outcome.framework} manifest (would put the ROCm torch at risk)"
        )
    for entry in outcome.invalid:
        print(
            f"{prefix}: cannot derive an import name for '{entry}' in the "
            f"{outcome.framework} manifest; declare one as '<spec>:<module>'"
        )
    if outcome.skipped_reason:
        print(f"{prefix}: {outcome.framework or '<unset>'} skipped ({outcome.skipped_reason})")
        return
    if outcome.would_install:
        print(f"{prefix}: would install {' '.join(outcome.would_install)}")
        return
    if outcome.installed:
        print(f"{prefix}: installed {' '.join(outcome.installed)}")
    if outcome.failed:
        print(
            f"{prefix}: {outcome.framework} unresolved: {' '.join(outcome.failed)}"
            " -- benchmarks importing them will fail until this is fixed"
        )
    elif not outcome.installed:
        print(f"{prefix}: {outcome.framework} already satisfied")


def main(argv: list[str] | None = None) -> int:
    """Installer entry point: ``python -m ...framework_deps --framework X``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--framework", default=os.environ.get("FRAMEWORK", ""))
    parser.add_argument("--python", dest="python_exe", default=sys.executable)
    # One flag per occurrence, attached with =: a pip flag starts with a dash, which argparse never consumes as the
    # value of a variadic option.
    parser.add_argument("--pip-extra", action="append", default=None, metavar="FLAG")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prefix", default="framework deps")
    args = parser.parse_args(argv)

    try:
        outcome = ensure(
            args.framework,
            python_exe=args.python_exe,
            pip_extra=tuple(args.pip_extra or ()),
            check_only=args.check_only,
            dry_run=args.dry_run,
        )
    except TorchClobberedError as exc:
        print(f"{args.prefix}: FATAL {exc}", file=sys.stderr)
        return 2
    report(outcome, prefix=args.prefix)
    # Unresolved packages stay fail-soft: the installer should not abort a whole pod setup over one optional build,
    # and the warning is already printed.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
