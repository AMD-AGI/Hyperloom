# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Ray smoke test preflight runs on a backend interpreter, and the pip specs built from what it reports.

``preflight.py`` owns running the script and the install; this module owns the script's text, the versions it
gates on, and the reading of its output back into the specs a failed install is retried with.
"""

from __future__ import annotations

import re
import subprocess

# A floor rather than an exact requirement: interpreters with no 2.44.1 wheel
# (cp314 postdates it) must be allowed to keep the newer release the
# kernel-agent installer resolved for them.
_RAY_MIN_VERSION = "2.44.1"
# Only 2.44.1's CLI requires the default Click ceiling. Explicit Ray/Click
# overrides retain the installer's requested ceiling on other releases too.
_RAY_CLICK_PINNED_VERSION = "2.44.1"
_RAY_CLI_CLICK_MAX_VERSION = "8.3.0"


_RAY_SMOKE_TEMPLATE = r"""
import importlib.metadata as md
import re
import sys

RAY_MIN_VERSION = "__RAY_MIN_VERSION__"
RAY_CLICK_PINNED_VERSION = "__RAY_CLICK_PINNED_VERSION__"
RAY_CLI_CLICK_MAX_VERSION = "__RAY_CLI_CLICK_MAX_VERSION__"
ray_version = sys.argv[1] if len(sys.argv) > 1 else ""
click_override = sys.argv[2] if len(sys.argv) > 2 else ""

def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = [int(p) for p in re.findall(r"\d+", version)[:3]]
    parts.extend([0] * (3 - len(parts)))
    return tuple(parts[:3])

try:
    import ray
except Exception as exc:
    print(f"ray import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

# Reported even when a later gate fails, so the caller can build a repair spec
# from what is installed rather than from the pin it wanted.
print(f"ray detected: {ray.__version__}", file=sys.stderr)

if ray_version:
    if ray.__version__ != ray_version:
        print(f"ray version mismatch: {ray.__version__} != {ray_version}", file=sys.stderr)
        raise SystemExit(1)
elif _version_tuple(ray.__version__) < _version_tuple(RAY_MIN_VERSION):
    print(f"ray too old: {ray.__version__} < {RAY_MIN_VERSION}", file=sys.stderr)
    raise SystemExit(1)

if ray_version or click_override or ray.__version__ == RAY_CLICK_PINNED_VERSION:
    click_max_version = click_override or RAY_CLI_CLICK_MAX_VERSION
    try:
        click_version = md.version("click")
    except md.PackageNotFoundError:
        print("click is not installed", file=sys.stderr)
        raise SystemExit(1)

    if _version_tuple(click_version) >= _version_tuple(click_max_version):
        print(
            f"click version incompatible with Ray CLI: {click_version} >= {click_max_version}",
            file=sys.stderr,
        )
        raise SystemExit(1)

try:
    from ray.scripts.scripts import main as _ray_cli_main  # noqa: F401
except Exception as exc:
    print(f"ray CLI import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(ray.__version__)
"""


_RAY_SMOKE = (
    _RAY_SMOKE_TEMPLATE.replace("__RAY_MIN_VERSION__", _RAY_MIN_VERSION)
    .replace("__RAY_CLICK_PINNED_VERSION__", _RAY_CLICK_PINNED_VERSION)
    .replace("__RAY_CLI_CLICK_MAX_VERSION__", _RAY_CLI_CLICK_MAX_VERSION)
)

_RAY_SMOKE_DETECTED_RE = re.compile(r"^ray detected: (\S+)$", re.MULTILINE)


def _ray_version_tuple(version: str) -> tuple[int, int, int]:
    """The first three numbers of ``version``, zero-padded — as the smoke reads them."""
    parts = [int(p) for p in re.findall(r"\d+", version)[:3]]
    parts.extend([0] * (3 - len(parts)))
    return (parts[0], parts[1], parts[2])


def _smoke_ray_version(check: subprocess.CompletedProcess) -> str:
    """The Ray version the smoke found on its interpreter; empty when Ray did not import."""
    match = _RAY_SMOKE_DETECTED_RE.search(check.stderr or "")
    return match.group(1) if match else ""


def _ray_retry_specs(installed_ray: str, click_spec: str, click_override: str) -> tuple[str, ...]:
    """What to retry with after the pinned Ray install failed, built from the observed state.

    A bare ``ray[default]>=_RAY_MIN_VERSION`` floor is already satisfied when the
    broken interpreter carries exactly that version: pip then exits 0 having
    installed nothing, and the Click ceiling — the part that has to move while
    Ray stays put — was dropped along with the pin. Ask for what this
    interpreter actually needs instead, never for something it already has.
    """
    if not installed_ray or _ray_version_tuple(installed_ray) < _ray_version_tuple(_RAY_MIN_VERSION):
        # Ray is absent or below the floor, so the floor installs something.
        # Preserve an explicit Click ceiling; the default only guards 2.44.1.
        return (f"ray[default]>={_RAY_MIN_VERSION}",) + ((click_spec,) if click_override else ())
    if click_override or installed_ray == _RAY_CLICK_PINNED_VERSION:
        # The smoke's Click gate applies to the Ray already installed here, so
        # Click is the thing to move — and a Ray spec would be the no-op.
        return (click_spec,)
    # Ray is past the floor with no Click ceiling of its own: only a strictly
    # newer Ray can repair it, and ``>`` cannot be satisfied by doing nothing.
    return (f"ray[default]>{installed_ray}",)
