"""GPU / framework environment probe.

Called once at conductor startup to fill in the env block executors
need (``GPU_COUNT``, ``GPU_TYPE``, ``FRAMEWORK_VERSION``) when the
operator hasn't explicitly set them on the command line.

Everything here is best-effort and side-effect-free: every probe
returns ``None`` instead of raising when the underlying tool is
missing or fails. The conductor's :meth:`_bootstrap` falls back to
operator-supplied defaults in that case.

Test seam: the public ``probe_environment(runner=...)`` accepts a
custom subprocess runner so unit tests can simulate amd-smi / rocm-smi
without spawning real binaries.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable


log = logging.getLogger(__name__)


# Type alias for the subprocess seam used by tests.
SubprocessRunner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Production runner. Times out aggressively so a misbehaving SMI
    can't stall the conductor's startup."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@dataclass
class EnvProbe:
    """One snapshot of the local GPU + framework environment."""

    gpu_count: int | None = None
    gpu_type: str | None = None  # e.g. "gfx942" / "MI355X" / ""
    rocm_smi: str | None = None
    amd_smi: str | None = None
    nvidia_smi: str | None = None
    framework: str | None = None  # "sglang" | "vllm" | None
    framework_version: str | None = None

    def to_env(self) -> dict[str, str]:
        """Return the subset that should be merged into the run env block.

        Only fields with non-empty values are returned so we never
        clobber operator-supplied env vars.
        """
        out: dict[str, str] = {}
        if self.gpu_count is not None:
            out["GPU_COUNT"] = str(self.gpu_count)
        if self.gpu_type:
            out["GPU_TYPE"] = self.gpu_type
        if self.framework:
            out["FRAMEWORK"] = self.framework
        if self.framework_version:
            out["FRAMEWORK_VERSION"] = self.framework_version
        return out


# --------------------------------------------------------------------------
# GPU detection
# --------------------------------------------------------------------------
def detect_gpu_count(env: dict[str, str], runner: SubprocessRunner) -> int | None:
    """Best-effort GPU count detection. Order of precedence:

        1. ``GPU_COUNT`` env (operator override)
        2. ``HIP_VISIBLE_DEVICES`` count
        3. ``ROCR_VISIBLE_DEVICES`` count
        4. ``CUDA_VISIBLE_DEVICES`` count
        5. ``amd-smi list`` row count
        6. ``rocm-smi --showid`` row count
        7. ``nvidia-smi -L`` line count
    """
    if env.get("GPU_COUNT"):
        try:
            return int(env["GPU_COUNT"])
        except ValueError:
            pass
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        v = env.get(var, "")
        if v:
            return len([x for x in v.split(",") if x.strip()])

    if shutil.which("amd-smi"):
        try:
            r = runner(["amd-smi", "list"])
            if r.returncode == 0:
                count = sum(
                    1 for line in r.stdout.splitlines()
                    if line.startswith("GPU:")
                )
                if count:
                    return count
        except (OSError, subprocess.SubprocessError):
            pass

    if shutil.which("rocm-smi"):
        try:
            r = runner(["rocm-smi", "--showid"])
            if r.returncode == 0:
                count = sum(
                    1 for line in r.stdout.splitlines()
                    if line.strip() and line.lstrip()[:1].isdigit()
                )
                if count:
                    return count
        except (OSError, subprocess.SubprocessError):
            pass

    if shutil.which("nvidia-smi"):
        try:
            r = runner(["nvidia-smi", "-L"])
            if r.returncode == 0:
                count = sum(1 for _ in r.stdout.splitlines() if _.strip())
                if count:
                    return count
        except (OSError, subprocess.SubprocessError):
            pass

    return None


def detect_gpu_type(env: dict[str, str], runner: SubprocessRunner) -> str | None:
    """Return a short architecture tag (``gfx942`` etc.) when discoverable."""
    if env.get("GPU_TYPE"):
        return env["GPU_TYPE"]
    if shutil.which("rocm-smi"):
        try:
            r = runner(["rocm-smi", "--showproductname"])
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "GFX Version" in line:
                        # Lines look like: "GPU[0]: GFX Version: gfx942"
                        for token in line.split():
                            if token.startswith("gfx"):
                                return token.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    if shutil.which("nvidia-smi"):
        try:
            r = runner([
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
            ])
            if r.returncode == 0:
                first = next(
                    (l.strip() for l in r.stdout.splitlines() if l.strip()),
                    "",
                )
                return first or None
        except (OSError, subprocess.SubprocessError):
            pass
    return None


# --------------------------------------------------------------------------
# Framework detection
# --------------------------------------------------------------------------
def detect_framework(env: dict[str, str]) -> tuple[str | None, str | None]:
    """Return ``(framework_name, version_string)``. Operator override
    via ``FRAMEWORK`` env wins."""
    requested = env.get("FRAMEWORK", "").strip().lower()
    if requested in ("sglang", "vllm"):
        return requested, _import_version(requested)
    # Auto-prefer sglang if both installed (matches the bundled scripts'
    # default).
    sg_v = _import_version("sglang")
    if sg_v:
        return "sglang", sg_v
    vl_v = _import_version("vllm")
    if vl_v:
        return "vllm", vl_v
    return None, None


def _import_version(module: str) -> str | None:
    try:
        m = __import__(module)
        return getattr(m, "__version__", None)
    except Exception:  # noqa: BLE001 — module just isn't installed
        return None


# --------------------------------------------------------------------------
# Top-level entry
# --------------------------------------------------------------------------
def probe_environment(
    *,
    env: dict[str, str] | None = None,
    runner: SubprocessRunner | None = None,
) -> EnvProbe:
    """Run all probes; return an :class:`EnvProbe` snapshot."""
    env = dict(env) if env is not None else dict(os.environ)
    run = runner or _default_runner
    gpu_count = detect_gpu_count(env, run)
    gpu_type = detect_gpu_type(env, run)
    framework, framework_version = detect_framework(env)
    return EnvProbe(
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        rocm_smi=shutil.which("rocm-smi"),
        amd_smi=shutil.which("amd-smi"),
        nvidia_smi=shutil.which("nvidia-smi"),
        framework=framework,
        framework_version=framework_version,
    )


def fill_default_env(
    base: dict[str, str], probe: EnvProbe
) -> dict[str, str]:
    """Return a *new* env block with auto-detected defaults applied where
    the operator hasn't already set them.

    Specifically populates:

        TP        = GPU_COUNT  (only when missing AND we detected a count)
        CONC      = sensible default by TP (4 / 32 / 64)
        ISL/OSL   = 1024 / 256                (sister default)
        FRAMEWORK = whatever we detected
        GPU_*     = from probe.to_env()
    """
    env = dict(base)
    env.update({k: v for k, v in probe.to_env().items() if k not in env})

    if "TP" not in env and probe.gpu_count:
        env["TP"] = str(probe.gpu_count)
    tp_val = env.get("TP", "")
    if "CONC" not in env:
        try:
            tp_n = int(tp_val) if tp_val else 0
        except ValueError:
            tp_n = 0
        if tp_n <= 1:
            env["CONC"] = "4"
        elif tp_n <= 4:
            env["CONC"] = "32"
        else:
            env["CONC"] = "64"
    env.setdefault("ISL", "1024")
    env.setdefault("OSL", "256")
    env.setdefault("PORT", "8888")
    return env


__all__ = [
    "EnvProbe",
    "SubprocessRunner",
    "detect_framework",
    "detect_gpu_count",
    "detect_gpu_type",
    "fill_default_env",
    "probe_environment",
]
