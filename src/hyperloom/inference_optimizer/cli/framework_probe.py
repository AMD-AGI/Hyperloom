# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preflight serving-framework probes: candidate interpreters, ROCm-build verdicts, and their reports."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hyperloom.common import provenance

from .device_init import _Probe, _block_on_unverified_build, _rocm_smi_visible_gpus


# Escape hatch for the serving-framework gate below, mirroring install_baremetal.sh's --skip-base-check.
SKIP_FRAMEWORK_CHECK_ENV = "HYPERLOOM_SKIP_FRAMEWORK_CHECK"

#: Frameworks ``install_baremetal.sh --install-framework`` accepts; it exits 2 on
#: anything else. A test asserts this stays equal to the installer's own list.
_SETUP_INSTALLABLE_FRAMEWORKS = frozenset({"sglang", "vllm", "atom"})


def _setup_install_command(framework: str) -> str:
    """The documented setup invocation for ``framework``, verbatim in shape."""
    extra = " --framework-env isolated" if framework == "vllm" else ""
    return f'PYTHONPATH="$REPO_ROOT" python3 -m hyperloom setup -- --install-framework {framework}{extra} --yes'


# Rootfs markers the runtimes drop: Docker writes the first, podman the second.
_CONTAINER_MARKER_FILES = ("/.dockerenv", "/run/.containerenv")

# Runtime names that appear in a cgroup v1 path (v2 hides them, see below).
_CGROUP_RUNTIME_MARKERS = ("docker", "containerd", "kubepods", "libpod")


def _pid1_at_cgroup_root(cgroup: str) -> bool:
    """Whether PID 1 sits at the root of every cgroup hierarchy."""
    paths = [line.rsplit(":", 1)[-1].strip() for line in cgroup.splitlines() if line.strip()]
    return bool(paths) and all(path == "/" for path in paths)


def _in_container() -> bool:
    """Best-effort containerization test (never raises)."""
    if any(Path(marker).exists() for marker in _CONTAINER_MARKER_FILES):
        return True
    # Injected into every pod; a host that merely talks to a cluster lacks it.
    if os.environ.get("KUBERNETES_SERVICE_HOST", "").strip():
        return True
    # Empty env so only on-disk evidence (image markers, PID 1's container id) counts.
    if provenance.detect_image({}, probe=True):
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    if any(marker in cgroup for marker in _CGROUP_RUNTIME_MARKERS):
        return True
    return _pid1_at_cgroup_root(cgroup)


def _framework_probe_interpreters(framework: str, benchmark_python: str) -> list[str]:
    """Interpreters that may hold the serving package, deduped in probe order."""
    if framework == "atom":
        # Magpie launches ATOM with python3 from PATH, not its benchmark client interpreter.
        python = shutil.which("python3")
        return [python] if python else []
    candidates: list[str] = []
    venv_root = os.environ.get("VLLM_VENV_ROOT", "").strip()
    venv_python = str(Path(venv_root) / "bin" / "python") if venv_root else ""
    if framework == "vllm" and venv_python and os.access(venv_python, os.X_OK):
        candidates.append(venv_python)
    candidates += [benchmark_python, sys.executable]
    out: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


# A ROCm import traceback can run to megabytes, so the tail is clipped twice: to the last few lines, and to the
# informative head of each.
_PROBE_TAIL_LINES = 4
_PROBE_TAIL_LINE_CHARS = 200


def _probe_stderr_tail(stderr: str | None) -> str:
    """Clipped tail of a probe's stderr; ``""`` when it wrote nothing."""
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    return "\n".join(
        line if len(line) <= _PROBE_TAIL_LINE_CHARS else f"{line[:_PROBE_TAIL_LINE_CHARS]} ..."
        for line in lines[-_PROBE_TAIL_LINES:]
    )


def _probe_detail_block(detail: str) -> str:
    """Indent a probe's diagnostics under a preflight line, or return ``""``."""
    if not detail:
        return ""
    body = "\n".join(f"    {line}" for line in detail.splitlines())
    return f"\n  probe diagnostics:\n{body}"


def _probe_failure_detail(exc: BaseException) -> str:
    """One-line reason a probe reached no verdict (no argv dump)."""
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"probe timed out after {exc.timeout}s"
    return f"{type(exc).__name__}: {exc}"


# Probe budgets. find_spec only stats the filesystem; importing torch matches build_utils.probe_torch_abi's 30s, and
# vLLM's platform import costs far more.
_IMPORT_PROBE_TIMEOUT_SEC = 20
_ROCM_PROBE_TIMEOUT_SEC = 30
_VLLM_ROCM_PROBE_TIMEOUT_SEC = 120


def _rocm_evidence(framework: str) -> str:
    """Name the signal the ROCm verdict for ``framework`` rests on."""
    return "the vllm platform" if framework == "vllm" else "torch.version.hip"


def _probe_rocm_build(framework: str, python_exe: str) -> _Probe:
    """Tri-state: is the ROCm stack behind ``framework`` in ``python_exe`` ROCm?"""
    # rc 1 means only "definitely not ROCm", so nothing else may produce it -- Python exits 1 on any uncaught
    # exception, and find_spec found the package without importing it, so "spec present but import explodes" is a
    # normal path, not a corner.
    probe = ["import sys", "def verdict():"]
    if framework == "atom":
        selected_python = os.environ.get("PYTHON", sys.executable)
        probe += [
            "    import os, subprocess",
            "    from pathlib import Path",
            f"    selected = {selected_python!r}",
            "    try:",
            "        prefix = subprocess.check_output([selected, '-c', 'import sys; print(sys.prefix)'],",
            "                                         text=True, stderr=subprocess.PIPE, timeout=20).strip()",
            "    except (OSError, subprocess.SubprocessError) as exc:",
            "        raise RuntimeError(f'Cannot execute selected PYTHON={selected!r}: {exc}') from exc",
            "    if Path(prefix).resolve() != Path(sys.prefix).resolve():",
            "        raise RuntimeError(f'python3 prefix {sys.prefix} differs from selected PYTHON prefix {prefix}')",
            "    venv = os.environ.get('VIRTUAL_ENV')",
            "    if venv and Path(venv).resolve() != Path(sys.prefix).resolve():",
            "        raise RuntimeError(f'VIRTUAL_ENV={venv} differs from python3 prefix {sys.prefix}')",
        ]
    probe += [
        "    import torch",
        "    if not getattr(torch.version, 'hip', None):",
        "        return 1",
    ]
    if framework == "vllm":
        # vLLM carries its own platform verdict, so a ROCm torch beside a CUDA vLLM is still caught.
        probe += [
            "    import vllm",
            "    from vllm.platforms import current_platform",
            "    ck = getattr(current_platform, 'is_rocm', None)",
            "    ok = bool(ck()) if callable(ck) else 'rocm' in f'{current_platform!r}'.lower()",
            "    return 0 if ok else 1",
        ]
    else:
        if framework == "atom":
            probe.append("    import atom")
        probe.append("    return 0")
    probe += [
        "try:",
        "    code = verdict()",
        "except BaseException:",
        "    import traceback; traceback.print_exc()",
        "    code = 3",
        "sys.exit(code)",
    ]
    timeout = _VLLM_ROCM_PROBE_TIMEOUT_SEC if framework == "vllm" else _ROCM_PROBE_TIMEOUT_SEC
    try:
        proc = subprocess.run(
            [python_exe, "-c", "\n".join(probe)], capture_output=True, text=True, errors="replace", timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _Probe(None, _probe_failure_detail(exc), isinstance(exc, subprocess.TimeoutExpired))
    stderr = getattr(proc, "stderr", "")
    detail = _probe_stderr_tail(stderr)
    if proc.returncode == 0:
        return _Probe(True, detail)
    # Absent torch (3) and a signal death (negative rc, which a broken ROCm stack can trigger on ``import torch``)
    # both mean no verdict was reached; the caller blocks on that rather than sort unclear exits into classes.
    if proc.returncode == 1:
        return _Probe(False, detail)
    return _Probe(None, detail)


def _framework_importable(framework: str, python_exe: str) -> _Probe:
    """Whether ``python_exe`` can locate ``framework``."""
    probe = f"import importlib.util as u, sys; sys.exit(0 if u.find_spec({framework!r}) else 1)"
    try:
        proc = subprocess.run(
            [python_exe, "-c", probe],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_IMPORT_PROBE_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _Probe(False, _probe_failure_detail(exc), isinstance(exc, subprocess.TimeoutExpired))
    return _Probe(proc.returncode == 0, _probe_stderr_tail(getattr(proc, "stderr", "")))


def _resolve_framework_build(framework: str, interpreters: list[str]) -> tuple[str | None, _Probe]:
    """Best (interpreter, probe) across every candidate."""
    refuted: tuple[str, _Probe] | None = None
    inconclusive: tuple[str, _Probe] | None = None
    missing = _Probe(False)
    for python_exe in interpreters:
        found = _framework_importable(framework, python_exe)
        if found.timed_out:
            return inconclusive or (None, found)
        if not found.verdict:
            # Keep the first probe that said something: a framework that is installed yet unreachable shows up only
            # here.
            missing = missing if missing.detail else found
            continue
        probe = _probe_rocm_build(framework, python_exe)
        if probe.verdict is True:
            return python_exe, probe
        if probe.verdict is None:
            # Every unclear probe blocks, so the first one is as good a report as any.
            inconclusive = inconclusive or (python_exe, probe)
            if probe.timed_out:
                # A wedged import costs the full budget per candidate; stop the scan at the first one.
                return inconclusive
        else:
            refuted = refuted or (python_exe, probe)
    return inconclusive or refuted or (None, missing)


def _report_framework_probe(
    framework: str, found: str | None, probe: _Probe, interpreters: list[str]
) -> dict[str, Any]:
    """Report a usable or unverifiable probe; anything else goes to ``_report_unusable_framework``."""
    evidence = _rocm_evidence(framework)
    if found and probe.verdict is True:
        print(f"Preflight: {framework} importable ({found}); {evidence} confirms a ROCm build")
        return {
            "status": "applied",
            "skip_reason": None,
            "target": framework,
            "detail": {"probe_interpreter": found, "rocm_verified": True},
        }
    if probe.verdict is None or probe.timed_out:
        # No verdict is not "probably fine": a reworded torch message, a device failure or a wedged import all
        # land here, and only rc 0 says the stack serving will import works. Report the evidence and stop.
        visible, _skip_reason = _rocm_smi_visible_gpus()
        _block_on_unverified_build(
            framework, found, _probe_detail_block(probe.detail), visible, SKIP_FRAMEWORK_CHECK_ENV
        )
    return _report_unusable_framework(framework, found, probe, interpreters)


def _report_unusable_framework(
    framework: str, found: str | None, probe: _Probe, interpreters: list[str]
) -> dict[str, Any]:
    """Warn about, or stop the run on, a serving framework that is missing here or is not a ROCm build."""
    evidence = _rocm_evidence(framework)
    # Every path below stops the run, so it needs a remedy that works.
    probed = "\n".join(f"  - {python_exe}" for python_exe in interpreters)
    if framework not in _SETUP_INSTALLABLE_FRAMEWORKS:
        state = (
            f"is importable ({found}) but {evidence} says it is not a ROCm build"
            if found
            else f"is not importable by:\n{probed}"
        )
        print(
            f"Preflight: WARNING — {framework} {state}{_probe_detail_block(probe.detail)}\n"
            f"setup cannot install {framework}, so it has to come from the image or an\n"
            "existing checkout on this host. Continuing; the benchmark will fail if it\n"
            f"genuinely needs {framework} here."
        )
        return {
            "status": "warned",
            "skip_reason": None,
            "target": framework,
            "message": probe.detail or "framework availability could not be verified",
            "detail": {"probe_interpreter": found, "rocm_verified": probe.verdict},
        }

    if found:
        print(
            f"\nERROR: {framework} is importable ({found}) but {evidence} says it is NOT a ROCm build."
            f"{_probe_detail_block(probe.detail)}\n\n"
            "The wheels on PyPI are the CUDA build: they import fine and then fail\n"
            "at GPU init. Reinstall the ROCm stack:\n"
            f"    {_setup_install_command(framework)}\n\n"
            f"To proceed anyway, set {SKIP_FRAMEWORK_CHECK_ENV}=1.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if _in_container():
        remedy = (
            "This process already runs in a container, so its image does not ship\n"
            f"{framework}. Restart it from a ROCm image that does, or install it here:\n"
            f"    {_setup_install_command(framework)}"
        )
    else:
        remedy = (
            "Pick ONE of:\n"
            f"  1. Install it on this host (ROCm {framework} is NOT on PyPI; setup pulls\n"
            "     it from the ROCm wheel index instead):\n"
            f"       {_setup_install_command(framework)}\n"
            "  2. Run Hyperloom inside a ROCm image that already ships it: set\n"
            "     HYPERLOOM_RUN_MODE=docker before setup, which hands image choice\n"
            "     and container startup to the demo skill.\n"
            "\nNote: HYPERLOOM_RUN_MODE selects where Hyperloom itself runs and is\n"
            "unrelated to Magpie's run_mode=local, which only means: do not start a\n"
            "second container. That stays correct in both cases."
        )
    print(
        f"\nERROR: serving framework {framework!r} is not importable by any candidate interpreter:\n"
        f"{probed}{_probe_detail_block(probe.detail)}\n\n"
        f"{remedy}\n\n"
        "If the server already runs elsewhere, benchmark against it instead:\n"
        "    export BENCHMARK_BASE_URL=http://<serving-host>:<port>\n"
        f"which skips this check. As a last resort, {SKIP_FRAMEWORK_CHECK_ENV}=1 drops it\n"
        "for a local run that is expected to serve from somewhere unprobed.",
        file=sys.stderr,
    )
    raise SystemExit(2)
