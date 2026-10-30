# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI ``_preflight`` cluster — auto-install/env-hygiene checks run before ``optimize`` starts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from hyperloom.common.agent_runtime_requirements import agent_runtime_requirements
from hyperloom.common.agent_cli import resolve_claude_executable, resolve_codex_executable
from hyperloom.common.llm_config import (
    AGENT_BACKEND_CODEX,
    CLAUDE_OAUTH_TOKEN_ENV,
    DEFAULT_CLAUDE_MODEL,
    anthropic_synthesizable_key,
    preferred_agent_backend,
    with_synthesized_anthropic_keys,
)
from hyperloom.common.fs_utils import is_network_fs
from hyperloom.common.llm_headers import expand_env_refs
from hyperloom.common.gpu_identity import AMD_GPU_BOARDS
from hyperloom.common.platform_probe import probe_cpu_platform
from hyperloom.common.pr_monitor_urls import kb_store_url
from hyperloom.common.provenance import (
    RESOLVED_FRAMEWORK_ENV,
    RESOLVED_FRAMEWORK_PYTHON_ENV,
    detect_gfx_arch,
)
from hyperloom.common.timeutil import now_iso
from hyperloom.common.visible_devices import hip_mask_within_rocr
from hyperloom.common.model_paths import resolve_serving_model_path

from .. import inferencex_checkout
from .bootstrap import parse_operator_extra_env
from .device_init import _rocm_smi_visible_gpus
from .framework_probe import (
    SKIP_FRAMEWORK_CHECK_ENV,
    _framework_probe_interpreters,
    _probe_detail_block,
    _report_framework_probe,
    _resolve_framework_build,
    _setup_install_command,
)
from .preflight_env import (
    _ANTHROPIC_FALLBACK_KEYS,
    _PROVIDER_FALLBACK_KEYS,
    _derive_runtime_paths,
    _load_dotenv_fallback,
    _load_kernel_agent_env_fallback,
    _normalize_legacy_deepseek_env,
    _provider_only_mode,
    _restore_provider_only_mode,
)
from .server_args import validate_server_args

from .credentials import (
    _is_stale_proxy_url,
    _resolve_llm_endpoints,
    _reset_claude_config_to_upstream,
    _sync_geak_config_base_url,
    _validate_credentials,
)
from ..session.paths import (
    DEFAULT_SESSION_DIR,
    ENV_USER_DATA_PATH,
    session_dir as _session_dir_resolve,
    workspace_root as _workspace_root_resolve,
)

log = logging.getLogger("hyperloom.inference_optimizer.cli")
_ASSETS = Path(__file__).resolve().parent.parent / "assets"

_CLONE_INFERENCEX_STEP = "clone_inferencex"


#: What a custom header may reference. Every child inherits the expanded header, so expanding any other name would
#: hand a child the value of a secret its environment allowlist strips, such as ``${GITHUB_TOKEN}``.
_HEADER_REF_ALLOWLIST = frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"})


def _expand_custom_header_env_refs() -> None:
    """Resolve the credential ``${VAR}`` references in the ``*_CUSTOM_HEADERS`` settings in place.

    Child processes (specialists, the Critic, GEAK on Ray) forward these verbatim, and an agent CLI sends them
    verbatim, so an unexpanded ``${ANTHROPIC_API_KEY}`` reaches the gateway as literal text and is rejected.
    References resolve against the same view ``claude_sdk_env_options`` uses, so a key it would fill in from the
    other Anthropic credential is not erased here. A reference outside ``_HEADER_REF_ALLOWLIST`` is left as
    written, which is what children received before this expansion existed.
    """
    source = with_synthesized_anthropic_keys(os.environ)
    for key in ("ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS"):
        raw = os.environ.get(key)
        if not raw or "${" not in raw:
            continue
        os.environ[key] = expand_env_refs(raw, source, only=_HEADER_REF_ALLOWLIST)
        if "${" in os.environ[key]:
            log.warning(
                "%s references a variable outside %s; child processes receive that reference unexpanded",
                key,
                ", ".join(sorted(_HEADER_REF_ALLOWLIST)),
            )


# /dev/shm threshold: below this, a launch can collide with stale vLLM/NCCL shm segments.
_DEV_SHM_MIN_FREE_BYTES = 16 * 1024 * 1024 * 1024  # 16 GiB


def _ensure_python_sdks(python_exe: str, pip_extra: list[str]) -> dict[str, Any]:
    """Probe-then-install runtime-imported Python SDKs using the same interpreter that imports them."""
    candidates = agent_runtime_requirements()
    installed: list[str] = []
    already_present: list[str] = []
    for module_name, pip_spec in candidates:
        check = subprocess.run(
            [python_exe, "-c", f"import {module_name}"],
            capture_output=True,
        )
        if check.returncode == 0:
            print(f"Preflight: {module_name} OK")
            already_present.append(pip_spec)
            continue
        print(f"Preflight: {module_name} not importable, installing {pip_spec} ...")
        subprocess.run(
            [python_exe, "-m", "pip", "install", "--quiet", *pip_extra, pip_spec],
            check=True,
        )
        print(f"Preflight: installed {pip_spec}")
        installed.append(pip_spec)
    return {
        "status": "applied" if installed else "already_present",
        "skip_reason": None,
        "target": ",".join(spec for _, spec in candidates),
        "interpreter": python_exe,
        "detail": {
            "installed": installed,
            "already_present": already_present,
        },
    }


# A floor rather than an exact requirement: interpreters with no 2.44.1 wheel
# (cp314 postdates it) must be allowed to keep the newer release the
# kernel-agent installer resolved for them.
_RAY_MIN_VERSION = "2.44.1"
# Ray 2.44.1's CLI does not import under click 8.3+ ("ValueError: ... is not a valid Sentinel"). The smoke
# exits with this code for that failure on that Ray alone, the one failure a fixed click hold repairs;
# install_kernel_tools.sh applies the same rule.
_RAY_CLICK_HOLD_VERSION = "2.44.1"
_RAY_CLICK_HOLD_SPEC = "click<8.3"
_RAY_SENTINEL_EXIT = 75
# A representative `ray start` argv of the orchestrator's Ray runtime: every option it passes to the head
# this step provisions, with values of its kind. Ray's own parser reading this argv is the certification, so
# a pair whose `ray start` parses what Hyperloom passes is never refused over a version number.
_RAY_START_ARGV = (
    *"--head --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265".split(),
    *"--ray-client-server-port=10001 --num-gpus=8 --resources".split(),
    '{"serving_slot": 1}',
)


_RAY_SMOKE_SCRIPT = Path(__file__).resolve().parent.parent / "assets" / "ray_cli_smoke.py"


def _ray_probe_env() -> dict[str, str]:
    """Ray refuses to import on ROCm when only ROCR_VISIBLE_DEVICES is set, and
    preflight clears HIP_VISIBLE_DEVICES for the benchmark path; restore a
    re-indexed value for Ray's own probes so they are not false negatives."""
    env = dict(os.environ)
    if env.get("HIP_VISIBLE_DEVICES"):
        return env
    visible = [part for part in env.get("ROCR_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if visible:
        env["HIP_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(len(visible)))
    return env


def _ray_smoke_command(python_exe: str, ray_version: str) -> list[str]:
    hold = (_RAY_CLICK_HOLD_VERSION, str(_RAY_SENTINEL_EXIT))
    return [python_exe, str(_RAY_SMOKE_SCRIPT), _RAY_MIN_VERSION, *hold, *_RAY_START_ARGV, ray_version]


def _ray_smoke(python_exe: str, ray_version: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        _ray_smoke_command(python_exe, ray_version),
        capture_output=True,
        text=True,
        env=_ray_probe_env(),
    )


def _pip_install_ray(python_exe: str, pip_extra: list[str], specs: tuple[str, ...]) -> subprocess.CompletedProcess:
    argv = [python_exe, "-m", "pip", "install", "--quiet", *pip_extra, *specs]
    return subprocess.run(argv, capture_output=True, text=True)


def _smoke_after_click_hold(
    python_exe: str, pip_extra: list[str], ray_version: str, check: subprocess.CompletedProcess
) -> subprocess.CompletedProcess:
    """Hold click below 8.3 and smoke again, but only for the Sentinel import failure of Ray 2.44.1."""
    if check.returncode != _RAY_SENTINEL_EXIT:
        return check
    print(f"Preflight: ray {_RAY_CLICK_HOLD_VERSION} cannot import under this click; holding {_RAY_CLICK_HOLD_SPEC}")
    _pip_install_ray(python_exe, pip_extra, (_RAY_CLICK_HOLD_SPEC,))
    return _ray_smoke(python_exe, ray_version)


def _ray_result(status: str, python_exe: str, spec: str, version_after: str, message: str | None) -> dict[str, Any]:
    """The record the Ray step reports, whether it found Ray or had to install it."""
    return {
        "status": status,
        "skip_reason": None,
        "target": "ray",
        "interpreter": python_exe,
        "spec": spec,
        "version_after": version_after,
        "message": message,
    }


def _install_ray(python_exe: str, pip_extra: list[str], ray_version: str, ray_spec: str, reason: str) -> dict[str, Any]:
    """Install Ray for an interpreter whose smoke test refused the Ray it has."""
    specs: tuple[str, ...] = (ray_spec,)
    pinned_detail = ""
    install = _pip_install_ray(python_exe, pip_extra, specs)
    if install.returncode != 0 and not ray_version:
        pinned_detail = (install.stderr or install.stdout or "no pip output").strip()
        specs = (f"ray[default]>={_RAY_MIN_VERSION}",)
        print(f"Preflight: {ray_spec} does not resolve for {python_exe}; retrying with {' '.join(specs)}")
        install = _pip_install_ray(python_exe, pip_extra, specs)
    if install.returncode != 0:
        detail = (install.stderr or install.stdout or "no pip output").strip()
        raise RuntimeError(f"Ray install failed for {' '.join(specs)}: {detail}")
    check = _smoke_after_click_hold(python_exe, pip_extra, ray_version, _ray_smoke(python_exe, ray_version))
    if check.returncode != 0:
        detail = (check.stderr or check.stdout or "unknown Ray smoke failure").strip()
        # The floor spec is satisfied by whatever Ray is already installed, so a
        # zero exit from pip does not mean anything was installed: report what
        # the repair really did rather than claiming it completed.
        earlier = f" after {ray_spec} failed to install ({pinned_detail})" if pinned_detail else ""
        raise RuntimeError(
            f"pip install {' '.join(specs)} succeeded{earlier} but the Ray smoke test still failed: {detail}"
        )
    version_after = (check.stdout or "").strip() or ray_version or _RAY_MIN_VERSION
    print(f"Preflight: ray installed OK ({version_after})")
    return _ray_result("applied", python_exe, " ".join(specs), version_after, reason)


def _ensure_ray(python_exe: str, pip_extra: list[str]) -> dict[str, Any]:
    """Probe-then-install Ray using the interpreter that will import it."""
    ray_version = os.environ.get("RAY_VERSION", "")
    ray_spec = f"ray[default]=={ray_version or _RAY_MIN_VERSION}"
    check = _ray_smoke(python_exe, ray_version)
    if check.returncode == _RAY_SENTINEL_EXIT:
        held = _smoke_after_click_hold(python_exe, pip_extra, ray_version, check)
        if held.returncode == 0:
            version_after = held.stdout.strip() or ray_version or _RAY_MIN_VERSION
            reason = "ray CLI Sentinel import failure"
            return _ray_result("applied", python_exe, _RAY_CLICK_HOLD_SPEC, version_after, reason)
        check = held
    if check.returncode == 0:
        print("Preflight: ray OK")
        version_after = check.stdout.strip() or ray_version or _RAY_MIN_VERSION
        return _ray_result("already_present", python_exe, ray_spec, version_after, None)
    reason = (check.stderr or check.stdout or "unknown Ray smoke failure").strip().splitlines()[-1]
    print(f"Preflight: ray unusable ({reason}), installing {ray_spec} ...")
    return _install_ray(python_exe, pip_extra, ray_version, ray_spec, reason)


# InferenceX benchmark_serving client-side deps.
_BENCH_SERVING_DEPS = (
    "aiohttp",
    "tqdm",
    "numpy",
    "requests",
    "transformers",
    "huggingface_hub",
    "datasets",
    "pandas",
)


def _ensure_bench_serving_deps(python_exe: str, pip_extra: list[str]) -> dict[str, Any]:
    """Probe-then-install the InferenceX benchmark_serving client deps in python_exe."""
    mods = list(_BENCH_SERVING_DEPS)
    probe = (
        "import importlib.util, sys; print('\\n'.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))"
    )
    result = subprocess.run([python_exe, "-c", probe, *mods], capture_output=True, text=True)
    if result.returncode != 0:
        # Probe itself failed unexpectedly; fall back to attempting all so a genuinely missing client is not silently
        # left uninstalled.
        missing = mods
    else:
        missing = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if not missing:
        print("Preflight: benchmark_serving client deps OK")
        return {
            "status": "already_present",
            "skip_reason": None,
            "target": "benchmark_serving client deps",
            "interpreter": python_exe,
            "detail": {"installed": [], "already_present": mods},
        }
    print(f"Preflight: installing benchmark_serving client deps: {' '.join(missing)} ...")
    subprocess.run(
        [python_exe, "-m", "pip", "install", "--quiet", "--no-cache-dir", *pip_extra, *missing],
        check=True,
    )
    print("Preflight: benchmark_serving client deps installed OK")
    return {
        "status": "applied",
        "skip_reason": None,
        "target": "benchmark_serving client deps",
        "interpreter": python_exe,
        "detail": {
            "installed": missing,
            "already_present": [module for module in mods if module not in missing],
        },
    }


def _check_serving_framework(args, benchmark_python: str) -> dict[str, Any]:
    """Fail fast when the selected serving framework is not importable here."""
    from hyperloom.inference_optimizer import framework_registry

    framework = (
        getattr(args, "framework", None) or os.environ.get("FRAMEWORK", "")
    ).strip().lower() or framework_registry.DEFAULT_FRAMEWORK

    if framework_registry.is_scriptable(framework):
        return {"status": "skipped", "skip_reason": "scriptable_framework", "target": framework}
    if os.environ.get(SKIP_FRAMEWORK_CHECK_ENV, "").strip():
        print(f"Preflight: {SKIP_FRAMEWORK_CHECK_ENV} set; skipping the {framework} importability check")
        return {"status": "skipped", "skip_reason": "skip_framework_check_env", "target": framework}
    remote_base_url = os.environ.get("BENCHMARK_BASE_URL", "").strip()
    if remote_base_url:
        print(f"Preflight: BENCHMARK_BASE_URL={remote_base_url}; skipping the local {framework} check (remote server)")
        return {"status": "skipped", "skip_reason": "remote_benchmark_url", "target": framework}

    from hyperloom.inference_optimizer.multi_node._internal.external_state import external_service_url
    from hyperloom.orchestrator.actions.executors._multi_node_env import is_multi_node

    if is_multi_node() and external_service_url():
        print(f"Preflight: external multi-node mode; skipping the local {framework} check (serving is on remote pods)")
        return {"status": "skipped", "skip_reason": "external_multi_node", "target": framework}

    interpreters = _framework_probe_interpreters(framework, benchmark_python)
    found, probe = _resolve_framework_build(framework, interpreters)
    if framework == "atom" and (not found or probe.verdict is not True):
        remedy = (
            "Select the existing ATOM Python and put its bin directory first on PATH, or install ATOM into it:\n"
            f"    {_setup_install_command(framework)}"
            if not found
            else "Select an ATOM Python whose torch is a ROCm build and put its bin directory first on PATH."
        )
        print(
            f"Preflight: ERROR — atom runtime check failed in python3 ({found or ', '.join(interpreters) or 'not on PATH'}). "
            "ATOM must import with a ROCm torch build (torch.version.hip) in the selected Python environment."
            f"{_probe_detail_block(probe.detail)}\n"
            f"{remedy}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    # Publish the interpreter this scan resolved to, so consumers that would otherwise re-derive it from
    # installer-written host state read the probed answer instead.
    if found and probe.verdict is True:
        os.environ[RESOLVED_FRAMEWORK_PYTHON_ENV] = found
        os.environ[RESOLVED_FRAMEWORK_ENV] = framework
    return _report_framework_probe(framework, found, probe, interpreters)


# Probed one subprocess each: the base package and the [api] extra can arrive from different places (image vs pip),
# and only the truly absent one is installed.
_LM_EVAL_DEPS = ("lm_eval", "tenacity")


def _probe_missing_lm_eval_deps(python_exe: str) -> list[str] | None:
    """Report which accuracy-gate modules ``python_exe`` cannot import."""
    try:
        # A missing or non-executable interpreter raises instead of returning a code, and preflight must not die on
        # it: absence stays unproven, which the caller reports without touching anything.
        liveness = subprocess.run([python_exe, "-c", "pass"], capture_output=True)
        if liveness.returncode != 0:
            return None
        missing: list[str] = []
        for module in _LM_EVAL_DEPS:
            probe = subprocess.run([python_exe, "-c", f"import {module}"], capture_output=True)
            if probe.returncode != 0:
                missing.append(module)
    except OSError:
        return None
    return missing


# Pin consulted on multi-node preflight only: ``_ensure_lm_eval_dep`` skips single-node
# installs (``single_node_runtime_install``) because InferenceX's ``benchmark_lib.sh``
# reinstalls its own hardcoded pre-#3293 ref before every accuracy round there. This
# constant therefore does *not* decide which harness a single-node round runs; the guard
# appended to ``lm_eval_sitecustomize.py`` in ``_inferencex_patcher.py`` does.
#
# v0.4.13 (``ddd6722``). The previous pin matched InferenceX's reinstall ref
# (2025-12-02, ``b315ef3``): its failure handler logs bare ``outputs``, so a refused
# connection raises ``UnboundLocalError`` over the real error
# (EleutherAI/lm-evaluation-harness#3293, fixed upstream 2026-02-24).
_LM_EVAL_PINNED_REF = "ddd67220430a2470529f25fd5c05a576ca1057a0"
_LM_EVAL_REPO = "github.com/EleutherAI/lm-evaluation-harness"
# git first, then the archive, because the sandbox may not ship a git binary.
_LM_EVAL_PINNED_SPECS = (
    ("git", f"lm_eval[api] @ git+https://{_LM_EVAL_REPO}.git@{_LM_EVAL_PINNED_REF}"),
    ("archive", f"lm_eval[api] @ https://{_LM_EVAL_REPO}/archive/{_LM_EVAL_PINNED_REF}.tar.gz"),
)
# Settled by install.sh (or the image) and load-bearing elsewhere in the stack: pandas for rocprof-compute's CSV
# converter, torch/triton for the ROCm build PyPI has no equivalent of, numpy because both pin against it.
_LM_EVAL_FROZEN_DEPS = ("torch", "pandas", "numpy", "triton")


def _frozen_constraints(python_exe: str) -> list[str]:
    """Pin the packages this install must not move, as ``pip -c`` arguments."""
    pins: list[str] = []
    for name in _LM_EVAL_FROZEN_DEPS:
        probe = subprocess.run(
            [python_exe, "-c", f"import importlib.metadata as m; print(m.version({name!r}))"],
            capture_output=True,
            text=True,
            check=False,
        )
        version = probe.stdout.strip()
        if probe.returncode == 0 and version:
            pins.append(f"{name}=={version}")
    if not pins:
        print("Preflight: WARNING — could not read installed versions; lm_eval install is unconstrained")
        return []
    # The name deliberately carries no package name: this path is spliced into a pip command line that callers assert
    # does not mention a package spec.
    handle, path = tempfile.mkstemp(prefix="hyperloom_pip_constraints_", suffix=".txt")
    with os.fdopen(handle, "w", encoding="utf-8") as fh:
        fh.write("\n".join(pins) + "\n")
    print(f"Preflight: constraining the lm_eval install to {' '.join(pins)}")
    return ["-c", path]


def _install_pinned_lm_eval(python_exe: str, pip_extra: list[str]) -> None:
    """Install the pinned ``lm_eval[api]``, falling back to the source archive."""
    constraints = _frozen_constraints(python_exe)
    for i, (source, spec) in enumerate(_LM_EVAL_PINNED_SPECS):
        is_last = i == len(_LM_EVAL_PINNED_SPECS) - 1
        proc = subprocess.run(
            [python_exe, "-m", "pip", "install", "--quiet", "--no-cache-dir", *constraints, *pip_extra, spec],
            check=is_last,
        )
        if proc.returncode == 0:
            return
        print(f"Preflight: WARNING — pinned lm_eval via {source} failed; falling back")


def _resumed_session_state(args: argparse.Namespace) -> dict[str, Any] | None:
    """The persisted ``state.json`` of the session ``--resume-from`` names, or ``None``.

    Preflight runs before the resume block loads the session, so anything preflight decides from a session-pinned
    setting has to read it here. ``None`` (no session named, or no readable state) leaves the decision to the flags;
    the resume block reports a missing or unreadable state itself.
    """
    raw = str(getattr(args, "resume_from", "") or "").strip()
    if not raw:
        return None
    try:
        state = json.loads((Path(raw).expanduser() / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def _supported_framework_names() -> tuple[str, ...]:
    """The framework names ``--framework`` accepts (the framework registry)."""
    from hyperloom.inference_optimizer import framework_registry

    return tuple(framework_registry.names())


def _refuse_resume_without_state(args: argparse.Namespace) -> None:
    """Refuse, before any install step, a resume of a session whose first launch never wrote ``state.json``.

    The resume block refuses such a session ("Coordinator never wrote SharedState"), but it runs after preflight, which
    would otherwise check the default framework and fail on that instead of on the real reason. The framework the
    session was created with is read from ``manifest.json`` only to name it in the refusal, and only if it is a
    registered framework name (the manifest lives on shared storage too). A missing session directory or manifest is
    left to the resume block, which reports those itself.
    """
    raw = str(getattr(args, "resume_from", "") or "").strip()
    if not raw:
        return
    session_dir = Path(raw).expanduser()
    if not (session_dir / "manifest.json").is_file() or (session_dir / "state.json").exists():
        return
    try:
        manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = None
    raw_framework = manifest.get("framework") if isinstance(manifest, dict) else None
    framework = raw_framework.strip().lower() if isinstance(raw_framework, str) else ""
    supported = _supported_framework_names()
    if framework in supported:
        hint = f"Start a fresh session with --framework {framework} instead."
    elif raw_framework not in (None, ""):
        hint = (
            f"Its manifest.json records --framework {raw_framework!r}, which is not a supported framework "
            f"({', '.join(supported)}). Start a fresh session instead."
        )
    else:
        hint = "Start a fresh session instead."
    print(
        f"ERROR: cannot resume this session -- {session_dir}/state.json missing (manifest exists but Coordinator "
        f"never wrote SharedState; the first launch stopped before the session had any state to resume). {hint}",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _pin_resumed_session_args(args: argparse.Namespace) -> None:
    """Apply the resumed session's pinned settings to ``args`` before any preflight check reads them.

    A session's framework is fixed at creation, and the resume block re-exports the persisted one; without this, a
    resume that does not re-pass ``--framework`` would have preflight install and probe the default framework instead
    of the session's. An explicit ``--framework`` that differs from the persisted one is refused: the resume would
    otherwise check one framework and run another. A kernel phase disabled at creation stays disabled, as the resume
    block also enforces, so the TraceLens requirement is judged on the same setting the run uses.
    """
    state = _resumed_session_state(args)
    if not state:
        _refuse_resume_without_state(args)
        return
    raw_persisted = state.get("framework")
    persisted = raw_persisted.strip().lower() if isinstance(raw_persisted, str) else ""
    if raw_persisted not in (None, "") and persisted not in _supported_framework_names():
        # state.json lives on shared storage; the framework name later becomes a manifest path that preflight
        # pip-installs from, so only a registered framework name may be pinned.
        print(
            f"ERROR: cannot resume this session -- its state.json records --framework {raw_persisted!r}, which is "
            f"not a supported framework ({', '.join(_supported_framework_names())}). The session state is invalid; "
            "start a fresh session.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if persisted:
        requested = str(getattr(args, "framework", None) or "").strip().lower()
        if requested and requested != persisted:
            print(
                f"ERROR: cannot resume this session -- it was created with --framework {persisted} and this resume "
                f"passes --framework {requested}; a session's framework cannot change. Drop --framework (or pass "
                f"--framework {persisted}) to resume it, or start a fresh session for {requested}.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        args.framework = persisted
    if state.get("kernel_enabled") is False:
        args.no_kernel = True


def _ensure_lm_eval_dep(
    python_exe: str,
    pip_extra: list[str],
) -> dict[str, Any]:
    """Probe-then-install ``lm_eval`` in python_exe when the accuracy gate is on."""
    from hyperloom.orchestrator.actions.executors._multi_node_env import is_multi_node

    if not is_multi_node():
        return {
            "status": "skipped",
            "skip_reason": "single_node_runtime_install",
            "target": "lm_eval[api]",
            "interpreter": python_exe,
            "message": "single-node InferenceX installs lm_eval on first use",
        }
    missing = _probe_missing_lm_eval_deps(python_exe)
    if missing is None:
        # Absence is unproven, so installing would be a guess that could replace an lm_eval the image ships.
        print("Preflight: WARNING — cannot run the lm_eval probe; leaving the interpreter untouched")
        return {
            "status": "warned",
            "skip_reason": None,
            "target": "lm_eval[api]",
            "interpreter": python_exe,
            "message": "lm_eval dependency probe could not run",
        }
    if not missing:
        print("Preflight: lm_eval[api] OK")
        return {
            "status": "already_present",
            "skip_reason": None,
            "target": "lm_eval[api]",
            "interpreter": python_exe,
        }
    if "lm_eval" in missing:
        print(f"Preflight: installing lm_eval[api]@{_LM_EVAL_PINNED_REF[:12]} (accuracy gate) ...")
        _install_pinned_lm_eval(python_exe, pip_extra)
        print("Preflight: lm_eval[api] installed OK")
        return {
            "status": "applied",
            "skip_reason": None,
            "target": "lm_eval[api]",
            "interpreter": python_exe,
            "detail": {"installed": list(missing)},
        }
    # The image already ships lm_eval.
    targets = missing
    print(f"Preflight: installing {' '.join(targets)} (accuracy gate; missing: {' '.join(missing)}) ...")
    subprocess.run(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            *_frozen_constraints(python_exe),
            *pip_extra,
            *targets,
        ],
        check=True,
    )
    print(f"Preflight: {' '.join(targets)} installed OK")
    return {
        "status": "applied",
        "skip_reason": None,
        "target": "lm_eval[api]",
        "interpreter": python_exe,
        "detail": {"installed": list(targets)},
    }


def _normalize_hip_visible_devices() -> None:
    """Re-index HIP within the device view selected by ROCR."""
    value = hip_mask_within_rocr(os.environ)
    if value is None:
        return
    previous = os.environ.get("HIP_VISIBLE_DEVICES")
    if previous == value:
        return
    os.environ["HIP_VISIBLE_DEVICES"] = value
    print(
        f"Preflight: WARNING — normalized HIP_VISIBLE_DEVICES={previous!r} to {value!r} "
        f"within ROCR_VISIBLE_DEVICES={os.environ['ROCR_VISIBLE_DEVICES']!r}"
    )


def _check_gpu_visibility() -> dict[str, Any]:
    """Best-effort informational check of visible GPU count vs ``$TP`` (silent when rocm-smi is absent)."""
    # External multi-node: GPUs are on remote pods, not this sandbox.
    from hyperloom.inference_optimizer.multi_node._internal.external_state import external_service_url
    from hyperloom.orchestrator.actions.executors._multi_node_env import is_multi_node

    if is_multi_node() and external_service_url():
        print("Preflight: external multi-node mode; skipping local GPU visibility check (GPUs are on remote pods)")
        return {
            "status": "skipped",
            "skip_reason": "external_multi_node",
            "detail": {"visible": None, "tp_requested": None, "warn": None},
        }
    visible, skip_reason = _rocm_smi_visible_gpus()
    if visible is None:
        return {
            "status": "skipped",
            "skip_reason": skip_reason,
            "detail": {"visible": None, "tp_requested": None, "warn": None},
        }
    try:
        wanted = int(os.environ.get("TP", "1") or "1")
    except ValueError:
        wanted = 1
    if visible == 0:
        warning = "rocm-smi sees 0 GPUs; benchmark will fail"
        print(f"Preflight: WARNING — {warning}")
        return {
            "status": "warned",
            "skip_reason": None,
            "detail": {"visible": visible, "tp_requested": wanted, "warn": warning},
        }
    if wanted > visible:
        warning = f"TP={wanted} but rocm-smi sees {visible} GPU(s); sglang/vllm may fail to load weights"
        print(f"Preflight: WARNING — {warning}. Lower TP or adjust ROCR_VISIBLE_DEVICES.")
        return {
            "status": "warned",
            "skip_reason": None,
            "detail": {"visible": visible, "tp_requested": wanted, "warn": warning},
        }
    return {
        "status": "applied",
        "skip_reason": None,
        "detail": {"visible": visible, "tp_requested": wanted, "warn": None},
    }


def _check_shm_disk() -> dict[str, Any]:
    """Warn (not fail-fast) on tight ``/dev/shm`` (vLLM/NCCL IPC needs headroom)."""
    try:
        usage = shutil.disk_usage("/dev/shm")  # nosec B108 - mountpoint probe, not temp file creation.
    except (FileNotFoundError, OSError):
        return {
            "status": "skipped",
            "skip_reason": "shm_unavailable",
            "detail": {"shm_free_gib": None, "min_gib": 16},
        }
    free_gb = usage.free / (1024**3)
    if usage.free < _DEV_SHM_MIN_FREE_BYTES:
        total_gb = usage.total / (1024**3)
        print(
            f"Preflight: WARNING — /dev/shm has {free_gb:.1f} GiB free of "
            f"{total_gb:.1f} GiB total (< 16 GiB threshold). vLLM IPC + "
            f"NCCL shm segments may collide with stale entries; if the "
            f"first server launch hangs >5min, clear /dev/shm/{{vllm,nccl,cuda}}*"
        )
    return {
        "status": "warned" if usage.free < _DEV_SHM_MIN_FREE_BYTES else "applied",
        "skip_reason": None,
        "detail": {"shm_free_gib": round(free_gb, 1), "min_gib": 16},
    }


def _check_gfx_arch_resolvable(gpu_type: str | None = None) -> None:
    """Warn when the GPU architecture cannot be resolved for provenance."""
    if detect_gfx_arch(os.environ, gpu_type=gpu_type):
        return
    boards = "/".join(sorted(AMD_GPU_BOARDS))
    print(
        "Preflight: WARNING — GPU architecture could not be resolved; provenance "
        "will record gfx_arch as null, so an archived report will not say which "
        f"ISA produced its numbers. Pass --gpu-type ({boards}), "
        "set HYPERLOOM_GFX_ARCH, or put rocminfo (/opt/rocm/bin) on PATH."
    )


def _check_platform_tuning() -> dict[str, Any]:
    """Record host CPU tuning state and warn on settings that skew results."""
    plat = probe_cpu_platform()
    if plat is None:
        return {
            "status": "skipped",
            "skip_reason": "platform_probe_unavailable",
            "detail": {"smt": None, "governor": "unknown", "cpb": None},
        }

    print(
        f"Preflight: platform [{socket.gethostname()}] — SMT {plat.smt or '?'}, "
        f"{plat.nps or 'unknown'} ({plat.numa_nodes or '?'} NUMA nodes / "
        f"{plat.sockets or '?'} sockets), governor {plat.governor}"
    )

    if plat.governor not in ("performance", "unknown"):
        print(
            f"Preflight: WARNING — cpufreq governor is {plat.governor!r}, not 'performance'; "
            f"clock ramp can distort latency percentiles and roofline measurements"
        )
    if plat.boost == "off":
        print(
            "Preflight: WARNING — Core Performance Boost is disabled; CPU-side "
            "work (sampling, scheduling, tokenization) will run below rated clocks"
        )
    warned = plat.governor not in ("performance", "unknown") or plat.boost == "off"
    return {
        "status": "warned" if warned else "applied",
        "skip_reason": None,
        "detail": {
            "smt": plat.smt,
            "governor": plat.governor,
            "cpb": {"on": True, "off": False}.get(plat.boost),
        },
    }


_TRACELENS_REQUIRED_CLIS: tuple[str, ...] = ("TraceLens_generate_perf_report_from_config",)


def _tracelens_required_at_preflight(no_kernel: bool, enable_roofline: bool) -> bool:
    """Return whether the TraceLens CLI must be present at preflight (hard-fail)."""
    return not (no_kernel and not enable_roofline)


def _check_tracelens_cli() -> dict[str, Any]:
    """Hard-gate TraceLens CLI presence — abort before Coordinator starts (SKILL IR-2)."""
    missing = [name for name in _TRACELENS_REQUIRED_CLIS if shutil.which(name) is None]
    if not missing:
        return {
            "status": "applied",
            "skip_reason": None,
            "target": "TraceLens",
            "message": None,
        }
    session_dir = str(_workspace_root_resolve())
    print(
        f"ERROR: TraceLens CLI(s) not on PATH: {missing}. The pod-local "
        f"/opt/venv/bin/TraceLens_* console_scripts are installed by "
        f"src/hyperloom/inference_optimizer/assets/install.sh and do NOT persist "
        f"across pod restarts. SKILL IR-2 requires running install.sh "
        f"before every launch (carve-out applies only to --resume-from in "
        f"the same shell that earlier ran install.sh). Re-run:\n"
        f"  bash $REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh\n"
        f"  . {session_dir}/runtime/kernel-agent.env.sh\n"
        f"then retry `python -m hyperloom optimize`. Refusing to start.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _check_tracelens_root_exists() -> dict[str, Any]:
    """Hard-gate a set ``TRACELENS_ROOT`` that is missing, or a ``TraceLens@<sha>`` other than the installer's pin."""
    override = os.environ.get("TRACELENS_ROOT")
    pin = re.search(r'^TRACELENS_REF="([0-9a-f]{40})"$', (_ASSETS / "install_kernel_tools.sh").read_text(), re.M)
    name = Path(override or "").name
    stale = pin is not None and name.startswith("TraceLens@") and name != f"TraceLens@{pin.group(1)}"
    if not override or (Path(override).is_dir() and not stale):
        return {"status": "applied", "skip_reason": None, "target": "TRACELENS_ROOT"}
    problem = f"is an earlier install's checkout, not TraceLens@{pin.group(1)}" if stale else "does not exist"
    print(
        f"ERROR: TRACELENS_ROOT={override} {problem} (a stale shell, .env or install). Re-run src/hyperloom/"
        f"inference_optimizer/assets/install.sh and source $KERNEL_AGENT_ENV, or unset it. Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


_KERNEL_TUNING_CLIS: dict[str, str] = {
    "hipblaslt-bench": "offline hipBLASLt GEMM solution tuning",
    "ckProfiler": "the Composable Kernel GEMM/attention instance sweep",
}


def _check_kernel_tuning_clis() -> dict[str, Any]:
    """WARN-only presence check for the GEMM tuning CLIs the kernel phase's backends call."""
    missing = [name for name in _KERNEL_TUNING_CLIS if shutil.which(name) is None]
    for name in missing:
        print(
            f"Preflight: WARNING — {name} not on PATH; {_KERNEL_TUNING_CLIS[name]} is unavailable to the kernel "
            "phase. To use it, build it for this ROCm and put it on PATH."
        )
    return {
        "status": "warned" if missing else "applied",
        "skip_reason": None,
        "detail": {"missing": missing},
    }


def _check_agent_cli() -> None:
    """WARN when the CLI this deployment's specialists run cannot be resolved; launching them will fail."""
    backend = preferred_agent_backend()
    resolve = resolve_codex_executable if backend == AGENT_BACKEND_CODEX else resolve_claude_executable
    if not resolve():
        print(f"Preflight: WARNING — no {backend} CLI; specialists cannot start without it (assets/install.sh).")


def _emit_preflight_diagnostics(
    *,
    magpie_python: str,
    anthropic_base_url: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """One canonical, grep-friendly diagnostics block at the end of preflight."""
    from hyperloom.orchestrator.actions.executors._aiter_jit import probe_aiter_jit_cache as _probe_aiter_jit_cache
    from hyperloom.orchestrator.actions.executors._subprocess_kill import resolve_benchmark_timeouts
    from ..session.paths import asset_root

    probe = _probe_aiter_jit_cache()
    silence_timeout, hard_timeout = resolve_benchmark_timeouts()
    if probe["probe_status"] == "found":
        kind = "COLD" if probe["is_cold"] else "WARM"
        cache_line = f"{probe['kernel_count']} .so / {probe['size_mb']} MB ({kind}) at {probe['path']}"
    else:
        cache_line = f"<probe_status={probe['probe_status']}>"

    print("Preflight diagnostics:")
    print(f"  asset_root          = {asset_root()}")
    print(
        f"  session_dir         = {_session_dir_resolve()}  "
        f"({ENV_USER_DATA_PATH}="
        f"{os.environ.get(ENV_USER_DATA_PATH, '<unset>')}, "
        f"default={DEFAULT_SESSION_DIR})"
    )
    print(f"  magpie_python       = {magpie_python}")
    print(f"  INFERENCEX_PATH     = {os.environ.get('INFERENCEX_PATH', '<unset>')}")
    print(f"  aiter jit cache     = {cache_line}")
    print(f"  benchmark_timeout   = {hard_timeout}s")
    print(f"  benchmark_silence   = {silence_timeout}s")
    print(f"  ANTHROPIC_BASE_URL  = {anthropic_base_url or '<unset>'}")
    pr_enabled = bool(getattr(args, "pr_monitor_enabled", True))
    kb_degraded_reason = getattr(args, "kb_degraded_reason", None)
    kb_reason = kb_degraded_reason or "-"
    pr_reason = getattr(args, "pr_degraded_reason", None) or "-"
    kb_status = f"DEGRADED ({kb_reason})" if kb_degraded_reason else "OK"
    pr_status = "OK" if pr_enabled else f"DEGRADED ({pr_reason})"
    print(f"  kb_status           = {kb_status}")
    print(f"  pr_monitor_status   = {pr_status}")
    print(f"  kb_degraded_reason  = {kb_reason}")
    print(f"  pr_degraded_reason  = {pr_reason}")
    return {
        "status": "applied",
        "skip_reason": None,
        "message": None,
        "detail": {
            "asset_root": str(asset_root()),
            "session_dir": str(_session_dir_resolve()),
            "magpie_python": magpie_python,
            "inferencex_path": os.environ.get("INFERENCEX_PATH") or None,
            "aiter_jit_cache": dict(probe),
            "benchmark_timeout_sec": hard_timeout,
            "benchmark_silence_timeout_sec": silence_timeout,
            "anthropic_base_url": anthropic_base_url,
        },
    }


def _ensure_eval_concurrency_compat(magpie_path: str, inferencex_path: str) -> bool:
    """Scrub the fatal ``--concurrent-requests`` eval flag from the resolved trees."""
    from hyperloom.orchestrator.actions.executors._magpie_patcher import (
        ensure_eval_concurrency_compat,
    )

    ok = ensure_eval_concurrency_compat(magpie_path or None, inferencex_path or None)
    if not ok:
        print(
            "Preflight: WARNING — could not remove the redundant "
            "'--concurrent-requests' flag from a Magpie benchmark script "
            f"(MAGPIE_PATH={magpie_path or '<unset>'}, "
            f"INFERENCEX_PATH={inferencex_path}). RUN_EVAL=true baselines will "
            'abort with "Unknown parameter: --concurrent-requests"; eval '
            "concurrency must flow via EVAL_CONCURRENT_REQUESTS/CONC instead."
        )
    return ok


def _report_inferencex_patch_anchors(inferencex_path: str) -> bool:
    """Report whether Hyperloom's InferenceX patches can still find their place."""
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
        verify_patch_anchors,
    )

    statuses = verify_patch_anchors(inferencex_path or None)
    broken = [status for status in statuses if not status.ok]
    if not broken:
        return True
    print(
        f"Preflight: WARNING — {len(broken)} of {len(statuses)} InferenceX patch "
        "anchors no longer match; those patches will not be applied:"
    )
    for status in broken:
        print(f"  {status.describe()}")
    print(
        "  Hyperloom patches InferenceX by matching exact upstream text, so this "
        "means the checkout drifted from the pinned revision. Re-anchor the "
        "patches in _inferencex_patcher.py or pin INFERENCEX_REF back to a "
        "revision they match. An accuracy-gate run aborts at launch on the ones "
        "that would void its score."
    )
    return False


def _ensure_client_trust_compat(magpie_path: str) -> bool:
    """Assert the custom-tokenizer trust patch on the resolved Magpie tree."""
    from hyperloom.orchestrator.actions.executors._multi_node_env import is_multi_node

    if not is_multi_node():
        return True
    from hyperloom.orchestrator.actions.executors._magpie_patcher import (
        ensure_client_trust_compat,
    )

    ok = ensure_client_trust_compat(magpie_path or None)
    if not ok:
        print(
            "Preflight: WARNING — could not apply the custom-tokenizer trust "
            "patch to a Magpie SGLang script "
            f"(MAGPIE_PATH={magpie_path or '<unset>'}). MAGPIE_TRUST_REMOTE_CODE=1 "
            "will not reach benchmark_serving.py, so a model shipping custom "
            "tokenizer code will fail to load its tokenizer before issuing any "
            "request. Models without remote code are unaffected."
        )
    return ok


def _begin_install_event(args: argparse.Namespace | None) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "install",
        "kind": "install",
        "status": "succeeded",
        "start_time": now_iso(timespec="seconds"),
        "end_time": "",
        "ext": {
            "run_kind": "resume" if bool(getattr(args, "resume_from", None)) else "fresh",
            "hard_fail_step_id": None,
            "runtime_snapshot": {},
            "steps": [],
        },
    }
    from ..session.sbd_v6 import set_pending_install_event

    set_pending_install_event(args, event)
    return event


@dataclass(frozen=True)
class InstallStep:
    """One install-event step: its id and category, the status it ended in, and why it skipped or what it said."""

    step_id: str
    category: str
    status: str
    skip_reason: str | None = None
    message: str | None = None


def _record_install_step(event: dict[str, Any], step: InstallStep, **fields: Any) -> None:
    row: dict[str, Any] = {
        "step_id": step.step_id,
        "category": step.category,
        "status": step.status,
        "skip_reason": step.skip_reason,
    }
    if step.message is not None:
        row["message"] = step.message
    row.update(fields)
    event["ext"]["steps"].append(row)


def _fail_install_step(
    event: dict[str, Any],
    *,
    step_id: str,
    category: str,
    exc: BaseException,
) -> None:
    failure_fields: dict[str, Any] = {}
    if isinstance(exc, SystemExit):
        failure_fields["detail"] = {"exit_code": exc.code}
    _record_install_step(
        event,
        InstallStep(step_id, category, "failed", message=str(exc) or type(exc).__name__),
        error_class=type(exc).__name__,
        **failure_fields,
    )
    event["status"] = "failed"
    event["end_time"] = now_iso(timespec="seconds")
    event["ext"]["hard_fail_step_id"] = step_id


def _mark_pending_install_event_failed(
    args: argparse.Namespace | None,
    exc: BaseException,
) -> None:
    """Mark an unwrapped preflight exception on the pending install event."""
    from ..session.sbd_v6 import pending_install_event

    event = pending_install_event(args)
    if event is None:
        event = _begin_install_event(args)
    if str(event.get("status") or "") != "failed":
        _fail_install_step(
            event,
            step_id="unhandled_preflight",
            category="check",
            exc=exc,
        )


def _run_install_step(
    event: dict[str, Any],
    *,
    step_id: str,
    category: str,
    action: Callable[[], Any],
    **success_fields: Any,
) -> Any:
    try:
        result = action()
    except BaseException as exc:
        try:
            _fail_install_step(event, step_id=step_id, category=category, exc=exc)
        except Exception:
            log.warning("failed to record SBD V6 install-step failure", exc_info=True)
        raise
    outcome = dict(result) if isinstance(result, dict) else {}
    status = str(outcome.pop("status", "applied") or "applied")
    skip_reason = outcome.pop("skip_reason", None)
    message = outcome.pop("message", None)
    fields = {**success_fields, **outcome}
    _record_install_step(
        event,
        InstallStep(step_id, category, status, skip_reason=skip_reason, message=message),
        **fields,
    )
    return result


def _resolved_provider_mode(resolved_urls: tuple[str, str]) -> str:
    anthropic_url, openai_url = resolved_urls
    if anthropic_url and openai_url:
        return "mixed"
    if anthropic_url:
        return "anthropic"
    return "openai"


@dataclass(frozen=True)
class PreflightRuntime:
    """The benchmark runtime preflight settled on: backend, interpreters, InferenceX checkout and LLM endpoints."""

    benchmark_backend: str
    benchmark_python: str
    magpie_python: str
    inferencex_path: str
    resolved_urls: tuple[str, str]


def _finish_install_event(event: dict[str, Any], *, args: argparse.Namespace, runtime: PreflightRuntime) -> None:
    no_kernel = bool(getattr(args, "no_kernel", False))
    enable_roofline = bool(getattr(args, "enable_roofline", True))
    tracelens_required = _tracelens_required_at_preflight(no_kernel, enable_roofline)
    event["ext"]["runtime_snapshot"] = {
        "benchmark_backend": runtime.benchmark_backend,
        "benchmark_interpreter": runtime.benchmark_python,
        "magpie_python": runtime.magpie_python,
        "inferencex_path": runtime.inferencex_path,
        "magpie_path": os.environ.get("MAGPIE_PATH") or None,
        "tracelens_required": tracelens_required,
        "provider_mode": _resolved_provider_mode(runtime.resolved_urls),
    }
    statuses = {str(step.get("status") or "") for step in event["ext"]["steps"]}
    if "failed" in statuses:
        event["status"] = "failed"
    elif "warned" in statuses:
        event["status"] = "degraded"
    elif statuses and statuses == {"skipped"}:
        event["status"] = "skipped"
    else:
        event["status"] = "succeeded"
    event["end_time"] = now_iso(timespec="seconds")


def _persist_install_event(args: argparse.Namespace | None, session_dir: Path) -> None:
    """Persist the pre-session install trace without changing launch behavior."""
    from ..session.sbd_v6 import persist_pending_install_event, record_write_warning

    try:
        path = persist_pending_install_event(args, session_dir)
    except (OSError, ValueError, TypeError) as exc:
        log.warning("failed to persist SBD V6 install event", exc_info=True)
        if not record_write_warning(session_dir, component="install.event", exc=exc):
            log.debug("failed to persist SBD V6 install-event write warning", exc_info=True)
        return
    if path is None:
        exc = RuntimeError("pending install event is unavailable")
        log.warning("failed to persist SBD V6 install event: %s", exc)
        if not record_write_warning(session_dir, component="install.event", exc=exc):
            log.debug("failed to persist SBD V6 install-event write warning", exc_info=exc)


def _export_llm_endpoints() -> tuple[str, str]:
    """Resolve and export the Anthropic and OpenAI base URLs; at least one is non-empty after credential validation."""
    anthropic_url, openai_url = _resolve_llm_endpoints()
    # A subscription token needs no endpoint: the Claude CLI knows where to go.
    skip_anthropic_export = (
        not os.environ.get("ANTHROPIC_BASE_URL", "").strip()
        and not anthropic_synthesizable_key()
        and bool(os.environ.get(CLAUDE_OAUTH_TOKEN_ENV, "").strip())
    )
    for var, want in (("ANTHROPIC_BASE_URL", anthropic_url), ("OPENAI_BASE_URL", openai_url)):
        if not want:
            continue
        if var == "ANTHROPIC_BASE_URL" and skip_anthropic_export:
            continue
        prev = os.environ.get(var, "")
        if prev != want:
            os.environ[var] = want
            print(f"Preflight: {var} {prev or '<unset>'} -> {want} (resolved endpoint)")
    # Claude CLI primary key: Anthropic-side credentials only, and only the synthesizable subset so a subscription
    # token never lands in config.json.
    _reset_claude_config_to_upstream(anthropic_synthesizable_key(), anthropic_url)
    if anthropic_url and not openai_url and not os.environ.get("GEAK_CLAUDE_MODEL"):
        geak_claude_model = os.environ.get("CLAUDE_MODEL", "").strip() or DEFAULT_CLAUDE_MODEL
        os.environ["GEAK_CLAUDE_MODEL"] = geak_claude_model
        print(f"Preflight: GEAK_CLAUDE_MODEL <unset> -> {geak_claude_model} (GEAKv4 Claude workflow)")
    _export_gateway_alias(openai_url)
    _sync_geak_config_endpoint()
    return anthropic_url, openai_url


def _export_gateway_alias(gateway_url: str) -> None:
    """Point ``LLM_API_BASE`` at the gateway.

    It addresses an OpenAI-protocol endpoint, so it defaults to the resolved
    OpenAI-side URL and stays unset when that side is not configured.
    """
    if not gateway_url:
        return
    for alias in ("LLM_API_BASE",):
        current = os.environ.get(alias, "").strip()
        if current and current != gateway_url:
            # A genuine operator override is preserved, but a leftover install-time proxy is unreachable and
            # force-rewritten.
            if _is_stale_proxy_url(current):
                os.environ[alias] = gateway_url
                print(
                    f"Preflight: {alias} {current} -> {gateway_url} "
                    "(stale install-time proxy; force-rewritten to gateway)"
                )
                continue
            print(f"Preflight: {alias} kept at {current} (operator override; not forced to gateway)")
            continue
        if os.environ.get(alias) != gateway_url:
            prev = os.environ.get(alias, "")
            os.environ[alias] = gateway_url
            print(f"Preflight: {alias} {prev or '<unset>'} -> {gateway_url} (direct to gateway)")


def _sync_geak_config_endpoint() -> None:
    """A supplied GEAK_CONFIG yaml may carry its own endpoint; sync it so an override reaches GEAK."""
    geak_cfg = os.environ.get("GEAK_CONFIG", "").strip()
    geak_url = os.environ.get("GEAK_BASE_URL", "").strip()
    if geak_cfg and geak_url and _sync_geak_config_base_url(geak_cfg, geak_url):
        print(f"Preflight: synced GEAK config base_url -> {geak_url} ({geak_cfg})")


def _ensure_magpie(
    install_event: Any,
    *,
    magpie_python: str,
    benchmark_backend: str,
    magpie_active: bool,
    pip_extra: list[str],
) -> None:
    """Install Magpie and resolve MAGPIE_PATH, unless another benchmark backend is active."""
    magpie_installed = False
    magpie_spec: str | None = None
    try:
        if not magpie_active:
            print(f"Preflight: benchmark backend is {benchmark_backend!r}; skipping Magpie install/import")
        else:
            magpie_ref = os.environ.get("MAGPIE_REF", "6a1ee26f9053363d3750b43e3d5afb5d1c4ce84e")
            checker = _ASSETS / "check_magpie.py"
            check = subprocess.run([magpie_python, str(checker), magpie_ref], capture_output=True)
            if check.returncode != 0 and not os.environ.get("MAGPIE_PATH"):
                magpie_repo = os.environ.get("MAGPIE_REPO", "https://github.com/AMD-AGI/Magpie.git")
                magpie_spec = os.environ.get("MAGPIE_PACKAGE_SPEC", f"magpie-eval @ git+{magpie_repo}@{magpie_ref}")
                print(f"Preflight: Magpie revision is missing; installing {magpie_spec} ...")
                subprocess.run(
                    [magpie_python, "-m", "pip", "install", "--quiet", *pip_extra, magpie_spec],
                    check=True,
                )
                magpie_installed = True
        if magpie_active and not os.environ.get("MAGPIE_PATH", "").strip():
            magpie_root = subprocess.run(
                [
                    magpie_python,
                    "-c",
                    "from pathlib import Path; import Magpie; print(Path(Magpie.__file__).resolve().parent.parent)",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if magpie_root:
                os.environ["MAGPIE_PATH"] = magpie_root
                print(f"Preflight: MAGPIE_PATH resolved from installed package: {magpie_root}")
    except BaseException as exc:
        _fail_install_step(install_event, step_id="ensure_magpie", category="install", exc=exc)
        raise
    _record_install_step(
        install_event,
        InstallStep(
            "ensure_magpie",
            "install",
            "skipped" if not magpie_active else "applied" if magpie_installed else "already_present",
            skip_reason=None if magpie_active else "benchmark_backend_not_magpie",
            message=(
                f"installed {magpie_spec}"
                if magpie_installed
                else f"benchmark backend is {benchmark_backend!r}"
                if not magpie_active
                else None
            ),
        ),
        target="magpie-eval",
        interpreter=magpie_python,
    )


def _fail_inferencex(install_event: Any, message: str) -> NoReturn:
    """Record a fatal InferenceX problem and stop: baseline cannot run without a usable checkout."""
    print(message, file=sys.stderr)
    exc = SystemExit(2)
    _fail_install_step(install_event, step_id=_CLONE_INFERENCEX_STEP, category="install", exc=exc)
    raise exc


def _ensure_inferencex(install_event: Any) -> str:
    """The InferenceX checkout this run uses, cloning one when no writable checkout at the pin exists.

    Required for the GSM8K accuracy eval; lm-eval deps auto-install at runtime
    via ``benchmark_lib.sh``.
    """
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip() or inferencex_checkout.detect()
    inferencex_cloned = False
    if not (inferencex_path and inferencex_checkout.checkout_ok(inferencex_path)):
        from ..session.paths import deps_cache_root as _open_source_default

        ref = os.environ.get("INFERENCEX_REF") or inferencex_checkout.DEFAULT_REF
        # Per-revision dir, matching install.sh: a shared name is what allowed a pre-bump clone to be reused forever.
        dest = _open_source_default() / inferencex_checkout.destination_name(ref)
        print(f"Preflight: no InferenceX checkout at {ref[:12]}; cloning into {dest} ...")
        inferencex_path = inferencex_checkout.clone(dest)
        inferencex_cloned = bool(inferencex_path)
        if not (inferencex_path and inferencex_checkout.checkout_ok(inferencex_path)):
            _fail_inferencex(
                install_event,
                "Preflight: ERROR — InferenceX checkout missing and clone "
                "failed. baseline cannot run without it. Set INFERENCEX_PATH "
                "to a writable checkout or re-run "
                "src/hyperloom/inference_optimizer/assets/install.sh.",
            )
    # Magpie stages benchmark scripts into this tree, so a non-writable one fails the run before server boot.
    if not os.access(inferencex_path, os.W_OK):
        _fail_inferencex(
            install_event,
            f"Preflight: ERROR — INFERENCEX_PATH={inferencex_path} is not "
            f"writable. Magpie stages benchmark scripts into it and will "
            f"fail with [Errno 30] Read-only file system. Point "
            f"INFERENCEX_PATH at a writable checkout (unset it to let "
            f"Hyperloom clone a fresh one).",
        )
    # Always overwrite (not setdefault): a stale/broken INFERENCEX_PATH must not survive into the child env.
    os.environ["INFERENCEX_PATH"] = inferencex_path
    # A round cd's into this checkout and bash reads the benchmark script off it for the whole run, so a revocable
    # mount that flaps fails the round on an exit code the measurement had nothing to do with. Recording it here is
    # what tells that apart from a variant that genuinely cannot serve.
    inferencex_network_fs = is_network_fs(inferencex_path)
    if inferencex_network_fs:
        print(
            f"Preflight: WARNING — INFERENCEX_PATH={inferencex_path} is on a network filesystem. A mount flap "
            f"mid-round exits the benchmark non-zero after it has already run. A round that served its whole "
            f"protocol is kept, but one the flap cut short is recorded as "
            f"magpie_nonzero_after_valid_measurement. Point INFERENCEX_PATH at local disk, or unset it and put "
            f"HYPERLOOM_CACHE_DIR on local disk.",
            file=sys.stderr,
        )
    _record_install_step(
        install_event,
        InstallStep(_CLONE_INFERENCEX_STEP, "install", "applied" if inferencex_cloned else "already_present"),
        target="InferenceX",
        version_after=inferencex_checkout.head_sha(inferencex_path) or None,
        detail={
            "ref": os.environ.get("INFERENCEX_REF") or inferencex_checkout.DEFAULT_REF,
            "dest": inferencex_path,
            "writable": os.access(inferencex_path, os.W_OK),
            "network_fs": inferencex_network_fs,
            "exit_code": 0,
        },
    )
    return inferencex_path


def _patch_magpie_compat(install_event: Any, inferencex_path: str, *, magpie_active: bool) -> None:
    """Apply the Magpie script patches install.sh would have.

    Preflight installs Magpie and clones InferenceX itself, entirely outside
    install.sh, so those patches are applied here as well. Trust patch first,
    mirroring install.sh: the eval-concurrency strip removes the very
    ``run_eval ... --concurrent-requests`` line the legacy MI300X trust patcher
    matches on, so the reverse order would leave a tree permanently unpatchable
    by that path.
    """
    try:
        if magpie_active:
            trust_ok = _ensure_client_trust_compat(os.environ.get("MAGPIE_PATH", ""))
            concurrency_ok = _ensure_eval_concurrency_compat(os.environ.get("MAGPIE_PATH", ""), inferencex_path)
        else:
            trust_ok = True
            concurrency_ok = True
        anchors_ok = _report_inferencex_patch_anchors(inferencex_path)
    except BaseException as exc:
        _fail_install_step(install_event, step_id="patch_magpie_eval_concurrency", category="patch", exc=exc)
        raise
    patch_ok = trust_ok and concurrency_ok and anchors_ok
    _record_install_step(
        install_event,
        InstallStep(
            "patch_magpie_eval_concurrency",
            "patch",
            "skipped" if not magpie_active else "applied" if patch_ok else "warned",
            skip_reason=None if magpie_active else "benchmark_backend_not_magpie",
        ),
        detail={
            "client_trust_compatible": trust_ok,
            "eval_concurrency_compatible": concurrency_ok,
            "inferencex_patch_anchors_ok": anchors_ok,
        },
    )


def _prepare_inferencex(install_event: Any, *, magpie_active: bool) -> str:
    """Prepare the synthetic runtime; native AgentX's checkout belongs to Magpie."""
    from hyperloom.common.agentx_workload import is_mlperf_backend
    from hyperloom.common.perf_metric import agentx_enabled

    if agentx_enabled() and not is_mlperf_backend():
        _record_install_step(
            install_event,
            InstallStep(_CLONE_INFERENCEX_STEP, "install", "skipped", skip_reason="magpie_manages_native_agentx"),
        )
        return ""
    inferencex_path = _ensure_inferencex(install_event)
    _patch_magpie_compat(install_event, inferencex_path, magpie_active=magpie_active)
    return inferencex_path


def _check_kernel_and_tracelens_clis(install_event: Any, args: argparse.Namespace) -> None:
    """Check the kernel-tuning and TraceLens CLIs before the Coordinator starts.

    A hard failure here catches launchers that skip install.sh, rather than
    letting a missing CLI surface mid-run -- or, for a stale TRACELENS_ROOT,
    ten hours later in trace_analyze. Both are skipped when the run asks for
    neither kernel work nor roofline.
    """
    no_kernel = getattr(args, "no_kernel", False)
    enable_roofline = getattr(args, "enable_roofline", True)
    if no_kernel:
        _record_install_step(
            install_event,
            InstallStep("check_kernel_tuning_clis", "check", "skipped", skip_reason="no_kernel"),
        )
    else:
        _run_install_step(
            install_event,
            step_id="check_kernel_tuning_clis",
            category="check",
            action=_check_kernel_tuning_clis,
        )
    if _tracelens_required_at_preflight(no_kernel, enable_roofline):
        _run_install_step(install_event, step_id="check_tracelens_cli", category="check", action=_check_tracelens_cli)
        _run_install_step(
            install_event,
            step_id="check_tracelens_root",
            category="check",
            action=_check_tracelens_root_exists,
        )
        return
    missing = [name for name in _TRACELENS_REQUIRED_CLIS if shutil.which(name) is None]
    if missing:
        print(
            f"Preflight: WARNING — TraceLens CLI(s) not on PATH: {missing} (skipped; --no-kernel + roofline disabled)"
        )
    _record_install_step(
        install_event,
        InstallStep(
            "check_tracelens_cli",
            "check",
            "skipped",
            skip_reason="no_kernel_and_roofline_disabled",
            message=f"missing CLIs: {missing}" if missing else None,
        ),
        target="TraceLens",
    )
    _record_install_step(
        install_event,
        InstallStep("check_tracelens_root", "check", "skipped", skip_reason="tracelens_not_required"),
        target="TRACELENS_ROOT",
    )


def _validate_explicit_server_limits(args: argparse.Namespace) -> None:
    effective = argparse.Namespace(**vars(args))
    state = _resumed_session_state(args) or {}
    pins = parse_operator_extra_env(effective) or state.get("operator_extra_env", {})
    if state:
        effective.server_args = str(getattr(args, "server_args", "") or "").strip() or state.get(
            "operator_server_args", ""
        )
        if getattr(args, "max_model_len", None) is None:
            effective.max_model_len = (
                pins["MAX_MODEL_LEN"]
                if "MAX_MODEL_LEN" in pins
                else state.get("max_model_len") or os.environ.get("MAX_MODEL_LEN", "").strip() or None
            )
    model = str(state.get("model_path") or getattr(args, "model", None) or os.environ.get("MODEL_PATH", ""))
    validate_server_args(effective, resolve_serving_model_path(model, model_base=pins.get("HL_MODEL_BASE")))


def _preflight(args: argparse.Namespace) -> tuple[str, str]:
    """Auto-install missing runtime deps and export auth aliases."""
    _pin_resumed_session_args(args)
    install_event = _begin_install_event(args)
    _run_install_step(
        install_event,
        step_id="load_dotenv",
        category="normalize",
        action=_load_dotenv_fallback,
    )
    # ``.env`` is operator configuration, so both the single-provider intent and the restore baseline are taken after
    # it loads.
    provider_mode = _provider_only_mode()
    provider_snapshot = {key: os.environ.get(key) for key in (*_PROVIDER_FALLBACK_KEYS, *_ANTHROPIC_FALLBACK_KEYS)}
    _run_install_step(
        install_event,
        step_id="load_kernel_agent_env",
        category="normalize",
        action=_load_kernel_agent_env_fallback,
    )
    if framework_arg := getattr(args, "framework", None):
        os.environ["FRAMEWORK"] = framework_arg.strip().lower()
    _run_install_step(
        install_event,
        step_id="validate_server_args",
        category="check",
        action=lambda: _validate_explicit_server_limits(args),
    )
    _derive_runtime_paths()
    _restore_provider_only_mode(provider_mode, provider_snapshot)
    _run_install_step(
        install_event,
        step_id="normalize_legacy_deepseek_env",
        category="normalize",
        action=_normalize_legacy_deepseek_env,
    )
    # After the legacy normalization, which can create the credentials a header references.
    _expand_custom_header_env_refs()

    # Fail fast on missing credentials after the fallback loaders.
    _run_install_step(
        install_event,
        step_id="validate_credentials",
        category="check",
        action=_validate_credentials,
        detail={"exit_code": 0},
    )

    # --- Resolve install interpreters --- Resolve the ACTIVE benchmark backend first so a bypass-only environment (no
    # Magpie / no /opt/venv) never routes installs through Magpie's interpreter.
    from hyperloom.orchestrator.actions.executors.benchmark_backend import (
        resolve_backend_name as _resolve_active_backend_name,
        resolve_benchmark_interpreter as _resolve_benchmark_interpreter,
    )

    benchmark_backend = _resolve_active_backend_name()
    _magpie_backend_active = benchmark_backend == "magpie"
    # Interpreter used for benchmark-runtime installs (Ray).
    benchmark_python = _resolve_benchmark_interpreter()

    # Outside a venv, add --break-system-packages so pip installs on bare-metal Debian/Ubuntu.
    pip_extra: list[str] = []
    if not (hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)):
        pip_extra = ["--break-system-packages"]

    # --- Python SDK auto-install (claude-agent-sdk / openai / httpx) --- Must precede Coordinator import
    # (ClaudeBackend lazy-imports the SDK).
    _run_install_step(
        install_event,
        step_id="ensure_python_sdks",
        category="install",
        action=lambda: _ensure_python_sdks(sys.executable, pip_extra),
    )

    # --- Resolve Anthropic + OpenAI base URLs (split entrypoints) ---
    resolved_urls = _export_llm_endpoints()

    # --- ROCm env hygiene + GPU/shm sanity (defensive WARN-only) ---
    _normalize_hip_visible_devices()
    _run_install_step(
        install_event,
        step_id="check_gpu_visibility",
        category="check",
        action=_check_gpu_visibility,
    )
    _run_install_step(
        install_event,
        step_id="check_shm_disk",
        category="check",
        action=_check_shm_disk,
    )
    _run_install_step(
        install_event,
        step_id="check_platform_tuning",
        category="check",
        action=_check_platform_tuning,
    )

    # --- Runtime dep install --- 1.
    _run_install_step(
        install_event,
        step_id="ensure_ray",
        category="install",
        action=lambda: _ensure_ray(benchmark_python, pip_extra),
    )

    # 1b.
    _run_install_step(
        install_event,
        step_id="ensure_bench_serving_deps",
        category="install",
        action=lambda: _ensure_bench_serving_deps(benchmark_python, pip_extra),
    )

    # 1c. lm_eval — GSM8K accuracy gate, multi-node only (the helper gates itself).
    _run_install_step(
        install_event,
        step_id="ensure_lm_eval",
        category="install",
        action=lambda: _ensure_lm_eval_dep(
            benchmark_python,
            pip_extra,
        ),
    )

    # 1d.
    _run_install_step(
        install_event,
        step_id="check_serving_framework",
        category="check",
        action=lambda: _check_serving_framework(args, benchmark_python),
    )

    # 2.
    magpie_python = benchmark_python
    _ensure_magpie(
        install_event,
        magpie_python=magpie_python,
        benchmark_backend=benchmark_backend,
        magpie_active=_magpie_backend_active,
        pip_extra=pip_extra,
    )

    inferencex_path = _prepare_inferencex(install_event, magpie_active=_magpie_backend_active)

    _check_agent_cli()

    _check_kernel_and_tracelens_clis(install_event, args)

    # --- IR-3: PR Monitor reachability probe (soft degrade) ---
    _run_install_step(
        install_event,
        step_id="ir3_pr_monitor_probe",
        category="degrade",
        action=lambda: _run_ir3_preflight(args),
    )

    # --- Single canonical diagnostics block ---
    _run_install_step(
        install_event,
        step_id="diagnostics_snapshot",
        category="diagnostic",
        action=lambda: _emit_preflight_diagnostics(
            magpie_python=magpie_python,
            anthropic_base_url=resolved_urls[0],
            args=args,
        ),
    )

    _finish_install_event(
        install_event,
        args=args,
        runtime=PreflightRuntime(benchmark_backend, benchmark_python, magpie_python, inferencex_path, resolved_urls),
    )

    return resolved_urls


def _run_ir3_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """IR-3 — PR Monitor reachability probe (soft degrade); never raises/exits."""
    explicit_kb = bool(getattr(args, "degraded_kb", False))
    explicit_pr = bool(getattr(args, "degraded_pr", False))

    args.kb_degraded_reason = "explicit_flag" if explicit_kb else None
    args.pr_monitor_enabled = not explicit_pr
    args.pr_degraded_reason = "explicit_flag" if explicit_pr else None

    if explicit_kb and explicit_pr:
        return {
            "status": "skipped",
            "skip_reason": "explicit_flag",
            "detail": {
                "pr_monitor": {"enabled": False, "reason": "explicit_flag"},
                "marker": None,
            },
        }

    user_data = _workspace_root_resolve()
    marker_path = user_data / "runtime" / "recipe_kb" / ".kb_preflight.json"
    script = _ASSETS / "preflight_kb.sh"
    env = os.environ.copy()
    resolved_kb_store_url = kb_store_url(env=env)
    if resolved_kb_store_url:
        # Local mode may derive the default without exporting KB_STORE_URL.
        env["KB_STORE_URL"] = resolved_kb_store_url
    if explicit_pr:
        env["SKIP_PR_PROBE"] = "1"

    try:
        subprocess.run(
            ["bash", str(script)],
            env=env,
            check=False,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("IR-3 preflight script error: %s", exc)
        marker: dict[str, Any] = {
            "kb_reachable": False,
            "pr_reachable": False,
            "kb_skipped": True,
            "pr_skipped": explicit_pr,
        }
    else:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("IR-3 marker unreadable: %s", exc)
            marker = {
                "kb_reachable": False,
                "pr_reachable": False,
                "kb_skipped": True,
                "pr_skipped": explicit_pr,
            }

    if not explicit_pr and not marker.get("pr_reachable", False) and not marker.get("pr_skipped", False):
        args.pr_monitor_enabled = False
        args.pr_degraded_reason = "ir3_auto"
    enabled = bool(args.pr_monitor_enabled)
    status = "skipped" if explicit_pr else "applied" if enabled else "warned"
    event_reason = "explicit_flag" if explicit_pr else "ir3_unreachable" if not enabled else None
    return {
        "status": status,
        "skip_reason": event_reason,
        "detail": {
            "pr_monitor": {
                "enabled": enabled,
                "reason": event_reason,
            },
            "marker": str(marker_path),
        },
    }
