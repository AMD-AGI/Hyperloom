"""CLI entry

Usage::

    inference_optimizer optimize \\
        --model /wekafs/models/<your-model> \\
        --target-gain 10 \\
        --max-hours 2

    # or via env (matches the rest of the pipeline / Dockerfile convention):
    export MODEL_PATH=/wekafs/models/<your-model>
    inference_optimizer optimize --target-gain 10 --max-hours 2

Single subcommand for now (``optimize``). Wires Claude+Codex backends,
registers all available action_executors, builds the requested objective,
and starts ``Coordinator.run()`` until target / time / SIGTERM.

Env vars consumed (besides the standard backend creds):

  MODEL_PATH                                   — required if --model not passed;
                                                 also exported back to subprocess
                                                 env so Magpie YAMLs get the
                                                 correct model path injected
                                                 instead of the YAML's hardcoded
                                                 fallback.
  OPENAI_BASE_URL + SAFE_API_KEY — canonical LiteLLM endpoint; compatibility aliases are exported for Claude/OOB/GEAK
  ROCR_VISIBLE_DEVICES                         — pin the GPU
  CLAUDE_MODEL                                 — default claude-opus-4-7
  CODEX_MODEL                                  — default gpt-5.4
  USER_DATA_PATH                               — override session dir
                                                 (default: /workspace/hyperloom).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cli_executors import (  # noqa: F401 - re-exported for callers/tests
    _NOOP_KINDS_KERNEL_ONLY,
    _REAL_EXECUTORS_FULL,
    _build_dynamic_action_executor,
    _build_specialist_executor,
    _noop_prep,
    _register_executors,
)
from .cli_kb import (  # noqa: F401 - re-exported for callers/tests
    _bootstrap_cortex_kb,
    _bootstrap_knowledge_plane,
    _build_recipe_kb_dispatcher,
    _resolve_local_kb_root,
)
from .cli_backends import (  # noqa: F401 - re-exported for callers/tests
    _MULTI_NODE_WORKLOAD_UID_ENV_KEYS,
    _build_backends,
    _build_proposal_scorer,
    _build_robustness_options,
    _robustness_server_configured,
)
from .orchestrator.backends import ClaudeBackend
from .manifest import load_manifest, write_manifest
from .orchestrator.action_registry import ActionRegistry
from .orchestrator.coordinator import Coordinator
from .orchestrator.proposal_scorer import DEFAULT_SCORER_MODELS
from .orchestrator.framework_paths import resolve_source_file_allowlist
from .orchestrator.objective import Objective, build_objective
from .orchestrator.shared_state import SharedState
from .orchestrator.system_prompts.prompt_builder import (
    build_orchestration_prompt,
    default_enabled_actions,
)
from .paths import (
    DEFAULT_SESSION_DIR,
    ENV_USER_DATA_PATH,
    _SESSION_SKELETON,
    asset_system_prompts_dir,
    make_session_dir,
    mn_profile_trace_root,
    session_dir as _session_dir_resolve,
    workspace_root as _workspace_root_resolve,
)
from .session_paths import (
    agent_prompt_snapshot,
)


log = logging.getLogger("inference_optimizer.cli")


class _RetiredFlag(argparse.Action):
    """Argparse action that hard-fails on a retired CLI flag.

    Reaching for the old flag spelling should produce an immediate,
    explicit error (``parser.error`` exits 2) with a one-line
    migration hint — silent aliases would mask the behaviour change
    when the underlying semantics differ.
    """

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        *,
        hint: str,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("nargs", 0)
        kwargs.setdefault("default", argparse.SUPPRESS)
        kwargs.setdefault("help", argparse.SUPPRESS)
        self._hint = hint
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        parser.error(f"{option_string} was removed. {self._hint}")


def _orchestration_rules_fragment_path() -> Path:
    """Path to the rules-only fragment consumed by ``prompt_builder``.

    ``orchestration.md`` is a small "rules + output protocol" fragment;
    the full system prompt is composed at runtime from
    :class:`ActionMetadata` and run-level parameters by
    :func:`build_orchestration_prompt`. Kernel-vs-no-kernel is a builder
    parameter, not a separate file.
    """
    return asset_system_prompts_dir() / "orchestration.md"


def _objective_summary_for_prompt(objective: Objective) -> tuple[str, float | str | None]:
    """Return ``(kind, value)`` strings consumed by the prompt builder."""
    kind = objective.kind()
    value: float | str | None = None
    if hasattr(objective, "target_gain_pct"):
        value = float(getattr(objective, "target_gain_pct"))
    elif hasattr(objective, "target_tput_per_gpu"):
        value = float(getattr(objective, "target_tput_per_gpu"))
    elif hasattr(objective, "baseline_dir"):
        value = str(getattr(objective, "baseline_dir"))
    return kind, value


def _build_orchestration_prompt(
    *,
    no_kernel: bool,
    framework: str,
    objective: Objective,
    max_minutes: int,
    action_registry: ActionRegistry | None = None,
) -> str:
    """Compose the Orchestration system prompt for this run.

    The legacy ``_load_orchestration_prompt(no_kernel)`` returned a
    hand-maintained markdown file; this replacement assembles the prompt
    from typed inputs so kernel-enabled / no-kernel split is just a
    parameter and every enabled action carries a 1-line description.

    Callers may still pass ``--orch-prompt`` to fully override the result.
    """
    registry = action_registry or ActionRegistry().load()
    enabled = default_enabled_actions(no_kernel=no_kernel)
    kind, value = _objective_summary_for_prompt(objective)
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=enabled,
        framework=framework,
        kernel_enabled=not no_kernel,
        objective_kind=kind,
        objective_value=value,
        max_minutes=int(max_minutes),
        rules_fragment_path=_orchestration_rules_fragment_path(),
        framework_source_roots=resolve_source_file_allowlist(),
    )


def _load_critic_prompt() -> str:
    """Return the Critic system prompt sourced from ``system_prompts/critic.md``."""
    return (asset_system_prompts_dir() / "critic.md").read_text(encoding="utf-8")


_DEFAULT_KERNEL_PROMPT = (
    "You are the Kernel agent — responder-only. You receive `request`\n"
    "events from Orchestration in your inbox.\n\n"
    "For every un-answered request, emit ONE `response` intent in reply.\n"
    "Schema:\n"
    "  intent_type: response\n"
    "  payload: {\n"
    "    in_reply_to: <request msg_id>,\n"
    "    kind:        '<request.kind>_done',\n"
    "    status:      'ok' | 'failed' | 'needs_review',\n"
    "    result:      { /* whatever the request asked for */ }\n"
    "  }\n\n"
    "Native-only rule: run_optimization must refuse runtime-generated\n"
    "torch.compile/Inductor/Triton cache kernels. Only reusable framework\n"
    "sources under stable repos (aiter/sglang/vllm source trees) are valid\n"
    "kernel-opt targets; otherwise return status='failed' with a clear reason.\n\n"
    "SESSION_DIR contract: every path you emit in result.* must be either\n"
    "verbatim from the request payload, prefixed by SESSION_DIR (injected\n"
    "per tick), or under one of `/sgl-workspace/aiter/`, `/sgl-workspace/\n"
    "sglang/`, `/sgl-workspace/vllm/` (the framework source allowlists).\n"
    "PolicyGate rejects responses whose path fields escape this set.\n\n"
    "If your inbox has no requests, emit one send_message{topic='heartbeat',\n"
    "body_md='ok'}. You may NOT propose, delegate, or initiate REQUESTs."
)


_GFX_TO_RUNNER: dict[str, str] = {
    # Mirror Magpie/modes/benchmark/image_selector.py:138-140. Listed here so
    # we can log the resolved value at session start instead of waiting for
    # Magpie subprocess output deep in the run.
    "gfx942":  "mi300x",
    "gfx950":  "mi355x",
}


def _gpu_runner_type(gpu_type: str) -> str:
    """Return the Magpie runner label for a resolved real GPU type."""
    normalized = str(gpu_type or "").strip().lower()
    if normalized == "mi325x":
        return "mi300x"
    return normalized


# ---------------------------------------------------------------------------
# Hard model allowlist. Orchestration MUST resolve to one of these two ids
# before Coordinator boots; anything else is a configuration error and the
# CLI refuses to start the service. Per operator direction (2026-05-09):
# only Opus 4-7 (preferred) or Opus 4-6 (fallback when 4-7 is missing from
# the gateway catalog) are acceptable for the long-running orchestration
# loop. Other catalog entries (haiku / opus 4-5) drift the agent's behaviour
# enough that prior runs degraded measurably.
_CLAUDE_PREFERRED_MODEL = "claude-opus-4-7"
_CLAUDE_FALLBACK_MODEL  = "claude-opus-4-6"
_CLAUDE_ALLOWED_MODELS  = (_CLAUDE_PREFERRED_MODEL, _CLAUDE_FALLBACK_MODEL)

# Catalog probe retry contract (gateway is documented-flaky; the launcher
# we replaced had to manually curl this endpoint a couple of times before it
# returned 200). Sleep N seconds before attempt i+1; len(_CATALOG_RETRY_DELAYS_SEC)
# is the retry count after the initial attempt.
_CATALOG_RETRY_DELAYS_SEC = (1.0, 3.0, 5.0)
_CATALOG_REQUEST_TIMEOUT_SEC = 5.0

# /dev/shm threshold: vLLM IPC + NCCL shm segments routinely need >8GB; when
# free space drops below this the next launch tends to collide with stale
# segments and hang for 5 minutes inside zmq.
_DEV_SHM_MIN_FREE_BYTES = 16 * 1024 * 1024 * 1024  # 16 GiB


# Critic-agent skill root resolution. Env wins; otherwise we look at the
# sibling ``critic-agent/`` directory next to this package's repo root.
# The runtime is invoked with ``cwd=<root>`` (mirrors critic-agent's own
# pytest.ini ``pythonpath = .``) so ``python -m runtime.cli`` resolves
# without needing a pip-install of critic-agent.
_CRITIC_AGENT_ROOT_ENV = "CRITIC_AGENT_ROOT"


def _resolve_critic_agent_root() -> Path | None:
    """Return the critic-agent skill root, or ``None`` if not found.

    Order:
    1. ``$CRITIC_AGENT_ROOT`` env var (operator override).
    2. Sibling ``critic-agent/`` next to the inference_optimizer package
       (i.e. ``$REPO_ROOT/critic-agent``).
    """
    override = os.environ.get(_CRITIC_AGENT_ROOT_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        return p if (p / "runtime" / "cli.py").is_file() else None
    # PACKAGE_ROOT/.. == repo root (since the package lives at $REPO/inference_optimizer/).
    from .paths import PACKAGE_ROOT
    candidate = PACKAGE_ROOT.parent / "critic-agent"
    return candidate if (candidate / "runtime" / "cli.py").is_file() else None


def _validate_critic_agent_runtime(root: Path) -> None:
    """Fail fast if ``python -m runtime.cli --help`` doesn't work.

    Raises :class:`SystemExit` with a clear message when the runtime cannot
    start; printed in the operator's voice so they know which env to fix.
    """
    cmd = [sys.executable, "-m", "runtime.cli", "--help"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(
            f"ERROR: critic-agent runtime sanity check failed: {exc!r}\n"
            f"  cwd={root}\n"
            f"  cmd={' '.join(cmd)}\n"
            f"Either fix CRITIC_AGENT_ROOT, install critic-agent at "
            f"$REPO_ROOT/critic-agent, or pass --critic-mock to bypass "
            f"critic-agent.",
            file=sys.stderr,
        )
        sys.exit(2)
    if proc.returncode != 0:
        print(
            f"ERROR: critic-agent runtime.cli --help exited rc={proc.returncode}\n"
            f"  cwd={root}\n"
            f"  stderr={proc.stderr.strip()[:500]}",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Robustness-agent runtime location resolution. Mirrors the critic-agent
# helpers above: hosts pick a backend through --robustness-mock /
# --robustness-agent (env override INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND),
# and when "agent" is selected we shell out to
# ``python -m robustness_agent.runtime.cli`` with cwd=<root> and
# PYTHONPATH=<root>/src.
_ROBUSTNESS_AGENT_ROOT_ENV = "ROBUSTNESS_AGENT_ROOT"


def _resolve_robustness_agent_root() -> Path | None:
    """Return the robustness-agent skill root, or ``None`` if not found.

    Order:
    1. ``$ROBUSTNESS_AGENT_ROOT`` env var (operator override).
    2. Sibling ``robustness-agent/`` next to the inference_optimizer package
       (i.e. ``$REPO_ROOT/robustness-agent``).
    """
    override = os.environ.get(_ROBUSTNESS_AGENT_ROOT_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        return p if (p / "src" / "robustness_agent" / "runtime" / "cli.py").is_file() else None
    from .paths import PACKAGE_ROOT
    candidate = PACKAGE_ROOT.parent / "robustness-agent"
    cli_module = candidate / "src" / "robustness_agent" / "runtime" / "cli.py"
    return candidate if cli_module.is_file() else None


def _validate_robustness_agent_runtime(root: Path) -> None:
    """Fail fast if ``python -m robustness_agent.runtime.cli --help`` doesn't work."""
    src = str(root / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    cmd = [sys.executable, "-m", "robustness_agent.runtime.cli", "--help"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(
            f"ERROR: robustness-agent runtime sanity check failed: {exc!r}\n"
            f"  cwd={root}\n"
            f"  cmd={' '.join(cmd)}\n"
            f"Either fix ROBUSTNESS_AGENT_ROOT, install robustness-agent at "
            f"$REPO_ROOT/robustness-agent, or pass --robustness-mock to bypass.",
            file=sys.stderr,
        )
        sys.exit(2)
    if proc.returncode != 0:
        print(
            f"ERROR: robustness-agent runtime.cli --help exited rc={proc.returncode}\n"
            f"  cwd={root}\n"
            f"  stderr={proc.stderr.strip()[:500]}",
            file=sys.stderr,
        )
        sys.exit(2)


def _apply_atom_auto_tighten(args: argparse.Namespace) -> list[str]:
    """Validate atom-specific CLI knob compatibility.

    The function's only responsibility is the multi-node fail-fast
    guard. The forward-looking alias :data:`_assert_atom_single_node`
    (defined below) resolves to this same callable — new call sites
    are encouraged to prefer the clearer name.

    What works on atom today (NO auto-tightening applied):

    * kernel-agent — atom source roots are wired into PolicyGate's
      allowlist, ``_REUSABLE_SOURCE_ROOTS``, and the server-flag
      pre-flight probe; ``--no-kernel`` is preserved at its False
      default.
    * framework-agent — atom's repo URL
      (https://github.com/ROCm/ATOM.git) is in
      ``framework_agent.repo_map``; ``--no-framework`` is preserved
      at its False default, so the FRAMEWORK_PR phase runs.
    * profile / roofline / TraceLens — atom's OpenAI-compatible
      server exposes /start_profile and /stop_profile HTTP endpoints,
      the atom engine takes a ``--torch-profiler-dir`` CLI flag, and
      Magpie's ``atom_mi*x.sh`` bridges ``PROFILE=1`` to that flag.
      atom writes standard ``*.pt.trace.json.gz`` chrome traces which
      TraceLens consumes unchanged.

    What still fails fast on atom:

    * ``--nodes >= 2`` — atom multi-node TP wiring is not yet
      implemented in either the Magpie wrapper or the atom server.
      ``sys.exit(2)`` so operators don't burn a ~6-min cold start on
      a doomed run.

    Returns the list of flag names auto-disabled — always empty since
    no defaults are flipped. The return type is preserved so callers
    that append to / log the list keep working unchanged.
    """
    auto_disabled: list[str] = []
    if int(getattr(args, "nodes", 1) or 1) >= 2:
        print(
            "ERROR: --framework atom does not support multi-node "
            "(--nodes >= 2). atom multi-node TP wiring is deferred; "
            "drop to --nodes 1 or pick --framework sglang/vllm.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        "  framework=atom: no auto-disable applied (kernel-agent + "
        "framework-agent + profile / roofline / TraceLens all wired "
        "for atom); --nodes>=2 guard active — see SKILL.md IR-8"
    )
    return auto_disabled


# Forward-looking alias. The name ``_apply_atom_auto_tighten`` is kept
# because tests / SKILL references still cite it. New call sites that
# want the clearer name can use this alias; the body is the same
# function — both names point at the same callable object so
# monkeypatching either still works.
_assert_atom_single_node = _apply_atom_auto_tighten


_DEPTH_GATE_THRESHOLD_KEYS = frozenset({
    "scout_runs_min", "prs_fetched_min", "pr_diffs_read_min",
    "nvidia_refs_min", "code_patches_min", "reverts_to_evaluate",
})


def _parse_depth_gate_thresholds(raw: str) -> dict[str, int]:
    """Parse ``--depth-gate-thresholds`` (``key=value,...``) into a dict.

    Unknown keys and non-integer values are skipped silently so a typo
    degrades to the defaults rather than aborting startup.
    """
    out: dict[str, int] = {}
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        key, _, value = token.partition("=")
        key = key.strip()
        if key not in _DEPTH_GATE_THRESHOLD_KEYS:
            continue
        try:
            out[key] = int(value.strip())
        except (TypeError, ValueError):
            continue
    return out


def _resolve_gpu_type(
    user_specified: str,
    probed: str,
) -> tuple[str, list[str]]:
    """Resolve the effective gpu_type given a user hint and a hardware probe.

    Pure function so it can be unit tested without monkey-patching half of
    ``_run_optimize``. ``user_specified`` and ``probed`` are expected to
    already be lower-cased and stripped.

    Probe always wins when both are present and disagree. This is the
    strict version of the historical "operator value > probe" priority,
    which silently produced corrupted baseline + KB rows when --gpu-type
    was wrong for the host (e.g. mi300x flag on an MI355X box). Probe
    failure (CPU sandbox, container without rocm-smi) is the only path
    that keeps the user-supplied value.

    Returns ``(effective_gpu_type, warnings)``. Effective gpu_type is the
    string the rest of the cli should write into ``args``/``state``/
    ``manifest``; ``warnings`` is a list of human-readable lines that the
    caller should ``print(..., file=sys.stderr)`` so the operator sees
    them but they do not pollute stdout (which now carries the machine-
    readable ``HYPERLOOM_LAUNCH`` sentinel).
    """
    warnings: list[str] = []
    if probed and user_specified and probed != user_specified:
        warnings.append(
            f"WARN: --gpu-type={user_specified!r} disagrees with probed "
            f"{probed!r}; using probed {probed!r}. The probe wins because "
            f"Magpie runner_type + KB recipe rows must match the actual "
            f"hardware to keep baseline numbers comparable across sessions."
        )
        return probed, warnings
    return (probed or user_specified), warnings


def _emit_launch_info(
    *,
    pid: int,
    session_dir: Path,
    session_id: str,
    run_log: str,
    gpu_type: str,
    framework: str,
    model: str,
    launch_info_file: str | None,
) -> dict[str, Any]:
    """Print the machine-readable HYPERLOOM_LAUNCH stdout line and
    optionally write the same payload to ``launch_info_file`` as JSON.

    Returns the launch_info dict so callers / tests can inspect what was
    emitted. Side effects:

    * ``print("HYPERLOOM_LAUNCH key=value ...")`` to stdout (single line,
      grep-friendly, sed/eval-parseable).
    * When ``launch_info_file`` is set, the same payload is JSON-dumped to
      that path (parent dirs created on demand) so launcher scripts can
      ``jq -r .pid`` instead of grepping the log.
    """
    launch_info: dict[str, Any] = {
        "event": "launch",
        "pid": pid,
        "session_dir": str(session_dir),
        "session_id": session_id,
        "run_log": run_log,
        "manifest": str(session_dir / "manifest.json"),
        "gpu_type": gpu_type,
        "framework": framework,
        "model": model,
    }
    kv_body = " ".join(
        f"{k}={shlex.quote(str(v))}" for k, v in launch_info.items()
    )
    print(f"HYPERLOOM_LAUNCH {kv_body}")
    if launch_info_file:
        path = Path(launch_info_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(launch_info, indent=2))
        print(f"Launch info file: {path}")
    return launch_info


def _clean_stale_aiter_locks(
    aiter_jit_dir: Path | None = None,
    stale_minutes: int = 5,
) -> dict[str, Any]:
    """Sweep aiter's JIT build dir for stale lock files left by killed runs.

    aiter implements its inter-process baton with a plain-file lock (not
    ``fcntl.flock``): the first compiling process creates ``lock_<module>``
    + ``<module>/build/lock`` + ``<module>/build/.ninja_lock`` and removes
    them on success. If the process is SIGKILL'd / SIGTERM'd / OOM-killed
    mid-compile, the files survive and the *next* run blocks forever on
    "[aiter] waiting for baton release ..." with the GPU idle and the
    sglang server.log frozen mid-CUDA-graph-capture. The failure looks
    indistinguishable from a slow first-time JIT compile, which is what
    burned three launch attempts on the Qwen3-30B-A3B mi355x run.

    Cleanup contract: only delete locks whose mtime is older than
    ``stale_minutes`` (default 5). Active compiles touch their lock /
    output files continuously while ninja runs, so a fresh lock is never
    deleted. 5 minutes is well above the cold-start MoE kernel build
    time on MI300X-class hardware (typically 60-180s) and below the
    cliff where users start suspecting a hang.

    Resolution order for the build dir:
      1. caller-supplied ``aiter_jit_dir``
      2. ``$INFERENCE_OPTIMIZER_AITER_JIT_DIR`` env (matches the existing
         probe in ``baseline.py``)
      3. dynamic ``<aiter>/jit/build`` via ``importlib.util.find_spec``
      4. legacy fallback list (``/sgl-workspace/aiter/aiter/jit/build``
         then site-packages variants)

    Returns a stats dict (``{dir, scanned, deleted, skipped_fresh,
    errors}``) so the caller can log a single line and tests can pin the
    behaviour without filesystem flakiness. Never raises: any OSError /
    permission issue is recorded in ``errors`` and the rest of the sweep
    proceeds.
    """
    stats: dict[str, Any] = {
        "dir": None,
        "scanned": 0,
        "deleted": 0,
        "skipped_fresh": 0,
        "errors": 0,
    }

    if aiter_jit_dir is None:
        candidates: list[str] = []
        override = os.environ.get(
            "INFERENCE_OPTIMIZER_AITER_JIT_DIR", "",
        ).strip()
        if override:
            override_path = Path(override)
            candidates.extend([str(override_path), str(override_path / "build")])
        try:
            import importlib.util as _il_util
            spec = _il_util.find_spec("aiter")
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin:
            aiter_root = Path(spec.origin).parent
            candidates.append(str(aiter_root / "jit" / "build"))
        candidates.extend([
            "/sgl-workspace/aiter/aiter/jit/build",
            "/usr/local/lib/python3.10/dist-packages/aiter/jit/build",
            "/usr/local/lib/python3.12/dist-packages/aiter/jit/build",
            "/opt/venv/lib/python3.10/site-packages/aiter/jit/build",
            "/opt/venv/lib/python3.12/site-packages/aiter/jit/build",
        ])
        chosen: Path | None = None
        for cand in candidates:
            p = Path(cand)
            if p.is_dir():
                chosen = p
                break
        if chosen is None:
            return stats
        aiter_jit_dir = chosen

    stats["dir"] = str(aiter_jit_dir)

    threshold_seconds = float(stale_minutes) * 60.0
    now = time.time()
    lock_names = {"lock", ".ninja_lock"}
    try:
        walker = os.walk(str(aiter_jit_dir))
    except OSError:
        stats["errors"] += 1
        return stats

    for root, _dirs, files in walker:
        for fname in files:
            if not (fname in lock_names or fname.startswith("lock_")):
                continue
            stats["scanned"] += 1
            fpath = Path(root) / fname
            try:
                age = now - fpath.stat().st_mtime
            except OSError:
                stats["errors"] += 1
                continue
            if age < threshold_seconds:
                stats["skipped_fresh"] += 1
                continue
            try:
                fpath.unlink()
                stats["deleted"] += 1
            except OSError:
                stats["errors"] += 1

    return stats


def _autodetect_gpu_type() -> str | None:
    """Return mi300x|mi325x|mi355x or None if undetectable.

    Tries `rocm-smi --showproductname` first (most reliable), then falls
    back to torch.cuda.get_device_properties(0).gcnArchName parsing. Both
    are best-effort — on CPU-only or non-ROCm boxes we silently return
    None so the caller can defer to Magpie's own detection layer.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5,
        ).stdout.upper()
        for tag in ("MI355X", "MI325X", "MI300X"):
            if tag in out:
                return tag.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        pass
    try:
        import torch
        arch = torch.cuda.get_device_properties(0).gcnArchName
        gfx = arch.split(":", 1)[0].lower()
        return _GFX_TO_RUNNER.get(gfx)
    except Exception:  # noqa: BLE001
        return None


def _resume_safe_flag(
    args: argparse.Namespace,
    arg_name: str,
    manifest: dict | None,
    manifest_key: str,
    *,
    default: bool,
    invert: bool = False,
) -> bool:
    """Resolve a boolean CLI flag with resume-safe manifest fallback.

    Resolution order:
      1. If ``args`` carries the flag *explicitly* (i.e. the user set
         ``store_true=True`` on this run), that value wins. We detect
         "explicitly set" as "non-default value" because argparse
         stores ``False`` by default for ``store_true``.
      2. Otherwise, fall back to ``manifest[manifest_key]`` (recorded
         on the first launch, persists across resume).
      3. Otherwise, fall back to ``default``.

    ``invert=True`` is for the common ``--no-*`` pattern: ``args.no_X``
    is ``True`` when user disables, but ``manifest.X_enabled`` is the
    positive form. The helper inverts on the way in.

    This is what makes robustness_monitor.sh resume preserve the
    operator's original ``--no-warm-replay`` / ``--no-fact-writes``
    intent without re-passing the flag.
    """
    raw_arg = getattr(args, arg_name, None)
    # ``--no-*`` flag → user passed it → True; default is False.
    # If ``invert``, ``True`` here means "disable feature".
    if isinstance(raw_arg, bool) and raw_arg:
        # User explicitly disabled on THIS launch — honor it.
        return (not raw_arg) if invert else raw_arg
    # User didn't pass the flag — check manifest.
    if manifest is not None and manifest_key in manifest:
        stored = manifest.get(manifest_key)
        if isinstance(stored, bool):
            return stored
    return default


def _resume_safe_numeric(
    args: argparse.Namespace,
    arg_name: str,
    manifest: dict | None,
    manifest_key: str,
    *,
    default: float,
) -> float:
    """Same as :func:`_resume_safe_flag` for float-valued CLI options.

    Resolution order: explicit non-default arg → manifest → default.
    Argparse fills in the registered default when the flag isn't
    passed, so we detect "explicitly set" as "value differs from
    default" — fragile but matches the existing pattern in cli.py.
    """
    raw_arg = getattr(args, arg_name, None)
    if raw_arg is not None:
        try:
            v = float(raw_arg)
        except (TypeError, ValueError):
            v = None
        # We use ``!= default`` as the "user explicitly set" heuristic.
        # When the manifest carries a different value, we still prefer
        # the explicit flag on the current command line.
        if v is not None and v != default:
            return v
    if manifest is not None and manifest_key in manifest:
        try:
            return float(manifest.get(manifest_key) or default)
        except (TypeError, ValueError):
            pass
    return default


def _load_model_arch(workspace_root: Path, model_name: str) -> dict:
    """Best-effort loader for the launcher's advisory ``model_arch`` profile.

    Reads ``<workspace_root>/model_arch.json`` (the convention path under
    ``$USER_DATA_PATH``) written pre-launch by the SKILL launcher. The
    profile is **advisory-only** — it is injected into prompts but drives
    no deterministic gating.

    Soft-degrade contract (never blocks launch):
      * file missing / unreadable / not valid JSON / not a dict -> WARN,
        return ``{}``.
      * stale-file guard: the convention path lives at the shared
        workspace root (parent of every model/session dir), so a leftover
        file from a previous run could otherwise leak into an unrelated
        launch. Require ``data["model_name"]`` basename to match the
        launched ``--model`` basename; mismatch -> WARN
        ``model_arch_stale_or_mismatch``, return ``{}``.
    """
    arch_path = workspace_root / "model_arch.json"
    try:
        raw = arch_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logging.warning("model_arch_unreadable: %s (%s)", arch_path, exc)
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("model_arch_invalid_json: %s (%s)", arch_path, exc)
        return {}
    if not isinstance(data, dict):
        logging.warning(
            "model_arch_not_a_dict: %s (got %s)", arch_path, type(data).__name__
        )
        return {}
    declared = str(data.get("model_name") or "").strip()
    if not declared:
        logging.warning(
            "model_arch_missing_model_name: %s (cannot verify freshness)", arch_path
        )
        return {}
    if Path(declared).name != Path(model_name).name:
        logging.warning(
            "model_arch_stale_or_mismatch: %s declares model_name=%r but "
            "launching %r — ignoring",
            arch_path,
            declared,
            model_name,
        )
        return {}
    return data


def _load_model_config_tags(model_path: str) -> dict:
    """Best-effort loader for KB architecture tags from ``config.json``.

    Reads ``<model_path>/config.json`` (the HF weights dir) and lifts the
    two architecture-identity fields the recipe-snapshot KB stamps as
    ``extras`` tags:

      * ``architectures`` — a list like ``["LlamaForCausalLM"]``.
      * ``model_type``    — a string like ``"llama"``.

    Both are shared by a base model and the models fine-tuned from it, so
    stamping them lets a fine-tuned recipe be recognised as carrying the
    base model's architecture identity.

    Soft-degrade contract (never blocks launch): a missing / unreadable /
    invalid-JSON / non-dict ``config.json`` returns ``{}``. Individual
    fields are normalised (``architectures`` -> list of non-empty strings;
    ``model_type`` -> stripped string) and a key is omitted entirely when
    its normalised value is empty, so callers can ``.get(key, default)``
    without re-checking for ``[]`` / ``""``.
    """
    if not model_path:
        return {}
    cfg_path = Path(model_path) / "config.json"
    try:
        raw = cfg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logging.warning("model_config_tags_unreadable: %s (%s)", cfg_path, exc)
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("model_config_tags_invalid_json: %s (%s)", cfg_path, exc)
        return {}
    if not isinstance(data, dict):
        logging.warning(
            "model_config_tags_not_a_dict: %s (got %s)",
            cfg_path,
            type(data).__name__,
        )
        return {}
    out: dict = {}
    arches_raw = data.get("architectures")
    if isinstance(arches_raw, list):
        arches = [str(a).strip() for a in arches_raw if str(a or "").strip()]
        if arches:
            out["architectures"] = arches
    elif isinstance(arches_raw, str) and arches_raw.strip():
        # Tolerate a scalar ``architectures`` by wrapping it in a list.
        out["architectures"] = [arches_raw.strip()]
    model_type = str(data.get("model_type") or "").strip()
    if model_type:
        out["model_type"] = model_type
    return out


def _seed_shared_state(
    session_dir: Path,
    args: argparse.Namespace,
    *,
    session_id: str,
) -> SharedState:
    # research_lane capacity is locked here for the lifetime of the
    # session. Clamp to [0, research-lane ceiling], where the ceiling
    # scales with the visible GPU count (``2 × GPU``). The cap protects
    # LLM quota and PR Monitor load.
    from inference_optimizer.orchestrator.policy import (
        research_lane_ceiling,
    )
    research_lane_capacity = int(
        getattr(args, "research_lane_capacity", 1) or 1
    )
    research_lane_capacity = max(
        0, min(research_lane_ceiling(), research_lane_capacity),
    )
    gpu_specialist_capacity_raw = getattr(
        args, "gpu_specialist_capacity", None,
    )
    try:
        gpu_specialist_capacity = max(
            0,
            int(gpu_specialist_capacity_raw)
            if gpu_specialist_capacity_raw is not None else 0,
        )
    except (TypeError, ValueError):
        gpu_specialist_capacity = 0
    # collect plateau threshold overrides into a single dict
    #. Only non-None CLI overrides
    # land in the dict; absent keys fall through to the
    # ``DEFAULT_PLATEAU_*`` library constants at phase-compute time.
    plateau_overrides: dict[str, Any] = {}
    if getattr(args, "plateau_explore_keep_gain", None) is not None:
        plateau_overrides["explore_keep_gain_pct"] = float(args.plateau_explore_keep_gain)
    if getattr(args, "plateau_explore_empty_streak", None) is not None:
        plateau_overrides["explore_empty_streak"] = int(args.plateau_explore_empty_streak)
    if getattr(args, "plateau_explore_lookback", None) is not None:
        plateau_overrides["explore_lookback"] = int(args.plateau_explore_lookback)
    if getattr(args, "plateau_kernel_revert_streak", None) is not None:
        plateau_overrides["kernel_revert_streak"] = int(args.plateau_kernel_revert_streak)
    if getattr(args, "plateau_kernel_keep_gain", None) is not None:
        plateau_overrides["kernel_keep_gain_pct"] = float(args.plateau_kernel_keep_gain)
    if getattr(args, "plateau_kernel_lookback", None) is not None:
        plateau_overrides["kernel_lookback"] = int(args.plateau_kernel_lookback)
    # EXPLORE HARD force-exit thresholds. Either condition fires
    # an ``explore_force_exit_low_budget`` exit (overrides plateau /
    # steward / LLM proposals).
    if getattr(args, "explore_force_exit_hours_remaining", None) is not None:
        plateau_overrides["force_exit_hours_remaining"] = float(
            args.explore_force_exit_hours_remaining
        )
    if getattr(args, "explore_force_exit_budget_pct", None) is not None:
        plateau_overrides["force_exit_budget_pct"] = float(
            args.explore_force_exit_budget_pct
        )
    # steward controls.
    if getattr(args, "steward_disabled", False):
        plateau_overrides["steward_disabled"] = True
    if getattr(args, "steward_continuation_cap", None) is not None:
        plateau_overrides["steward_continuation_cap"] = int(
            args.steward_continuation_cap
        )

    # Resolve workload metadata from CLI flags first, then env. The
    # same fields are mirrored into manifest.json by manifest.build_manifest
    # (single source of truth); we duplicate the parse here so SharedState
    # can be populated without re-reading the file we just wrote.
    def _int_env_or_arg(arg_name: str, env_name: str) -> int:
        val = getattr(args, arg_name, None)
        if val is None or val == 0:
            raw = (os.environ.get(env_name, "") or "").strip()
            return int(raw) if raw.isdigit() else 0
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    def _resolve_framework_version(args_in: Any) -> str:
        """Resolve ``framework_version`` for the recipe-snapshot canonical id.

        Three-tier ladder so the operator never has to think about it
        unless they specifically want to pin a version:

        1. Explicit CLI override — ``--framework-version=<slug>`` /
           ``$FRAMEWORK_VERSION`` (operator-pinned, highest priority).
        2. Auto-detect — import the framework's top-level package and
           read ``__version__`` (sglang / vllm / atom supported).
        3. Fall through to empty string — SharedState then carries
           ``""`` and the canonical_id helper substitutes the
           ``unknown_version`` slug. Recipe row is still created;
           just less specific.

        Auto-detect runs only when both CLI and env are empty so a
        process that has *intentionally* set ``$FRAMEWORK_VERSION=""``
        (e.g. a CI smoke test that doesn't want sglang imported) still
        gets the empty-string outcome instead of an unexpected import.
        """
        explicit = (
            (getattr(args_in, "framework_version", None) or "").strip()
            or (os.environ.get("FRAMEWORK_VERSION", "") or "").strip()
        )
        if explicit:
            return explicit
        framework = (
            (getattr(args_in, "framework", None) or "").strip()
            or (os.environ.get("FRAMEWORK", "") or "").strip()
        )
        if not framework:
            return ""
        from .recipe_snapshot_constants import (
            DEFAULT_FRAMEWORK_VERSION_SLUG,
            detect_framework_version,
        )

        detected = detect_framework_version(framework)
        # Treat the failure-slug as "no info" rather than persisting it
        # on SharedState — the canonical_id helper will redo the same
        # fallback on its own at use time.
        return "" if detected == DEFAULT_FRAMEWORK_VERSION_SLUG else detected

    # --explore-overtime-kill-ratio: mirror the CLI value into the
    # fresh SharedState so the ExploreExecutor can read it via the
    # Coordinator-injected task.params on the very first explore round.
    # ``0`` (and any non-positive) disables the gate.
    explore_overtime_kill_ratio_raw = getattr(
        args, "explore_overtime_kill_ratio", None,
    )
    try:
        explore_overtime_kill_ratio = (
            float(explore_overtime_kill_ratio_raw)
            if explore_overtime_kill_ratio_raw is not None else 1.10
        )
    except (TypeError, ValueError):
        explore_overtime_kill_ratio = 1.10

    # ``--explore-variant-timeout-sec`` mirror. ``0`` (default) lets the
    # ExploreExecutor auto-derive the cap from baseline_runtime_sec; any
    # positive value pins it (CI smoke / debug).
    explore_variant_timeout_raw = getattr(
        args, "explore_variant_timeout_sec", None,
    )
    try:
        explore_variant_timeout_sec_override = max(
            0,
            int(explore_variant_timeout_raw)
            if explore_variant_timeout_raw is not None else 0,
        )
    except (TypeError, ValueError):
        explore_variant_timeout_sec_override = 0

    # ``--explore-variant-timeout-safety-margin`` mirror. Headroom as a
    # fraction of baseline_runtime_sec on top of the soft kill ratio when
    # the auto-derive path computes the cap. Negative values clamp to 0
    # (which collapses the hard cap onto the soft kill, no headroom).
    explore_variant_timeout_safety_margin_raw = getattr(
        args, "explore_variant_timeout_safety_margin", None,
    )
    try:
        explore_variant_timeout_safety_margin = max(
            0.0,
            float(explore_variant_timeout_safety_margin_raw)
            if explore_variant_timeout_safety_margin_raw is not None else 0.5,
        )
    except (TypeError, ValueError):
        explore_variant_timeout_safety_margin = 0.5

    # ``--explore-roofline-hard-gate`` mirror (opt-in roofline filter).
    explore_roofline_hard_gate = bool(getattr(
        args, "explore_roofline_hard_gate", False,
    ))

    # KB architecture tags from the model weights' config.json
    # (``architectures`` + ``model_type``). Fresh-launch only — resume
    # rehydrates the persisted values from state.json (see load_or_init).
    _cfg_tags = _load_model_config_tags(str(args.model))

    state = SharedState(
        session_id=session_id,
        claw_session_id=(os.environ.get("CLAW_SESSION_ID") or "").strip(),
        sandbox_user_id=(os.environ.get("SANDBOX_USER_ID") or "").strip(),
        model_name=Path(args.model).name,
        model_path=str(args.model),
        model_class=args.model_class or "",
        # Advisory architecture profile (launcher-supplied convention file).
        # Fresh-launch only: resume rehydrates the persisted value from
        # state.json (see load_or_init below), so we must NOT clobber it
        # with a possibly newer/older convention file. Soft-degrade to {}.
        model_arch=_load_model_arch(
            _workspace_root_resolve(), Path(args.model).name
        ),
        # Architecture-identity tags lifted from config.json; stamped into
        # the recipe-snapshot ``extras`` so a fine-tuned model carries the
        # base model's identity (see ``_load_model_config_tags``).
        model_architectures=_cfg_tags.get("architectures", []),
        model_type=_cfg_tags.get("model_type", ""),
        framework=os.environ.get("FRAMEWORK", "sglang"),
        gpu_type=str(getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")),
        # Workload metadata mirrored from CLI / env at fresh-session time
        # so downstream consumers (specialist prompt builder,
        # orchestration tick prompt) see the real values. Without this
        # the specialist prompt's "## 2. HARDWARE CONTEXT" silently uses
        # the SpecialistPromptInputs dataclass defaults (e.g. TP=1) and
        # comm_specialist self-vetoes on TP=8 sessions.
        tp=_int_env_or_arg("tp", "TP"),
        # ``ep`` mirrors the EP env var so resume in a fresh shell
        # still recovers the value — KB warm-start queries depend on
        # it for the same-shape filter.
        ep=_int_env_or_arg("ep", "EP"),
        precision=(
            str(getattr(args, "precision", None) or os.environ.get("PRECISION", "") or "").strip()
        ),
        framework_version=_resolve_framework_version(args),
        conc=_int_env_or_arg("conc", "CONC"),
        isl=_int_env_or_arg("isl", "ISL"),
        osl=_int_env_or_arg("osl", "OSL"),
        max_model_len=_int_env_or_arg("max_model_len", "MAX_MODEL_LEN"),
        kernel_enabled=not getattr(args, "no_kernel", False),
        continue_kernel_after_gemm=bool(
            getattr(args, "continue_kernel_after_gemm", True)
        ),
        target_summary=args.target_summary or _default_target_summary(args),
        baseline_tput=0.0,
        cumulative_gain=0.0,
        max_minutes=int((args.max_hours or 0) * 60),
        research_lane_capacity=research_lane_capacity,
        gpu_specialist_capacity=gpu_specialist_capacity,
        plateau_overrides=plateau_overrides,
        explore_overtime_kill_ratio=explore_overtime_kill_ratio,
        enable_roofline=bool(
            getattr(args, "enable_roofline", True),
        ),
        # Standalone FRAMEWORK_PR phase (PRELUDE → FRAMEWORK_PR →
        # EXPLORE). ``--no-framework`` skips it; default on. Mirrors
        # the ``--no-kernel`` / ``kernel_enabled`` pattern.
        framework_phase_enabled=not bool(getattr(args, "no_framework", False)),
        # ``--no-explore`` skips the EXPLORE phase entirely (PRELUDE /
        # FRAMEWORK_PR → KERNEL, or → SWEEP when kernel is also off).
        explore_enabled=not bool(getattr(args, "no_explore", False)),
        explore_variant_timeout_sec_override=explore_variant_timeout_sec_override,
        explore_variant_timeout_safety_margin=explore_variant_timeout_safety_margin,
        explore_roofline_hard_gate=explore_roofline_hard_gate,
        research_scout_enabled=bool(getattr(args, "research_scout", True)),
        research_scout_interval=max(
            1, int(getattr(args, "research_scout_interval", 3) or 3)
        ),
        target_advisory_enabled=bool(getattr(args, "target_advisory", True)),
        recipe_sediment_enabled=bool(getattr(args, "recipe_sediment", True)),
        depth_gate_thresholds=_parse_depth_gate_thresholds(
            getattr(args, "depth_gate_thresholds", "") or ""
        ),
        # SWEEP-phase post-sweep concurrency sweep flags (on by default).
        # See ``orchestrator/conc_sweep.py`` + coordinator hook
        # ``_enqueue_internal_conc_sweep_task``.
        conc_sweep_enabled=bool(getattr(args, "enable_conc_sweep", True)),
        conc_sweep_concs=_parse_conc_sweep_concs(args),
        conc_sweep_total_budget_sec=int(
            getattr(args, "conc_sweep_total_budget_sec", 9000) or 0,
        ),
        conc_sweep_variant_timeout_sec=int(
            getattr(args, "conc_sweep_timeout_sec", 1800) or 1800,
        ),
    )
    state.set_depth_gate_enabled(bool(getattr(args, "depth_gate", True)))
    state.save(session_dir)
    return state


def _parse_conc_sweep_concs(args: argparse.Namespace) -> list[int]:
    """Parse ``--conc-sweep-concs '1,2,4,8'`` into a list[int]. Drops
    non-integer tokens with a warning rather than crashing the CLI
    boot so a typo doesn't take the whole session down. Empty list
    (e.g. ``--conc-sweep-concs ''``) lets ``SharedState`` default-factory
    populate the canonical 1..128 ladder."""
    raw = str(getattr(args, "conc_sweep_concs", "") or "").strip()
    if not raw:
        return [1, 2, 4, 8, 16, 32, 64, 128]
    out: list[int] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except ValueError:
            log.warning("conc_sweep: ignoring non-integer CONC token %r", t)
    return out or [1, 2, 4, 8, 16, 32, 64, 128]


def _print_session_skeleton(session_dir: Path) -> None:
    """Echo the freshly-created skeleton so launchers see the exact layout."""
    print(f"Session layout under {session_dir}:")
    for sub in _SESSION_SKELETON:
        marker = "ok" if (session_dir / sub).is_dir() else "MISSING"
        print(f"  [{marker}] {sub}/")
    print(f"  [ok] manifest.json (written first)")


def _snapshot_system_prompts(
    session_dir: Path,
    *,
    prompts: dict[str, str],
) -> None:
    """Persist each persistent agent's effective system prompt.

    Writes to ``agents/<role>/system_prompt.snapshot.md`` so resume
    runs and post-mortem inspection can compare against the in-memory
    prompt without re-deriving it from CLI args.
    """
    for role, body in prompts.items():
        target = agent_prompt_snapshot(session_dir, role)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body or "(empty)", encoding="utf-8")


def _default_target_summary(args: argparse.Namespace) -> str:
    if args.target_gain:
        return (
            f"Establish baseline on {Path(args.model).name} then drive "
            f"cumulative_gain to >= {args.target_gain}% within "
            f"{args.max_hours}h."
        )
    if args.target_tput:
        return (
            f"Establish baseline on {Path(args.model).name} then reach "
            f"{args.target_tput} tok/s/GPU within {args.max_hours}h."
        )
    return f"Optimize {Path(args.model).name} for up to {args.max_hours}h (no target)."


def _print_final_summary(state: SharedState, stop_reason: str) -> None:
    print()
    print("================ Final summary ================")
    print(f"  stop_reason          : {stop_reason}")
    print(f"  session_id           : {state.session_id}")
    print(f"  model                : {state.model_name}")
    print(f"  baseline_tput        : {state.baseline_tput:.1f} tok/s/GPU")
    print(
        f"  cumulative_gain      : {state.cumulative_gain:.2f}% "
        f"(per-round sum — informational)"
    )
    if state.cumulative_gain_validated_ts:
        stale = (
            " ⚠ stack changed since validation"
            if len(state.optimization_stack) > state.cumulative_gain_validated_stack_len
            else ""
        )
        print(
            f"  cumulative_gain_val  : {state.cumulative_gain_validated:.2f}% "
            f"(validated_at_stack_len={state.cumulative_gain_validated_stack_len}, "
            f"ts={state.cumulative_gain_validated_ts}){stale}"
        )
    else:
        print(
            f"  cumulative_gain_val  : 0.00% "
            f"⚠ never validated — no `explore` stack-rebench has succeeded yet"
        )
    print(f"  current_best         : {state.current_best}")
    print(f"  pruned_families      : {state.pruned_families}")
    print(f"  crash_count          : {state.crash_count}")
    _print_kernel_opt_summary_line(state)
    print("===============================================")


def _reconcile_crash_count(state: SharedState, session_dir: Path) -> None:
    """Make the persisted ``crash_count`` agree with the authoritative
    in-memory value printed in the final summary.

    The ReportExecutor and the breakdown writer both reload state from
    ``state.json`` (``SharedState.load_or_init``), so a reactor-pass
    crash increment that lost a write race against a later save would
    leave ``state.json`` / ``reports/final.json`` showing a stale (lower)
    count while the live coordinator object — and therefore the console
    summary — shows the true count. Reconcile both disk artifacts to the
    live value at teardown so all three sources never disagree. We only
    ever raise the persisted value (``max``); we never lower a count that
    disk recorded but memory somehow missed. Best-effort: never fatal.
    """
    live = int(getattr(state, "crash_count", 0) or 0)

    # 1) state.json — reload, bump if stale, atomic re-save.
    try:
        disk_state = SharedState.load_or_init(session_dir)
        if int(disk_state.crash_count or 0) < live:
            disk_state.crash_count = live
            disk_state.save(session_dir)
    except Exception:  # noqa: BLE001
        log.exception("crash_count reconcile (state.json) failed (non-fatal)")

    # 2) reports/final.json — patch the single field in place if present.
    try:
        from .session_paths import reports_dir
        final_json = reports_dir(session_dir) / "final.json"
        if final_json.exists():
            data = json.loads(final_json.read_text(encoding="utf-8"))
            if int(data.get("crash_count") or 0) < live:
                data["crash_count"] = live
                final_json.write_text(
                    json.dumps(data, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
    except Exception:  # noqa: BLE001
        log.exception("crash_count reconcile (final.json) failed (non-fatal)")


def _print_kernel_opt_summary_line(state: SharedState) -> None:
    """One-line forensic readout of kernel_opt attempts at session end.

    Pulls the same aggregation used by
    ``reports/kernel_optimization_summary.json`` so stdout matches the
    on-disk report. Best-effort: any failure is swallowed (this is a
    summary print, not a critical path).
    """
    try:
        from .orchestrator.kernel_attempt_summary import (
            build_kernel_optimization_summary,
        )
        session_dir = _resolve_session_dir_for_summary(state)
        if session_dir is None:
            return
        summary = build_kernel_optimization_summary(state, session_dir)
        totals = summary.get("totals") or {}
        attempted = int(totals.get("attempted") or 0)
        if attempted == 0 and int(totals.get("unattempted") or 0) == 0:
            return
        integrated = int(totals.get("integrated") or 0)
        rejected = int(totals.get("rejected") or 0)
        unattempted = int(totals.get("unattempted") or 0)
        print(
            f"  kernel_opt           : {attempted} attempted "
            f"({integrated} integrated, {rejected} rejected), "
            f"{unattempted} unattempted in top candidates"
        )
        takeaways = summary.get("top_takeaways") or []
        if len(takeaways) >= 2:
            print(f"  kernel_opt_top_cause : {takeaways[1]}")
        report_path = (
            Path(session_dir) / "reports" / "kernel_optimization_summary.json"
        )
        if report_path.is_file():
            print(f"  kernel_opt_report    : {report_path}")
    except Exception:  # noqa: BLE001 — stdout print must never fail the run
        pass


def _resolve_session_dir_for_summary(state: SharedState) -> Path | None:
    """Best-effort session_dir lookup for the stdout kernel_opt line.

    SharedState does not carry session_dir directly, so we use the same
    discovery the SessionInit path uses: HYPERLOOM_SESSION_DIR env when
    set, else the cwd-anchored default. Returns ``None`` if neither
    resolves to an existing directory.
    """
    env_sd = os.environ.get("HYPERLOOM_SESSION_DIR", "").strip()
    if env_sd:
        p = Path(env_sd).expanduser()
        if p.is_dir():
            return p
    return None


def _derive_anthropic_base_url(openai_base_url: str) -> str:
    """Derive ``ANTHROPIC_BASE_URL`` from ``OPENAI_BASE_URL``.

    The Anthropic SDK appends ``/v1`` itself, so we strip a trailing
    ``/v1`` from the OpenAI-style URL. Returns the input verbatim when
    there is no ``/v1`` suffix to strip.
    """
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(openai_base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunparse(parsed._replace(path=path))


def _reset_claude_config_to_upstream(
    safe_key: str, anthropic_base_url: str
) -> None:
    """Point ``~/.claude/config.json`` ``customApiUrl`` at the upstream gateway.

    Used after the auth-proxy was removed: a stale ``127.0.0.1:4002``
    value would make the Claude CLI dial a port no longer bound. We
    rewrite to the upstream Anthropic URL so the CLI talks directly to
    the gateway with its ``x-api-key`` header (the gateway accepts it).
    """
    import json as _json

    if not anthropic_base_url:
        return
    claude_config_path = Path.home() / ".claude" / "config.json"
    config_data: dict = {}
    if claude_config_path.exists():
        try:
            config_data = _json.loads(
                claude_config_path.read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            config_data = {}
        current_url = config_data.get("customApiUrl", "")
        if current_url == anthropic_base_url:
            print("Preflight: ~/.claude/config.json already points at upstream")
            return

    config_data.setdefault("theme", "dark")
    config_data.setdefault("hasCompletedOnboarding", True)
    if safe_key:
        config_data["primaryApiKey"] = safe_key
    elif "primaryApiKey" not in config_data:
        config_data["primaryApiKey"] = ""
    config_data["customApiUrl"] = anthropic_base_url
    claude_config_path.parent.mkdir(parents=True, exist_ok=True)
    claude_config_path.write_text(
        _json.dumps(config_data, indent=2) + "\n", encoding="utf-8",
    )
    claude_config_path.chmod(0o600)
    print(
        f"Preflight: updated ~/.claude/config.json customApiUrl -> "
        f"{anthropic_base_url}"
    )


def _validate_credentials() -> None:
    """Fail fast when SAFE_API_KEY or OPENAI_BASE_URL is missing.

    Mirrors the gate in ``kernel-agent/scripts/install.sh`` and
    ``inference_optimizer/scripts/install.sh`` so the same missing-credentials
    failure surfaces at the same point regardless of whether the operator
    ran the installers or jumped straight to ``inference_optimizer optimize``.
    Without this, the cli would happily import Coordinator, spin up KB / Ray /
    catalog probes, and only blow up when a specialist agent or the catalog
    probe finally tries to authenticate.

    Strict mode by design: no bypass flag or env var. Specialist agents,
    GEAK, and the catalog probe all require live credentials, so a run
    without them cannot finish anyway. Failing here keeps the failure
    message attached to the actual cause.
    """
    missing: list[str] = []
    if not os.environ.get("SAFE_API_KEY"):
        missing.append("SAFE_API_KEY")
    if not os.environ.get("OPENAI_BASE_URL"):
        missing.append("OPENAI_BASE_URL")
    if not missing:
        return
    repo_root = os.environ.get("REPO_ROOT") or os.getcwd()
    env_file = Path(repo_root) / ".env"
    env_status = "present" if env_file.exists() else "not found"
    print(
        "\nERROR: Missing required credential(s): "
        f"{', '.join(missing)}\n\n"
        "Tried loading from:\n"
        "  - shell environment\n"
        f"  - $REPO_ROOT/.env  ({env_status}: {env_file})\n\n"
        "Fix one of:\n"
        "  1. Copy .env from a working worktree into this one:\n"
        f"       cp /path/to/main-worktree/.env {env_file}\n"
        "  2. Export directly into the shell before re-running:\n"
        "       export SAFE_API_KEY=sk-xxxxx\n"
        "       export OPENAI_BASE_URL=https://gateway.example.com/v1",
        file=sys.stderr,
    )
    sys.exit(2)


def _is_placeholder_tracelens_path(value: str) -> bool:
    """Treat .env.template's bare ``\\`` and whitespace-only values as unset."""
    stripped = value.strip()
    return stripped in ("", "\\")


def _load_dotenv_fallback() -> None:
    """Env always wins over .env. If SAFE_API_KEY or OPENAI_BASE_URL is
    missing from os.environ, source ``$REPO_ROOT/.env`` (defaults to
    ``os.getcwd()``) but never overwrite a key that is already in env.
    """
    if os.environ.get("SAFE_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
        return
    repo_root = os.environ.get("REPO_ROOT") or os.getcwd()
    env_file = Path(repo_root) / ".env"
    if not env_file.exists():
        return
    loaded = 0
    for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key in ("TRACELENS_ROOT", "TRACELENS_INTERNAL_ROOT") and _is_placeholder_tracelens_path(value):
            continue
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    if loaded:
        print(f"Preflight: loaded {loaded} missing var(s) from {env_file} (env wins)")


def _load_kernel_agent_env_fallback() -> None:
    """If ``HYPERLOOM_KERNEL_AGENT_ROOT`` is unset, auto-source the
    kernel-agent env file produced by ``inference_optimizer/scripts/
    install.sh`` at ``$USER_DATA_PATH/runtime/kernel-agent.env.sh``
    (overridable via ``$KERNEL_AGENT_ENV``).

    Background: a run once stalled because the launcher only sourced
    the user's basic ``.env`` and missed kernel-agent.env.sh.
    ``RooflineExecutor``'s trace_analyze sub-step imports
    ``kernel_request_handlers.HYPERLOOM_KERNEL_AGENT_ROOT`` at module
    load — that read happens before any user code can fix the env, so
    the only way to recover without a restart is to source the file
    here, before any orchestrator import. Setting the env in this
    process also propagates to all subprocesses launched by Magpie /
    TraceLens / GEAK runners.

    Hard-fail contract (a wrong-path env file used to WARN-only and let
    trace_analyze silently fail for hours). The function now:

    * Looks ONLY at ``$KERNEL_AGENT_ENV`` (if set) or
      ``$USER_DATA_PATH/runtime/kernel-agent.env.sh``. USER_DATA_PATH
      MUST be the workspace root (``runtime/`` is workspace-shared,
      not per-session). No parent-dir fallback.
    * If the file is missing OR parses 0 vars OR
      HYPERLOOM_KERNEL_AGENT_ROOT is still unset after sourcing,
      ``sys.exit(2)`` with a clear actionable message instead of
      silently warning and letting trace_analyze fail later. Fail
      early, fail loud: the operator sees the problem in the first
      30 seconds, not 10 hours in.
    * Skip entirely when HYPERLOOM_KERNEL_AGENT_ROOT is already set
      (operator pre-sourced manually, or running from a sandbox that
      injected it — both fine).
    """
    if os.environ.get("HYPERLOOM_KERNEL_AGENT_ROOT"):
        return
    candidate = os.environ.get("KERNEL_AGENT_ENV")
    if not candidate:
        user_data = os.environ.get("USER_DATA_PATH")
        if not user_data:
            print(
                "Preflight: ERROR — neither $HYPERLOOM_KERNEL_AGENT_ROOT "
                "nor $KERNEL_AGENT_ENV nor $USER_DATA_PATH is set. Cannot "
                "resolve kernel-agent.env.sh. Run "
                "inference_optimizer/scripts/install.sh and export "
                "USER_DATA_PATH=/path/to/sessions first.",
                file=sys.stderr,
            )
            sys.exit(2)
        candidate = str(Path(user_data) / "runtime" / "kernel-agent.env.sh")
    env_path = Path(candidate)
    if not env_path.is_file():
        print(
            f"Preflight: ERROR — kernel-agent env file not found at "
            f"{env_path}. USER_DATA_PATH must be the workspace root "
            f"(parent of <model>/<ts>/ per-session subdirs); runtime/ "
            f"is workspace-shared, not per-session. Either "
            f"(a) re-run inference_optimizer/scripts/install.sh under "
            f"USER_DATA_PATH={os.environ.get('USER_DATA_PATH','?')}, "
            f"(b) set $KERNEL_AGENT_ENV to point at an existing file, or "
            f"(c) set $HYPERLOOM_KERNEL_AGENT_ROOT directly to skip this "
            f"fallback entirely. Aborting now (was: silently warning and "
            f"letting trace_analyze fail 10h in).",
            file=sys.stderr,
        )
        sys.exit(2)
    loaded = 0
    try:
        text = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(
            f"Preflight: ERROR — failed to read {env_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    if "HYPERLOOM_KERNEL_AGENT_ROOT" not in os.environ:
        print(
            f"Preflight: ERROR — sourced {env_path} ({loaded} vars) but "
            f"HYPERLOOM_KERNEL_AGENT_ROOT is still unset. The env file is "
            f"malformed or stale. Re-run inference_optimizer/scripts/"
            f"install.sh to regenerate it.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        f"Preflight: loaded {loaded} kernel-agent var(s) from "
        f"{env_path} (env wins, HYPERLOOM_KERNEL_AGENT_ROOT="
        f"{os.environ['HYPERLOOM_KERNEL_AGENT_ROOT']})"
    )


def _ensure_python_sdks(python_exe: str, pip_extra: list[str]) -> None:
    """Install Python SDKs that ``inference_optimizer`` imports at runtime.

    Even though they're declared in ``pyproject.toml``, a sandbox that only
    pulled the source tree (or used ``pip install`` without resolving the
    dep tree) lands here without them. ``ClaudeBackend`` / ``CodexBackend``
    lazy-import them, so without this guard the first reactor tick fails
    with ``BackendError: claude-agent-sdk not installed`` after baseline
    has already burned 5+ minutes of wall time.

    We probe-then-install per package using the SAME interpreter that
    will later import them (``sys.executable``); cross-interpreter installs
    are the precise failure mode that "claude-agent-sdk was not installed
    in /opt/venv → installed 0.1.77" reports come from.
    """
    candidates = (
        ("claude_agent_sdk", "claude-agent-sdk>=0.1.65"),
        ("openai",           "openai>=1.50"),
        ("httpx",             "httpx>=0.27"),
    )
    for module_name, pip_spec in candidates:
        check = subprocess.run(
            [python_exe, "-c", f"import {module_name}"],
            capture_output=True,
        )
        if check.returncode == 0:
            print(f"Preflight: {module_name} OK")
            continue
        print(f"Preflight: {module_name} not importable, installing {pip_spec} ...")
        subprocess.run(
            [python_exe, "-m", "pip", "install", "--quiet",
             *pip_extra, pip_spec],
            check=True,
        )
        print(f"Preflight: installed {pip_spec}")


def _unset_hip_visible_devices() -> None:
    """Drop ``HIP_VISIBLE_DEVICES`` if ``ROCR_VISIBLE_DEVICES`` is set.

    Long-standing ROCm gotcha (mirrored in ``inference_optimizer/SKILL.md``
    §"GPU Runner Type"): when both vars are set, ``HIP_VISIBLE_DEVICES``
    can mask devices in a way that makes ``torch.cuda.is_available()``
    return False inside the Magpie subprocess, which then logs "No
    accelerator" and exits non-zero. We use ``ROCR_VISIBLE_DEVICES`` as the
    canonical pinning; ``HIP_VISIBLE_DEVICES`` is a footgun left over from
    upstream CUDA scripts that copy-pasted into ROCm sandboxes.
    """
    if "HIP_VISIBLE_DEVICES" not in os.environ:
        return
    if "ROCR_VISIBLE_DEVICES" not in os.environ:
        return
    value = os.environ.pop("HIP_VISIBLE_DEVICES")
    print(
        f"Preflight: WARNING — unset HIP_VISIBLE_DEVICES={value!r} "
        f"(ROCR_VISIBLE_DEVICES wins on ROCm; HIP_VISIBLE_DEVICES can "
        f"make torch.cuda.is_available() false inside Magpie subprocess)"
    )


def _check_gpu_visibility() -> None:
    """Best-effort sanity check on visible GPU count vs ``$TP``.

    rocm-smi may be missing on CPU-only test boxes; we return silently in
    that case and let Magpie's own detection complain later. The check is
    purely informational — TP can also be set inside the Magpie YAML, in
    which case this WARN is a false positive but cheap.
    """
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        return
    if proc.returncode != 0:
        return
    # ``rocm-smi --showid`` emits multiple lines per GPU (Device Name,
    # Device ID, Device Rev, Subsystem ID, GUID, ...). Counting raw lines
    # that start with ``GPU[`` overcounts by ~6x — a 4-GPU pod looked like
    # it had 24 visible GPUs and this WARN never fired. Deduplicate by
    # GPU index instead.
    visible_indices: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("GPU["):
            idx, _, _ = stripped[4:].partition("]")
            if idx:
                visible_indices.add(idx)
    visible = len(visible_indices)
    try:
        wanted = int(os.environ.get("TP", "1") or "1")
    except ValueError:
        wanted = 1
    if visible == 0:
        print("Preflight: WARNING — rocm-smi sees 0 GPUs; benchmark will fail")
        return
    if wanted > visible:
        print(
            f"Preflight: WARNING — TP={wanted} but rocm-smi sees {visible} "
            f"GPU(s); sglang/vllm may fail to load weights. Lower TP or "
            f"adjust ROCR_VISIBLE_DEVICES."
        )


def _check_shm_disk() -> None:
    """Warn on tight ``/dev/shm`` (vLLM/NCCL IPC needs headroom).

    ``_kill_stale_servers()`` already clears stale segments before each
    Magpie variant, but if the partition itself is small the very first
    launch can still collide. We don't fail-fast — partitions can be
    short-lived (test sandboxes use tmpfs sized to the workload).
    """
    try:
        usage = shutil.disk_usage("/dev/shm")
    except (FileNotFoundError, OSError):
        return
    if usage.free < _DEV_SHM_MIN_FREE_BYTES:
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        print(
            f"Preflight: WARNING — /dev/shm has {free_gb:.1f} GiB free of "
            f"{total_gb:.1f} GiB total (< 16 GiB threshold). vLLM IPC + "
            f"NCCL shm segments may collide with stale entries; if the "
            f"first server launch hangs >5min, clear /dev/shm/{{vllm,nccl,cuda}}*"
        )


_TRACELENS_REQUIRED_CLIS: tuple[str, ...] = (
    "TraceLens_generate_perf_report_pytorch_inference",
)


def _check_tracelens_cli() -> None:
    """Hard-gate TraceLens CLI presence — abort BEFORE Coordinator starts.

    Why this is hard-fail (vs ``_check_node_claude_cli`` which is WARN-only):

    The ``TraceLens_*`` console_scripts are installed by
    ``kernel-agent/scripts/install.sh`` (chained from
    ``inference_optimizer/scripts/install.sh``) into the *pod-local*
    ``/opt/venv/bin/`` — they do NOT persist across pod restarts even
    when ``$USER_DATA_PATH`` (typically ``/workspace/hyperloom``) is a
    WekaFS-backed session dir that survives pod recycling. SKILL
    therefore requires running ``install.sh`` before every launch; the
    only carve-out is ``--resume`` in the *same shell* that earlier ran
    install.sh.

    Brain-generated launchers that source only
    ``runtime/kernel-agent.env.sh`` and skip install.sh (fresh-start
    ``--model`` path) land here with no TraceLens CLI on PATH. Until this
    gate was added, the missing-CLI failure was surfaced only by the
    robustness agent's HIGH-severity ``tracelens_cli_missing`` signal
    at tick ~6 — after baseline had already completed
    (or hung) and a multi-minute setup cost was wasted.
    ``trace_analyze`` / ``kernel_opt`` then fail downstream when they
    shell out to ``tracelens_analysis.py``.

    Moving discovery to launch — mirroring ``_gate_claude_model`` —
    turns a delayed silent strike into a
    fail-fast with an actionable error pointing at the install.sh fix.
    """
    missing = [
        name for name in _TRACELENS_REQUIRED_CLIS
        if shutil.which(name) is None
    ]
    if not missing:
        return
    session_dir = os.environ.get("USER_DATA_PATH", "/workspace/hyperloom")
    print(
        f"ERROR: TraceLens CLI(s) not on PATH: {missing}. The pod-local "
        f"/opt/venv/bin/TraceLens_* console_scripts are installed by "
        f"kernel-agent/scripts/install.sh (chained from "
        f"inference_optimizer/scripts/install.sh) and do NOT persist "
        f"across pod restarts. SKILL IR-2 requires running install.sh "
        f"before every launch (carve-out applies only to --resume in "
        f"the same shell that earlier ran install.sh). Re-run:\n"
        f"  bash $REPO_ROOT/inference_optimizer/scripts/install.sh\n"
        f"  . {session_dir}/runtime/kernel-agent.env.sh\n"
        f"then retry `inference_optimizer optimize`. Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


def _check_node_claude_cli() -> None:
    """WARN-only presence check for the bundled agent CLIs and ``@cursor/sdk``.

    ``claude_agent_sdk`` typically shells out to the bundled
    ``@anthropic-ai/claude-code`` CLI; without it on PATH the SDK falls
    back to a direct HTTP path against the upstream gateway, so this
    is informational rather than fatal. Same for ``codex`` (used by
    ``CriticAgentBackend`` review reasoning and ``--kernel-codex``).
    ``node`` is a
    transitive dep — if it's missing, npm-based recovery via
    ``kernel-agent/scripts/install.sh`` won't work either.

    The cursor backend talks to Cursor's own gateway via the ``@cursor/sdk``
    Node library (not a CLI). We probe it via ``require.resolve`` against
    ``$(npm root -g)`` since ``shutil.which('cursor')`` would always miss.
    """
    missing = [t for t in ("node", "claude", "codex") if shutil.which(t) is None]
    if missing:
        print(
            f"Preflight: WARNING — CLI(s) not on PATH: {missing}. "
            f"ClaudeBackend / CodexBackend may fall back to direct HTTP. "
            f"Run kernel-agent/scripts/install.sh to bring them in."
        )
    # @cursor/sdk presence — probe via Node since it's a library, not a CLI.
    if shutil.which("node") is not None and shutil.which("npm") is not None:
        try:
            npm_root = subprocess.run(
                ["npm", "root", "-g"], capture_output=True, text=True, timeout=10,
            )
            global_modules = (npm_root.stdout or "").strip()
            probe = subprocess.run(
                ["node", "-e", "require.resolve('@cursor/sdk')"],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "NODE_PATH": global_modules} if global_modules else None,
            )
            if probe.returncode != 0:
                print(
                    "Preflight: WARNING — @cursor/sdk not resolvable; cursor "
                    "backend will fail to start. Run kernel-agent/scripts/"
                    "install.sh to install it globally via npm."
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            print(
                "Preflight: WARNING — could not probe @cursor/sdk presence; "
                "cursor backend may be unavailable."
            )


def _emit_preflight_diagnostics(
    *,
    magpie_python: str,
    anthropic_base_url: str | None,
    args: argparse.Namespace | None = None,
) -> None:
    """One canonical diagnostics block at the end of preflight.

    Replaces the half-dozen scattered ``print(f"Preflight: ...")`` lines
    with a single grep-friendly section that operators (and the launcher
    LLM) can paste verbatim into status reports without spelunking the
    source for env var names.
    """
    from .orchestrator.action_executors.baseline import (
        BASELINE_COLD_START_TIMEOUT_SEC,
        BASELINE_DEFAULT_TIMEOUT_SEC,
        _probe_aiter_jit_cache,
    )
    from .paths import asset_root

    probe = _probe_aiter_jit_cache()
    cold_cap = os.environ.get(
        "INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC",
        str(BASELINE_COLD_START_TIMEOUT_SEC),
    )
    if probe["probe_status"] == "found":
        kind = "COLD" if probe["is_cold"] else "WARM"
        cache_line = (
            f"{probe['kernel_count']} .so / {probe['size_mb']} MB "
            f"({kind}) at {probe['path']}"
        )
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
    print(
        f"  INFERENCEX_PATH     = "
        f"{os.environ.get('INFERENCEX_PATH', '<unset>')}"
    )
    print(f"  aiter jit cache     = {cache_line}")
    print(f"  cold_start_timeout  = {cold_cap}s")
    print(f"  warm_timeout        = {BASELINE_DEFAULT_TIMEOUT_SEC}s")
    if anthropic_base_url:
        print(f"  ANTHROPIC_BASE_URL  = {anthropic_base_url} (direct to gateway)")
    else:
        print(
            "  ANTHROPIC_BASE_URL  = <unset> — OPENAI_BASE_URL missing; "
            "Claude SDK will fail"
        )
    if args is not None:
        kb_enabled = bool(getattr(args, "cortex_enabled", True))
        pr_enabled = bool(getattr(args, "pr_monitor_enabled", True))
        kb_reason = getattr(args, "kb_degraded_reason", None) or "-"
        pr_reason = getattr(args, "pr_degraded_reason", None) or "-"
        kb_status = "OK" if kb_enabled else f"DEGRADED ({kb_reason})"
        pr_status = "OK" if pr_enabled else f"DEGRADED ({pr_reason})"
        print(f"  kb_status           = {kb_status}")
        print(f"  pr_monitor_status   = {pr_status}")
        print(f"  kb_degraded_reason  = {kb_reason}")
        print(f"  pr_degraded_reason  = {pr_reason}")

    # Issue-H (Saturday May 2026): surface Cortex KB offline-queue
    # state. A dead-letter pile-up is the canonical signal for "the
    # prior session's KB writes were rejected (HTTP 4xx schema), this
    # session is starting cold." Operators have no in-band way to see
    # this today — they discover it only when specialists return empty
    # proposal_set.
    try:
        _print_cortex_kb_queue_status()
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"  cortex_kb_queue     = <probe_failed: {exc!r}>")


def _print_cortex_kb_queue_status() -> None:
    """Emit a one-line summary of the Cortex KB offline NDJSON queue.

    Pure visibility helper. We count rows so the operator sees
    ``pending=N dead=M`` next to the rest of the preflight diagnostics.
    The dead-letter count is the 422-style permanent-reject signal that
    telegraphs an upcoming cold-start session for new models.
    """
    from .session_paths import (
        cortex_dead_letter_ndjson,
        cortex_flushed_ndjson,
        cortex_pending_ndjson,
    )

    sd = _session_dir_resolve()
    pending = cortex_pending_ndjson(sd)
    dead = cortex_dead_letter_ndjson(sd)
    flushed = cortex_flushed_ndjson(sd)

    def _count(p: Path) -> int:
        if not p.exists():
            return 0
        try:
            with p.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    p_n, d_n, f_n = _count(pending), _count(dead), _count(flushed)
    print(
        f"  cortex_kb_queue     = pending={p_n} dead_letter={d_n} "
        f"flushed={f_n} (root={pending.parent})"
    )
    if d_n > 0:
        print(
            f"                        ⚠ {d_n} dead-letter row(s) — "
            f"prior KB writes permanently rejected (4xx schema). "
            f"Specialists for affected anchors will start cold "
            f"(no priors). See {dead}."
        )


def _probe_llm_catalog(
    *,
    base_url: str,
    api_key: str,
) -> set[str] | None:
    """Probe ``<base_url>/models`` with retry; return set of model ids or None.

    Mirrors what the launcher had to do by hand (``terminals/6.txt``):

        curl -H "Authorization: Bearer $SAFE_API_KEY" \
             "https://gateway/api/v1/llm-proxy/v1/models" | jq '.data[].id'

    The gateway has a documented flake rate; we retry up to
    ``len(_CATALOG_RETRY_DELAYS_SEC)`` times with exponential backoff.

    TLS verification is ON by default. For internal gateways with self-
    signed certs, set ``INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE=1`` to
    skip verification (a warning is printed since the probe also sends
    ``Authorization: Bearer <api_key>``).
    """
    import time

    if not base_url:
        return None

    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError:
        # _ensure_python_sdks should have installed it; if it didn't, we
        # cannot probe — return None so the caller can decide how to react.
        print(
            "Preflight: WARNING — httpx not importable, skipping catalog "
            "probe. _ensure_python_sdks should have installed it."
        )
        return None

    insecure = os.environ.get(
        "INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE", "",
    ).strip().lower() in ("1", "true", "yes")
    if insecure:
        print(
            "Preflight: WARNING — INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE=1 "
            "is set; catalog probe will skip TLS verification while sending "
            "an Authorization: Bearer header. Use only against trusted internal "
            "gateways with self-signed certs."
        )
        try:
            import urllib3  # type: ignore[import-not-found]
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:  # noqa: BLE001
            pass

    probe_url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    delays = (0.0, *_CATALOG_RETRY_DELAYS_SEC)
    last_err: str = ""
    for i, delay in enumerate(delays):
        if delay > 0:
            time.sleep(delay)
        try:
            resp = httpx.get(
                probe_url,
                headers=headers,
                timeout=_CATALOG_REQUEST_TIMEOUT_SEC,
                verify=not insecure,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            print(
                f"Preflight: catalog probe attempt {i + 1}/{len(delays)} "
                f"failed: {last_err}"
            )
            continue
        if resp.status_code != 200:
            last_err = (
                f"HTTP {resp.status_code}: "
                f"{(resp.text or '')[:200]}"
            )
            print(
                f"Preflight: catalog probe attempt {i + 1}/{len(delays)} "
                f"got {last_err}"
            )
            continue
        try:
            data = resp.json()
        except ValueError as exc:
            last_err = f"JSON decode: {exc}"
            print(
                f"Preflight: catalog probe attempt {i + 1}/{len(delays)} "
                f"returned non-JSON: {last_err}"
            )
            continue
        ids = {
            m["id"] for m in data.get("data") or []
            if isinstance(m, dict) and isinstance(m.get("id"), str)
        }
        if not ids:
            last_err = "empty data[]"
            continue
        return ids

    print(
        f"Preflight: catalog probe exhausted {len(delays)} attempts "
        f"({last_err}); cannot validate model availability"
    )
    return None


def _validate_and_resolve_claude_model(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str] | None,
) -> set[str] | None:
    """Hard-gate Claude model selection. Mutates ``args.claude_model``.

    Per operator direction (2026-05-09):

    * ``--claude-model`` MUST be one of ``_CLAUDE_ALLOWED_MODELS`` (4-7 or
      4-6). Any other value aborts boot before Coordinator starts —
      orchestration drift on opus-4-5 / haiku silently degraded prior runs.
    * Probe the gateway catalog with retry (gateway flakes). If the chosen
      model is missing but the fallback (4-6) is in catalog, rewrite the
      arg + WARN. Otherwise sys.exit(2) — refuse to start the service.

    Returns the catalog id set on success (so the codex smoke-test can
    reuse it without re-probing); returns None if catalog probe failed but
    the chosen model was already in ``_CLAUDE_ALLOWED_MODELS`` AND we
    decide to proceed (we don't — gateway unreachable means abort).
    """
    chosen = (args.claude_model or "").strip()
    if chosen not in _CLAUDE_ALLOWED_MODELS:
        print(
            f"ERROR: --claude-model={chosen!r} is not allowed. "
            f"Orchestration model must be one of {list(_CLAUDE_ALLOWED_MODELS)} "
            f"(preferred: {_CLAUDE_PREFERRED_MODEL}, "
            f"fallback: {_CLAUDE_FALLBACK_MODEL}). Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Catalog probe uses GET <base>/models against the upstream gateway URL.
    # ``INFERENCE_OPTIMIZER_CATALOG_PROBE_URL`` remains an explicit override
    # path for operators who need to point the probe at a different host.
    base_url = (
        os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "")
    )
    if not base_url and resolved_urls is not None:
        base_url = resolved_urls[1]

    api_key = (
        os.environ.get("SAFE_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )

    catalog_ids = _probe_llm_catalog(base_url=base_url, api_key=api_key)
    if catalog_ids is None:
        print(
            "ERROR: gateway catalog unreachable after retries; cannot "
            "verify Claude model availability. Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    if chosen in catalog_ids:
        print(f"Preflight: Claude model {chosen!r} confirmed in gateway catalog")
        return catalog_ids

    if _CLAUDE_FALLBACK_MODEL in catalog_ids:
        print(
            f"Preflight: WARNING — {chosen!r} not in gateway catalog; "
            f"falling back to {_CLAUDE_FALLBACK_MODEL!r}"
        )
        args.claude_model = _CLAUDE_FALLBACK_MODEL
        return catalog_ids

    print(
        f"ERROR: neither {_CLAUDE_PREFERRED_MODEL!r} nor "
        f"{_CLAUDE_FALLBACK_MODEL!r} present in gateway catalog "
        f"(catalog has {sorted(m for m in catalog_ids if m.startswith('claude-'))}). "
        f"Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


def _smoke_test_codex_model(
    args: argparse.Namespace,
    catalog_ids: set[str] | None,
) -> None:
    """WARN-only catalog check for ``--codex-model`` (no hard gate).

    Operator pinned only Claude; Codex is allowed to use whatever is
    requested. We still want to flag obvious typos / missing models BEFORE
    the Coordinator starts ticking, since the Codex backend's
    ``__post_init__`` does not pre-validate against the catalog.
    """
    if catalog_ids is None:
        return
    # Codex is needed by the Kernel agent (when kernel-codex is on) and by
    # the critic-agent path (which calls Codex for review reasoning).
    critic_uses_codex = args.critic_backend == "agent"
    needs_codex = critic_uses_codex or (
        args.kernel_codex and not getattr(args, "no_kernel", False)
    )
    if not needs_codex:
        return
    chosen = (args.codex_model or "").strip()
    if chosen in catalog_ids:
        print(f"Preflight: Codex model {chosen!r} confirmed in gateway catalog")
        return
    print(
        f"Preflight: WARNING — codex model {chosen!r} not in gateway catalog "
        f"({sorted(m for m in catalog_ids if m.startswith('gpt-'))}); "
        f"CodexBackend will fail at first turn. Pass --codex-model with a "
        f"value in the catalog or use --critic-mock / --kernel-claude to "
        f"avoid the Codex path entirely."
    )


def _preflight(
    args: argparse.Namespace | None = None,
) -> tuple[str, str] | None:
    """Auto-install missing runtime deps and export auth aliases.

    1. Credentials fallback: env > $REPO_ROOT/.env (env always wins).
    2. Auth aliases for Claude/Codex CLIs from SAFE_API_KEY/OPENAI_BASE_URL.
    3. Python SDK (claude-agent-sdk / openai / httpx) auto-install.
    4. Resolve ANTHROPIC_BASE_URL from OPENAI_BASE_URL (strip trailing /v1)
       and reset ``~/.claude/config.json`` ``customApiUrl`` to the upstream
       gateway. The auth-proxy on :4002 has been retired — the AMD
       primus-safe gateway accepts both ``x-api-key`` and Bearer auth so
       the proxy rewrite step is no longer needed.
    5. ROCm env hygiene (HIP_VISIBLE_DEVICES unset, GPU/shm sanity).
    6. ray + Magpie + InferenceX auto-install.
    7. node / claude / codex CLI presence check (WARN-only).
    8. Single canonical diagnostics block.

    Returns ``(anthropic_base_url, openai_base_url)`` — the resolved
    upstream URLs that ``_run_optimize`` uses for the catalog probe and
    diagnostics. ``None`` only when ``OPENAI_BASE_URL`` is missing.
    """
    _load_dotenv_fallback()
    _load_kernel_agent_env_fallback()

    # Fail fast when credentials are missing. Must run after the fallback
    # loaders (so .env / $USER_DATA_PATH/runtime/kernel-agent.env.sh have
    # had their chance) but before any code that would otherwise burn
    # cycles on auth-alias propagation, SDK install, ROCm probing, and
    # the upstream catalog probe.
    _validate_credentials()

    # --- Auth alias export ---
    safe_key = os.environ.get("SAFE_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if safe_key:
        for alias in ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                      "ANTHROPIC_API_KEY", "OOB_API_KEY", "GEAK_API_KEY",
                      "LLM_API_KEY", "AMD_LLM_API_KEY"):
            if os.environ.get(alias) != safe_key:
                os.environ[alias] = safe_key
                print(f"Preflight: refreshed {alias} from SAFE_API_KEY")
    # OOB / GEAK / LLM_API_BASE inherit the upstream URL verbatim.
    if base_url:
        for alias in ("OOB_BASE_URL", "GEAK_BASE_URL", "LLM_API_BASE"):
            if os.environ.get(alias) != base_url:
                prev = os.environ.get(alias, "")
                os.environ[alias] = base_url
                print(
                    f"Preflight: {alias} {prev or '<unset>'} -> {base_url} "
                    f"(direct to gateway)"
                )

    # --- Resolve install interpreters ---
    from .orchestrator.action_executors._grid_runner import _resolve_magpie_python
    magpie_python = _resolve_magpie_python()

    # Detect whether we're in a venv; if not, add --break-system-packages
    # so pip doesn't refuse to install on bare-metal Debian/Ubuntu hosts.
    pip_extra: list[str] = []
    if not (hasattr(sys, "real_prefix") or
            (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)):
        pip_extra = ["--break-system-packages"]

    # --- Python SDK auto-install (claude-agent-sdk / openai / httpx) ---
    # Must happen BEFORE Coordinator import, since ClaudeBackend lazy-imports
    # claude_agent_sdk at construction time and the catalog probe later in
    # this preflight needs httpx. Use sys.executable so the package lands
    # in the same site-packages this process imports from.
    _ensure_python_sdks(sys.executable, pip_extra)

    # --- Resolve ANTHROPIC_BASE_URL + reset ~/.claude/config.json ---
    # The auth-proxy on :4002 has been removed. We force-override both URL
    # vars to keep them consistent and prevent stale shell/.env/k8s values
    # (e.g. a legacy 127.0.0.1:4002 leftover) from reaching the CLIs.
    resolved_urls: tuple[str, str] | None = None
    if base_url:
        anthropic_url = _derive_anthropic_base_url(base_url)
        orig_anthropic = os.environ.get("ANTHROPIC_BASE_URL", "")
        orig_openai = os.environ.get("OPENAI_BASE_URL", "")
        for var, want, prev in (
            ("ANTHROPIC_BASE_URL", anthropic_url, orig_anthropic),
            ("OPENAI_BASE_URL", base_url, orig_openai),
        ):
            if os.environ.get(var) != want:
                os.environ[var] = want
                print(
                    f"Preflight: {var} {prev or '<unset>'} -> {want} "
                    f"(direct to gateway)"
                )
        _reset_claude_config_to_upstream(safe_key, anthropic_url)
        resolved_urls = (anthropic_url, base_url)
    else:
        print(
            "Preflight: WARNING — OPENAI_BASE_URL unset; "
            "Claude/Codex SDKs will fail at first call"
        )

    # --- ROCm env hygiene + GPU/shm sanity (defensive WARN-only) ---
    _unset_hip_visible_devices()
    _check_gpu_visibility()
    _check_shm_disk()

    # --- Runtime dep install ---
    # 1. Ray — needed by Magpie for task scheduling even without kernel-agent.
    if shutil.which("ray") is None:
        print("Preflight: ray not found, installing ray[default]==2.44.1 + click<8.3.0 ...")
        subprocess.run(
            [magpie_python, "-m", "pip", "install", "--quiet",
             *pip_extra, "ray[default]==2.44.1", "click<8.3.0"],
            check=True,
        )
        print("Preflight: ray installed OK")

    # 2. Magpie — the benchmark engine all executors shell out to.
    # ``$MAGPIE_DIR`` is the operator override; install.sh defaults it
    # to ``$HYPERLOOM_RUNTIME_DIR/Magpie`` (= ``$USER_DATA_PATH/runtime/
    # Magpie``) so a missing Magpie auto-clones into the session tree.
    # Falls back to legacy ``/workspace/Magpie`` for environments still
    # on pre-migration launchers.
    check = subprocess.run(
        [magpie_python, "-c", "import Magpie"],
        capture_output=True,
    )
    if check.returncode != 0:
        magpie_env = os.environ.get("MAGPIE_DIR")
        magpie_env_explicit = bool(magpie_env)
        if magpie_env:
            magpie_dir = Path(magpie_env)
        else:
            from .paths import magpie_dir as _magpie_default
            magpie_dir = _magpie_default(_session_dir_resolve())
        magpie_dir.parent.mkdir(parents=True, exist_ok=True)
        if not (magpie_dir / "setup.py").exists() and not (magpie_dir / "pyproject.toml").exists():
            # Refuse-to-clobber guard: when $MAGPIE_DIR was set explicitly by
            # the operator (e.g. a wekafs fork with un-pushed local commits),
            # cloning Magpie main on top would silently destroy local work.
            # Auto-clone only happens for the default path resolved from
            # `paths.magpie_dir(session_dir)`.
            if magpie_env_explicit:
                print(
                    f"Preflight: ERROR — $MAGPIE_DIR={magpie_dir} has no "
                    f"setup.py/pyproject.toml; refusing to clone Magpie "
                    f"main on top of an operator-supplied path. Fix the "
                    f"env or unset $MAGPIE_DIR to fall back to the "
                    f"session-default location.",
                    file=sys.stderr,
                )
                raise FileNotFoundError(
                    f"$MAGPIE_DIR={magpie_dir} is not a valid Magpie checkout"
                )
            print(f"Preflight: Magpie not importable and not found at {magpie_dir}; cloning ...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/AMD-AGI/Magpie.git", str(magpie_dir)],
                check=True,
            )
        print(f"Preflight: installing Magpie from {magpie_dir} ...")
        subprocess.run(
            [magpie_python, "-m", "pip", "install", "--quiet",
             *pip_extra, "-e", str(magpie_dir)],
            check=True,
        )
        print("Preflight: Magpie installed OK")

    # 3. InferenceX + lm-eval — required for accuracy evaluation (GSM8K).
    # Magpie's benchmark scripts call `run_eval` which sources
    # InferenceX/benchmarks/benchmark_lib.sh → _install_lm_eval_deps.
    # We just ensure the InferenceX checkout exists; lm-eval deps are
    # auto-installed by benchmark_lib.sh at runtime.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "")
    if not inferencex_path:
        from .paths import (
            magpie_dir as _magpie_default,
            runtime_dir as _runtime_default,
        )
        runtime_root = _runtime_default(_session_dir_resolve())
        magpie_root = (
            Path(os.environ["MAGPIE_DIR"])
            if os.environ.get("MAGPIE_DIR")
            else _magpie_default(_session_dir_resolve())
        )
        # InferenceX detection order: Magpie's own InferenceX submodule
        # first (the canonical layout after install.sh), then the
        # standalone runtime checkout, then legacy host-level mounts so
        # existing pre-migration pods keep working.
        for candidate in (
            magpie_root / "InferenceX",
            runtime_root / "InferenceX",
            Path("/wekafs/hyperloom/InferenceX"),
            Path("/opt/hyperloom/InferenceX"),
            Path("/wekafs/fully-local/inference_optimization/InferenceX"),
        ):
            if candidate.is_dir():
                inferencex_path = str(candidate)
                break
    if inferencex_path and Path(inferencex_path).is_dir():
        os.environ.setdefault("INFERENCEX_PATH", inferencex_path)
    else:
        print("Preflight: WARNING — InferenceX not found. GSM8K accuracy "
              "eval will fail. Set INFERENCEX_PATH or clone Magpie with "
              "InferenceX submodule.")

    # --- node / claude / codex CLI presence (WARN-only) ---
    _check_node_claude_cli()

    # --- TraceLens CLI presence (HARD-FAIL; SKILL Step 2 step 8.5) ---
    # Catches brain-generated launchers that source only env.sh and skip
    # install.sh — without this gate the missing-CLI symptom would not
    # surface until the robustness probe fires at tick ~6, after baseline.
    _check_tracelens_cli()

    # --- IR-3: Cortex KB + PR Monitor reachability (soft degrade) ---
    if args is not None:
        _run_ir3_preflight(args)

    # --- Single canonical diagnostics block ---
    _emit_preflight_diagnostics(
        magpie_python=magpie_python,
        anthropic_base_url=(resolved_urls[0] if resolved_urls is not None else None),
        args=args,
    )

    return resolved_urls


def _run_ir3_preflight(args: argparse.Namespace) -> None:
    """IR-3 — Cortex KB + PR Monitor reachability probe (soft degrade).

    Mutates ``args``:
    * ``cortex_enabled`` / ``pr_monitor_enabled`` (bool) — final
      enablement after applying explicit flag + IR-3 marker.
    * ``kb_degraded_reason`` / ``pr_degraded_reason`` —
      ``None | "explicit_flag" | "ir3_auto"``.

    Never raises and never ``sys.exit``s — KB/PR outages downgrade to
    NDJSON fallback / specialist no-op, not an aborted launch.
    """
    explicit_kb = bool(getattr(args, "degraded_kb", False))
    explicit_pr = bool(getattr(args, "degraded_pr", False))

    args.cortex_enabled = True
    args.pr_monitor_enabled = True
    args.kb_degraded_reason = None
    args.pr_degraded_reason = None

    if explicit_kb and explicit_pr:
        args.cortex_enabled = False
        args.kb_degraded_reason = "explicit_flag"
        args.pr_monitor_enabled = False
        args.pr_degraded_reason = "explicit_flag"
        return

    user_data = Path(os.environ.get("USER_DATA_PATH", "/workspace/hyperloom"))
    marker_path = user_data / "runtime" / "cortex" / ".kb_preflight.json"
    script = (
        Path(__file__).resolve().parent / "scripts" / "preflight_kb.sh"
    )
    env = os.environ.copy()
    # Only probe a remote KB the operator explicitly configured. A
    # ``--cortex-kb-url`` flag isn't visible to the probe script via
    # the environment unless we inject it here; with neither flag nor
    # ``$CORTEX_KB_URL`` set the script sees an empty URL and skips the
    # KB branch (local-only — there is no hard-coded default to probe).
    cortex_url = (getattr(args, "cortex_kb_url", None) or "").strip()
    if cortex_url:
        env["CORTEX_KB_URL"] = cortex_url
    if explicit_kb:
        env["SKIP_KB_PROBE"] = "1"
    if explicit_pr:
        env["SKIP_PR_PROBE"] = "1"

    try:
        subprocess.run(
            ["bash", str(script)], env=env,
            check=False, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        # The script itself died — treat both branches as unreachable
        # so soft-degrade kicks in (unless explicit flag overrides).
        log.warning("IR-3 preflight script error: %s", exc)
        marker: dict[str, Any] = {
            "kb_reachable": False, "pr_reachable": False,
            "kb_skipped": explicit_kb, "pr_skipped": explicit_pr,
        }
    else:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("IR-3 marker unreadable: %s", exc)
            marker = {
                "kb_reachable": False, "pr_reachable": False,
                "kb_skipped": explicit_kb, "pr_skipped": explicit_pr,
            }

    if explicit_kb:
        args.cortex_enabled = False
        args.kb_degraded_reason = "explicit_flag"
    elif not marker.get("kb_reachable", False) and not marker.get("kb_skipped", False):
        args.cortex_enabled = False
        args.kb_degraded_reason = "ir3_auto"

    if explicit_pr:
        args.pr_monitor_enabled = False
        args.pr_degraded_reason = "explicit_flag"
    elif not marker.get("pr_reachable", False) and not marker.get("pr_skipped", False):
        args.pr_monitor_enabled = False
        args.pr_degraded_reason = "ir3_auto"


# ---------------------------------------------------------------------------
# Default critic backend. Step D of the critic-agent integration plan flipped
# this from "mock" to "agent" once CriticAgentBackend is wired and tested.
# Override via env (operator runbook) or the --critic-mock /
# --critic-agent flags. Step C tests pin a specific value via the CLI flag
# so they're insulated from default drift.
DEFAULT_CRITIC_BACKEND = os.environ.get(
    "INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND", "agent",
)
_VALID_CRITIC_BACKENDS = ("mock", "agent")


def _resolve_critic_choice(args: argparse.Namespace) -> str:
    """Resolve the active critic backend choice.

    ``args.critic_backend`` is set by the matching CLI flag (or ``None`` if
    the operator passed nothing), in which case we fall back to
    :data:`DEFAULT_CRITIC_BACKEND`. Hard-fails on an invalid value.
    """
    chosen = args.critic_backend
    if chosen is None:
        chosen = DEFAULT_CRITIC_BACKEND
    if chosen not in _VALID_CRITIC_BACKENDS:
        print(
            f"ERROR: critic backend {chosen!r} not in {_VALID_CRITIC_BACKENDS!r} "
            f"(set by --critic-mock / --critic-agent or "
            f"INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND)",
            file=sys.stderr,
        )
        sys.exit(2)
    return chosen


# Default robustness backend. Production runs use the real subprocess
# transport; operators can still force the heartbeat-only mock with
# --robustness-mock or the env override below.
DEFAULT_ROBUSTNESS_BACKEND = os.environ.get(
    "INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND", "agent",
)
_VALID_ROBUSTNESS_BACKENDS = ("mock", "agent")


def _resolve_robustness_choice(args: argparse.Namespace) -> str:
    """Resolve the active robustness backend choice.

    Mirrors :func:`_resolve_critic_choice`. Operator picks via
    ``--robustness-mock`` / ``--robustness-agent`` (sets
    ``args.robustness_backend``); ``None`` falls back to
    :data:`DEFAULT_ROBUSTNESS_BACKEND`. Hard-fails on an invalid value.

    Multi-node policy: when ``args.nodes >= 2`` the robustness-agent's
    ``LocalProbeSource`` family targets sandbox-local resources only
    (``ray status``, the inference server health URL, GPU / FD / disk /
    shm metrics, ...). On multi-node every one of those lives in a
    separate Kubernetes pod, unreachable from the sandbox by design, so
    each probe failure surfaces as a HIGH-severity false positive
    (``ray_head_dead``, ``local_server_unreachable``, ...).

    The fix is to source signals from robustness-server instead of the
    local sandbox. So on multi-node:

    * if a robustness-server is configured (``--robustness-server-url``
      or ``ROBUSTNESS_SERVER_URL``) we keep the ``agent`` backend — it
      runs with ``disable_local_probe`` / ``enable_cluster_pod_metrics``
      (see :func:`_build_robustness_options`) and the cluster source
      replaces the sandbox-local probes;
    * if no server is configured the agent has no cluster-wide view and
      would fall back to the noisy LocalProbe, so we auto-downgrade to
      the heartbeat-only ``mock``.

    Operators who explicitly pass ``--robustness-agent`` without a
    server on multi-node get a WARNING; passing ``--robustness-mock``
    suppresses it. Single-node behaviour is unchanged.
    """
    chosen = getattr(args, "robustness_backend", None)
    explicit = chosen is not None
    if chosen is None:
        chosen = DEFAULT_ROBUSTNESS_BACKEND
    if chosen not in _VALID_ROBUSTNESS_BACKENDS:
        print(
            f"ERROR: robustness backend {chosen!r} not in "
            f"{_VALID_ROBUSTNESS_BACKENDS!r} (set by --robustness-mock / "
            f"--robustness-agent or "
            f"INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND)",
            file=sys.stderr,
        )
        sys.exit(2)
    nodes = int(getattr(args, "nodes", 1) or 1)
    if nodes >= 2 and chosen == "agent" and not _robustness_server_configured(args):
        if explicit:
            print(
                f"WARN: --robustness-agent selected but nodes={nodes} and "
                f"no robustness-server configured — the agent's LocalProbe "
                f"family targets sandbox-local resources (ray, inference "
                f"server, GPU, ...) that all live in separate pods on "
                f"multi-node and surface as HIGH false positives. "
                f"Auto-downgrading to --robustness-mock; configure "
                f"--robustness-server-url / ROBUSTNESS_SERVER_URL to keep "
                f"the agent backend, or pass --robustness-mock explicitly "
                f"to suppress this warning. See "
                f"inference_optimizer/multi_node/SKILL.md "
                f"(Robustness limitation in multi-node mode).",
                file=sys.stderr,
            )
        chosen = "mock"
    return chosen


def _reset_state_file(session_dir: Path) -> None:
    """Back up the existing ``state.json`` and start fresh.

    The backup name is ``state.json.preReset.<unix_ts>`` so multiple
    resets in the same session_dir don't clobber each other. Symbolic
    of the operator's nuclear option:
    the Cortex KB cross-session knowledge is *not* touched here — only
    the per-session fact-layer is reset. ``Coordinator`` will reseed
    its dataclass defaults on the next ``load_or_init`` call.
    """
    state_path = session_dir / "state.json"
    if not state_path.exists():
        return
    import time as _time
    ts = int(_time.time())
    backup_path = session_dir / f"state.json.preReset.{ts}"
    try:
        state_path.replace(backup_path)
    except OSError as exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "v0.8 §3.10 --reset-state: could not move %s → %s: %s",
            state_path, backup_path, exc,
        )
        return
    import logging as _logging
    _logging.getLogger(__name__).info(
        "v0.8 §3.10 --reset-state: backed up state.json to %s; "
        "session starts blank.", backup_path.name,
    )


def _gc_old_profile_traces(
    root: str | None = None,
    retention_days: int = 7,
    keep: str | None = None,
) -> None:
    """Best-effort GC of stale per-RayJob profile-trace dirs on the shared FS.

    Removes only top-level subdirectories whose mtime is older than
    ``retention_days``. The active session's dir (just mkdir'd seconds ago)
    is always young enough; ``keep`` adds an explicit name-match guard so
    the current run is never collected even if a clock skew flipped mtime.

    ``root`` defaults to :func:`mn_profile_trace_root` (anchored on
    ``$USER_DATA_PATH``); callers may pass an explicit override for
    tests or migration scenarios.

    Failure is logged + swallowed: GC must never block optimizer startup.

    Override knobs (env):
      HYPERLOOM_MN_TRACE_RETENTION_DAYS -- int days, default 7
      HYPERLOOM_MN_TRACE_GC_DISABLE     -- "1" disables GC entirely
    """
    if os.environ.get("HYPERLOOM_MN_TRACE_GC_DISABLE", "").strip() in (
        "1", "true", "yes",
    ):
        return
    try:
        retention_days = int(
            os.environ.get("HYPERLOOM_MN_TRACE_RETENTION_DAYS") or retention_days
        )
    except ValueError:
        retention_days = 7
    base = Path(root) if root is not None else mn_profile_trace_root()
    if not base.is_dir():
        return
    cutoff = time.time() - retention_days * 86400
    keep_name = Path(keep).name if keep else ""
    removed = 0
    kept = 0
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if keep_name and child.name == keep_name:
                kept += 1
                continue
            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue
            if mtime >= cutoff:
                kept += 1
                continue
            try:
                shutil.rmtree(child)
                removed += 1
            except OSError as exc:
                print(
                    f"WARN multi-node GC: failed to rm {child}: {exc}",
                    file=sys.stderr,
                )
    except OSError as exc:
        print(f"WARN multi-node GC: scan failed under {base}: {exc}", file=sys.stderr)
        return
    if removed or kept:
        print(
            f"multi-node: GC profile-traces removed={removed} kept={kept} "
            f"retention={retention_days}d root={base}"
        )


def _provision_multi_node_rayjob_stack(args: argparse.Namespace) -> None:
    """When ``--nodes >= 2``, create/reuse SaFE RayJob, bootstrap once, export RAY_ADDRESS."""
    nodes = max(1, int(args.nodes))
    if nodes < 2:
        return

    from .multi_node.cli import cmd_bootstrap, cmd_create_rayjob, _load_state
    from .orchestrator.action_executors._multi_node_env import export_ray_address_to_os

    state_path = Path(os.environ.get("MULTI_NODE_STATE_FILE", "/tmp/multi_node_state.json"))
    image = (
        (getattr(args, "rayjob_image", None) or "").strip()
        or os.environ.get("INFERENCE_OPTIMIZER_RAYJOB_IMAGE", "").strip()
    )
    if not image and state_path.is_file():
        try:
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            image = str((prior.get("last_create_request") or {}).get("image") or "").strip()
        except (OSError, json.JSONDecodeError, TypeError):
            image = ""
    if not image:
        print(
            "ERROR: --nodes >= 2 requires a RayJob container image. Pass "
            "--rayjob-image <harbor/...> or set INFERENCE_OPTIMIZER_RAYJOB_IMAGE.",
            file=sys.stderr,
        )
        sys.exit(2)

    gpn = getattr(args, "rayjob_gpus_per_node", None)
    if gpn is None:
        try:
            gpn = int(os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8)
        except ValueError:
            gpn = 8

    # Forward agent-supplied prompt env (verbatim, no defaults). When the
    # state file already has a non-terminal rayjob_id, cmd_create_rayjob
    # reuses it and skips POST CreateWorkload -- so passing extra_env on
    # a reuse call is a no-op (the original create-time env stays in
    # effect). To inject env into an existing RayJob the caller must
    # `stop-rayjob --clear-state` first or invoke create-rayjob with
    # --recreate. See multi_node/SKILL.md.
    rayjob_extra_env = list(getattr(args, "rayjob_extra_env", None) or [])

    ns_create = argparse.Namespace(
        workspace=None,
        image=image,
        nodes=nodes,
        gpus_per_node=int(gpn),
        cpus_per_node=96,
        mem_per_node=1024,
        ephemeral_per_node=400,
        display_name=None,
        description=None,
        owner_id=None,
        extra_env=rayjob_extra_env,
        extra_label=[],
        no_wait=False,
        recreate=False,
        poll_interval=6,
        poll_timeout=int(
            os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "110") or 110
        ),
    )
    rc = cmd_create_rayjob(ns_create)
    if rc != 0:
        sys.exit(rc)

    state = _load_state()
    if not state.get("last_bootstrap_submission_id"):
        ns_boot = argparse.Namespace(
            script=None,
            force=False,
            print_logs=False,
            poll_interval=6,
            poll_timeout=int(
                os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "110") or 110
            ),
        )
        rc_boot = cmd_bootstrap(ns_boot)
        if rc_boot != 0:
            sys.exit(rc_boot)

    export_ray_address_to_os()
    ra = os.environ.get("RAY_ADDRESS", "")
    if ra:
        print(f"multi-node: exported RAY_ADDRESS={ra} for kernel-agent Ray tasks")

    # Multi-node only: server-side sglang/vllm pods (head + worker) must
    # write torch traces to a path that the sandbox-side profile_executor
    # can read. Per-pod /tmp is invisible to the sandbox; wekafs is the
    # only fs both sides mount. Namespace the dir by ``rayjob_id`` so a
    # restart of the same RayJob reuses the directory and a new RayJob
    # gets a fresh one. This env is consumed by:
    #   * multi_node.cli._build_multinode_launch_entrypoint -> --torch-profiler-dir
    #   * orchestrator.action_executors.profile.ProfileExecutor.__call__
    state_after = _load_state()
    rid = (state_after.get("rayjob_id") or "").strip()
    if rid:
        # Anchor torch-profile shared root on $USER_DATA_PATH so the
        # operator only has to point one knob at a cluster-shared
        # filesystem (e.g. wekafs); the sandbox and the RayJob pods then
        # both see traces under the same path. See
        # ``inference_optimizer.paths.mn_profile_trace_root`` docstring
        # for the multi-node USER_DATA_PATH caveat.
        trace_root_path = mn_profile_trace_root() / rid / "torch_trace"
        trace_root = str(trace_root_path)
        try:
            trace_root_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                f"WARN multi-node: cannot mkdir {trace_root}: {exc}; "
                f"server traces will fall back to per-pod /tmp",
                file=sys.stderr,
            )
        else:
            os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"] = trace_root
            print(
                f"multi-node: exported HYPERLOOM_MN_PROFILE_TRACE_DIR={trace_root}"
            )
            # Best-effort GC of older sibling RayJob trace dirs. Runs only
            # AFTER the current session's dir is mkdir'd, so the active
            # rayjob_id is name-guarded and its mtime is fresh.
            _gc_old_profile_traces(keep=rid)

    # RayJob recreate path: when an existing session's RayJob was killed
    # (OOM, manual recreate, SaFE rescheduling) and we provisioned a
    # fresh one above, any kernel-agent patches previously applied to
    # the OLD pods are gone from the new pods' filesystems (the pod
    # local fs is destroyed with the old pod). The sandbox-side
    # SharedState.optimization_stack still records every promoted patch
    # though, so we replay them here before any executor's first
    # restart_server_for_round. Best-effort; failures degrade to
    # warnings — the orchestrator will notice missing speedups in the
    # next baseline and re-run kernel-agent on the affected kernels.
    _replay_kernel_patches_for_multi_node(args)


def _replay_kernel_patches_for_multi_node(args: argparse.Namespace) -> None:
    """Replay every applied kernel-agent patch onto the (possibly new)
    RayJob pods.

    Scans the session's ``kernel-agent-workspace`` tree for manifest
    files with ``status="applied"`` and a ``multinode`` block, then
    invokes ``python3 -m inference_optimizer.multi_node apply-patch``
    once per patch. The fan-out is idempotent — re-running apply on
    pods that already have the new file produces a fresh backup of
    the (now-current) source and overwrites with the same bytes.

    Run only when ``--nodes >= 2``. Best-effort: stdout/stderr from
    each replay is mirrored verbatim; per-patch failures emit a
    warning but do not raise so a single broken patch doesn't block
    other replays or the subsequent optimize loop.
    """
    nodes = max(1, int(getattr(args, "nodes", 1) or 1))
    if nodes < 2:
        return
    session_dir = _session_dir_resolve()
    workspace_root = session_dir / "kernel-agent-workspace"
    if not workspace_root.is_dir():
        return

    manifests: list[Path] = sorted(workspace_root.rglob("manifest.json"))
    if not manifests:
        return

    replayed = 0
    skipped = 0
    failed = 0
    for mpath in manifests:
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"WARN multi-node patch replay: skipping unreadable "
                f"manifest {mpath}: {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        if str(data.get("status", "")).lower() != "applied":
            continue
        mn = data.get("multinode") or {}
        if not mn:
            continue
        target_file = data.get("target_file") or ""
        patch_path = data.get("patch_path") or ""
        kernel_id = data.get("kernel_id") or ""
        backup_dir_on_pod = mn.get("backup_dir_on_pod") or "/var/kernel_patch_backups"
        if not target_file or not patch_path:
            skipped += 1
            continue
        if not Path(patch_path).is_file():
            print(
                f"WARN multi-node patch replay: source patch missing for "
                f"{target_file} (manifest={mpath} patch_path={patch_path}); "
                f"skipping",
                file=sys.stderr,
            )
            skipped += 1
            continue
        cmd = [
            sys.executable, "-m", "inference_optimizer.multi_node",
            "apply-patch",
            "--patch-file", str(patch_path),
            "--target-path", str(target_file),
            "--backup-dir", str(backup_dir_on_pod),
            "--kernel-id", str(kernel_id),
        ]
        print(
            f"multi-node patch replay: target={target_file} kernel_id={kernel_id!r} "
            f"(from {mpath})",
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            failed += 1
            print(
                f"WARN multi-node patch replay failed for {target_file} "
                f"rc={proc.returncode} stderr={(proc.stderr or '')[-1000:]!r}",
                file=sys.stderr,
            )
            continue
        replayed += 1
    if replayed or failed or skipped:
        print(
            f"multi-node patch replay: applied={replayed} "
            f"skipped={skipped} failed={failed} "
            f"(scanned {len(manifests)} manifest(s) under {workspace_root})"
        )


def _argv_has_option(argv: list[str], option: str) -> bool:
    """Return True when argv explicitly carries ``option``."""
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in argv)


def _export_workload_envs_for_optimize(
    args: argparse.Namespace,
    *,
    nodes_resolved: int,
    tp_resolved: int,
    ep_resolved: int,
    argv: list[str] | None = None,
) -> None:
    """Mirror explicit workload CLI flags into env for executors.

    Single-node runs normally preserve YAML workload defaults. When the
    operator explicitly passes ``--tp`` / ``--conc`` / ``--ep``, those values
    must still win because the action executors read these knobs from
    ``os.environ`` while materializing Magpie YAMLs.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if nodes_resolved >= 2 or _argv_has_option(argv, "--tp"):
        os.environ["TP"] = str(tp_resolved)
    if nodes_resolved >= 2 or _argv_has_option(argv, "--conc"):
        os.environ["CONC"] = str(max(1, int(getattr(args, "conc", 8) or 8)))
    if nodes_resolved >= 2 or _argv_has_option(argv, "--ep"):
        os.environ["EP"] = str(ep_resolved)


async def _run_optimize(args: argparse.Namespace) -> int:
    # Surface --nodes to the rest of the process (preflight diagnostics
    # and any executor that wants to short-circuit when the optimizer is
    # in single-node mode) by exporting it before _preflight runs. We
    # re-export even when the env var was already set so the CLI flag
    # always wins, matching the documented resolution order.
    nodes_resolved = max(1, int(args.nodes))
    tp_resolved = max(1, int(getattr(args, "tp", 1) or 1))
    ep_resolved = max(1, int(getattr(args, "ep", 1) or 1))
    # Resolve gpus_per_node with the same priority chain
    # `_provision_multi_node_rayjob_stack` uses (CLI > env > 8) so the
    # validation here matches what SaFE actually ends up seeing.
    gpn_attr = getattr(args, "rayjob_gpus_per_node", None)
    if gpn_attr is not None:
        gpus_per_node_resolved = int(gpn_attr)
    else:
        try:
            gpus_per_node_resolved = int(
                os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8,
            )
        except ValueError:
            gpus_per_node_resolved = 8
    total_gpus = nodes_resolved * gpus_per_node_resolved

    # Topology sanity gates — multi-node ONLY (nodes >= 2). Single-pod
    # runs keep their legacy code paths untouched (Magpie owns the
    # server lifecycle there, and the TP/EP knobs flow through env
    # rather than the multi_node CLI). Rejected at CLI parse time so
    # the agent gets an immediate, attributable error instead of a
    # cryptic sglang/vllm launcher crash 30 minutes into a cold start.
    if nodes_resolved >= 2:
        # Gate 1: total cluster GPUs must hold the model's TP shards.
        #   nodes * gpus_per_node >= tp
        # Anything less means at least one TP rank has no GPU to land on.
        if total_gpus < tp_resolved:
            print(
                f"ERROR: TP={tp_resolved} exceeds total GPU count "
                f"({nodes_resolved} nodes * {gpus_per_node_resolved} "
                f"gpus_per_node = {total_gpus}). Either lower --tp, raise "
                "--nodes, or use a larger --rayjob-gpus-per-node pod "
                "template.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Gate 2: expert-parallel size cannot exceed TP — sglang/vllm
        # cannot place more expert shards than ranks. Helper repeats
        # this check at server-restart time, but failing here keeps
        # the agent from spending bootstrap minutes on an unrunnable
        # topology.
        if ep_resolved > tp_resolved:
            print(
                f"ERROR: EP={ep_resolved} > TP={tp_resolved}. Expert-parallel "
                "size must be <= tensor-parallel size. Either lower --ep or "
                "raise --tp.",
                file=sys.stderr,
            )
            sys.exit(2)

    os.environ["INFERENCE_OPTIMIZER_NODES"] = str(nodes_resolved)
    # Re-export $TP / $CONC / $EP from resolved CLI args when the user
    # explicitly supplied them, plus always for multi-node where child
    # workers need the values. Avoid exporting argparse defaults in
    # single-node mode because YAML workload defaults remain supported.
    _export_workload_envs_for_optimize(
        args,
        nodes_resolved=nodes_resolved,
        tp_resolved=tp_resolved,
        ep_resolved=ep_resolved,
    )
    # User-declared grid skip list. Resolution order is already enforced
    # by argparse default (--skip-variants > $SKIP_VARIANTS); we re-export
    # so executors started later via subprocess (multi-node orchestrator,
    # sweep child workers) inherit the same spec without re-parsing argv.
    # Empty string is intentional: it clears any stale value left by a
    # prior session in the same shell.
    skip_variants_resolved = (getattr(args, "skip_variants", "") or "").strip()
    os.environ["SKIP_VARIANTS"] = skip_variants_resolved
    # Surface PD_* knobs the same way for executors / helper. Empty
    # string means "let helper resolve from state.json"; pd_mode
    # always exported so colocated runs explicitly clear any stale
    # PD_MODE the operator may have left set.
    pd_mode = (getattr(args, "pd_mode", "") or "colocated").lower()
    # nodes_resolved is already computed above (TP/EP gates).
    if pd_mode == "disaggregated" and nodes_resolved < 2:
        # PD disaggregation logically requires at least one prefill pod
        # and one decode pod; with NODES=1 that would mean co-hosting
        # both server roles on the same GPU set, which defeats the
        # latency-isolation purpose and is not supported by sglang/vllm
        # in the current architecture (also rejected later by
        # _resolve_pd_args, but we fail at CLI parse so the agent gets
        # an immediate, attributable error).
        print(
            f"ERROR: --pd-mode disaggregated requires --nodes >= 2 "
            f"(got --nodes {nodes_resolved}). PD splits the cluster "
            "into prefill + decode groups; a single pod cannot host "
            "both. Either drop --pd-mode (defaults to colocated) or "
            "raise --nodes.",
            file=sys.stderr,
        )
        sys.exit(2)
    os.environ["PD_MODE"] = pd_mode
    if pd_mode == "disaggregated":
        for cli_attr, env_key in (
            ("pd_prefill_nodes", "PD_PREFILL_NODES"),
            ("pd_decode_nodes", "PD_DECODE_NODES"),
            ("pd_prefill_tp", "PD_PREFILL_TP"),
            ("pd_decode_tp", "PD_DECODE_TP"),
        ):
            v = int(getattr(args, cli_attr, 0) or 0)
            if v > 0:
                os.environ[env_key] = str(v)
        for cli_attr, env_key in (
            ("pd_transfer_backend", "PD_TRANSFER_BACKEND"),
            ("pd_ib_device", "PD_IB_DEVICE"),
        ):
            v = (getattr(args, cli_attr, "") or "").strip()
            if v:
                os.environ[env_key] = v

    await asyncio.to_thread(_provision_multi_node_rayjob_stack, args)

    # Stale aiter JIT lock sweep. aiter does not implement its
    # inter-process baton with fcntl.flock, so a SIGKILL'd previous run
    # leaves lock files behind that block every subsequent sglang /
    # vllm start with "[aiter] waiting for baton release ..." and a
    # silently idle GPU. This was the root cause of the three failed
    # launches on the Qwen3-30B-A3B mi355x run; running the sweep
    # unconditionally here costs ~milliseconds when there is nothing to
    # delete. Locks younger than 5 minutes are preserved so an in-flight
    # compile in another shell is never disturbed.
    aiter_sweep = _clean_stale_aiter_locks()
    if aiter_sweep["dir"] and aiter_sweep["deleted"]:
        print(
            f"Stale aiter locks cleared: "
            f"dir={aiter_sweep['dir']} "
            f"deleted={aiter_sweep['deleted']} "
            f"skipped_fresh={aiter_sweep['skipped_fresh']} "
            f"errors={aiter_sweep['errors']}"
        )

    resolved_urls = _preflight(args)

    # Hard-gate Claude model BEFORE any session work. Mutates args.claude_model
    # in-place when falling back to opus-4-6; aborts with sys.exit(2) if the
    # gateway catalog cannot be probed or neither allowed model is present.
    catalog_ids = _validate_and_resolve_claude_model(args, resolved_urls)
    _smoke_test_codex_model(args, catalog_ids)

    # `--resume-from <path>` implies `--resume` (operator convenience).
    if args.resume_from and not args.resume:
        args.resume = True

    if args.resume:
        # Resume mode: USER_DATA_PATH stays at workspace level so
        # runtime/ + logs/ resolution doesn't break
        # (`_load_kernel_agent_env_fallback` looks at $USER_DATA_PATH/
        # runtime/kernel-agent.env.sh, which only exists at workspace
        # level after install.sh ran). We then pick the per-session
        # subdir to resume from via either:
        #
        # * --resume-from <path>  : explicit operator choice
        # * --resume alone        : auto-pick LATEST per-session subdir
        #                            under workspace_root/<model>/<ts>/.
        #                            Falls back to workspace_root itself
        #                            (legacy flat layout) when no
        #                            per-session subdir exists.
        #
        # We pin INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR to the picked
        # subdir so every paths.session_dir() call later (and every
        # subprocess that inherits env) resolves consistently.
        from .paths import (
            ENV_CURRENT_SESSION_DIR, find_latest_per_session_dir,
            workspace_root,
        )
        ws = workspace_root()
        if args.resume_from:
            session_dir = Path(args.resume_from).expanduser().resolve()
            try:
                session_dir.relative_to(ws.resolve())
            except ValueError:
                print(
                    f"ERROR: --resume-from {session_dir!r} is not under "
                    f"$USER_DATA_PATH={ws}. Move USER_DATA_PATH to the "
                    f"workspace root (the parent of the per-session subdirs) "
                    f"and pass the per-session subdir via --resume-from.",
                    file=sys.stderr,
                )
                sys.exit(2)
            if not session_dir.is_dir():
                print(
                    f"ERROR: --resume-from {session_dir!r} does not exist.",
                    file=sys.stderr,
                )
                sys.exit(2)
        else:
            picked = find_latest_per_session_dir()
            if picked is not None:
                session_dir = picked
                print(f"  --resume: auto-picked latest per-session subdir")
            else:
                # Legacy flat layout — workspace_root itself is the
                # session_dir. Validate it has manifest.json + state.json
                # the same way the per-session branch does below.
                session_dir = ws
                print(
                    f"  --resume: no per-session subdir found under "
                    f"{ws}/<model>/<ts>/; falling back to flat layout "
                    f"({ws})"
                )
        # Pin so every paths.session_dir() / subprocess inherits the
        # resolved location BEFORE Coordinator/SharedState load.
        os.environ[ENV_CURRENT_SESSION_DIR] = str(session_dir)
        # Make sure per-session skeleton exists (mkdir -p semantics,
        # idempotent; doesn't disturb existing files).
        for sub in __import__(
            "inference_optimizer.paths", fromlist=["_SESSION_SKELETON"]
        )._SESSION_SKELETON:
            (session_dir / sub).mkdir(parents=True, exist_ok=True)

        try:
            manifest = load_manifest(session_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: --resume failed: {exc}", file=sys.stderr)
            sys.exit(2)
        if not (session_dir / "state.json").exists():
            print(
                f"ERROR: --resume failed: {session_dir}/state.json missing "
                f"(manifest exists but Coordinator never wrote SharedState)",
                file=sys.stderr,
            )
            sys.exit(2)
        state = SharedState.load_or_init(session_dir)
        prior_stop = state.stop_reason
        print(f"Resuming session: {session_dir}")
        print(f"  manifest.session_id    : {manifest.get('session_id')}")
        print(f"  prior baseline_tput   : {state.baseline_tput:.1f}")
        print(f"  prior cumul_gain      : {state.cumulative_gain:.2f}%")
        print(f"  prior current_best    : "
              f"{(state.current_best or {}).get('action')}/"
              f"{(state.current_best or {}).get('tput')}")
        print(f"  prior stop_reason     : {prior_stop or '(none)'}")

        # Re-export session-level env vars from persisted state so the
        # executors (baseline / profile / sweep / backends / params) resolve
        # model / framework / gpu_type correctly. Without this, a resume
        # in a fresh shell would fall back to YAML hardcoded defaults,
        # potentially benchmarking the wrong model on the wrong framework.
        if state.model_path:
            os.environ["MODEL_PATH"] = state.model_path
            print(f"  re-exported MODEL_PATH: {state.model_path}")
        if state.framework:
            os.environ["FRAMEWORK"] = state.framework
            print(f"  re-exported FRAMEWORK : {state.framework}")
        if state.gpu_type:
            runner_gpu_type = _gpu_runner_type(state.gpu_type)
            os.environ["TARGET_GPU_TYPE"] = state.gpu_type
            os.environ["GPU_TYPE"] = runner_gpu_type
            print(f"  re-exported GPU_TYPE  : {state.gpu_type}")
            if runner_gpu_type != state.gpu_type:
                print(f"  Magpie runner GPU_TYPE: {runner_gpu_type}")
        # Re-export workload metadata (TP / CONC / ISL / OSL /
        # MAX_MODEL_LEN / PRECISION) from SharedState so a resume in
        # a fresh shell sees the same workload contract baseline ran
        # under. Without this, the resumed shell's executors fall
        # back to YAML defaults (TP=1, CONC=8, ISL=256, OSL=256) and
        # the specialist prompt builder renders TP=1 (which makes
        # comm_specialist self-veto on TP=8 sessions).
        for state_attr, env_name in (
            ("tp", "TP"),
            ("conc", "CONC"),
            ("isl", "ISL"),
            ("osl", "OSL"),
            ("max_model_len", "MAX_MODEL_LEN"),
        ):
            val = getattr(state, state_attr, 0) or 0
            if val:
                os.environ[env_name] = str(val)
                print(f"  re-exported {env_name:<14s}: {val}")
        if state.precision:
            os.environ["PRECISION"] = state.precision
            print(f"  re-exported PRECISION     : {state.precision}")
        if getattr(state, "framework_version", ""):
            os.environ["FRAMEWORK_VERSION"] = state.framework_version
            print(
                f"  re-exported FRAMEWORK_VERSION: "
                f"{state.framework_version}"
            )
        # Honour persisted kernel_enabled flag on resume; CLI --no-kernel
        # can still override on a previously-enabled session.
        if not state.kernel_enabled:
            args.no_kernel = True
            print("  kernel agent          : DISABLED (persisted from original run)")
        # Same persistence contract for the FRAMEWORK_PR phase toggle.
        if not bool(getattr(state, "framework_phase_enabled", True)):
            args.no_framework = True
            print("  framework phase       : DISABLED (persisted from original run)")
        elif bool(getattr(args, "no_framework", False)):
            # Inverse direction (P2.d): persisted state still has the
            # phase enabled, but the operator passed ``--no-framework``
            # on resume. Only honour this when the original session has
            # not yet entered FRAMEWORK_PR (or anything past it) —
            # retroactively skipping a phase we are in/past is
            # incoherent.
            cur_phase = (getattr(state, "phase", "") or "").strip().upper()
            if cur_phase in ("", "PRELUDE"):
                state.framework_phase_enabled = False
                # Persist immediately: the later conditional save only
                # runs when there was a prior stop_reason / crash, so a
                # clean resume would otherwise drop this toggle on the
                # next disk reload.
                state.save(session_dir)
                print(
                    "  framework phase       : DISABLING for resume "
                    "(--no-framework + phase=PRELUDE)"
                )
            else:
                print(
                    f"  framework phase       : WARN --no-framework ignored; "
                    f"session is already in phase={cur_phase!r} "
                    f"(cannot retroactively skip)"
                )
        # Same persistence contract for the EXPLORE phase toggle.
        if not bool(getattr(state, "explore_enabled", True)):
            args.no_explore = True
            print("  explore phase         : DISABLED (persisted from original run)")
        elif bool(getattr(args, "no_explore", False)):
            # Operator passed --no-explore on resume of an explore-enabled
            # session. Only honour it before EXPLORE has been entered —
            # retroactively skipping a phase we are in/past is incoherent.
            cur_phase = (getattr(state, "phase", "") or "").strip().upper()
            if cur_phase in ("", "PRELUDE", "FRAMEWORK_PR"):
                state.explore_enabled = False
                print(
                    "  explore phase         : DISABLING for resume "
                    f"(--no-explore + phase={cur_phase or 'PRELUDE'})"
                )
            else:
                print(
                    f"  explore phase         : WARN --no-explore ignored; "
                    f"session is already in phase={cur_phase!r} "
                    f"(cannot retroactively skip)"
                )

        # CRITICAL: a leftover stop_reason from the prior run (most often
        # "time_exhausted") fools Orchestration into thinking the work is
        # already done — it just heartbeats forever. Clear it so the new
        # run has a clean signal. The Coordinator's run() always re-sets
        # stop_reason at exit anyway.
        prior_crash = state.crash_count

        # Issue-G (Saturday May 2026): the SKILL.md "Run-time signals"
        # section says ``no_more_leverage`` and ``target_reached`` are
        # intentional terminal states — "only resume if the user
        # changes workload / search space / model / strategy." The
        # original auto-clear silently nuked both, which is how an
        # over-eager monitor or a habitual ``--resume`` would push a
        # session past a steward verdict the operator never reviewed.
        # We now require ``--force-resume`` to push past those two
        # vocab terms; everything else (time_exhausted, max_ticks,
        # crash recovery) auto-clears as before.
        force_resume = bool(getattr(args, "force_resume", False))
        gated_terminal = {"no_more_leverage", "target_reached"}
        if prior_stop in gated_terminal and not force_resume:
            print(
                f"\nERROR: --resume blocked by terminal stop_reason="
                f"{prior_stop!r}.\n"
                f"\n"
                f"  SKILL.md (Run-time signals): {prior_stop!r} is a "
                f"deliberate terminal state.\n"
                f"  The optimizer will not auto-resume past it because "
                f"the prior run\n"
                f"  declared exhaustion — picking up where it left off "
                f"only repeats\n"
                f"  the same exhaustion verdict.\n"
                f"\n"
                f"  Override paths:\n"
                f"  1. Pass ``--force-resume`` if you have changed the "
                f"workload /\n"
                f"     search space / model / strategy and want to "
                f"continue regardless.\n"
                f"  2. Start a fresh session (different "
                f"$USER_DATA_PATH) for a clean run.\n"
                f"\n"
                f"  Reports for the prior run live under "
                f"{session_dir}/reports/.\n",
                file=sys.stderr,
            )
            sys.exit(2)

        if prior_stop or prior_crash >= 3:
            state.stop_reason = ""
            state.closing_phase = False
            state.closing_started_unix = 0.0
            state.closing_report_task_id = ""
            # Reset persisted crash_count so a fresh resume isn't immediately
            # tripped into "emergency" by accumulated failures from prior runs
            # (e.g. authentication errors before .env was loaded).
            state.crash_count = 0
            # Reset start_ts to "now" so the elapsed_minutes value the
            # orchestration agent sees in its prompt reflects this resume's
            # budget rather than the original session start; otherwise an
            # old session that exhausted its budget will look "already
            # over budget" on resume and the LLM will refuse to propose
            # any work.
            from datetime import datetime, timezone
            state.start_ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            state.save(session_dir)
            override_note = (
                " (--force-resume override)"
                if force_resume and prior_stop in gated_terminal
                else ""
            )
            print(
                f"  → cleared stop_reason and reset crash_count "
                f"(was {prior_crash}) for fresh resume{override_note}"
            )
            print(f"  → reset start_ts to {state.start_ts} (resume budget)")
        # re-bootstrap the Cortex KB client. The KB session protocol
        # was retired (fact writes are session-less), so this just
        # re-creates the client + reruns T0 warm-start; ``resume=True``
        # is preserved for the banner label.
        cortex_client = _bootstrap_cortex_kb(
            args, session_dir=session_dir, manifest=manifest, resume=True,
        )
        # KnowledgePlane facade. Bootstrapped
        # alongside the cortex client so a resumed session also gets the
        # PR Monitor + KB readonly tools wired into specialist dispatch.
        # Returns a plane that fail-soft degrades when PR Monitor or
        # Cortex is unreachable; specialists then see empty pr_feed /
        # kb_subgraph and continue. ``None`` only when --degraded-kb.
        knowledge_plane = (
            None if not getattr(args, "cortex_enabled", True)
            else _bootstrap_knowledge_plane(
                args,
                cortex_client=cortex_client,
                session_dir=session_dir,
            )
        )
        # No resume backfill is required for the roofline comparison
        # pipeline: PR #321 retired the ``last_trace_analyze_baseline``
        # baseline-freeze field in favour of the append-only
        # ``roofline_snapshots`` history, which is restored verbatim by
        # ``SharedState.from_dict`` (missing key → empty list default).
    else:
        # Resolve model path from --model first, then $MODEL_PATH env. Without
        # either, fail fast: silently falling back to the YAML's hardcoded
        # `/wekafs/models/Qwen-Qwen3-8B` was the cause of "the optimizer ran
        # the wrong model" reports — explicit > implicit.
        if not args.model:
            args.model = os.environ.get("MODEL_PATH") or ""
        if not args.model:
            print(
                "ERROR: model is required. Pass --model <path> or set "
                "MODEL_PATH env (or use --resume to continue an existing "
                "session at the canonical session_dir).",
                file=sys.stderr,
            )
            sys.exit(2)
        # Re-export the resolved value so downstream subprocess executors
        # (baseline / profile / sweep / backends / params) inject it into
        # the Magpie YAML instead of trusting the YAML's hardcoded `model:`.
        os.environ["MODEL_PATH"] = str(args.model)

        # Resolve framework: --framework > $FRAMEWORK env > "sglang".
        # Session-wide; mixing frameworks in one session is not supported.
        framework = (
            (args.framework or os.environ.get("FRAMEWORK", "")).strip().lower()
            or "sglang"
        )
        if framework not in ("sglang", "vllm", "atom"):
            print(
                f"ERROR: --framework must be sglang, vllm, or atom "
                f"(got {framework!r}); set $FRAMEWORK accordingly or pass "
                "--framework",
                file=sys.stderr,
            )
            sys.exit(2)
        os.environ["FRAMEWORK"] = framework
        print(f"Framework       : {framework}")

        # B3: --framework atom auto-tightens incompatible phases. See
        # _apply_atom_auto_tighten for rationale.
        if framework == "atom":
            _apply_atom_auto_tighten(args)

        # Resolve real target GPU. Order of precedence:
        #   1. rocm-smi / torch probe (always runs when hardware is reachable)
        #   2. --gpu-type CLI flag / $GPU_TYPE env (used as a hint only)
        #   3. --gpu-type-force / HYPERLOOM_GPU_TYPE_FORCE=1 lets the operator
        #      override a successful probe (CI, mock, cross-arch testing).
        #
        # Rationale: the old "operator-specified > probe" priority was a
        # silent footgun. A typo like --gpu-type mi300x on an MI355X host
        # made Magpie pick the wrong runner_type without any warning, which
        # produced bogus baseline numbers and corrupted the KB. Probing
        # first and treating the flag as a hint catches that immediately;
        # the force-override knob preserves the escape hatch for legitimate
        # use cases (containers that mock rocm-smi, dev laptops without a
        # GPU, etc.).
        user_specified = (
            (args.gpu_type or os.environ.get("GPU_TYPE", "")).strip().lower()
        )
        probed = _autodetect_gpu_type() or ""
        gpu_type, gpu_warnings = _resolve_gpu_type(
            user_specified=user_specified,
            probed=probed,
        )
        for line in gpu_warnings:
            print(line, file=sys.stderr)
        if probed and not user_specified:
            print(f"GPU type        : {gpu_type} (auto-detected)")
        runner_gpu_type = _gpu_runner_type(gpu_type)
        if gpu_type and runner_gpu_type != gpu_type:
            print(
                "WARN: mi325x uses mi300x as Magpie runner_type (same arch; "
                "Magpie has no sglang_mi325x.sh / vllm_mi325x.sh yet)",
                file=sys.stderr,
            )
        args.gpu_type = gpu_type or None
        if runner_gpu_type:
            os.environ["TARGET_GPU_TYPE"] = gpu_type
            os.environ["GPU_TYPE"] = runner_gpu_type
            print(f"GPU type        : {gpu_type}")
            print(f"Magpie runner   : {runner_gpu_type} (will inject runner_type into Magpie YAML)")
        else:
            os.environ.pop("TARGET_GPU_TYPE", None)
            os.environ.pop("GPU_TYPE", None)
            args.gpu_type = None
            print("GPU type        : <unset> (Magpie will auto-detect)")

        # Compute MAX_MODEL_LEN = ISL + OSL + 4096 headroom, export for yaml injection.
        max_model_len = args.isl + args.osl + 4096
        os.environ["MAX_MODEL_LEN"] = str(max_model_len)
        os.environ["ISL"] = str(args.isl)
        os.environ["OSL"] = str(args.osl)
        os.environ["PRECISION"] = args.precision
        # Mirror the resolved framework_version into the env so kernel /
        # framework executors and any subprocess (sglang / vllm CLI
        # wrappers) see the same value SharedState carries. Resolution
        # mirrors :func:`_resolve_framework_version`: explicit override
        # wins, otherwise auto-detect, otherwise leave the env unset.
        _fw_version_for_env = (
            (getattr(args, "framework_version", None) or "").strip()
            or (os.environ.get("FRAMEWORK_VERSION", "") or "").strip()
        )
        if not _fw_version_for_env:
            from .recipe_snapshot_constants import (
                DEFAULT_FRAMEWORK_VERSION_SLUG,
                detect_framework_version,
            )

            _detected = detect_framework_version(
                (getattr(args, "framework", None) or "").strip()
                or os.environ.get("FRAMEWORK", "")
            )
            if _detected and _detected != DEFAULT_FRAMEWORK_VERSION_SLUG:
                _fw_version_for_env = _detected
        if _fw_version_for_env:
            os.environ["FRAMEWORK_VERSION"] = _fw_version_for_env
        print(f"Workload        : ISL={args.isl} OSL={args.osl} "
              f"MAX_MODEL_LEN={max_model_len} PRECISION={args.precision} "
              f"FRAMEWORK_VERSION={_fw_version_for_env or '<unset>'}")

        # session_dir is <workspace_root>/<model>/<UTC ts>/ by default
        # (per-model + per-launch). Workspace_root is
        # $USER_DATA_PATH (fallback /workspace/hyperloom). To restore
        # the legacy flat layout, set INFERENCE_OPTIMIZER_SESSION_LAYOUT=
        # flat. `make_session_dir(model_name=...)` does the layout
        # decision + pins the result via $INFERENCE_OPTIMIZER_CURRENT_
        # SESSION_DIR for every subprocess.
        session_dir = make_session_dir(model_name=args.model)
        manifest = write_manifest(session_dir, args=args)
        print(f"Session dir     : {session_dir}")
        print(f"Session id      : {manifest['session_id']}  (manifest label only)")
        _print_session_skeleton(session_dir)

        # Machine-readable launch info. The human-friendly prints above are
        # column-aligned for log reading; this one-liner gives launcher
        # scripts (robustness_monitor, health-check loops, kill scripts)
        # a stable single point to harvest pid + session_dir + run_log
        # without resorting to ``pgrep -af inference_optimizer`` and
        # ``ls -d $USER_DATA_PATH/<model>/*T*Z | tail -1``, both of which
        # break when several sessions overlap on the same host. See
        # ``_emit_launch_info`` for the wire format.
        _emit_launch_info(
            pid=os.getpid(),
            session_dir=session_dir,
            session_id=str(manifest["session_id"]),
            run_log=os.environ.get("INFERENCE_OPTIMIZER_RUN_LOG", ""),
            gpu_type=gpu_type or "",
            framework=args.framework or "",
            model=str(args.model) if args.model else "",
            launch_info_file=getattr(args, "launch_info_file", None),
        )
        _seed_shared_state(
            session_dir, args, session_id=manifest["session_id"],
        )
        # Cortex KB T0 anchor. Must run after the SharedState
        # seed (so model_name / gpu_type / framework are populated for
        # recipe_canonical_id derivation) but before Coordinator is
        # constructed (the Coordinator stores the client + threads it
        # into its KB hooks). Fails fast unless --degraded-kb.
        cortex_client = _bootstrap_cortex_kb(
            args, session_dir=session_dir, manifest=manifest, resume=False,
        )
        # KnowledgePlane facade for specialist
        # sub-agents. Wraps cortex_client (already T0'd above) + a
        # PR Monitor REST client; fail-soft on either side so specialist
        # dispatch always has a non-None plane to consult.
        knowledge_plane = (
            None if not getattr(args, "cortex_enabled", True)
            else _bootstrap_knowledge_plane(
                args,
                cortex_client=cortex_client,
                session_dir=session_dir,
            )
        )

    objective = build_objective({
        "MAX_HOURS": str(args.max_hours),
        "TARGET_GAIN_PCT": str(args.target_gain) if args.target_gain else "",
        "TARGET_TPUT_PER_GPU": str(args.target_tput) if args.target_tput else "",
        "TARGET_DIR": args.target_baseline_dir or "",
    })
    print(f"Objective       : kind={objective.kind()} {objective.describe()}")
    no_kernel = getattr(args, "no_kernel", False)
    no_explore = getattr(args, "no_explore", False)
    if no_explore and no_kernel:
        print(
            "WARNING: --no-explore and --no-kernel are both set; the run "
            "collapses to baseline -> SWEEP over an empty optimization_stack "
            "(no EXPLORE param search, no KERNEL rewrites). SWEEP only "
            "re-validates the baseline recipe. Continuing as requested.",
            file=sys.stderr,
        )
    elif no_explore:
        print("Explore phase   : DISABLED (--no-explore); "
              f"{'baseline -> SWEEP' if no_kernel else 'baseline -> KERNEL -> SWEEP'}")
    if bool(getattr(args, "research_scout", True)):
        print(
            "Research scout  : ENABLED at PRELUDE (re-dispatch every "
            f"{max(1, int(getattr(args, 'research_scout_interval', 3) or 3))} "
            "explore rounds)"
        )
    else:
        print("Research scout  : DISABLED (--no-research-scout)")
    if bool(getattr(args, "target_advisory", True)):
        print("Target advisory : ENABLED (External target gap injected into "
              "prompts; advisory-only)")
    else:
        print("Target advisory : DISABLED (--no-target-advisory)")
    if bool(getattr(args, "recipe_sediment", True)):
        print("Recipe sediment : ENABLED (KEEP/REVERT provenance written to "
              "persistent recipe)")
    else:
        print("Recipe sediment : DISABLED (--no-recipe-sediment)")
    if bool(getattr(args, "depth_gate", True)):
        print("Depth gate      : ENABLED (stop/advance blocked while "
              "exploration is too shallow; IR-6 budget overrides)")
    else:
        print("Depth gate      : DISABLED (--no-depth-gate); legacy "
              "single steward continuation")
    if bool(getattr(args, "allow_empty_kernel_shape", False)):
        os.environ["HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE"] = "1"
        print("Kernel shape    : empty-shape dispatch ALLOWED "
              "(--allow-empty-kernel-shape)")
    else:
        os.environ.pop("HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE", None)
        print("Kernel shape    : non-empty trace shape REQUIRED for "
              "kernel-opt dispatch")

    # Resolve critic backend choice + critic-agent runtime root before
    # _build_backends (which constructs CriticAgentBackend immediately and
    # would otherwise blow up on missing runtime). Fail-fast policy: if the
    # operator selected --critic-agent (or it's the default) but the
    # critic-agent runtime is unreachable, we abort with rc=2 instead of
    # silently falling back to mock.
    critic_choice = _resolve_critic_choice(args)
    critic_agent_root: Path | None = None
    critic_kb_mode = os.environ.get("CRITIC_KB_CLIENT_MODE", "inmemory").lower()
    if critic_kb_mode not in ("inmemory", "live"):
        print(
            f"ERROR: CRITIC_KB_CLIENT_MODE={critic_kb_mode!r} not in "
            "{'inmemory','live'}",
            file=sys.stderr,
        )
        sys.exit(2)
    if critic_choice == "agent":
        critic_agent_root = _resolve_critic_agent_root()
        if critic_agent_root is None:
            print(
                f"ERROR: --critic-agent selected but critic-agent runtime not "
                f"found.\n"
                f"  Set ${_CRITIC_AGENT_ROOT_ENV} to the directory containing "
                f"runtime/cli.py, or install critic-agent at "
                f"$REPO_ROOT/critic-agent/.\n"
                f"  Bypass with --critic-mock.",
                file=sys.stderr,
            )
            sys.exit(2)
        _validate_critic_agent_runtime(critic_agent_root)
        if critic_kb_mode == "live" and not os.environ.get("KB_BASE_URL"):
            print(
                "ERROR: CRITIC_KB_CLIENT_MODE=live but KB_BASE_URL is not "
                "set. Either export KB_BASE_URL or unset "
                "CRITIC_KB_CLIENT_MODE to fall back to inmemory.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Default WORKSPACE_PATH for the critic-agent runtime if the
        # operator hasn't already pinned it. NOTE: in the critic-agent
        # runtime this env names the SKILL static-asset root (not a
        # writable artefact directory), so it points at the repo root.
        # The legacy "artefact root" meaning of WORKSPACE_PATH used by
        # kernel-agent tools was retired during the
        # all-artefacts-under-USER_DATA_PATH migration.
        os.environ.setdefault("WORKSPACE_PATH", str(Path(__file__).resolve().parents[1]))

    # Resolve robustness backend choice + runtime root, mirroring critic.
    robustness_choice = _resolve_robustness_choice(args)
    robustness_agent_root: Path | None = None
    robustness_options = _build_robustness_options(args)
    if robustness_choice == "agent":
        robustness_agent_root = _resolve_robustness_agent_root()
        if robustness_agent_root is None:
            print(
                f"ERROR: --robustness-agent selected but robustness-agent "
                f"runtime not found.\n"
                f"  Set ${_ROBUSTNESS_AGENT_ROOT_ENV} to the directory "
                f"containing src/robustness_agent/runtime/cli.py, or install "
                f"robustness-agent at $REPO_ROOT/robustness-agent/.\n"
                f"  Bypass with --robustness-mock.",
                file=sys.stderr,
            )
            sys.exit(2)
        _validate_robustness_agent_runtime(robustness_agent_root)

    backends = _build_backends(
        claude_model=args.claude_model,
        codex_model=args.codex_model,
        kernel_codex=args.kernel_codex,
        critic_choice=critic_choice,
        session_dir=session_dir,
        critic_agent_root=critic_agent_root,
        critic_kb_mode=critic_kb_mode,
        robustness_choice=robustness_choice,
        robustness_agent_root=robustness_agent_root,
        robustness_options=robustness_options,
        no_kernel=no_kernel,
    )
    # Bug A fix: expose the active session_dir to in-process executors
    # (e.g. ReportExecutor) that don't get session_dir threaded through
    # task.params. This is read in report.py::_resolve_session_dir.
    os.environ["USER_DATA_PATH"] = str(session_dir)
    # Production: enable strict path-containment checks in PolicyGate so
    # any LLM-emitted intent whose path field escapes session_dir lands
    # as `policy_denied` in its inbox. Tests omit this and keep the
    # legacy lenient mode for fixture paths under /tmp.
    os.environ["INFERENCE_OPTIMIZER_STRICT_PATHS"] = "1"
    # flip on PolicyGate R1 phase_incompatible enforcement for
    # production runs. Tests construct PolicyGate directly with strict_phase
    # left at the dataclass default (False) so legacy fixtures aren't
    # broken; this env var only affects the cli boot path.
    if getattr(args, "strict_phase", True):
        os.environ["INFERENCE_OPTIMIZER_STRICT_PHASE"] = "1"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_STRICT_PHASE", None)
    # propagate the ``--legacy-action-scores`` choice so
    # ``SharedState.from_dict`` (called from anywhere — Coordinator,
    # breakdown, resume probes) handles the drop / warn behavior
    # uniformly. Default ``drop`` matches the env-unset case.
    legacy_mode = str(
        getattr(args, "legacy_action_scores", "drop") or "drop",
    ).strip().lower()
    if legacy_mode == "warn":
        os.environ["INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES"] = "warn"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", None)
    # propagate ``--migration-mode``. SharedState.from_dict
    # consults this env var to decide whether a fact-layer
    # discrepancy is fatal (strict) or a downgraded WARNING (lenient).
    migration_mode = str(
        getattr(args, "migration_mode", "strict") or "strict",
    ).strip().lower()
    if migration_mode == "lenient":
        os.environ["INFERENCE_OPTIMIZER_MIGRATION_MODE"] = "lenient"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_MIGRATION_MODE", None)
    # ``--reset-state`` nukes the existing state.json
    # (backing it up to ``state.json.preReset.<unix_ts>``) so the
    # session starts blank. Done BEFORE Coordinator is constructed.
    if getattr(args, "reset_state", False):
        _reset_state_file(session_dir)
    # propagate ``--breakdown-include-transcripts`` so
    # any end-of-session breakdown emitted from this run picks up the
    # inline / path-only choice.
    transcripts_flag = str(
        getattr(args, "breakdown_include_transcripts", "false") or "false",
    ).strip().lower()
    if transcripts_flag == "true":
        os.environ["INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS"] = "1"
    else:
        os.environ.pop(
            "INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", None,
        )

    # Build the phase budget pct dict from CLI flags; ``None`` values
    # fall back to library defaults inside Coordinator.
    phase_budget_pct: dict[str, float] = {}
    for cli_field, phase_name in (
        ("phase_budget_prelude_pct", "PRELUDE"),
        ("phase_budget_explore_pct", "EXPLORE"),
        ("phase_budget_kernel_pct",  "KERNEL"),
        ("phase_budget_sweep_pct",   "SWEEP"),
        ("phase_budget_close_pct",   "CLOSE"),
    ):
        val = getattr(args, cli_field, None)
        if val is not None:
            phase_budget_pct[phase_name] = float(val)

    # When kernel is disabled, strip it from the role registry so
    # Coordinator does not tick a non-existent agent and PolicyGate does
    # not expect a backend for it.
    role_registry = None
    if no_kernel:
        from .orchestrator.agent_role import default_role_registry
        role_registry = {
            k: v for k, v in default_role_registry().items() if k != "kernel"
        }

    coordinator = Coordinator(
        session_dir, backends=backends, role_registry=role_registry,
        compare_against_gpu=getattr(args, "compare_against_gpu", None),
        model_class=(
            getattr(args, "model_class", None)
            or os.environ.get("MODEL_CLASS")
            or ""
        ),
        cortex_kb=cortex_client,
        phase_budget_pct=phase_budget_pct or None,
        # KnowledgePlane facade.
        # ``None`` when --degraded-kb; otherwise wraps Cortex KB +
        # PR Monitor for specialist prompt assembly.
        knowledge_plane=knowledge_plane,
        # Advisory multi-model specialist-proposal scorer. ``None`` when
        # --no-proposal-scoring or an empty model list; otherwise scores
        # each proposal_set and surfaces the results to Orchestration as
        # one reference among many (never gates anything).
        proposal_scorer=_build_proposal_scorer(args),
        # Warm-recipe replay controls. Default ON, fires when
        # warm_start_recipe.confidence >= min_confidence and the
        # measured gain reproduces at least min_reproduce_pct of the
        # recipe's historical claim. Manifest is the persistent
        # authority across restarts (resume-safe).
        warm_replay_enabled=_resume_safe_flag(
            args, "no_warm_replay", manifest, "warm_replay_enabled",
            default=True, invert=True,
        ),
        warm_replay_min_confidence=_resume_safe_numeric(
            args, "warm_replay_min_confidence", manifest,
            "warm_replay_min_confidence", default=0.7,
        ),
        warm_replay_min_reproduce_pct=_resume_safe_numeric(
            args, "warm_replay_min_reproduce_pct", manifest,
            "warm_replay_min_reproduce_pct", default=0.8,
        ),
    )
    framework_for_prompt = (
        os.environ.get("FRAMEWORK", "").strip().lower() or "sglang"
    )
    max_minutes_for_prompt = int(round(float(args.max_hours) * 60))
    prompts: dict[str, str] = {
        "orchestration": args.orch_prompt or _build_orchestration_prompt(
            no_kernel=no_kernel,
            framework=framework_for_prompt,
            objective=objective,
            max_minutes=max_minutes_for_prompt,
        ),
        "critic":        args.critic_prompt or _load_critic_prompt(),
    }
    if not no_kernel:
        prompts["kernel"] = args.kernel_prompt or _DEFAULT_KERNEL_PROMPT
    coordinator.system_prompt_overrides = prompts
    # ``fa phase-discover`` timeout override. ``framework_agent_client``
    # uses DEFAULT_FA_PHASE_TIMEOUT_SEC (180s) when this is missing /
    # falsy.
    try:
        coordinator.framework_pr_discover_timeout_sec = float(
            getattr(args, "framework_pr_discover_timeout_sec", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        coordinator.framework_pr_discover_timeout_sec = 0.0
    # Build specialist executor when the research_lane capacity is
    # non-zero. ``args.research_lane_capacity`` is already clamped to
    # [0, 32] by ``_seed_shared_state``; a value of 0 means "degrade to
    # the LLM-direct grid", and we keep ``specialist_executor=None`` so
    # the dispatcher falls back to the ``no_executor`` rejection (which
    # PolicyGate R2 also short-circuits in practice).
    specialist_capacity = int(
        getattr(args, "research_lane_capacity", 1) or 0
    )
    specialist_executor: "Any" = None
    if specialist_capacity > 0:
        specialist_executor = _build_specialist_executor(
            args,
            session_dir=session_dir,
            knowledge_plane=knowledge_plane,
        )
    dynamic_action_executor: "Any" = _build_dynamic_action_executor(args)
    _register_executors(
        coordinator,
        no_kernel=no_kernel,
        compare_against_gpu=getattr(args, "compare_against_gpu", None),
        session_dir=session_dir,
        specialist_executor=specialist_executor,
        dynamic_action_executor=dynamic_action_executor,
    )
    # Persist effective system prompts for resume / drift inspection.
    _snapshot_system_prompts(session_dir, prompts=prompts)

    kernel_str = "DISABLED" if no_kernel else (
        f"{'Codex' if args.kernel_codex else 'Claude'}"
    )
    if critic_choice == "mock":
        critic_str = "mock"
    else:  # "agent"
        critic_str = (
            f"critic-agent(kb={critic_kb_mode}, codex={args.codex_model}, "
            f"root={critic_agent_root})"
        )
    if robustness_choice == "mock":
        robustness_str = "mock"
    else:
        robustness_str = f"robustness-agent(root={robustness_agent_root})"
        if robustness_options:
            kvs = ",".join(f"{k}={v!r}" for k, v in sorted(robustness_options.items()))
            robustness_str += f"[{kvs}]"
    print(f"Backends        : "
          f"orchestration=Claude({args.claude_model}), "
          f"kernel={kernel_str}, "
          f"critic={critic_str}, "
          f"robustness={robustness_str}")
    print(f"Max ticks       : {args.max_ticks or 'unlimited'} "
          f"(budget = {args.max_hours}h)")
    print(f"Tick interval   : {args.tick_interval_sec}s")
    print()

    if not (getattr(args, "compare_against_gpu", None) or "").strip():
        print(
            "[target_analysis] no --compare-against-gpu set; will write a "
            "marker JSON at $SESSION_DIR/target_analysis/target_baseline.json "
            "(reason=no_target_gpu_configured) — set --compare-against-gpu "
            "to fetch real InferenceX reference data.",
            file=sys.stderr,
        )

    try:
        stop_reason = await coordinator.run(
            objective=objective,
            max_minutes=args.max_hours * 60.0,
            tick_interval_sec=args.tick_interval_sec,
            max_ticks=args.max_ticks,
            install_signal_handlers=True,
            closing_grace_sec=args.closing_grace_sec,
        )
    finally:
        await coordinator.stop()
        # End-of-session safety net: always materialize session_breakdown.json
        # for downstream consumers (claw-stats-service / hyperloom-results-
        # service / offline analysis). Best-effort — a failure here MUST NOT
        # mask the actual stop_reason, so we swallow exceptions and log.
        #
        # When the CLOSE phase sequencer ran to completion, step 2
        # already wrote the same artifact via
        # the standard session_breakdown executor. Skip the duplicate
        # write here so the cli.finally path doesn't clobber the
        # sequencer's output (which includes the full CLOSE-step
        # evidence the sequencer stamped on phase_history). The flag
        # is locked in CORE_STATE_FIELDS so an LLM can't trick us
        # into skipping the safety net for a non-CLOSE termination.
        sequencer_done = getattr(
            coordinator.shared_state, "close_sequence_done", False,
        )
        if sequencer_done:
            print(
                "Session breakdown : (already written by CLOSE phase "
                "sequencer; skipping cli.finally safety-net write)"
            )
        else:
            try:
                from .breakdown import write_breakdown_json
                breakdown_path = write_breakdown_json(session_dir)
                print(f"Session breakdown : {breakdown_path}")
            except Exception:  # noqa: BLE001
                log.exception(
                    "session_breakdown finalize failed (non-fatal)"
                )
            # Issue-I (Saturday May 2026): mirror the same safety-net
            # for ``reports/final.md``. The full ReportExecutor walks
            # the message bus (rich highlights), but if the close
            # sequencer never reached step 1 the operator is left
            # with an empty ``reports/`` directory and has to manually
            # piece together stop_reason / current_best from
            # state.json. ``write_minimal_final_report`` is no-op when
            # the sequencer's final.md already exists.
            try:
                from .breakdown import write_minimal_final_report
                final_md = write_minimal_final_report(session_dir)
                print(f"Final report      : {final_md}")
            except Exception:  # noqa: BLE001
                log.exception(
                    "emergency final report write failed (non-fatal)"
                )

    _reconcile_crash_count(coordinator.shared_state, session_dir)
    # NOTE: conc_sweep used to run here as a post-hook. It is now a
    # real SWEEP-phase action auto-enqueued by the Coordinator after
    # the SWEEP-entry sweep task lands (see
    # ``coordinator._enqueue_internal_conc_sweep_task``). The CLI
    # flags still live below and are mirrored onto SharedState at
    # session init.

    _print_final_summary(coordinator.shared_state, stop_reason)
    return 0 if stop_reason in (
        "target_reached",
        "no_more_leverage",
        "time_exhausted",
        "max_ticks",
    ) else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inference_optimizer",
        description="Inference Optimizer v0.6 — multi-agent SGLang/vLLM optimization",
    )
    p.add_argument("--verbose", "-v", action="count", default=0,
                    help="Verbose logging (-v INFO, -vv DEBUG)")
    sub = p.add_subparsers(dest="command", required=True)

    opt = sub.add_parser("optimize",
                          help="Drive a multi-agent optimization run on a model")
    opt.add_argument("--model", "-m", type=Path, default=None,
                      help="Model path (required for new runs; ignored when "
                           "--resume is set — model is read from manifest.json/"
                           "state.json)")
    opt.add_argument(
        "--gpu-type", choices=["mi300x", "mi325x", "mi355x"], default=None,
        help="Hint for the real target GPU. The rocm-smi probe always "
             "wins when both are present and disagree; a WARN is "
             "emitted to stderr so the operator sees the typo. Used "
             "verbatim only when the probe fails (CPU sandbox / no "
             "rocm-smi). Magpie runner_type is derived separately; "
             "mi325x currently runs with mi300x runner scripts because "
             "Magpie does not yet ship sglang_mi325x.sh / vllm_mi325x.sh.",
    )
    opt.add_argument(
        "--framework", choices=["sglang", "vllm", "atom"], default=None,
        help="Inference framework to benchmark / optimize. Resolution order: "
             "--framework > $FRAMEWORK env > sglang (default). Selection is "
             "session-wide; mixing frameworks in a single session is not "
             "supported. NOTE: --framework atom is single-node-only "
             "(``--nodes>=2`` fails fast); profile / roofline, "
             "kernel-agent, and framework-agent are all enabled on atom. "
             "The auto-tighten guard only enforces ``--nodes 1``.",
    )
    opt.add_argument(
        "--nodes", type=int,
        # Resolution: --nodes (CLI) > $INFERENCE_OPTIMIZER_NODES > $NODES > 1.
        # Brain / SaFE sandbox prompts conventionally export ``NODES=2`` in
        # ``optimizer.env`` (not the ``INFERENCE_OPTIMIZER_NODES`` form), so
        # without the ``NODES`` fallback the optimizer silently fell back to
        # single-node mode and the brain-supplied nodes value never reached
        # ``args.nodes``. The downstream ``nodes_resolved >= 2`` gate in
        # ``_run_optimize`` (and ``is_multi_node()`` in
        # ``_multi_node_env.py``) still hard-requires >= 2 before any
        # multi-node code path runs, so falling through to ``NODES=1``
        # produces the same single-pod behaviour as before this fallback
        # existed.
        default=int(
            os.environ.get("INFERENCE_OPTIMIZER_NODES")
            or os.environ.get("NODES")
            or "1"
        ),
        help="Total number of GPU nodes for the inference RayJob. "
             "1 (default) keeps the legacy single-pod path. "
             ">=2: `optimize` provisions the SaFE RayJob before preflight "
             "(unless already in /tmp/multi_node_state.json), runs bootstrap "
             "once, and exports RAY_ADDRESS for kernel-agent. Does not stop the "
             "RayJob on exit; run `python3 -m inference_optimizer.multi_node "
             "stop-rayjob` when you want to release it. Requires "
             "--rayjob-image or INFERENCE_OPTIMIZER_RAYJOB_IMAGE. "
             "Resolution: --nodes > $INFERENCE_OPTIMIZER_NODES > $NODES > 1.",
    )
    opt.add_argument(
        "--rayjob-image",
        default=None,
        help="Container image for the multi-node RayJob (head+workers). "
             "Required when --nodes>=2 unless INFERENCE_OPTIMIZER_RAYJOB_IMAGE "
             "is set or state file last_create_request.image is present.",
    )
    opt.add_argument(
        "--rayjob-gpus-per-node",
        type=int,
        default=None,
        help="GPUs per RayJob pod (default: INFERENCE_OPTIMIZER_GPUS_PER_NODE "
             "or 8). Passed to multi_node create-rayjob.",
    )
    # --rayjob-extra-env is a PROMPT-DRIVEN pass-through: this CLI never
    # invents or hardcodes env keys/values. The agent maps each line of
    # the user prompt's `env:` block into one `--rayjob-extra-env K=V`
    # (same contract as `multi_node create-rayjob --extra-env`). Forwarded
    # verbatim to `_provision_multi_node_rayjob_stack` -> cmd_create_rayjob
    # -> workload_spec.env. Reserved keys (RAY_JOB_ENTRYPOINT) are stripped
    # by the multi_node layer. Credential keys (*_API_KEY / *_BASE_URL)
    # are still auto-injected by _credential_fanout() — do NOT also pass
    # them here; the prompt block deliberately excludes them per the
    # multi_node SKILL contract.
    opt.add_argument(
        "--rayjob-extra-env",
        action="append",
        default=[],
        metavar="K=V",
        help="Extra env entries to inject into the multi-node RayJob "
             "(repeatable). Agent maps each line of the user prompt's "
             "`env:` block into one --rayjob-extra-env K=V; the CLI "
             "does not own any default. Skip *_API_KEY / *_BASE_URL "
             "(auto-injected by _credential_fanout) and RAY_JOB_ENTRYPOINT "
             "(reserved by workload_spec). Only takes effect when "
             "--nodes>=2 and this run actually creates the RayJob; "
             "idempotent reuse of an existing rayjob_id keeps the env "
             "set at original create time.",
    )
    opt.add_argument(
        "--tp", type=int,
        default=int(os.environ.get("TP", "1") or 1),
        help="Tensor parallel size. Resolution: --tp > $TP env > 1. "
             "Symmetric with --ep — historically TP only flowed in via "
             "$TP env (read by _workload_envs); the CLI flag was added "
             "so the agent can pass `--tp N` directly from the prompt's "
             "Environment block instead of having to `export TP=N` "
             "first. Either path still works.",
    )
    opt.add_argument(
        "--conc", type=int,
        default=int(os.environ.get("CONC", "8") or 8),
        help="Magpie client concurrency cap (max in-flight requests). "
             "Resolution: --conc > $CONC env > 8. Symmetric with --tp; "
             "agent can pass `--conc N` directly from the prompt.",
    )
    opt.add_argument(
        "--ep", type=int,
        default=int(os.environ.get("EP", "1") or 1),
        help="Expert-parallel size for MoE inference. 1 (default) keeps "
             "experts sharded by TP (legacy behaviour). >=2 enables true "
             "expert parallelism: sglang adds `--expert-parallel-size N`, "
             "vllm adds `--enable-expert-parallel`. Typical: EP=TP for "
             "DSr1/DSv3 on multi-node. Resolution: --ep > $EP env > 1. "
             "EP > TP is rejected at server-restart time.",
    )
    opt.add_argument(
        "--pd-mode",
        choices=("colocated", "disaggregated"),
        default="colocated",
        help="Prefill-Decode disaggregation mode. ALWAYS defaults to "
             "`colocated` regardless of any inherited $PD_MODE env, so "
             "PD only turns on when the agent explicitly passes "
             "`--pd-mode disaggregated` (driven by the prompt's "
             "Environment block having a PD_MODE=disaggregated line). "
             "Stale env from a previous restart cannot accidentally "
             "re-enable PD.",
    )
    opt.add_argument(
        "--pd-prefill-nodes", type=int,
        default=int(os.environ.get("PD_PREFILL_NODES", "0") or 0),
        help="Number of prefill nodes (disaggregated only); pn+dn=nodes",
    )
    opt.add_argument(
        "--pd-decode-nodes", type=int,
        default=int(os.environ.get("PD_DECODE_NODES", "0") or 0),
        help="Number of decode nodes (disaggregated only)",
    )
    opt.add_argument(
        "--pd-prefill-tp", type=int,
        default=int(os.environ.get("PD_PREFILL_TP", "0") or 0),
        help="TP for prefill group (disaggregated only); default = --tp",
    )
    opt.add_argument(
        "--pd-decode-tp", type=int,
        default=int(os.environ.get("PD_DECODE_TP", "0") or 0),
        help="TP for decode group (disaggregated only); default = --tp",
    )
    opt.add_argument(
        "--pd-transfer-backend", type=str,
        default=os.environ.get("PD_TRANSFER_BACKEND", ""),
        help="sglang: mooncake|nixl ; vllm: NixlConnector|...; empty = default",
    )
    opt.add_argument(
        "--pd-ib-device", type=str,
        default=os.environ.get("PD_IB_DEVICE", ""),
        help="comma-separated IB/RoCE device list (e.g. mlx5_0,mlx5_1). "
             "Empty = use $NCCL_IB_HCA from RayJob pod env at server-launch time.",
    )
    opt.add_argument(
        "--skip-variants", type=str,
        default=os.environ.get("SKIP_VARIANTS", ""),
        help="Comma/whitespace-separated list of variant names or fnmatch "
             "globs to drop from the backends/params grids before launch. "
             "Examples: `attn_aiter` (exact), `attn_aiter,sched_dfs` (two "
             "exacts), `attn_*,vllm_aiter_*` (globs). Resolution: "
             "--skip-variants > $SKIP_VARIANTS > empty. Exported back into "
             "$SKIP_VARIANTS so all executors and the multi-node orchestrator "
             "subprocess see the same value. Dropped variants surface in "
             "state.json under each action's `dropped_variants` field tagged "
             "`source=user_skip`.",
    )
    opt.add_argument("--max-hours", type=float, default=2.0,
                      help="Wall-clock budget in hours (default 2.0)")
    opt.add_argument(
        "--closing-grace-sec",
        type=float,
        default=None,
        help=(
            "Extra seconds after the wall-clock deadline for Coordinator to "
            "flush a deterministic report task (no LLM). Default: "
            "min(120, max_hours * 60 * 0.02). Pass 0 to disable closing phase."
        ),
    )
    opt.add_argument("--isl", type=int, default=int(os.environ.get("ISL", "256")),
                      help="Input sequence length (default $ISL or 256)")
    opt.add_argument("--osl", type=int, default=int(os.environ.get("OSL", "256")),
                      help="Output sequence length (default $OSL or 256)")
    opt.add_argument("--precision", type=str,
                      default=os.environ.get("PRECISION", "bf16"),
                      help="Model precision (default $PRECISION or bf16)")
    opt.add_argument(
        "--framework-version",
        dest="framework_version",
        type=str,
        default=None,
        help=(
            "Framework version slug for the recipe-snapshot canonical id "
            "(scopes recipes to a specific framework release — sglang 0.4.5 "
            "and sglang 0.5.x have different scheduler defaults so they "
            "deserve separate KB rows). When omitted, auto-detected via "
            "importing the framework's top-level package and reading "
            "``__version__`` (sglang/vllm/atom supported); auto-detect "
            "failure degrades to 'unknown_version'. Override with "
            "--framework-version=0.4.5 to pin a specific tag for the run."
        ),
    )
    opt.add_argument(
        "--continue-kernel-after-gemm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After FP8 GEMM tuning succeeds, continue into source-level "
            "kernel_opt. Use --no-continue-kernel-after-gemm for a "
            "GEMM-only validation run."
        ),
    )
    grp = opt.add_mutually_exclusive_group()
    grp.add_argument("--target-gain", type=float, default=None,
                      help="Stop when cumulative_gain >= N%% over baseline")
    grp.add_argument("--target-tput", type=float, default=None,
                      help="Stop when current best tok/s/GPU >= N")
    grp.add_argument("--target-baseline-dir", type=str, default=None,
                      help="Stop when current best matches the baseline in DIR")
    opt.add_argument("--resume", action="store_true", default=False,
                      help="Resume an existing session. Without --resume-from, "
                           "auto-picks the latest per-session subdir under "
                           "$USER_DATA_PATH/<model>/<UTC ts>/ (N17 layout) or "
                           "falls back to $USER_DATA_PATH (legacy flat layout). "
                           "USER_DATA_PATH MUST stay at workspace level "
                           "(/wekafs/.../sessions/, not the per-session subdir) "
                           "so runtime/ resolution works. Skips the SharedState "
                           "seed and lets the Coordinator replay the prior "
                           "event log + state.json.")
    opt.add_argument("--resume-from", type=str, default=None,
                      help="Explicit per-session subdir to resume from. Use "
                           "when multiple per-launch ts dirs exist under the "
                           "same model and the latest is not what you want. "
                           "Must be an absolute path under $USER_DATA_PATH "
                           "(workspace_root). Implies --resume.")
    opt.add_argument(
        "--force-resume", action="store_true", default=False,
        help=(
            "Allow ``--resume`` to push past a terminal "
            "``stop_reason='no_more_leverage'`` or ``'target_reached'``. "
            "Without this flag the resume aborts (Issue-G guard, per "
            "SKILL.md 'Run-time signals': those terminals require an "
            "operator-side workload / strategy change before resuming). "
            "No-op outside ``--resume``."
        ),
    )
    opt.add_argument(
        "--model-class", type=str,
        default=os.environ.get("MODEL_CLASS", None),
        help=(
            "Categorical model-class key. It is the deterministic key for "
            "several consumers: the atom explore seed grid "
            "(action_executors/explore.py), the framework-agent gap search "
            "token (action_executors/_framework_gap_composer.py), the recipe "
            "key, and the orchestration prompt label. Recognised values "
            "(case-insensitive, with -/+/space tolerated): dense / moe_mla / "
            "moe_swa / moe_mla_nsa. When unset, Coordinator boot infers and "
            "persists it from config.json (num_experts / architectures) or "
            "model-path family keywords, replacing the deleted ``classify`` "
            "action's lightweight state-write role. For richer *advisory* "
            "model context (attention variant, KV/token, experts, MTP, ...) "
            "the SKILL launcher writes $USER_DATA_PATH/model_arch.json, which "
            "is injected into prompts but drives no gating."
        ),
    )
    opt.add_argument("--target-summary", type=str, default=None,
                      help="Free-text goal summary surfaced in prompts")
    opt.add_argument(
        "--compare-against-gpu", type=str, default=None,
        help=(
            "Reference GPU hardware key for external baseline comparison "
            "(e.g. b300 / mi355x / h200). target_analysis ALWAYS runs as "
            "TODO 0 and always writes "
            "$SESSION_DIR/target_analysis/target_baseline.json + a short "
            "MD report. When this flag is set, the JSON carries the "
            "matching InferenceX (https://inferencex.semianalysis.com) "
            "reference data point; when unset, the JSON carries a "
            "structured reason='no_target_gpu_configured' marker so the "
            "report still has a deterministic 'External baseline' "
            "section. The data is REPORT-ONLY: it does not influence "
            "Objective, scoring, or any agent prompt. Other dimensions "
            "(model / framework / precision / ISL / OSL) are derived "
            "from --model and the standard FRAMEWORK / PRECISION / ISL / "
            "OSL env vars."
        ),
    )
    opt.add_argument("--max-ticks", type=int, default=None,
                      help="Hard tick cap (None = unlimited; mostly for tests)")
    opt.add_argument("--tick-interval-sec", type=float, default=0.0,
                      help="Sleep between ticks (0 = no sleep)")
    opt.add_argument("--claude-model", type=str,
                      default=os.environ.get("CLAUDE_MODEL", "claude-opus-4-7"))
    opt.add_argument("--codex-model", type=str,
                      default=os.environ.get("CODEX_MODEL", "gpt-5.4"))
    opt.add_argument("--no-kernel", action="store_true", default=False,
                      help="Disable the Kernel agent entirely. The run will "
                           "only do baseline + params + backends + sweep (pure "
                           "parameter search). Useful when GEAK/OOB/GPU "
                           "compile env is unavailable or you just want the "
                           "quick-win parameter path. Default: kernel enabled.")
    opt.add_argument("--no-explore", action="store_true", default=False,
                      help="Skip the EXPLORE phase entirely. PRELUDE (and "
                           "FRAMEWORK_PR, if enabled) route straight to KERNEL "
                           "— or to SWEEP when --no-kernel is also set. Useful "
                           "for a baseline -> kernel-only run, or to validate "
                           "the current recipe via SWEEP without a serving-"
                           "param search. Default: explore enabled.")
    opt.add_argument(
        "--launch-info-file", type=str, default=None,
        help="Write a JSON file with the launched session's pid, "
             "session_dir, session_id, run_log, manifest path, gpu_type, "
             "framework and model. Launcher scripts can ``jq -r .pid`` / "
             "``jq -r .session_dir`` instead of grepping stdout or "
             "pgrep'ing. Always emitted alongside the ``HYPERLOOM_LAUNCH "
             "<key=value> ...`` single-line sentinel that is printed to "
             "stdout for stream-based parsers.",
    )
    opt.add_argument("--framework-pr-discover-timeout-sec", type=float,
                      default=0.0,
                      help="Override the per-call timeout for "
                           "``fa phase-discover``. 0 (the default) uses "
                           "framework_agent_client.DEFAULT_FA_PHASE_TIMEOUT_SEC "
                           "(180s). The Coordinator retries discover up to "
                           "DISCOVER_FAILURE_RETRY_LIMIT (3) consecutive "
                           "failures before marking FRAMEWORK_PR done.")
    opt.add_argument("--no-framework", action="store_true",
                      default=os.environ.get(
                          "INFERENCE_OPTIMIZER_NO_FRAMEWORK", "0",
                      ).strip() in ("1", "true", "True", "TRUE", "yes"),
                      help="Skip the FRAMEWORK_PR phase (PRELUDE → EXPLORE "
                           "directly). The phase pre-scans upstream sglang/"
                           "vllm PRs via framework-agent and lands KEPT "
                           "patches before EXPLORE starts. Disable when "
                           "the framework-agent toolchain is unavailable "
                           "or you want a faster cold start. Also read from "
                           "$INFERENCE_OPTIMIZER_NO_FRAMEWORK=1. "
                           "Default: framework phase enabled.")
    opt.add_argument("--kernel-codex", action="store_true", default=True,
                      help="Use Codex backend for Kernel agent (default — faster). "
                           "Pass --kernel-claude to switch.")
    opt.add_argument("--kernel-claude", action="store_false", dest="kernel_codex",
                      help="Use Claude backend for Kernel agent")
    # Critic backend selection. ``critic_backend`` is the canonical
    # attribute; the flags below are convenience aliases that all
    # set the same dest. Default is filled by the CLI default block in
    # ``_resolve_critic_choice``; a single explicit flag wins, conflicts
    # are caught at runtime (mutual exclusion checked there because
    # argparse mutually-exclusive groups don't compose with default values
    # cleanly when we want one flag to be the implicit default).
    opt.add_argument(
        "--critic-mock",
        dest="critic_backend",
        action="store_const",
        const="mock",
        default=None,
        help="Force the always-approve mock Critic (offline / smoke tests).",
    )
    opt.add_argument(
        "--critic-agent",
        dest="critic_backend",
        action="store_const",
        const="agent",
        help="Force the critic-agent runtime backend (KB + session memory + "
             "review_constraints). Requires CRITIC_AGENT_ROOT or a sibling "
             "$REPO_ROOT/critic-agent/ directory.",
    )
    # ----- Robustness backend selection (mirrors critic) ------------------
    opt.add_argument(
        "--robustness-mock",
        dest="robustness_backend",
        action="store_const",
        const="mock",
        default=None,
        help="Force the heartbeat-only mock Robustness backend.",
    )
    opt.add_argument(
        "--robustness-agent",
        dest="robustness_backend",
        action="store_const",
        const="agent",
        help="Force the robustness-agent runtime backend (subprocess + JSON, "
             "mirrors critic-agent transport). Requires ROBUSTNESS_AGENT_ROOT "
             "or a sibling $REPO_ROOT/robustness-agent/ directory.",
    )
    opt.add_argument(
        "--robustness-server-url",
        dest="robustness_server_url",
        type=str,
        default=None,
        help="Override the robustness-server base URL forwarded into "
             "request.options. Honoured only when --robustness-agent is "
             "selected.",
    )
    opt.add_argument(
        "--robustness-llm-rca",
        dest="robustness_llm_rca",
        action="store_true",
        default=None,
        help="Forward llm_rca_enabled=true into request.options. The agent "
             "still falls back to NoopRcaEngine when LLM credentials aren't "
             "set in the runtime env.",
    )
    opt.add_argument(
        "--no-robustness-llm-rca",
        dest="robustness_llm_rca",
        action="store_false",
        help="Forward llm_rca_enabled=false into request.options.",
    )
    opt.add_argument(
        "--robustness-workload-uid",
        dest="robustness_workload_uid",
        type=str,
        default=None,
        help="Forward workload_uid into request.options. The robustness-server "
             "resolves it to every pod (head + workers) backing the RayJob via "
             "the cluster/workloads/{uid}/hierarchy endpoint. Falls back to "
             "$CLAW_WORKLOAD_UID / $WORKLOAD_UID / $RAY_JOB_ID when unset.",
    )
    opt.add_argument(
        "--robustness-disable-local-probe",
        dest="robustness_disable_local_probe",
        action="store_true",
        default=None,
        help="Force disable_local_probe=true. The robustness-agent silences "
             "its LocalProbe fallback so per-pod sandbox checks (ps, rocm-smi, "
             "local HTTP) cannot emit false-positive symptoms.",
    )
    opt.add_argument(
        "--no-robustness-disable-local-probe",
        dest="robustness_disable_local_probe",
        action="store_false",
        help="Force disable_local_probe=false (keep the LocalProbe fallback "
             "even in multi-node mode).",
    )
    opt.add_argument(
        "--robustness-enable-cluster-pod-metrics",
        dest="robustness_enable_cluster_pod_metrics",
        action="store_true",
        default=None,
        help="Force enable_cluster_pod_metrics=true so the robustness-agent "
             "fans out per-pod metrics through robustness-server and feeds "
             "the local_health rules with cluster-decoded GPU snapshots.",
    )
    opt.add_argument(
        "--no-robustness-enable-cluster-pod-metrics",
        dest="robustness_enable_cluster_pod_metrics",
        action="store_false",
        help="Force enable_cluster_pod_metrics=false.",
    )
    opt.add_argument(
        "--robustness-pod-metrics-categories",
        dest="robustness_pod_metrics_categories",
        type=str,
        default=None,
        help="Comma-separated metric categories forwarded into "
             "pod_metrics_categories (e.g. 'gpu,memory'). Default 'gpu' is "
             "applied by the runtime when this flag is omitted.",
    )
    opt.add_argument("--orch-prompt", type=str, default=None,
                      help="Override Orchestration system prompt (file path or inline)")
    opt.add_argument("--critic-prompt", type=str, default=None,
                      help="Override Critic system prompt")
    opt.add_argument("--kernel-prompt", type=str, default=None,
                      help="Override Kernel system prompt")
    # ------------------------------------------------------------------
    # Cortex KB integration flags
    # ------------------------------------------------------------------
    # The defaults wire Cortex *on*. ``--degraded-kb`` is a debug escape
    # hatch that fully bypasses the KB hooks so a fresh sandbox can
    # reproduce the behaviour without any KB writes. ``--cortex-kb-url``
    # overrides the env value (``CORTEX_KB_URL``) without exporting one
    # process-wide. ``--cortex-strict-fingerprint`` enforces the
    # manifest stack_fingerprint matches a recipe before warm_start is
    # consumed.
    opt.add_argument(
        "--cortex-kb-url",
        dest="cortex_kb_url",
        type=str,
        default=None,
        help="Remote recipe-snapshot KB URL (read-only) for this run; "
             "also settable via $CORTEX_KB_URL. Leave it UNSET to run "
             "fully local — there is no default endpoint, so the "
             "optimizer never connects to a remote KB unless you pass "
             "this explicitly. Writes always go to --local-kb-root "
             "regardless; an explicitly-configured but unreachable URL "
             "degrades the dispatcher to local-only transparently (no "
             "need to also pass --degraded-kb).",
    )
    opt.add_argument(
        "--local-kb-root",
        dest="local_kb_root",
        type=str,
        default=None,
        help="Filesystem root for the local recipe-snapshot KB store. "
             "All writes (put_recipe / append_attempt / delete_recipe) go "
             "here regardless of --cortex-kb-url. Defaults to "
             "$HYPERLOOM_LOCAL_KB_ROOT, then $USER_DATA_PATH/kb, "
             "then /workspace/hyperloom/kb. Layout is a 5-level "
             "directory tree keyed by canonical_id components "
             "(model -> hardware -> framework -> framework_version -> "
             "precision); each leaf holds recipe.json + history/ + "
             "attempts.ndjson + .lock. See "
             "inference_optimizer/recipe_kb/local_store.py for the "
             "on-disk contract.",
    )
    opt.add_argument(
        "--degraded-kb",
        dest="degraded_kb",
        action="store_true",
        default=False,
        help="Skip the Cortex KB integration entirely (T0/T2/T3/T4 become "
             "no-ops). Also short-circuits the IR-3 KB probe. IR-3 sets "
             "this automatically when kb-service is unreachable (soft "
             "degrade); manifest records the reason as ``explicit_flag`` "
             "vs ``ir3_auto``.",
    )
    opt.add_argument(
        "--cortex-strict-fingerprint",
        dest="cortex_strict_fingerprint",
        action="store_true",
        default=False,
        help="When set, T0 refuses warm_start_recipe rows whose "
             "stack_fingerprint does not match the current pod (recorded "
             "in manifest.json). Default: lenient (M1 records the flag "
             "in manifest only; consumed by M5 specialist assembly).",
    )
    # Warm-recipe replay (PRELUDE auto-applies the KB best_config
    # before EXPLORE starts). Three flags control the behavior:
    #
    # 1. ``--no-warm-replay``         — disable the auto-enqueue entirely.
    # 2. ``--warm-replay-min-confidence`` — minimum warm_start tier conf
    #    to trigger (default 0.7 → only T1 / T2 fires).
    # 3. ``--warm-replay-min-reproduce-pct`` — fraction of the
    #    recipe's historical gain we need to reproduce to count as
    #    "reproduced" (default 0.8 → +25% historical, +20% counts).
    opt.add_argument(
        "--no-warm-replay",
        dest="no_warm_replay",
        action="store_true",
        default=False,
        help="Disable the PRELUDE auto-replay of KB warm-start "
             "``best_config``. The warm_start_recipe is still rendered "
             "into the specialist prompt as priors, but the Coordinator "
             "will NOT auto-run the historical best_config. Use this "
             "for cold debugging / ablation runs.",
    )
    opt.add_argument(
        "--warm-replay-min-confidence",
        dest="warm_replay_min_confidence",
        type=float,
        default=0.7,
        help="Minimum ``warm_start_recipe.confidence`` required to "
             "trigger the auto-replay. Default 0.7 means an ``exact`` "
             "5-tuple hit (conf 1.0) and a server-returned ``relative`` "
             "match (conf 0.7) both fire, while a ``miss`` (conf 0.0) "
             "does not. Raise it above 0.7 to require an exact hit "
             "before spending a verify on the warm config.",
    )
    opt.add_argument(
        "--warm-replay-min-reproduce-pct",
        dest="warm_replay_min_reproduce_pct",
        type=float,
        default=0.8,
        help="Minimum fraction of the recipe's recorded gain we need "
             "to reproduce to count as ``status=reproduced`` and push "
             "the warm config onto the optimization stack. Default "
             "0.8 — a recipe claiming +25%% counts if we measure "
             "+20%% or more. Below the threshold we record "
             "``status=drift`` and continue with the regular EXPLORE "
             "flow without inheriting the warm config.",
    )
    # ------------------------------------------------------------------
    # PR Monitor REST + MCP
    # ------------------------------------------------------------------
    # ``--pr-monitor-url`` overrides the in-cluster default; the
    # marathon pod is typically in a different cluster from
    # primus-cortex, so an operator running outside the primus-cortex
    # k8s namespace must port-forward + pass a localhost URL.
    # ``--degraded-pr`` switches the KnowledgePlane.pr_feed_warm
    # path to a no-op and strips ``mcp__pr_monitor__*`` from the
    # specialist tool whitelist (degrade-to-empty).
    opt.add_argument(
        "--pr-monitor-url",
        dest="pr_monitor_url",
        type=str,
        default=None,
        help="Override PR Monitor REST URL for this run. Default: "
             "http://primus-cortex-pr-api.primus-cortex.svc.cluster.local"
             "/v1 (env: PR_MONITOR_URL). Pair with --pr-monitor-mcp-url "
             "when port-forwarding for local debug.",
    )
    opt.add_argument(
        "--pr-monitor-mcp-url",
        dest="pr_monitor_mcp_url",
        type=str,
        default=None,
        help="Override PR Monitor MCP URL handed to specialist LLM "
             "backends. Default mirrors --pr-monitor-url with /mcp/ "
             "suffix; the trailing slash is mandatory.",
    )
    opt.add_argument(
        "--degraded-pr",
        dest="degraded_pr",
        action="store_true",
        default=False,
        help="Disable the PR Monitor integration entirely. "
             "pr_feed_warm returns empty; the specialist tool "
             "whitelist drops mcp__pr_monitor__* tools. Short-circuits "
             "the IR-3 PR Monitor probe; IR-3 sets this automatically "
             "when PR Monitor is unreachable (soft degrade).",
    )
    opt.add_argument(
        "--pr-feed-window-days",
        dest="pr_feed_window_days",
        type=int,
        default=int(
            os.environ.get("PR_FEED_WINDOW_DAYS", "30") or "30"
        ),
        help="Look-back window for the PR feed warmup (days). "
             "Default: 30.",
    )
    # ------------------------------------------------------------------
    # specialist research_lane capacity
    # ------------------------------------------------------------------
    # ``--research-lane-capacity`` locks the number of LLM specialists
    # that may run concurrently on the research_lane:
    #   * 0   → no specialist dispatch; EXPLORE uses the default_grid
    #           path.
    #   * 1   → single specialist at a time.
    #   * 4   → default — enough headroom
    #           for the Orchestration LLM to fan out one specialist per
    #           top-K gap inside one tick (multi-emit shape) and have
    #           the dispatcher actually run them in parallel.
    #   * The upper bound scales with the visible GPU count
    #           (``2 × GPU``); operator values above the ceiling are
    #           silently clamped down (see ``research_lane_ceiling`` in
    #           ``orchestrator/policy.py``).
    # Locked at session start (mirrored into manifest + SharedState);
    # PolicyGate denies mid-flight mutation via CORE_STATE_FIELDS.
    opt.add_argument(
        "--research-lane-capacity",
        dest="research_lane_capacity",
        type=int,
        default=int(
            os.environ.get("INFERENCE_OPTIMIZER_RESEARCH_LANE_CAPACITY", "4")
            or "4"
        ),
        help="Max concurrent LLM specialist sub-agents on the "
             "research_lane. 0 disables specialist "
             "dispatch entirely (degrades to LLM-direct grid); 4 is "
             "the default. The upper bound scales with the visible GPU "
             "count (2 x GPU); values above it are silently clamped "
             "down. Locked at session start.",
    )
    opt.add_argument(
        "--gpu-specialist-capacity",
        dest="gpu_specialist_capacity",
        type=int,
        default=int(
            os.environ.get("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "0")
            or "0"
        ),
        help="Number of GPUs available to specialists that request "
             "needs_gpu=true. 0 disables GPU specialists (default). "
             "Set INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES to a "
             "comma-separated GPU id pool when the specialist pool should "
             "not use device ids 0..N-1. Locked at session start.",
    )
    # ------------------------------------------------------------------
    # Advisory specialist-proposal scorer (ProposalScorer). Scores each
    # specialist proposal_set with one or more gateway models (single
    # 0-10 composite + one-line reason) and surfaces the results to
    # Orchestration as one reference among many — never gates anything.
    # Adding a model = appending its gateway slug to the comma list.
    # ------------------------------------------------------------------
    opt.add_argument(
        "--proposal-scorer-models",
        dest="proposal_scorer_models",
        type=str,
        default=",".join(DEFAULT_SCORER_MODELS),
        help="Comma-separated gateway model slugs that independently "
             "score each specialist proposal_set (advisory only; never "
             "gates; rater identities are anonymized in the orchestration "
             "prompt). Default 'claude-opus-4-8,gpt-5.5,"
             "dvue-aoai-005-Kimi-K2.6,gemini/gemini-3.1-pro-preview'. "
             "Add a model by "
             "appending its slug. Empty list disables scoring.",
    )
    opt.add_argument(
        "--no-proposal-scoring",
        dest="no_proposal_scoring",
        action="store_true",
        help="Disable the advisory specialist-proposal scorer entirely.",
    )
    # ------------------------------------------------------------------
    # specialist sub-agent
    # backend selection. Specialists run via Claude (default) and inherit
    # the orchestration model unless overridden. Per-task turn / time
    # caps protect against runaway LLM consumption.
    # ------------------------------------------------------------------
    opt.add_argument(
        "--specialist-model",
        dest="specialist_model",
        type=str,
        default=os.environ.get("INFERENCE_OPTIMIZER_SPECIALIST_MODEL", "")
        or None,
        help="Claude model used for specialist sub-agents (defaults to "
             "the orchestration --claude-model). KB_design §3.5 §6.",
    )
    opt.add_argument(
        "--dynamic-action-model",
        dest="dynamic_action_model",
        type=str,
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_DYNAMIC_ACTION_MODEL", "",
        ) or None,
        help="Claude model used for dynamic_action sub-agents (defaults "
             "to --claude-model).",
    )
    opt.add_argument(
        "--dynamic-action-turn-cap",
        dest="dynamic_action_turn_cap",
        type=int,
        default=int(
            os.environ.get(
                "INFERENCE_OPTIMIZER_DYNAMIC_ACTION_TURN_CAP", "0",
            ) or "0",
        ) or None,
        help="Hard cap on ReAct turns per dynamic_action dispatch "
             "(default 12). Per dynamic_action_runner.DEFAULT_TURN_CAP.",
    )
    opt.add_argument(
        "--dynamic-action-wall-clock-sec",
        dest="dynamic_action_wall_clock_sec",
        type=float,
        default=float(
            os.environ.get(
                "INFERENCE_OPTIMIZER_DYNAMIC_ACTION_WALL_CLOCK_SEC", "0",
            ) or "0",
        ) or None,
        help="Wall-clock budget per dynamic_action dispatch (default "
             "900s = 15 min). Per "
             "dynamic_action_runner.DEFAULT_WALL_CLOCK_BUDGET_SEC.",
    )
    opt.add_argument(
        "--specialist-max-turns",
        dest="specialist_max_turns",
        type=int,
        default=int(
            os.environ.get("INFERENCE_OPTIMIZER_SPECIALIST_MAX_TURNS", "8")
            or "8"
        ),
        help="Hard cap on LLM turns per specialist task (KB_design "
             "§3.5 §6). On exhaustion the runner synthesises an empty "
             "specialist_done (Inv-5.3).",
    )
    opt.add_argument(
        "--specialist-per-turn-max-seconds",
        dest="specialist_per_turn_max_seconds",
        type=float,
        default=float(
            os.environ.get(
                "INFERENCE_OPTIMIZER_SPECIALIST_PER_TURN_MAX_SECONDS", "600"
            )
            or "600"
        ),
        help="Wall-clock cap per specialist turn (default 600s). Used "
             "by the robustness stale-scan to detect stuck specialists "
             ".",
    )
    # ------------------------------------------------------------------
    # specialist dispatch shape
    # ------------------------------------------------------------------
    opt.add_argument(
        "--specialist-dispatch-mode",
        dest="specialist_dispatch_mode",
        type=str,
        choices=("subprocess", "inprocess"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_SPECIALIST_DISPATCH_MODE", "subprocess",
        ).strip() or "subprocess",
        help="Specialist execution shape. 'subprocess' (default) spawns "
             "a fresh `claude` CLI per task inside a per-task git worktree "
             "for isolation (PR-A2). 'inprocess' keeps the legacy M5 path "
             "(claude-agent-sdk in the orchestrator process) for tests / "
             "environments without the claude binary.",
    )
    opt.add_argument(
        "--specialist-mcp-config",
        dest="specialist_mcp_config",
        type=str,
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_SPECIALIST_MCP_CONFIG", "",
        ).strip() or None,
        help="Optional path to an MCP config JSON forwarded to the "
             "subprocess claude (`--mcp-config`). Used to wire kb / pr "
             "MCP servers into specialists. Default: None.",
    )
    # ------------------------------------------------------------------
    # Integration toggles. The roofline composite + watermark-driven
    # refresh path is unconditional now: roofline auto-fires at PRELUDE
    # (after baseline) and on every 10% ``cumulative_gain_validated``
    # crossing over ``last_roofline_tput``. The composite/deny toggles
    # that gated the legacy single-step ``profile`` path are gone.
    # ------------------------------------------------------------------
    def _env_default_on(env_var: str) -> bool:
        return os.environ.get(env_var, "1").strip() != "0"

    opt.add_argument(
        "--allow-empty-kernel-shape",
        dest="allow_empty_kernel_shape",
        action="store_true",
        default=os.environ.get(
            "HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE", "0",
        ).strip().lower() in {"1", "true", "yes", "on"},
        help="Escape hatch (default off): allow kernel optimization to "
             "dispatch a candidate with no trace-anchored shape. Normally "
             "a shapeless candidate is rejected with a structured error so "
             "the run returns to ``trace_analyze`` instead of burning a "
             "GEAK / OOB budget on an unanchored kernel. Env: "
             "HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE=1.",
    )
    opt.add_argument(
        "--enable-roofline",
        dest="enable_roofline",
        action=argparse.BooleanOptionalAction,
        default=_env_default_on("INFERENCE_OPTIMIZER_ENABLE_ROOFLINE"),
        help="Select which analysis action the Coordinator enqueues at "
             "PRELUDE bootstrap and on every +10%% watermark crossing. "
             "Default on: ``roofline`` (composite profile + "
             "trace_analyze + analysis.md). Pass ``--no-enable-roofline`` "
             "to use plain ``profile`` instead (lighter — captures the "
             "trace only, skips trace_analyze). Behaviour is otherwise "
             "identical (same idempotency keys, same pending-task "
             "dispatch gate, same watermark anchor update). Env: "
             "INFERENCE_OPTIMIZER_ENABLE_ROOFLINE=0.",
    )
    opt.add_argument(
        "--research-scout",
        dest="research_scout",
        action=argparse.BooleanOptionalAction,
        default=_env_default_on("INFERENCE_OPTIMIZER_RESEARCH_SCOUT"),
        help="Auto-dispatch a read-only research scout at PRELUDE (and "
             "every --research-scout-interval EXPLORE rounds) that "
             "collects proven priors — reference launch scripts, model "
             "config.json architecture features, and cross-framework / "
             "NVIDIA research — into ``research_hints.md`` and seeds "
             "high-priority gaps. Default on; pass ``--no-research-scout`` "
             "to disable the whole feature. Env: "
             "INFERENCE_OPTIMIZER_RESEARCH_SCOUT=0.",
    )
    opt.add_argument(
        "--research-scout-interval",
        dest="research_scout_interval",
        type=int,
        default=3,
        help="Re-dispatch the research scout every N EXPLORE rounds with "
             "the current bottleneck context (append-only). Default 3. "
             "Ignored when ``--no-research-scout`` is set.",
    )
    opt.add_argument(
        "--recipe-sediment",
        dest="recipe_sediment",
        action=argparse.BooleanOptionalAction,
        default=_env_default_on("INFERENCE_OPTIMIZER_RECIPE_SEDIMENT"),
        help="Sediment KEEP/REVERT provenance into the persistent recipe: "
             "KEEP optimizations traceable to a research hint carry their "
             "source + measured gain into ``what_worked``; REVERTs land in "
             "``what_failed`` so the next warm-start avoids re-testing them. "
             "Default on; pass ``--no-recipe-sediment`` to keep the recipe "
             "purely ephemeral. Env: INFERENCE_OPTIMIZER_RECIPE_SEDIMENT=0.",
    )
    opt.add_argument(
        "--target-advisory",
        dest="target_advisory",
        action=argparse.BooleanOptionalAction,
        default=_env_default_on("INFERENCE_OPTIMIZER_TARGET_ADVISORY"),
        help="Inject an advisory 'External target gap' block (throughput / "
             "TPOT / interactivity gap vs the LLM-authored competitor "
             "target) into the orchestration and specialist prompts; when "
             "the TPOT ratio dominates it nudges toward latency-reducing "
             "directions. Advisory only — never gates Objective or scoring. "
             "Default on; pass ``--no-target-advisory`` to disable. Env: "
             "INFERENCE_OPTIMIZER_TARGET_ADVISORY=0.",
    )
    opt.add_argument(
        "--depth-gate",
        dest="depth_gate",
        action=argparse.BooleanOptionalAction,
        default=_env_default_on("INFERENCE_OPTIMIZER_DEPTH_GATE"),
        help="Deterministic exploration-depth guard. When the session "
             "steward recommends stop / advance-to-kernel but the run "
             "has not explored deeply enough (too few scout runs, no "
             "code patch, too few PRs/diffs/NVIDIA refs), the verdict is "
             "rewritten to continue_explore with a deepening hint — as "
             "many times as needed, bounded only by the IR-6 budget "
             "gate. Default on; ``--no-depth-gate`` restores the legacy "
             "single-continuation steward behaviour. Env: "
             "INFERENCE_OPTIMIZER_DEPTH_GATE=0.",
    )
    opt.add_argument(
        "--depth-gate-thresholds",
        dest="depth_gate_thresholds",
        default="",
        help="Override depth-gate minimums as comma-separated key=value "
             "pairs. Keys: scout_runs_min, prs_fetched_min, "
             "pr_diffs_read_min, nvidia_refs_min, code_patches_min, "
             "reverts_to_evaluate. Defaults: 2,5,3,2,1 evaluated after "
             "reverts_to_evaluate=3. Example: "
             "``--depth-gate-thresholds scout_runs_min=1,code_patches_min=2``.",
    )
    # ------------------------------------------------------------------
    # Post-optimization concurrency sweep (on by default).
    #
    # Runs an extra "baseline vs optimized" Magpie grid across CONC
    # values after the close-sequence report has been written. The
    # output is JSON/CSV in ``reports/conc_sweep_summary.json``; the
    # frontend pairs it with the roofline ceiling to visualise the
    # full curve. See ``orchestrator/conc_sweep.py``.
    #
    # Costs ~30 min/point on an 8xMI300 box and is bounded by
    # ``--conc-sweep-total-budget-sec`` (default 2.5h) plus the
    # remaining session wall-clock. Skip conditions (no baseline, no
    # ``current_best``, no ISL/OSL) short-circuit before any subprocess
    # is launched and are recorded in the JSON envelope. Disable with
    # ``--no-enable-conc-sweep`` or
    # ``INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP=0``.
    # ------------------------------------------------------------------
    opt.add_argument(
        "--enable-conc-sweep",
        dest="enable_conc_sweep",
        action=argparse.BooleanOptionalAction,
        default=(
            os.environ.get(
                "INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP", "",
            ).strip().lower() not in ("0", "false", "no", "off")
        ),
        help="Run a post-optimization concurrency sweep (baseline vs "
             "current_best across CONC) and write "
             "reports/conc_sweep_summary.json + conc_sweep_raw.csv. "
             "On by default; disable with --no-enable-conc-sweep or "
             "INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP=0.",
    )
    opt.add_argument(
        "--conc-sweep-concs",
        dest="conc_sweep_concs",
        type=str,
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_CONC_SWEEP_CONCS", "1,2,4,8,16,32,64,128",
        ),
        help="Comma-separated CONC ladder for --enable-conc-sweep. "
             "Default 1,2,4,8,16,32,64,128.",
    )
    opt.add_argument(
        "--conc-sweep-timeout-sec",
        dest="conc_sweep_timeout_sec",
        type=int,
        default=int(os.environ.get(
            "INFERENCE_OPTIMIZER_CONC_SWEEP_TIMEOUT_SEC", "1800",
        )),
        help="Per-variant timeout (seconds) for --enable-conc-sweep. "
             "Default 1800 (~30 min). Per-variant cap is also clamped "
             "by the remaining --conc-sweep-total-budget-sec.",
    )
    opt.add_argument(
        "--conc-sweep-total-budget-sec",
        dest="conc_sweep_total_budget_sec",
        type=int,
        default=int(os.environ.get(
            "INFERENCE_OPTIMIZER_CONC_SWEEP_TOTAL_BUDGET_SEC", "9000",
        )),
        help="Total wall-clock budget (seconds) for the whole conc-sweep "
             "action, independent of the per-variant Magpie timeout. "
             "Once exhausted, remaining variants are recorded as "
             "status=skipped / error_class=budget_exhausted and the "
             "JSON envelope carries budget_exhausted=true. Default 9000 "
             "(~2.5h); set to 0 to disable. Also bounded above by the "
             "main session wall-clock deadline since conc_sweep runs as "
             "a SWEEP-phase action.",
    )
    # Retired flags that operator scripts may still pass. We hard-fail
    # at argparse time with a one-line migration hint instead of
    # silently aliasing — silent aliases hide the behaviour change from
    # the single ``--enable-roofline`` mode-select (the old composite /
    # direct-profile bifurcation no longer exists, and the
    # PRELUDE-initial roofline is unconditional).
    _retired_hint = (
        "Use ``--enable-roofline`` (default on) / ``--no-enable-roofline`` "
        "instead. The PRELUDE-initial analysis task is unconditional and "
        "the composite/direct-profile bifurcation has been removed."
    )
    for _retired in (
        "--use-roofline-composite",
        "--no-use-roofline-composite",
        "--deny-direct-profile",
        "--no-deny-direct-profile",
        "--force-roofline-after-baseline",
        "--no-force-roofline-after-baseline",
    ):
        opt.add_argument(
            _retired,
            action=_RetiredFlag,
            hint=_retired_hint,
        )
    # ------------------------------------------------------------------
    # Per-variant explore overtime kill ratio.
    #
    # Mirrored into :attr:`SharedState.explore_overtime_kill_ratio`
    # and read by :class:`ExploreExecutor` via task.params (Coordinator
    # injects ``baseline_runtime_sec`` + ``explore_overtime_kill_ratio``
    # alongside ``base_extra_args`` / ``base_tput`` on every explore
    # task).
    #
    # Default 1.10: kill a single-variant Magpie run once its
    # wall-clock exceeds the baseline run's wall-clock by +10 %. The
    # killed variant is recorded with ``outcome='KILLED_OVERTIME'`` +
    # ``runtime_sec`` + ``wall_clock_ratio_vs_baseline`` (no tput) so
    # the orchestration LLM can tell "ran too slow → early kill" from
    # "benchmark crashed" or "hit the hard variant timeout".
    #
    # Pass ``0`` (or any non-positive value) to disable the gate; the
    # legacy ``variant_timeout_sec`` hard cap is still enforced in
    # that case. The gate ONLY applies to the per-variant single-run
    # step; the inlined stack rebench (Q4) intentionally uses the
    # legacy timeout because a deep stack legitimately runs slower
    # than the bare baseline.
    # ------------------------------------------------------------------
    def _env_float_or(default: float, env_var: str) -> float:
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    def _env_int_or(default: int, env_var: str) -> int:
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            return int(default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return int(default)
    opt.add_argument(
        "--explore-overtime-kill-ratio",
        dest="explore_overtime_kill_ratio",
        type=float,
        default=_env_float_or(
            1.10, "INFERENCE_OPTIMIZER_EXPLORE_OVERTIME_KILL_RATIO",
        ),
        help="Per-variant explore overtime kill: each single-variant "
             "Magpie run in the explore loop is reaped once its "
             "wall-clock exceeds ``baseline_runtime_sec * RATIO``. The "
             "variant is recorded with outcome=KILLED_OVERTIME + "
             "runtime_sec + wall_clock_ratio_vs_baseline (no tput) so "
             "the LLM can distinguish it from a hard timeout / crash. "
             "Default 1.10 (kill at +10%% over baseline wall-clock). "
             "Pass 0 to disable. Env: "
             "INFERENCE_OPTIMIZER_EXPLORE_OVERTIME_KILL_RATIO.",
    )
    # ------------------------------------------------------------------
    # Explore variant hard timeout — operator override for the
    # auto-derived cap.
    #
    # The legacy class default (2400 s) is a smoke-workload floor. On
    # slow workloads (Qwen3-32B TP=1 BF16 CONC=64 ISL/OSL=1024 NUM_PROMPTS
    # =320 → ~70 min baseline) the floor under-budgets every variant
    # before it can produce a measurement. ExploreExecutor now
    # auto-derives the cap from ``baseline_runtime_sec * (kill_ratio +
    # 0.5)`` so the hard cap stays above the soft kill ratio (preserves
    # the layered design instead of inverting it).
    #
    # Operators can pin an explicit value here (e.g. CI smoke runs that
    # want a tight bound, or workloads where the auto-derived value is
    # too generous). ``0`` (default) leaves the auto-derive in charge.
    # Mirrored to ``SharedState.explore_variant_timeout_sec_override``;
    # the Coordinator injects it as ``params['variant_timeout_sec']``
    # on every explore task, taking precedence over the auto-derive.
    # ------------------------------------------------------------------
    opt.add_argument(
        "--explore-variant-timeout-sec",
        dest="explore_variant_timeout_sec",
        type=int,
        default=_env_int_or(
            0, "INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SEC",
        ),
        help="Pin the per-variant hard timeout (seconds) inside the "
             "EXPLORE phase. ``0`` (default) auto-derives from "
             "``baseline_runtime_sec * (--explore-overtime-kill-ratio + "
             "--explore-variant-timeout-safety-margin)`` once baseline "
             "lands, with a 2400-14400 s range guard. Set to a positive "
             "integer to pin (CI smoke runs / debugging). Env: "
             "INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SEC.",
    )
    opt.add_argument(
        "--explore-variant-timeout-safety-margin",
        dest="explore_variant_timeout_safety_margin",
        type=float,
        default=_env_float_or(
            0.5, "INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SAFETY_MARGIN",
        ),
        help="Headroom (as a fraction of baseline_runtime_sec) added on "
             "top of --explore-overtime-kill-ratio when the EXPLORE hard "
             "cap is auto-derived. Default 0.5 (≈ 50%% of baseline as "
             "buffer for variant cold starts: torch.compile AOTI compile, "
             "fresh aiter shapes, spec-decoding draft load). Bump for "
             "workloads with heavy compile cost; lower to tighten the "
             "backstop. No effect when --explore-variant-timeout-sec is "
             "set to a positive value. Env: "
             "INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SAFETY_MARGIN.",
    )
    opt.add_argument(
        "--explore-roofline-hard-gate",
        dest="explore_roofline_hard_gate",
        action="store_true",
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_EXPLORE_ROOFLINE_HARD_GATE", "0",
        ).strip() == "1",
        help="(Opt-in, off by default.) Drop EXPLORE variants whose flags "
             "target only roofline directions that the latest snapshot "
             "shows saturated above 80%%. Saves variant slots on slow "
             "workloads where (e.g.) host-overhead reducers cannot help "
             "a memory-bound model. The existing soft "
             "``--roofline-saturation-advisory`` prompt hint is unchanged "
             "either way. Env: "
             "INFERENCE_OPTIMIZER_EXPLORE_ROOFLINE_HARD_GATE=1.",
    )
    # ------------------------------------------------------------------
    # drop scoreboard
    # ------------------------------------------------------------------
    # The legacy ``action_scores`` decision system is retired. The
    # flag below controls how a resumed session's leftover
    # scoreboard data is handled:
    #
    # * ``drop`` (default): silently strip ``action_scores`` /
    #   ``params_no_promote_streak`` / ``score_violation`` etc. from
    #   the loaded state.json. The Coordinator never writes them
    #   again so subsequent saves are clean.
    # * ``warn``: same drop behaviour PLUS a ``WARNING`` log line +
    #   a ``breakdown.warnings`` entry so the operator sees the
    #   migration even on a fresh dashboard.
    #
    # No third "keep" mode is offered: carrying the field forward is
    # forbidden (the prompt builder no longer reads
    # it; keeping the bytes only inflates state.json).
    opt.add_argument(
        "--legacy-action-scores",
        dest="legacy_action_scores",
        type=str,
        choices=("drop", "warn"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", "drop",
        ).strip() or "drop",
        help="Resume-mode handling of the legacy scoreboard "
             "(``action_scores`` and friends). 'drop' (default) "
             "silently discards. 'warn' logs a WARNING + adds a "
             "breakdown.warnings entry. KB_design §3.9 §7.",
    )
    # ------------------------------------------------------------------
    # SharedState evolution
    # ------------------------------------------------------------------
    # ``--migration-mode`` controls how non-fatal schema migration
    # discrepancies are surfaced:
    #
    # * ``strict`` (default): a missing fact-layer field (baseline_tput
    #   / current_best / cumulative_gain / optimization_stack) inside
    #   a non-empty state.json is fatal — CLI exits 1 so the operator
    #   can investigate before any new write corrupts the audit trail.
    # * ``lenient``: same discrepancies are downgraded to WARNING and
    #   the run continues with default values. Useful when an operator
    #   is intentionally salvaging a partially-corrupted session.
    #
    # ``--reset-state`` is the nuclear option: when set, the existing
    # ``state.json`` is backed up to ``state.json.preReset.<ts>`` and
    # the session starts fresh. Cortex KB cross-session knowledge is
    # untouched.
    opt.add_argument(
        "--migration-mode",
        dest="migration_mode",
        type=str,
        choices=("strict", "lenient"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_MIGRATION_MODE", "strict",
        ).strip() or "strict",
        help="Strictness of the legacy → v0.8 state.json migration. "
             "'strict' (default) aborts on fact-layer field loss; "
             "'lenient' logs WARNING and continues. KB_design §3.10 §5.3.",
    )
    opt.add_argument(
        "--reset-state",
        dest="reset_state",
        action="store_true",
        default=False,
        help="Back up the existing ``state.json`` (if any) to "
             "``state.json.preReset.<unix_ts>`` and start the session "
             "from a blank SharedState. Cortex KB is NOT touched. "
             "KB_design §3.10 §5.3.",
    )
    # ------------------------------------------------------------------
    # observability
    # ------------------------------------------------------------------
    # ``--breakdown-include-transcripts`` controls whether the per-task
    # specialist transcript bodies are inlined under
    # ``specialist_runs[i].transcripts[j].body`` (true) or only
    # referenced by path (false, default). Default false keeps the
    # ``session_breakdown.json`` payload small for the cluster
    # dashboards; operators can flip it on for offline replay.
    opt.add_argument(
        "--breakdown-include-transcripts",
        dest="breakdown_include_transcripts",
        type=str,
        choices=("true", "false"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", "false",
        ).strip().lower() or "false",
        help="Inline specialist transcript bodies into "
             "``specialist_runs`` (true) or reference them by path "
             "only (false, default). KB_design §3.12 §7.",
    )
    # ------------------------------------------------------------------
    # plateau threshold tuning
    # ------------------------------------------------------------------
    # These flags swap the library default plateau thresholds for the
    # ``compute_plateau_explore`` / ``compute_plateau_kernel`` pure
    # functions. They land in :attr:`SharedState.plateau_overrides`
    # at session boot; ``phase_state.compute_next_phase`` consults
    # the overrides every tick. Locked at session start so resume
    # uses the same threshold the original run picked.
    opt.add_argument(
        "--plateau-explore-keep-gain",
        dest="plateau_explore_keep_gain",
        type=float,
        default=None,
        help="EXPLORE plateau: max cumulative KEEP-gain (%%) across the "
             "lookback window below which the AND condition fires. "
             "Default 0.5.",
    )
    opt.add_argument(
        "--plateau-explore-empty-streak",
        dest="plateau_explore_empty_streak",
        type=int,
        default=None,
        help="EXPLORE plateau: required count of *consecutive* specialist "
             "rounds with empty proposal_set before the AND condition "
             "fires. Default 3.",
    )
    opt.add_argument(
        "--plateau-explore-lookback",
        dest="plateau_explore_lookback",
        type=int,
        default=None,
        help="EXPLORE plateau: number of trailing rounds the gain sum is "
             "computed over. Default 5.",
    )
    opt.add_argument(
        "--plateau-kernel-revert-streak",
        dest="plateau_kernel_revert_streak",
        type=int,
        default=None,
        help="KERNEL plateau: consecutive REVERT / NEEDS_REVIEW integrate "
             "attempts to count as plateau (one half of the OR). "
             "Default 3.",
    )
    opt.add_argument(
        "--plateau-kernel-keep-gain",
        dest="plateau_kernel_keep_gain",
        type=float,
        default=None,
        help="KERNEL plateau: max cumulative KEEP-gain (%%) across the "
             "lookback window below which the OR fires. Default 0.5.",
    )
    opt.add_argument(
        "--plateau-kernel-lookback",
        dest="plateau_kernel_lookback",
        type=int,
        default=None,
        help="KERNEL plateau: number of trailing integrate attempts the "
             "gain sum is computed over. Default 5.",
    )
    # ------------------------------------------------------------------
    # IR-6 — EXPLORE HARD force-exit thresholds
    # ------------------------------------------------------------------
    # Either condition fires an ``explore_force_exit_low_budget`` exit
    # which routes EXPLORE → KERNEL (or SWEEP when --no-kernel). This
    # gate is non-negotiable — the steward / plateau / LLM cannot extend
    # EXPLORE past either threshold. Locked at session start into
    # ``SharedState.plateau_overrides``.
    opt.add_argument(
        "--explore-force-exit-hours-remaining",
        dest="explore_force_exit_hours_remaining",
        type=float,
        default=None,
        help="EXPLORE force-exit: total wall-clock remaining (hours) "
             "below which EXPLORE exits immediately to the next phase, "
             "regardless of plateau / steward. Default 3.0 (IR-6).",
    )
    opt.add_argument(
        "--explore-force-exit-budget-pct",
        dest="explore_force_exit_budget_pct",
        type=float,
        default=None,
        help="EXPLORE force-exit: phase-budget remaining fraction "
             "(0..1) below which EXPLORE exits immediately. Default "
             "0.20 (IR-6).",
    )
    # ------------------------------------------------------------------
    # IR-7 — session_steward_specialist controls
    # ------------------------------------------------------------------
    # The steward is the soft gate on plateau (HARD IR-6 still wins on
    # low budget). Operators can disable it for smoke runs or cap its
    # continuation-granting power.
    opt.add_argument(
        "--steward-disabled",
        dest="steward_disabled",
        action="store_true",
        default=False,
        help="Disable session_steward_specialist; plateau directly "
             "exits EXPLORE without the steward gate (IR-7).",
    )
    opt.add_argument(
        "--steward-continuation-cap",
        dest="steward_continuation_cap",
        type=int,
        default=None,
        help="Max times the steward may return 'continue_explore' in "
             "this session (default 1). Beyond the cap, "
             "continue_explore is coerced to advance_to_kernel.",
    )
    # ------------------------------------------------------------------
    # phase budget percentages
    # ------------------------------------------------------------------
    # Each phase claims a fraction of the total wall-clock budget. The
    # numbers below are caps — the Coordinator may exit a phase earlier
    # if the plateau judge fires. Sum need not equal 1.0 (a deliberate
    # padding is encouraged).
    opt.add_argument(
        "--max-minutes-prelude-pct",
        dest="phase_budget_prelude_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for PRELUDE as a fraction of "
             "--max-hours. Default: 0.05.",
    )
    opt.add_argument(
        "--max-minutes-explore-pct",
        dest="phase_budget_explore_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for EXPLORE. Default: 0.40.",
    )
    opt.add_argument(
        "--max-minutes-kernel-pct",
        dest="phase_budget_kernel_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for KERNEL. Default: 0.35.",
    )
    opt.add_argument(
        "--max-minutes-sweep-pct",
        dest="phase_budget_sweep_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for SWEEP. Default: 0.18.",
    )
    opt.add_argument(
        "--max-minutes-close-pct",
        dest="phase_budget_close_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for CLOSE. Default: 0.02.",
    )
    opt.add_argument(
        "--strict-phase",
        dest="strict_phase",
        action="store_true",
        default=True,
        help="(v0.8 M2 default) Enforce PolicyGate R1 phase_incompatible. "
             "Action proposals outside the current phase's allowlist "
             "return policy_denied so the LLM self-corrects.",
    )
    opt.add_argument(
        "--no-strict-phase",
        dest="strict_phase",
        action="store_false",
        help="Disable R1 enforcement (warn-only). Useful for "
             "back-compat smoke tests; production should stay strict.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    if args.command == "optimize":
        # Resolve any --*-prompt that point at a file.
        for attr in ("orch_prompt", "critic_prompt", "kernel_prompt"):
            v = getattr(args, attr)
            if v and Path(v).exists():
                setattr(args, attr, Path(v).read_text(encoding="utf-8"))
        return asyncio.run(_run_optimize(args))
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
