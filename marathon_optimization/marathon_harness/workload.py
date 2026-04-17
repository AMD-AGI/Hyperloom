"""InferenceX workload integration — single benchmark source of truth.

Wraps InferenceX's benchmark_serving.py and server lifecycle as direct
subprocess calls.  No LLM tokens for deterministic measurement.

Usage:
    wl = InferenceXWorkload.from_sprint_handoff(handoff_dir, inferencex_path)
    await wl.apply_patches(handoff_dir / "patches")
    await wl.start_server()
    result = await wl.run_benchmark()
    print(result.tput_per_gpu)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Content patterns used to classify .sh scripts by what they *do*,
# not by what they're *named*.
_LAUNCH_CONTENT = (
    "vllm serve", "sglang.launch_server", "launch_server",
    "--tensor-parallel", "--tp ", "entrypoints.openai",
    "--served-model", "uvicorn", "api_server",
)
_BENCH_CONTENT = (
    "benchmark_serving", "bench_serving", "--num-prompts",
    "--request-rate", "run_bench",
)
_PATCH_CONTENT = (
    "git apply", "git am ", "apply_patches", "apply_patch",
    "patch -p", ".patch", "git merge", "git cherry-pick",
)

_ORCHESTRATOR_SIGNALS = ("bash \"$SCRIPT_DIR/", "bash $SCRIPT_DIR/", "Phase 1:", "Phase 2:")


def _classify_script(path: Path) -> str:
    """Classify a shell script by scoring which category has the most pattern hits.

    Uses non-comment lines only.  Ties broken by priority:
    orchestrator > launch > bench > patch > other.
    """
    try:
        raw = path.read_text(errors="ignore")[:4096]
    except OSError:
        return "other"
    # Strip comment lines and echo/printf strings to avoid matching doc references
    lines = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # Remove echo/printf string contents to avoid matching incidental mentions
        import re
        cleaned = re.sub(r'(echo|printf)\s+["\'].*?["\']', '', stripped)
        lines.append(cleaned)
    text = "\n".join(lines)

    if any(p in text for p in _ORCHESTRATOR_SIGNALS):
        return "orchestrator"

    scores = {
        "launch": sum(1 for p in _LAUNCH_CONTENT if p in text),
        "bench":  sum(1 for p in _BENCH_CONTENT if p in text),
        "patch":  sum(1 for p in _PATCH_CONTENT if p in text),
    }
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "other"


def _find_script_by_content(directory: Path, kind: str,
                            tp: int | None = None) -> Path | None:
    """Return the best .sh file in *directory* classified as *kind*.

    For launch scripts with a *tp* hint, prefer scripts whose content
    contains ``--tensor-parallel-size=<tp>`` or whose filename contains
    ``tp<tp>``.  Orchestrator wrapper scripts (that call other scripts
    rather than launching a server directly) are excluded from launch
    matches.
    """
    if not directory.is_dir():
        return None
    candidates: list[Path] = []
    for p in sorted(directory.iterdir()):
        if p.suffix != ".sh":
            continue
        cls = _classify_script(p)
        if cls == kind:
            candidates.append(p)
        elif kind == "launch" and cls == "orchestrator":
            continue
    if not candidates:
        return None
    if kind == "launch" and tp is not None:
        for c in candidates:
            text = c.read_text(errors="ignore")[:4096]
            if f"--tensor-parallel-size={tp}" in text or f"tp{tp}" in c.name.lower():
                return c
    return candidates[0]


HEALTH_POLL_S = 3
SERVER_KILL_WAIT_S = 10
DEFAULT_SERVER_TIMEOUT_S = 1500  # GLM-5 on MI355X: ~3min weight load + ~15min CUDA graph capture

# Path to the singleton server PID file.  Only the orchestrator writes this.
_SERVER_PID_FILE = Path("/tmp/.marathon_server.pid")
_SERVER_OWNER_SENTINEL = "marathon-orchestrator"


def _read_server_pid() -> int | None:
    """Read the PID of the server that the orchestrator launched."""
    try:
        data = json.loads(_SERVER_PID_FILE.read_text())
        pid = int(data.get("pid", 0))
        if pid > 0 and Path(f"/proc/{pid}").exists():
            return pid
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return None


def _write_server_pid(pid: int, port: int) -> None:
    """Record the orchestrator-launched server PID."""
    _SERVER_PID_FILE.write_text(json.dumps({
        "pid": pid,
        "port": port,
        "owner": _SERVER_OWNER_SENTINEL,
        "started_at": time.time(),
    }))
    log.info("Server PID file written: pid=%d port=%d", pid, port)


def _clear_server_pid() -> None:
    """Remove the server PID file after killing the server."""
    try:
        _SERVER_PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


_marathon_owns_server = False

async def kill_rogue_servers(port: int = 8888) -> int:
    """Kill any vLLM/sglang servers NOT launched by the orchestrator.

    Returns the number of rogue processes killed.
    """
    global _marathon_owns_server
    authorized_pid = _read_server_pid()

    proc = await asyncio.create_subprocess_shell(
        "ps -eo pid,args | grep -E 'vllm serve|vllm.entrypoints|sglang.srt|sglang.launch_server' | grep -v grep",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    if not stdout:
        return 0

    pids_found = []
    for line in stdout.decode(errors="replace").strip().splitlines():
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            pids_found.append((int(parts[0]), parts[1] if len(parts) > 1 else "?"))
        except ValueError:
            continue

    if not pids_found:
        return 0

    # If the marathon owns a server and the PID file is stale/missing,
    # re-register the first matching process rather than killing it.
    if _marathon_owns_server and authorized_pid is None and len(pids_found) >= 1:
        new_pid = pids_found[0][0]
        _write_server_pid(new_pid, port)
        authorized_pid = new_pid

    killed = 0
    for pid, cmd in pids_found:
        if pid == authorized_pid:
            continue
        log.warning("Killing ROGUE server process: pid=%d cmd=%s", pid, cmd[:120])
        try:
            os.kill(pid, 9)
            killed += 1
        except OSError:
            pass

    if killed:
        await asyncio.sleep(3)
        log.info("Killed %d rogue server process(es)", killed)
    return killed

# System-level Python packages that Claw agents may modify.  We snapshot these
# before each action so we can auto-revert if a patch corrupts the server.
_WATCHED_SYSTEM_FILES = [
    "/usr/local/lib/python3.12/dist-packages/aiter/fused_moe.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py",
    "/usr/local/lib/python3.12/dist-packages/aiter/ops/shuffle.py",
    "/usr/local/lib/python3.12/dist-packages/aiter/ops/norm.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/compilation/backends.py",
]


@dataclass
class BenchmarkResult:
    output_throughput: float = 0.0
    total_token_throughput: float = 0.0
    tput_per_gpu: float = 0.0
    request_throughput: float = 0.0
    mean_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    num_prompts: int = 0
    result_file: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchResult:
    patch_file: str = ""
    applied: bool = False
    already_applied: bool = False
    error: str = ""


class InferenceXWorkload:
    """Wraps InferenceX as direct subprocess — deterministic, no LLM."""

    def __init__(
        self,
        inferencex_path: str,
        model: str,
        tp: int = 8,
        port: int = 8888,
        host: str = "0.0.0.0",
        isl: int = 1024,
        osl: int = 1024,
        concurrency: int = 64,
        num_prompts_multiplier: int = 8,
        framework: str = "sglang",
        extra_launch_flags: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        result_dir: str = "",
    ):
        self.inferencex_path = Path(inferencex_path)
        self.model = model
        self.tp = tp
        self.port = port
        self.host = host
        self.isl = isl
        self.osl = osl
        self.concurrency = concurrency
        self.num_prompts_multiplier = num_prompts_multiplier
        self.framework = framework
        self.extra_launch_flags = extra_launch_flags or []
        self.env_vars = env_vars or {}
        self.result_dir = result_dir or "/tmp/marathon_bench"
        self._server_proc: asyncio.subprocess.Process | None = None
        self._server_log_path = ""

    @classmethod
    def from_sprint_handoff(cls, handoff_dir: str | Path,
                            inferencex_path: str | Path,
                            result_dir: str = "") -> "InferenceXWorkload":
        """Create workload from Sprint's handoff config.json.

        Loads launch_flags, env_vars, benchmark params, and checks for launch scripts.
        """
        handoff = Path(handoff_dir)
        config_path = handoff / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Sprint config.json not found at {config_path}")

        config = json.loads(config_path.read_text())
        bench_params = config.get("benchmark_params", {})

        wl = cls(
            inferencex_path=str(inferencex_path),
            model=config.get("model_path", config.get("model_name", "")),
            tp=config.get("tp", 8),
            framework=config.get("framework", "sglang"),
            extra_launch_flags=config.get("launch_flags", []),
            env_vars=config.get("env_vars", {}),
            isl=bench_params.get("input_len", 1024),
            osl=bench_params.get("output_len", 1024),
            concurrency=bench_params.get("max_concurrency", 64),
            result_dir=result_dir,
        )

        # Check for launch script in handoff dir
        wl._discover_scripts(handoff)
        return wl

    @classmethod
    def from_sprint_repo(cls, repo_dir: str | Path,
                         inferencex_path: str | Path,
                         result_dir: str = "",
                         tp_hint: int | None = None) -> "InferenceXWorkload":
        """Create workload from a standalone Sprint output repo.

        Sprint produces repos like Agentic-InferenceX/DeepSeek-R1-0528-optimized/
        with scripts/serve_tp1.sh, scripts/bench_sweep.sh, and results/.
        This method uses those scripts directly — they have all the flags, env
        exports, model paths, and config variants baked in.
        """
        repo = Path(repo_dir)
        scripts_dir = repo / "scripts"
        if not scripts_dir.is_dir():
            raise FileNotFoundError(f"No scripts/ directory in Sprint repo: {repo}")

        launch_script = _find_script_by_content(scripts_dir, "launch", tp=tp_hint)

        # Try to extract model/tp from launch script for metadata
        model = ""
        tp = tp_hint or 8
        framework = "sglang"
        if launch_script and launch_script.exists():
            content = launch_script.read_text()
            import re
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("MODEL=") and ":-" in stripped:
                    model = stripped.split(":-")[1].rstrip('}"')
                elif stripped.startswith("TP=") and ":-" in stripped:
                    try:
                        tp = int(stripped.split(":-")[1].rstrip('}"'))
                    except ValueError:
                        pass
            # Also check --tensor-parallel-size=N in the actual vllm/sglang command
            tp_match = re.search(r'--tensor-parallel-size[=\s]+(\d+)', content)
            if tp_match:
                tp = int(tp_match.group(1))
            if "vllm serve" in content or "vllm.entrypoints" in content:
                framework = "vllm"
            elif "sglang" in content:
                framework = "sglang"
        log.info("Parsed from launch script: tp=%d, framework=%s, model=%s",
                 tp, framework, model[:60] if model else "(empty)")

        wl = cls(
            inferencex_path=str(inferencex_path),
            model=model,
            tp=tp,
            framework=framework,
            result_dir=result_dir or str(repo / "results"),
        )

        wl._sprint_repo_dir = str(repo)

        # Sprint repo scripts are the source of truth
        if launch_script and launch_script.exists():
            wl._sprint_launch_script = str(launch_script)
            log.info("Using Sprint repo launch script: %s", launch_script)
        else:
            log.error(
                "No launch script found in %s — server will start with "
                "bare-minimum flags and throughput WILL be degraded", scripts_dir)

        patch_script = _find_script_by_content(scripts_dir, "patch")
        if patch_script:
            wl._sprint_patch_script = str(patch_script)
            log.info("Using Sprint repo patch script: %s", patch_script)

        bench_script = _find_script_by_content(scripts_dir, "bench")
        if bench_script:
            wl._sprint_benchmark_script = str(bench_script)
            log.info("Using Sprint repo benchmark script: %s", bench_script)

        sweep_script = scripts_dir / "run_sweep.sh"
        if sweep_script.exists():
            wl._sprint_sweep_script = str(sweep_script)

        # Check for handoff/ subdirectory (newer Sprint outputs)
        handoff_config = repo / "handoff" / "config.json"
        if handoff_config.exists():
            config = json.loads(handoff_config.read_text())
            wl.model = config.get("model_path", wl.model)
            wl.tp = config.get("tp", wl.tp)
            wl.env_vars = config.get("env_vars", {})
            wl.extra_launch_flags = config.get("launch_flags", [])
            bench_params = config.get("benchmark_params", {})
            wl.isl = bench_params.get("input_len", wl.isl)
            wl.osl = bench_params.get("output_len", wl.osl)
            wl.concurrency = bench_params.get("max_concurrency", wl.concurrency)

        return wl

    def _discover_scripts(self, search_dir: Path) -> None:
        """Find Sprint-provided launch, patch, and benchmark scripts by content."""
        self._sprint_repo_dir = str(search_dir)
        for d in (search_dir, search_dir / "scripts"):
            if not d.is_dir():
                continue
            if not getattr(self, "_sprint_launch_script", None):
                found = _find_script_by_content(d, "launch")
                if found:
                    self._sprint_launch_script = str(found)
                    log.info("Found Sprint launch script: %s", found)
            if not getattr(self, "_sprint_patch_script", None):
                found = _find_script_by_content(d, "patch")
                if found:
                    self._sprint_patch_script = str(found)
                    log.info("Found Sprint patch script: %s", found)
            if not getattr(self, "_sprint_benchmark_script", None):
                found = _find_script_by_content(d, "bench")
                if found:
                    self._sprint_benchmark_script = str(found)
                    log.info("Found Sprint benchmark script: %s", found)

    # ------------------------------------------------------------------
    # System file snapshot / rollback — protection against bad Claw patches
    # ------------------------------------------------------------------

    def _snapshot_dir(self) -> Path:
        return Path(self.result_dir) / ".system_snapshots"

    def snapshot_system_files(self) -> list[str]:
        """Copy watched system files to a snapshot directory.

        Called before handing off to the Claw agent so we can revert if
        the agent's patches break the inference server.
        """
        snap_dir = self._snapshot_dir()
        snap_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        for src_path in _WATCHED_SYSTEM_FILES:
            src = Path(src_path)
            if src.exists():
                dest = snap_dir / src.name
                try:
                    shutil.copy2(str(src), str(dest))
                    saved.append(src_path)
                except OSError as exc:
                    log.warning("Failed to snapshot %s: %s", src_path, exc)
        if saved:
            log.info("Snapshotted %d system file(s) for rollback safety", len(saved))
        return saved

    def rollback_system_files(self) -> list[str]:
        """Restore system files from the last snapshot."""
        snap_dir = self._snapshot_dir()
        if not snap_dir.is_dir():
            return []
        restored: list[str] = []
        for src_path in _WATCHED_SYSTEM_FILES:
            src = Path(src_path)
            backup = snap_dir / src.name
            if backup.exists():
                try:
                    shutil.copy2(str(backup), str(src))
                    restored.append(src_path)
                    log.info("Rolled back: %s", src_path)
                except OSError as exc:
                    log.error("Failed to rollback %s: %s", src_path, exc)
        return restored

    def system_files_changed(self) -> list[str]:
        """Return list of watched files that differ from their snapshot."""
        snap_dir = self._snapshot_dir()
        if not snap_dir.is_dir():
            return []
        changed: list[str] = []
        for src_path in _WATCHED_SYSTEM_FILES:
            src = Path(src_path)
            backup = snap_dir / src.name
            if not src.exists() or not backup.exists():
                continue
            try:
                if src.read_bytes() != backup.read_bytes():
                    changed.append(src_path)
            except OSError:
                pass
        return changed

    # ------------------------------------------------------------------
    # Framework launch registry
    # ------------------------------------------------------------------

    _FRAMEWORK_LAUNCH = {
        "sglang": ("sglang.launch_server", "--model-path"),
        "vllm": ("vllm.entrypoints.openai.api_server", "--model"),
        "atom": ("atom.entrypoints.openai.api_server", "--model"),
        "lmdeploy": ("lmdeploy.serve.openai.api_server", "--model-path"),
        "tensorrt_llm": ("tensorrt_llm.serve", "--model"),
    }

    def _build_server_cmd(self) -> list[str]:
        """Bare-minimum server command — FALLBACK ONLY.

        The Sprint launch script (detected by content, not name) is the
        single source of truth for optimization flags, env vars, and
        model-specific config.  This method only runs when no Sprint
        script was found and produces a minimal command that will start
        but is almost certainly missing critical flags.
        """
        fw = self.framework.lower()
        if fw in self._FRAMEWORK_LAUNCH:
            module, model_flag = self._FRAMEWORK_LAUNCH[fw]
        else:
            module = f"{fw}.entrypoints.openai.api_server"
            model_flag = "--model"
            log.warning("Unknown framework %r — guessing launch module %s", fw, module)

        log.error(
            "PERFORMANCE WARNING: No Sprint launch script found — using "
            "bare-minimum server command.  This WILL be missing model-specific "
            "optimization flags (allreduce fusion, KV cache dtype, backend "
            "selection, etc.) and throughput will be degraded.  Fix: ensure "
            "the Sprint repo has a scripts/*.sh launch script."
        )

        return [
            "python3", "-m", module,
            model_flag, self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tp),
            "--trust-remote-code",
        ]

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _write_server_owner_pid(self) -> None:
        """Write a marker file so the watchdog knows who owns the server."""
        marker = Path(self.result_dir) / ".server_owner_pid"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(os.getpid()))

    async def _reset_runtime_workspaces(self) -> None:
        """Reset sglang and aiter repos to their clean HEAD state.

        Previous marathon/agent runs may have left stale modifications
        (e.g. stream-overlap patches that break CUDA graph capture).
        We reset to HEAD so the Sprint's apply_patches.sh starts from a
        known-good baseline.  Also clears __pycache__ to avoid stale bytecode.
        """
        for repo in ["/sgl-workspace/sglang", "/sgl-workspace/aiter"]:
            if not Path(repo).is_dir():
                continue
            proc = await asyncio.create_subprocess_shell(
                f"cd {repo} && git checkout -- . && "
                f"find . -name __pycache__ -type d -exec rm -rf {{}} + 2>/dev/null; true",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                log.info("Reset %s to clean HEAD", repo)
            else:
                log.warning("git checkout failed for %s: %s",
                            repo, (stdout or b"").decode()[:200])

    async def _apply_extra_patch_files(self, sprint_dir: str) -> None:
        """Apply any .patch files in the Sprint repo's patches/ dir that
        weren't covered by the main apply_patches.sh script.

        This handles cases where the Sprint repo has patches that were added
        after the apply script was written (e.g. glm5_model_type_fix.patch).
        """
        patches_dir = Path(sprint_dir) / "patches"
        if not patches_dir.is_dir():
            return

        sglang_dir = "/sgl-workspace/sglang"
        aiter_dir = "/sgl-workspace/aiter"

        for pf in sorted(patches_dir.glob("*.patch")):
            # Try sglang first, then aiter
            applied = False
            for repo_name, repo_dir in [("sglang", sglang_dir), ("aiter", aiter_dir)]:
                if not Path(repo_dir).is_dir():
                    continue
                check = await asyncio.create_subprocess_shell(
                    f"cd {repo_dir} && git apply --check {pf} 2>/dev/null",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await check.communicate()
                if check.returncode == 0:
                    apply = await asyncio.create_subprocess_shell(
                        f"cd {repo_dir} && git apply {pf}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                    out, _ = await apply.communicate()
                    if apply.returncode == 0:
                        log.info("Applied extra patch %s to %s", pf.name, repo_name)
                        applied = True
                        break
            if not applied:
                log.debug("Patch %s: already applied or not applicable", pf.name)

    async def apply_sprint_patches(self) -> bool:
        """Reset runtime repos to clean state, run the Sprint patch script,
        then apply any extra .patch files from the Sprint repo.

        Must be called BEFORE start_server.  Returns True on success.
        """
        sprint_dir = getattr(self, '_sprint_repo_dir', None)

        # Step 1: reset sglang/aiter to clean HEAD
        await self._reset_runtime_workspaces()

        # Step 2: run the Sprint's apply_patches.sh if detected
        patch_script = getattr(self, '_sprint_patch_script', None)
        if patch_script and Path(patch_script).exists():
            log.info("Applying Sprint patches: %s", patch_script)
            env = {**os.environ, **self.env_vars}
            proc = await asyncio.create_subprocess_shell(
                f"bash {patch_script}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
            except asyncio.TimeoutError:
                proc.kill()
                log.error("Patch script timed out after 300s")
                return False

            output = (stdout or b"").decode(errors="replace")
            if proc.returncode != 0:
                log.error("Patch script failed (exit %d): %s",
                          proc.returncode, output[-500:])
                return False
            log.info("Patches applied successfully")

        # Step 3: apply any extra .patch files not covered by the script
        if sprint_dir:
            await self._apply_extra_patch_files(sprint_dir)

        # Step 4: container compatibility hotfixes
        await self._apply_container_hotfixes()

        return True

    async def _apply_container_hotfixes(self) -> None:
        """Apply runtime compatibility fixes for the container environment.

        Handles models whose tokenizer_config.json specifies a tokenizer class
        (e.g. TokenizersBackend) not recognized by the installed transformers.
        """
        hf_utils = Path("/sgl-workspace/sglang/python/sglang/srt/utils/"
                        "hf_transformers_utils.py")
        if not hf_utils.exists():
            return

        content = hf_utils.read_text()
        if "PreTrainedTokenizerFast(tokenizer_object=" in content:
            return

        # The upstream sglang ValueError handler re-raises unconditionally
        # when trust_remote_code=True.  Replace with a fallback that loads
        # the tokenizer directly from tokenizer.json if available.
        old = (
            '        else:\n'
            '            raise e\n'
            '\n'
            '    if not isinstance(tokenizer, PreTrainedTokenizerFast):'
        )
        new = (
            '        else:\n'
            '            from pathlib import Path as _P\n'
            '            _tj = _P(tokenizer_name) / "tokenizer.json"\n'
            '            if _tj.exists():\n'
            '                logger.warning("Tokenizer class not found (%s). "\n'
            '                    "Falling back to PreTrainedTokenizerFast.", e)\n'
            '                from tokenizers import Tokenizer as _Tok\n'
            '                tokenizer = PreTrainedTokenizerFast(\n'
            '                    tokenizer_object=_Tok.from_file(str(_tj)))\n'
            '                import json as _j\n'
            '                _cp = _P(tokenizer_name) / "tokenizer_config.json"\n'
            '                if _cp.exists():\n'
            '                    _cc = _j.loads(_cp.read_text())\n'
            '                    for _k in ("eos_token", "pad_token"):\n'
            '                        if _cc.get(_k):\n'
            '                            setattr(tokenizer, _k, _cc[_k])\n'
            '            else:\n'
            '                raise e\n'
            '\n'
            '    if not isinstance(tokenizer, PreTrainedTokenizerFast):'
        )
        if old in content:
            content = content.replace(old, new, 1)
            hf_utils.write_text(content)
            log.info("Applied tokenizer fallback hotfix to hf_transformers_utils.py")
        else:
            log.debug("Tokenizer hotfix: target pattern not found, skipping")

    async def start_server(self, extra_args: list[str] | None = None,
                           timeout_s: float = DEFAULT_SERVER_TIMEOUT_S) -> bool:
        """Start the inference server and wait until healthy.

        If Sprint provided a launch script (detected by content, not name),
        use it directly — it has all the flags, env vars, and model path
        baked in.  Patches are applied first if a patch script was detected.
        """
        await self.kill_server()
        await asyncio.sleep(5)  # let processes fully exit before rogue scan
        rogues = await kill_rogue_servers(port=self.port)
        if rogues:
            log.warning("Cleaned up %d rogue server(s) before starting", rogues)
        self._write_server_owner_pid()

        # Apply patches before launching (launch scripts assume patches are applied)
        await self.apply_sprint_patches()

        sprint_script = getattr(self, '_sprint_launch_script', None)
        if sprint_script and Path(sprint_script).exists():
            cmd = f"bash {sprint_script} --background"
            log.info("Using Sprint launch script: %s", sprint_script)
            return await self._start_via_sprint_script(cmd, timeout_s)

        all_args = list(self.extra_launch_flags) + (extra_args or [])
        cmd_parts = self._build_server_cmd()
        cmd_parts.extend(all_args)
        cmd = " ".join(cmd_parts)

        env = {**os.environ, **self.env_vars,
               "INFERENCEX_PATH": str(self.inferencex_path),
               "MODEL_PATH": self.model}
        log_path = Path(self.result_dir) / "server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._server_log_path = str(log_path)

        log.info("Starting server: %s", cmd[:200])
        log_fh = open(log_path, "w")
        try:
            self._server_proc = await asyncio.create_subprocess_shell(
                cmd, stdout=log_fh, stderr=asyncio.subprocess.STDOUT, env=env,
            )
            if self._server_proc.pid:
                _write_server_pid(self._server_proc.pid, self.port)

            healthy = await self.wait_healthy(timeout_s=timeout_s)
            if not healthy:
                log.error("Server failed to become healthy within %ds", timeout_s)
                return False
            return True
        finally:
            log_fh.close()

    async def _start_via_sprint_script(self, cmd: str, timeout_s: float,
                                      _is_rollback_retry: bool = False) -> bool:
        """Run a Sprint launch script that has its own --background health loop.

        The script forks the server, polls /health internally, and exits 0
        when healthy or 1 on failure.  If the server fails to start and
        system files were modified by a Claw agent, automatically rolls back
        and retries once.
        """
        env = {**os.environ, **self.env_vars,
               "INFERENCEX_PATH": str(self.inferencex_path),
               "MODEL_PATH": self.model}
        log_path = Path(self.result_dir) / "server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._server_log_path = str(log_path)

        log.info("Starting server (sprint script): %s", cmd[:200])
        log_fh = open(log_path, "w")
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=log_fh, stderr=asyncio.subprocess.STDOUT, env=env,
        )

        try:
            retcode = await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            log.error("Sprint launch script timed out after %ds", timeout_s)
            retcode = -1
        finally:
            log_fh.close()

        # The Sprint script backgrounds the server via nohup then runs a
        # health-check loop with curl.  If the script has 'set -e', curl
        # failing with exit 7 (connection refused) kills the script even
        # though the server process is alive.  Treat any non-zero exit as
        # "script died early" and fall through to our own health polling.
        if retcode == 0:
            if await self._health_check():
                log.info("Server healthy (confirmed after sprint script)")
                await self._record_running_server_pid()
                return True

        if retcode != 0:
            log.warning(
                "Sprint script exited %d (likely set -e killed the health "
                "loop); checking if server process is alive", retcode)

        # The server was forked by the script — give it time to load the
        # model, warm up CUDA graphs, and become healthy (up to 15 min).
        log.info("Polling for server health (up to 900s)...")
        if await self.wait_healthy(timeout_s=900):
            await self._record_running_server_pid()
            return True

        # Server truly failed. Check if Claw modified system files.
        if not _is_rollback_retry:
            changed = self.system_files_changed()
            if changed:
                log.warning(
                    "Server failed to start and %d system file(s) were modified by "
                    "Claw agent: %s — rolling back and retrying",
                    len(changed), changed,
                )
                self.rollback_system_files()
                await self.kill_server()
                return await self._start_via_sprint_script(
                    cmd, timeout_s, _is_rollback_retry=True,
                )

        log.error("Sprint launch script failed (exit %d)", retcode)
        return False

    async def _record_running_server_pid(self) -> None:
        """Find the actual server process PID (after a sprint-script fork) and record it."""
        global _marathon_owns_server
        _marathon_owns_server = True
        proc = await asyncio.create_subprocess_shell(
            "ps -eo pid,args | grep -E 'vllm serve|vllm.entrypoints|sglang.srt|sglang.launch_server' | grep -v grep | head -1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        if stdout:
            parts = stdout.decode(errors="replace").strip().split(None, 1)
            try:
                pid = int(parts[0])
                _write_server_pid(pid, self.port)
            except (ValueError, IndexError):
                pass

    async def wait_healthy(self, timeout_s: float = 300) -> bool:
        elapsed = 0.0
        while elapsed < timeout_s:
            if await self._health_check():
                log.info("Server healthy after %.1fs", elapsed)
                return True
            await asyncio.sleep(HEALTH_POLL_S)
            elapsed += HEALTH_POLL_S
        return False

    async def _health_check(self) -> bool:
        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                f"curl -sf http://{self.host}:{self.port}/health",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5)
            return proc.returncode == 0
        except asyncio.TimeoutError:
            if proc is not None:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    pass
            return False
        except Exception:
            return False

    async def kill_server(self) -> bool:
        global _marathon_owns_server
        _marathon_owns_server = False
        log.info("Killing inference server …")
        kill_patterns = ["sglang.srt", "sglang.launch_server", "vllm.entrypoints"]
        fw = self.framework.lower()
        if fw not in ("sglang", "vllm"):
            kill_patterns.append(f"{fw}.")
        for pattern in kill_patterns:
            proc = await asyncio.create_subprocess_shell(
                f"pkill -f '{pattern}' || true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        await asyncio.sleep(SERVER_KILL_WAIT_S)

        proc = await asyncio.create_subprocess_shell(
            "fuser -v /dev/dri/renderD* 2>&1 || true",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        if stdout and b"python" in stdout.lower():
            log.warning("GPU processes lingering, force-killing")
            await asyncio.create_subprocess_shell(
                "fuser -k /dev/dri/renderD* 2>/dev/null || true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.sleep(3)

        self._server_proc = None
        _clear_server_pid()
        return not await self._health_check()

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    async def run_benchmark(self, num_prompts: int | None = None,
                            result_filename: str = "",
                            timeout_s: float = 1200) -> BenchmarkResult:
        """Run benchmark using InferenceX benchmark_serving.py directly.

        Always uses the deterministic direct path so baseline and DFS
        benchmarks have identical parameters (ISL, OSL, CONC, num_prompts,
        backend, warmups).  Sprint's run_benchmark.sh is intentionally
        bypassed — it may embed different defaults that skew comparisons.
        """
        return await self._run_inferencex_benchmark(num_prompts, result_filename, timeout_s)

    async def run_quick_sweep(
        self,
        concurrency_points: list[int] | None = None,
        num_prompts_per_point: int | None = None,
        result_tag: str = "quick_sweep",
        timeout_per_point_s: float = 300,
    ) -> list[dict[str, Any]]:
        """Quick 3-point concurrency sweep for multi-regime validation.

        Returns list of {concurrency, tput_per_gpu, mean_tpot_ms, mean_ttft_ms, result}.
        """
        points = concurrency_points or [4, 32, 128]
        results: list[dict[str, Any]] = []
        original_conc = self.concurrency

        for conc in points:
            self.concurrency = conc
            np = num_prompts_per_point or conc * 3
            tag = f"{result_tag}_conc{conc}"
            try:
                br = await self.run_benchmark(
                    num_prompts=np, result_filename=tag, timeout_s=timeout_per_point_s)
                results.append({
                    "concurrency": conc,
                    "tput_per_gpu": br.tput_per_gpu,
                    "output_throughput": br.output_throughput,
                    "mean_tpot_ms": br.mean_tpot_ms,
                    "mean_ttft_ms": br.mean_ttft_ms,
                    "p99_tpot_ms": br.p99_tpot_ms,
                    "p99_ttft_ms": br.p99_ttft_ms,
                    "num_prompts": br.num_prompts,
                    "result_file": br.result_file,
                })
                log.info("Sweep point conc=%d: tput=%.1f tok/s/GPU, TPOT=%.1fms",
                         conc, br.tput_per_gpu, br.mean_tpot_ms)
            except Exception as exc:
                log.error("Sweep point conc=%d failed: %s", conc, exc)
                results.append({
                    "concurrency": conc,
                    "tput_per_gpu": 0.0,
                    "error": str(exc),
                })

        self.concurrency = original_conc
        return results

    async def run_micro_benchmark(
        self,
        num_prompts: int = 8,
        timeout_s: float = 120,
        result_tag: str = "micro_oracle",
    ) -> BenchmarkResult:
        """Fast micro-benchmark (8 prompts, ~30s) as quick sanity filter.

        Used by the orchestrator's micro-oracle to decide whether a change
        is worth a full E2E benchmark.
        """
        original_conc = self.concurrency
        self.concurrency = 4
        try:
            result = await self.run_benchmark(
                num_prompts=num_prompts,
                result_filename=f"{result_tag}_{int(time.time())}",
                timeout_s=timeout_s,
            )
            return result
        finally:
            self.concurrency = original_conc

    @staticmethod
    def compare_sweep_results(
        baseline: list[dict[str, Any]],
        candidate: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare two sweep results and determine if candidate is better.

        Returns {verdict, overall_delta_pct, per_point, regime_notes}.
        verdict is one of: "keep", "revert", "conditional-keep"
        """
        if not baseline or not candidate:
            return {"verdict": "keep" if candidate else "revert",
                    "overall_delta_pct": 0.0, "per_point": []}

        per_point: list[dict[str, Any]] = []
        total_delta = 0.0
        regressions = 0
        improvements = 0

        baseline_by_conc = {b.get("concurrency"): b for b in baseline}
        candidate_by_conc = {c.get("concurrency"): c for c in candidate}
        all_concs = sorted(set(baseline_by_conc) | set(candidate_by_conc))
        for conc in all_concs:
            b = baseline_by_conc.get(conc)
            c = candidate_by_conc.get(conc)
            if not b or not c:
                continue
            b_tput = b.get("tput_per_gpu", 0) or 0.001
            c_tput = c.get("tput_per_gpu", 0) or 0.001
            delta_pct = (c_tput - b_tput) / b_tput * 100
            total_delta += delta_pct
            point = {
                "concurrency": b.get("concurrency"),
                "baseline_tput": b_tput,
                "candidate_tput": c_tput,
                "delta_pct": round(delta_pct, 2),
            }
            per_point.append(point)
            if delta_pct < -2.0:
                regressions += 1
            elif delta_pct > 1.0:
                improvements += 1

        avg_delta = total_delta / len(per_point) if per_point else 0
        regime_notes = []
        if regressions > 0 and improvements > 0:
            regime_notes.append("regime-dependent: gains at some concurrencies, regressions at others")

        if avg_delta > 1.0 and regressions == 0:
            verdict = "keep"
        elif avg_delta > 2.0 and regressions <= 1:
            verdict = "conditional-keep"
            regime_notes.append(f"{regressions} regime(s) regressed but overall positive")
        elif avg_delta < -1.0:
            verdict = "revert"
        else:
            verdict = "keep" if avg_delta >= 0 else "revert"

        return {
            "verdict": verdict,
            "overall_delta_pct": round(avg_delta, 2),
            "per_point": per_point,
            "regime_notes": regime_notes,
        }

    async def _run_sprint_benchmark(self, script_path: str,
                                    result_filename: str,
                                    timeout_s: float,
                                    num_prompts: int | None = None) -> BenchmarkResult:
        """Run Sprint's run_benchmark.sh which has all params baked in."""
        result_dir = Path(self.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        env = {
            **os.environ,
            **self.env_vars,
            "CONC": str(self.concurrency),
            "ISL": str(self.isl),
            "OSL": str(self.osl),
            "RESULT_DIR": str(result_dir),
            "TAG": result_filename or f"marathon_{int(time.time())}",
            "INFERENCEX_PATH": str(self.inferencex_path),
        }
        if num_prompts is not None:
            env["NUM_PROMPTS"] = str(num_prompts)

        log.info("Running Sprint benchmark script: %s (CONC=%d, ISL=%d, OSL=%d)",
                 script_path, self.concurrency, self.isl, self.osl)

        proc = await asyncio.create_subprocess_shell(
            f"bash {script_path} {self.concurrency}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env=env,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            log.error("Sprint benchmark timed out after %ds", timeout_s)
            return BenchmarkResult()

        if proc.returncode != 0:
            log.error("Sprint benchmark failed (exit %d): %s",
                      proc.returncode, (stdout or b"").decode()[:500])
            return BenchmarkResult()

        # Find the result JSON — Sprint scripts use a naming convention
        tag = result_filename or f"marathon_{int(time.time())}"
        pattern = f"*tp{self.tp}_conc{self.concurrency}_isl{self.isl}_osl{self.osl}*.json"
        candidates = sorted(result_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            candidates = sorted(result_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return self.parse_result(str(candidates[0]))

        log.warning("No result JSON found after Sprint benchmark")
        return BenchmarkResult()

    async def _run_inferencex_benchmark(self, num_prompts: int | None,
                                        result_filename: str,
                                        timeout_s: float) -> BenchmarkResult:
        """Run InferenceX benchmark_serving.py directly with controlled params."""
        if num_prompts is None:
            num_prompts = self.concurrency * self.num_prompts_multiplier

        warmups = min(self.concurrency * 2, num_prompts // 2)

        if not result_filename:
            result_filename = f"bench_{int(time.time())}"

        bench_script = self.inferencex_path / "utils" / "bench_serving" / "benchmark_serving.py"
        if not bench_script.exists():
            bench_script = self.inferencex_path / "benchmarks" / "benchmark_serving.py"
        if not bench_script.exists():
            raise FileNotFoundError(
                f"benchmark_serving.py not found in {self.inferencex_path}")

        result_dir = Path(self.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        backend_name = self.framework.lower()

        cmd = [
            "python3", str(bench_script),
            "--model", self.model,
            "--backend", backend_name,
            "--base-url", f"http://{self.host}:{self.port}",
            "--dataset-name", "random",
            "--random-input-len", str(self.isl),
            "--random-output-len", str(self.osl),
            "--random-range-ratio", "1.0",
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(self.concurrency),
            "--request-rate", "inf",
            "--ignore-eos",
            "--num-warmups", str(warmups),
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--save-result",
            "--result-dir", str(result_dir),
            "--result-filename", f"{result_filename}.json",
        ]

        log.info("Running benchmark: %d prompts, ISL=%d, OSL=%d, CONC=%d, "
                 "warmups=%d, backend=%s",
                 num_prompts, self.isl, self.osl, self.concurrency,
                 warmups, backend_name)

        await self._kill_stale_benchmarks()

        if not await self._check_server_alive():
            log.error("Inference server not responding on %s:%s — skipping benchmark",
                      self.host, self.port)
            return BenchmarkResult()

        env = {**os.environ, **self.env_vars}
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env=env,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                self._monitor_benchmark(proc, timeout_s),
                timeout=timeout_s + 30,  # extra headroom for monitor cleanup
            )
        except asyncio.TimeoutError:
            proc.kill()
            log.error("Benchmark timed out after %ds", timeout_s)
            return BenchmarkResult()

        if proc.returncode != 0:
            log.error("Benchmark failed (exit %d): %s",
                      proc.returncode, (stdout or b"").decode()[:500])
            return BenchmarkResult()

        result_path = result_dir / f"{result_filename}.json"
        bench_result = self.parse_result(str(result_path))

        # Augment benchmark JSON with reproducibility metadata
        if result_path.exists() and bench_result.output_throughput > 0:
            try:
                raw = json.loads(result_path.read_text())
                _REPRO_ENV_KEYS = [
                    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                    "AMDGCN_USE_BUFFER_OPS", "VLLM_ROCM_USE_AITER",
                    "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM",
                    "VLLM_ROCM_USE_AITER_TRITON_ROPE",
                    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
                    "AITER_CONFIG_FMOE", "CU_NUM",
                ]
                raw["_marathon_meta"] = {
                    "tp": self.tp,
                    "resolved_tp": bench_result.tput_per_gpu and round(
                        bench_result.output_throughput / bench_result.tput_per_gpu),
                    "isl": self.isl,
                    "osl": self.osl,
                    "concurrency": self.concurrency,
                    "port": self.port,
                    "framework": self.framework,
                    "env_vars": {k: os.environ.get(k, "") for k in _REPRO_ENV_KEYS},
                    "serve_script": getattr(self, '_sprint_launch_script', None),
                }
                result_path.write_text(json.dumps(raw, indent=2))
            except Exception:
                pass

        return bench_result

    async def _check_server_alive(self, retries: int = 3, delay: float = 5.0) -> bool:
        """Deep health check: verify /health AND a single-token inference round-trip.

        A server can return /health 200 while being unable to process inference
        (e.g. corrupted model weights from a bad patch, GPU context errors).
        This sends a tiny completions request to prove the full pipeline works.
        """
        import aiohttp

        health_url = f"http://{self.host}:{self.port}/health"
        completions_url = f"http://{self.host}:{self.port}/v1/completions"

        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    # Phase 1: basic /health check
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"/health returned {resp.status}")

                    # Phase 2: single-token inference probe
                    probe_payload = {
                        "model": self.model,
                        "prompt": "test",
                        "max_tokens": 1,
                    }
                    async with session.post(
                        completions_url,
                        json=probe_payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status == 200:
                            return True
                        body = await resp.text()
                        raise RuntimeError(f"Inference probe returned {resp.status}: {body[:200]}")
            except Exception as exc:
                level = logging.WARNING if attempt < retries - 1 else logging.ERROR
                log.log(level,
                        "Server alive check failed (attempt %d/%d): %s",
                        attempt + 1, retries, exc)
            if attempt < retries - 1:
                await asyncio.sleep(delay)
        return False

    @staticmethod
    async def _get_gpu_utilization() -> float:
        """Return max GPU utilization (0-100) across all visible GPUs, or -1 on error."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "rocm-smi", "--showuse",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            max_util = 0.0
            for line in stdout.decode().splitlines():
                if "GPU use (%)" in line:
                    try:
                        val = float(line.split(":")[-1].strip())
                        max_util = max(max_util, val)
                    except ValueError:
                        pass
            return max_util
        except Exception:
            pass
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            vals = [float(x.strip()) for x in stdout.decode().splitlines() if x.strip()]
            return max(vals) if vals else -1
        except Exception:
            return -1

    async def _monitor_benchmark(
        self,
        proc: asyncio.subprocess.Process,
        timeout_s: float,
    ) -> tuple[bytes, None]:
        """Wrap proc.communicate() with periodic GPU idle detection.

        If GPU utilization stays at 0% for GPU_IDLE_KILL_S consecutive
        seconds, the benchmark is stuck and we kill it early instead of
        waiting for the full timeout.
        """
        GPU_IDLE_KILL_S = 90
        GPU_CHECK_INTERVAL_S = 15

        communicate_task = asyncio.ensure_future(proc.communicate())
        idle_seconds = 0
        check_start = time.monotonic()

        while not communicate_task.done():
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(communicate_task),
                    timeout=GPU_CHECK_INTERVAL_S,
                )
                return result
            except asyncio.TimeoutError:
                pass

            elapsed = time.monotonic() - check_start
            if elapsed < 30:
                continue

            gpu_util = await self._get_gpu_utilization()
            if gpu_util < 0:
                continue
            if gpu_util < 1.0:
                idle_seconds += GPU_CHECK_INTERVAL_S
                log.warning("GPU idle during benchmark (%d/%ds before kill)",
                            idle_seconds, GPU_IDLE_KILL_S)
                if idle_seconds >= GPU_IDLE_KILL_S:
                    log.error("GPU idle for %ds — benchmark is stuck, killing", idle_seconds)
                    proc.kill()
                    try:
                        return await asyncio.wait_for(communicate_task, timeout=5)
                    except asyncio.TimeoutError:
                        communicate_task.cancel()
                        return (b"", None)
            else:
                idle_seconds = 0

            if elapsed > timeout_s:
                break

        if not communicate_task.done():
            proc.kill()
            try:
                return await asyncio.wait_for(communicate_task, timeout=5)
            except asyncio.TimeoutError:
                communicate_task.cancel()
                return (b"", None)
        return communicate_task.result()

    @staticmethod
    async def _kill_stale_benchmarks() -> None:
        """Kill any orphaned benchmark_serving.py processes before starting a new run."""
        try:
            find = await asyncio.create_subprocess_exec(
                "pgrep", "-f", "benchmark_serving",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(find.communicate(), timeout=5)
            pids = stdout.decode().split()
            if pids:
                log.warning("Killing %d stale benchmark process(es): %s",
                            len(pids), " ".join(pids))
                kill = await asyncio.create_subprocess_exec(
                    "kill", "-9", *pids,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(kill.communicate(), timeout=5)
                await asyncio.sleep(1)
        except Exception:
            pass

    def parse_result(self, result_path: str) -> BenchmarkResult:
        """Parse InferenceX benchmark JSON result file."""
        p = Path(result_path)
        if not p.exists():
            log.warning("Benchmark result file not found: %s", result_path)
            return BenchmarkResult()

        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to parse benchmark result: %s", exc)
            return BenchmarkResult()

        output_tput = float(data.get("output_throughput", 0))
        total_tput = float(data.get("total_token_throughput", 0))

        # Resolve TP: prefer live server value, fall back to self.tp
        tp = self.tp
        try:
            import urllib.request
            resp = urllib.request.urlopen(
                f"http://{self.host}:{self.port}/v1/models", timeout=5)
            models = json.loads(resp.read())
            live_tp = models.get("data", [{}])[0].get("tensor_parallel_size")
            if live_tp and int(live_tp) != tp:
                log.warning("TP mismatch: self.tp=%d but server reports tp=%d — using server value",
                            tp, int(live_tp))
                tp = int(live_tp)
        except Exception:
            pass
        tput_per_gpu = total_tput / max(tp, 1)

        return BenchmarkResult(
            output_throughput=output_tput,
            total_token_throughput=total_tput,
            tput_per_gpu=tput_per_gpu,
            request_throughput=float(data.get("request_throughput", 0)),
            mean_ttft_ms=float(data.get("mean_ttft_ms", 0)),
            mean_tpot_ms=float(data.get("mean_tpot_ms", 0)),
            p99_ttft_ms=float(data.get("p99_ttft_ms", 0)),
            p99_tpot_ms=float(data.get("p99_tpot_ms", 0)),
            num_prompts=int(data.get("num_prompts", 0)),
            result_file=result_path,
            raw=data,
        )

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    async def run_profile(self, profiler: str = "rocprof",
                          output_dir: str = "",
                          timeout_s: float = 300) -> dict[str, Any]:
        """Run GPU profiler and return raw output path."""
        out_dir = Path(output_dir or self.result_dir) / "profiles"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())

        if profiler == "rocprof" and shutil.which("rocprof"):
            out_file = out_dir / f"rocprof_{ts}.csv"
            cmd = (
                f"rocprof --stats --timestamp on "
                f"-o {out_file} "
                f"timeout 30 python3 -c 'import requests; "
                f'requests.post("http://{self.host}:{self.port}/generate", '
                f'json={{"text":"Hello","sampling_params":{{"max_new_tokens":64}}}})\''
            )
        elif shutil.which("nsys"):
            out_file = out_dir / f"nsys_{ts}"
            cmd = (
                f"nsys profile --stats=true --output={out_file} "
                f"timeout 30 python3 -c 'import requests; "
                f'requests.post("http://{self.host}:{self.port}/generate", '
                f'json={{"text":"Hello","sampling_params":{{"max_new_tokens":64}}}})\''
            )
        else:
            log.warning("No profiler found (rocprof/nsys)")
            return {"error": "no profiler available"}

        log.info("Running %s profile …", profiler)
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return {"error": "profile timed out"}

        return {
            "profiler": profiler,
            "output_file": str(out_file),
            "returncode": proc.returncode,
        }

    # ------------------------------------------------------------------
    # Patch management
    # ------------------------------------------------------------------

    async def apply_patches(self, patch_dir: str | Path) -> list[PatchResult]:
        """Apply git patches from a directory. Returns per-patch results."""
        results: list[PatchResult] = []
        pdir = Path(patch_dir)
        if not pdir.exists():
            log.info("No patch directory at %s", patch_dir)
            return results

        for patch_file in sorted(pdir.glob("*.patch")):
            pr = PatchResult(patch_file=str(patch_file))

            repo_name = patch_file.stem
            repo_candidates = [
                Path("/sgl-workspace") / repo_name,
                Path("/workspace") / repo_name,
            ]
            repo_path = None
            for candidate in repo_candidates:
                if candidate.exists():
                    repo_path = candidate
                    break

            if not repo_path:
                pr.error = f"repo directory not found for {repo_name}"
                results.append(pr)
                continue

            # Check if already applied via reverse-apply check
            check = subprocess.run(
                ["git", "apply", "-R", "--check", str(patch_file)],
                cwd=str(repo_path), capture_output=True,
            )
            if check.returncode == 0:
                pr.applied = True
                pr.already_applied = True
                log.info("Patch already applied: %s", patch_file.name)
                results.append(pr)
                continue

            # Try to apply
            apply = subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=str(repo_path), capture_output=True,
            )
            if apply.returncode == 0:
                pr.applied = True
                log.info("Applied patch: %s", patch_file.name)
            else:
                pr.error = apply.stderr.decode(errors="replace")[:200]
                log.warning("Failed to apply %s: %s", patch_file.name, pr.error)

            results.append(pr)

        return results
