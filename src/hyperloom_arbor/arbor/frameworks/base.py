"""What a serving framework gives the harness: launch, find, stop, probe.

The harness launches a server, finds its process, reads its log, stops it and
probes its health, and each of those named vLLM or SGLang where it was done:
the launch in arbor/policy/claude_loop.py, the process and log patterns in
arbor/measure/armcheck.py, the stop sweep in arbor/execute/preflight.py. An
adapter holds one framework's half of each, so a third framework is one module
in this package rather than an edit at every site.

The two frameworks moved here with no change in behaviour: every command,
pattern and default below is what those sites produced, and
arbor/tests/test_frameworks_parity.py pins them. Plain classes and no
dependencies, so anything that imports armcheck still imports on a laptop.
"""

from __future__ import annotations

import re
import shlex

# Where the server's stdout and stderr go inside the container. The launch
# redirects there, armcheck tails it, and the server agent is told to watch it.
SERVER_LOG = "/tmp/server.log"

# SGLang's HiCache trace (SGLANG_KV_TRACE_PATH), which claude_loop's
# _collect_kv_trace copies out of the container after the bench.
KV_TRACE_PATH = "/tmp/kv_trace.jsonl"

# The ROCm stack under every framework: the env families amd-knobs.md §0 rule 1
# greps besides the framework's own.
STACK_ENV_PREFIXES = (
    "AITER_", "ROCM_", "HSA_", "HIP_", "GPU_", "NCCL_", "RCCL_", "TORCH_", "PYTORCH_",
    "TRITON_", "MIOPEN_", "ATOM_", "MORI_", "AMDGCN_",
)


def env_prefix(server_env: dict) -> str:
    """``env K=V ... `` for the front of a command, or "" when there is nothing to set.

    Sorted by key, each value shell-quoted. A key with a leading underscore is a
    note in the config, not env.
    """
    env_s = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in sorted(server_env.items())
                     if not k.startswith("_"))
    return f"env {env_s} " if env_s else ""


class FrameworkAdapter:
    """One serving framework, as the harness launches, finds, stops and probes its server."""

    # The value of a config's "framework" that names it, lower-case.
    name = ""
    # The server's command before its arguments, unless the host gives its own (`entrypoint`).
    default_entrypoint = ""
    # The server process, for `pgrep -af`. A bracket in it keeps pgrep's own
    # command line from matching.
    server_pattern = ""
    # `pkill -TERM -f` patterns: the server, at once ...
    stop_patterns: tuple[str, ...] = ()
    # ... and what it leaves running, once the grace period has passed.
    post_grace_patterns: tuple[str, ...] = ()
    # Whether the server survives SIGKILL without leaking GPU memory. stop_server
    # sends SIGTERM only, whatever this says.
    allow_sigkill = False
    health_path = "/health"
    # The env armcheck captures from the live process. A key outside them is not
    # captured, so it is never reported missing.
    env_prefixes: tuple[str, ...] = STACK_ENV_PREFIXES
    # Log lines armcheck keeps beside the archived tail.
    log_patterns: tuple[re.Pattern, ...] = ()
    server_log = SERVER_LOG
    # The framework's compiled extension: a tree that has the package but cannot import
    # this does not serve (a source-only worktree; component mode refuses it at start).
    native_module = ""

    def entrypoint(self, config: dict) -> str:
        """The server's command before its arguments: the host's ``component.entrypoint`` when it
        gives one (the entrypoint of its base launch, arbor/base_launch.py), else the framework's
        own. Component mode stops its server by PID, so another entrypoint needs no stop pattern."""
        block = (config or {}).get("component")
        given = block.get("entrypoint") if isinstance(block, dict) else None
        if isinstance(given, list) and given:
            return shlex.join(str(t) for t in given)
        return self.default_entrypoint

    def serve_cmd(self, config: dict, server_flags: str) -> str:
        """The server's own command line, in the foreground, flags last."""
        raise NotImplementedError

    def launch_cmd(self, config: dict, server_env: dict, server_flags: str,
                   log_path: str | None = None, pid_file: str | None = None) -> str:
        """The harness's launch: ``serve_cmd`` in the background, the arm's env in
        front of it as ``env K=V ...``, its output in ``server_log``, its PID echoed.

        ``log_path`` replaces ``server_log``. ``pid_file`` is component mode's launch
        (arbor/component.py): the server starts in a session and process group of its
        own (``setsid``), and its shell writes its own PID there before ``exec``-ing it,
        so the PID is the server's and its group's whatever ``setsid`` had to do. Arbor
        then stops that group and nothing else: inside MB, standalone's pattern
        ``pkill`` stopped MB's own servers. Without either, the command is byte for
        byte what it always was (arbor/tests/test_frameworks_parity.py).
        """
        serve = f"{env_prefix(server_env)}{self.serve_cmd(config, server_flags)}"
        if pid_file is None:
            log = self.server_log if log_path is None else shlex.quote(str(log_path))
            return f"nohup {serve} > {log} 2>&1 & echo $!"
        inner = f"echo $$ > {shlex.quote(str(pid_file))}; exec {serve}"
        log = shlex.quote(str(log_path or self.server_log))
        return f"setsid bash -c {shlex.quote(inner)} > {log} 2>&1 < /dev/null & echo $!"

    def help_cmd(self) -> str:
        """What lists the server's flags (the server agent greps it)."""
        raise NotImplementedError

    def extra_env(self, server_flags: str) -> dict[str, str]:
        """Env the server's flags call for, set over the arm's own.

        One case: HiCache (`--enable-hierarchical-cache`, an SGLang flag) turns on
        SGLang's KV trace. _start_server set it whatever the framework, so it
        stays on the base and vLLM inherits it; vLLM has no such flag and does not
        start with it, so there it never mattered. Making it SGLang's alone would
        change a vLLM command, which this move does not do.
        """
        if "--enable-hierarchical-cache" in server_flags:
            return {"SGLANG_KV_TRACE_PATH": KV_TRACE_PATH}
        return {}

    def device_env(self, env: dict, gpu_ids) -> dict:
        """``env`` with the server confined to ``gpu_ids`` ("4,5,6,7"); ``env`` as it
        is when none are given.

        ROCR_VISIBLE_DEVICES selects the cards. HIP indexes into the list ROCR has
        already filtered, so a HIP_VISIBLE_DEVICES copied from gpu_ids double-filters:
        with "0,1,2,3" that is the identity, with any other selection the server
        silently gets the wrong devices. So HIP_VISIBLE_DEVICES is set to the identity
        on ROCR's list, "0,..,n-1", and a configured one is replaced. It must be set:
        Ray, which vLLM 0.27 imports at start, refuses ROCR_VISIBLE_DEVICES without it
        ("Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES"), and every
        server the harness itself started failed so (every A/B of one run).
        """
        if not gpu_ids:
            return env
        ids = [g.strip() for g in str(gpu_ids).split(",") if g.strip()]
        return {**env, "ROCR_VISIBLE_DEVICES": ",".join(ids),
                "HIP_VISIBLE_DEVICES": ",".join(str(k) for k in range(len(ids)))}
