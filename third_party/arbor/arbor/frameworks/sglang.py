"""SGLang: `python3 -m sglang.launch_server`, the harness's first framework and still its default."""

from __future__ import annotations

from arbor.frameworks.base import STACK_ENV_PREFIXES, FrameworkAdapter


class SGLang(FrameworkAdapter):
    name = "sglang"
    server_pattern = "[s]glang.launch_server"
    native_module = "sgl_kernel"
    stop_patterns = ("sglang.launch_server",)
    # NEVER SIGKILL: SGLang holds KFD GPU memory until the Python process exits
    # cleanly. SIGKILL skips that cleanup, leaving 250 GB of zombie VRAM that
    # blocks the next launch with a memory-imbalance error.
    allow_sigkill = False
    env_prefixes = ("SGLANG_",) + STACK_ENV_PREFIXES
    # No log lines of SGLang's own are catalogued. armcheck reads every log with
    # the union of all adapters' patterns, so an SGLang log is read with vLLM's.

    def serve_cmd(self, config: dict, server_flags: str) -> str:
        # One served name, and no TP: SGLang takes it in the flags (`--tp`).
        return (
            f"python3 -m sglang.launch_server "
            f"--model-path {config.get('model_path', '')} "
            f"--served-model-name {config.get('model_name', '')} "
            f"--host 0.0.0.0 --port {config.get('port', 8888)} "
            f"{server_flags}"
        )

    def help_cmd(self) -> str:
        return "python3 -m sglang.launch_server --help"
