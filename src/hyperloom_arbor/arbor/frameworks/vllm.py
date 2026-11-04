"""vLLM: `vllm serve`, with the config's TP and both the served name and the model path."""

from __future__ import annotations

import re

from arbor.frameworks.base import STACK_ENV_PREFIXES, FrameworkAdapter


class VLLM(FrameworkAdapter):
    name = "vllm"
    server_pattern = "[v]llm serve"
    native_module = "vllm._C"
    # `vllm serve` is the API server, which takes its engine down on SIGTERM; an
    # EngineCore or worker it leaves behind renames itself `VLLM::...`, so those
    # get SIGTERM too once the grace period has passed. Model M B on Cluster B
    # (2026-10-07) kept a server and its EngineCore holding ~265 GB per GPU
    # through every "stop" when only SGLang was signalled.
    stop_patterns = ("vllm serve",)
    post_grace_patterns = ("VLLM::",)
    env_prefixes = ("VLLM_",) + STACK_ENV_PREFIXES
    # vLLM's own list of the arguments it was given, and the lines that name what
    # it selected ("Using AITER MoE backend", "Using ROCM_AITER_FA attention
    # backend", "Selected backend ..."). Deliberately short: armcheck archives the
    # whole tail beside them, these are what a reader looks at.
    log_patterns = (
        re.compile(r"non-default args"),
        re.compile(r"\bUsing\b.*\bbackend\b", re.IGNORECASE),
        re.compile(r"attention backend", re.IGNORECASE),
        re.compile(r"\bselected\b.*\bbackend\b|\bbackend\b.*\bselected\b", re.IGNORECASE),
    )

    def serve_cmd(self, config: dict, server_flags: str) -> str:
        model_path = config.get("model_path", "")
        model_name = config.get("model_name", "")
        tp = config.get("tp", 1)
        # Register BOTH the served name and the model path. The pack's
        # correctness.sh posts {"model": MODEL_PATH} while bench_agentx.sh posts
        # the served name; a server registering only one 404s the other.
        names = model_name if model_path in ("", model_name) else f"{model_name} {model_path}"
        return (
            f"vllm serve {model_path} "
            f"--served-model-name {names} "
            f"--host 0.0.0.0 --port {config.get('port', 8888)} "
            f"--tensor-parallel-size {tp} "
            f"{server_flags}"
        )

    def help_cmd(self) -> str:
        return "vllm serve --help"
