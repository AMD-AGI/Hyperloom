"""Multi-node inference optimization helpers.

Used when the optimization session needs more GPU memory than a single
pod provides. The agent inside the Claw sandbox calls this CLI (via
``python3 -m inference_optimizer.multi_node <subcommand>``) to:

1. Create one session-scoped SaFE RayJob with N GPU pods.
2. Bootstrap the toolchain (oob/claude/codex/tracelens) inside it.
3. Restart vllm/sglang servers as many times as the optimization needs,
   without ever recreating the underlying Ray pods (so aiter JIT cache
   on each pod's rootfs survives every restart).
4. Stop the RayJob when the session finishes.

Sandbox <-> SaFE: REST. Sandbox <-> inference RayJob: Ray Dashboard REST
(port 8265 on the head pod's PodIp). Orchestration code in this package
must not ``import ray`` or ``ray.init(address=...)`` for that cluster — see
ADDENDUM-02. Installing ``ray`` in the sandbox for other uses is fine.

See ``inference_optimizer/multi_node/SKILL.md`` for the LLM-facing
playbook.
"""
