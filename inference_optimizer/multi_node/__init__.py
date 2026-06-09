# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Multi-node inference optimization helpers.

Used when a session needs more GPU memory than one pod provides. The
sandbox agent drives this CLI (``python3 -m inference_optimizer.multi_node
<subcommand>``) to create one session-scoped SaFE RayJob with N GPU pods,
bootstrap the toolchain, restart servers without recreating pods (so the
aiter JIT cache survives), and stop the RayJob at session end.

Control channels: sandbox↔SaFE via REST; sandbox↔inference RayJob via Ray
Dashboard REST (port 8265). Per ADDENDUM-02, this package must not
``import ray`` / ``ray.init(address=...)`` for that cluster.
"""
