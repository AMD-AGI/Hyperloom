# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node inference optimization helpers.

Used when a session needs more GPU memory than one pod provides. The platform
provisions the cluster (RayJob or InferaDeployment) and hands it over through
``HYPERLOOM_MN_EXT_*``; the sandbox agent then drives this CLI (``python3 -m
hyperloom.inference_optimizer.multi_node <subcommand>``) to bootstrap the
toolchain and restart servers without recreating pods (so the aiter JIT cache
survives). It never creates or releases the cluster.

Control channels: sandbox↔inference RayJob via Ray Dashboard REST (port 8265),
sandbox↔Infera pods via SSH. This package must not ``import ray`` /
``ray.init(address=...)`` for that cluster.
"""
