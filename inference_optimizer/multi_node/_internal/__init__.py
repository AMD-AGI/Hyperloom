# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Internal helpers for ``inference_optimizer.multi_node``.

Everything in here is implementation detail: ``log``, ``safe_client``,
``ray_dashboard``, ``workload_spec``. The only public surface is the
``inference_optimizer.multi_node`` CLI, invoked as
``python3 -m inference_optimizer.multi_node <subcommand>`` from inside
the sandbox.
"""
