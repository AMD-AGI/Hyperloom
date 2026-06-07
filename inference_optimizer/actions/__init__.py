# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Optimization action catalogue.

Layout:

* ``_meta/<name>.yaml`` — machine-readable action metadata loaded by
  :class:`inference_optimizer.orchestrator.action_registry.ActionRegistry`
* ``<name>.md`` — agent-facing playbook (loaded lazily by SubAgentRunner
  when composing a sub-agent prompt; not required for PolicyGate)

v0.6 ships **19 OptimizationActions** total. The 5 "kernel-owned" actions
(kernel_opt / integrate / deep_kernel_analysis / operator_tuning /
vendor_kernel_config) are reachable only via REQUEST(target_agent="kernel")
— PolicyGate rejects direct delegate of any name in
:data:`inference_optimizer.orchestrator.policy.KERNEL_OWNED_ACTIONS`.
"""
