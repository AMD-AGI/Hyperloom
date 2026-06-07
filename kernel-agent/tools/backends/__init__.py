# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Self-contained backend wrappers for kernel-agent.

These modules provide GEAK and OOB submission via Ray (preferred) or direct
CLI fallback. They live inside the kernel-agent skill so the skill is fully
self-contained.
"""
