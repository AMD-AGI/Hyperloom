<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

Port the fixed AttnRes score operator in kernel.py to kernel.s, then optimize
only its assembly. During PORT, copy the attributed seed/score.s to kernel.s
and seed/launcher.py to kernel.py. The seed is Neha's published implementation,
not a Forge-discovered optimization. Verify it against the independent driver
before changing instructions. The public interface is Score() followed by
Score.__call__(prefix, bank, weight, output) and close(). Keep the constructor
outside timing and graph capture. The driver and seed directory are read-only.
After PORT, keep the launcher fixed; try one measured scheduling or register
allocation hypothesis at a time. Report source, initial ASM and optimized ASM
timings separately; a successful port does not imply a speedup.
