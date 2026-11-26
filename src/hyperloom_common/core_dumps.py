# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep a process that faults the GPU from dumping its memory to disk.

After a GPU fault ROCr writes the process's GPU memory into its working directory (``gpucore.<pid>.gpu``), one file per
process and as large as the memory it held: 35 GB for one generated tuner, 154 GiB for each rank of a large model's
server. ROCr reads ``HSA_DISABLE_COREDUMP_ON_EXCEPTION`` once at runtime load, and also skips the dump whenever
``RLIMIT_CORE``'s soft limit is 0 at fault time.
"""

from __future__ import annotations

import resource

NO_GPU_COREDUMP_ENV = {"HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1"}


def forbid_core_dumps() -> None:
    """Set this process's core file limit to 0, the hard limit too, so nothing it starts can raise it back; meant as a
    child's ``preexec_fn``."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


__all__ = ["NO_GPU_COREDUMP_ENV", "forbid_core_dumps"]
