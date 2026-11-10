# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every test here runs the same on a GPU host and on a CPU host, busy or idle.

The local peers build the real local toolbox, whose card pool asks the kernel's list who holds a card before it lends
one: a test that read the host's own list would find no free card wherever other work is on the cards. Every test
here reads an idle machine of its own and no GPU tool of the host (:mod:`hyperloom.measure.tests.gpu_free_host`).
"""

from hyperloom.measure.tests.gpu_free_host import (  # noqa: F401  (autouse fixtures)
    idle_card_list,
    machine_without_readable_gpus,
)
