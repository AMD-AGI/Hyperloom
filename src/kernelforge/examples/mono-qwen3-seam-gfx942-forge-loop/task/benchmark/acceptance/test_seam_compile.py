# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every width compiles for this GPU, and its whole grid is resident on it."""

import torch
from ranks import TP, WEIGHT_SEED
from runner import SeamRunner
from weights import random_layer

from qwen3_seam import seam


def test_every_width_compiles_with_a_resident_grid():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    # warming makes launches that return at once, which never read the peers: a null table stands in
    runner = SeamRunner(TP, 0, 0, torch.zeros(TP, dtype=torch.int64, device=device), device)
    compute_units = torch.cuda.get_device_properties(device).multi_processor_count
    runner.warm(random_layer(TP, 0, device, WEIGHT_SEED), compute_units)
    assert sorted(runner.residency) == list(range(1, seam.MAX_TOKENS + 1))
    short = {s: n for s, n in runner.residency.items() if n * compute_units < seam.BLOCKS}
    assert not short, f"widths whose {seam.BLOCKS}-CTA grid is not resident (CTAs per CU): {short}"
