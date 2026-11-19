# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a launch's answer is held to the reference's: ``res_out`` bit for bit, ``x`` within
``X_ULP_MAX`` bf16 steps on at most ``X_ULP_FRAC_MAX`` of its elements, every guard error word clear.

Every operation that produces ``res_out`` is fixed by the contract, order included, so nothing excuses
a bit. ``x`` divides by a sum of squares the launch may reduce in any order, and a few f32 ulps in its
rstd carry an element across a bf16 rounding boundary now and then."""

import torch

X_ULP_MAX = 1
X_ULP_FRAC_MAX = 1e-3


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in bf16 steps: each value's bits as a point on one ordered integer line."""

    def line(t: torch.Tensor) -> torch.Tensor:
        bits = t.view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (line(a) - line(b)).abs()


def judge(got, want, error_words: list[int]) -> tuple[bool, dict[str, int | float | str]]:
    """``got`` and ``want``: (x, res_out). Returns the verdict and the numbers it rests on."""
    (x, res_out), (x_ref, res_ref) = got, want
    res_exact = torch.equal(res_out.view(torch.int16), res_ref.view(torch.int16))
    finite = bool(torch.isfinite(x).all())
    ulp = ulp_distance(x, x_ref)
    ulp_max, ulp_frac = int(ulp.max()), float((ulp > 0).float().mean())
    passed = res_exact and finite and ulp_max <= X_ULP_MAX and ulp_frac <= X_ULP_FRAC_MAX and not any(error_words)
    return passed, {
        "res_exact": int(res_exact),
        "x_finite": int(finite),
        "x_ulp_max": ulp_max,
        "x_ulp_frac": ulp_frac,
        "errors": "-".join(map(str, error_words)),
    }
