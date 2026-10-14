# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notebook's request and result rows: the measurement contract's own types, as the notebook stores them.

The request row carries the priority the queue gave the test. A ``measurement_failed`` test is still written as a
withdrawal; the notebook would also take it as a result row that releases the claim.
"""

from __future__ import annotations

from dataclasses import replace

from hyperloom.contracts import measurement as m


def notebook_request(request: m.TestRequest, priority: int) -> m.TestRequest:
    return replace(request, priority=priority)


def notebook_result(result: m.TestResult) -> m.TestResult:
    """The row for a measured test; ``measurement_failed`` is withdrawn instead, never written as a result."""

    if result.validity is m.Validity.MEASUREMENT_FAILED:
        raise ValueError(f"a {result.validity.value} test has no notebook result row")
    return result
