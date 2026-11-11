# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every test here keeps the scratch of its helpers that are not fixtures (a fake toolbox's submit log and blob store)
below pytest's base temporary directory, so a run that is killed leaves nothing in the host's temporary directory."""

from hyperloom.measure.tests.gpu_free_host import own_test_scratch  # noqa: F401  (autouse fixture)
