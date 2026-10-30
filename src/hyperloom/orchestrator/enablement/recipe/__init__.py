# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recipe projection: the ordered enablement replay contract and its sufficiency.

``steps`` serializes the setup -> build -> patch ordering enablement executes
but never recorded; ``sufficiency`` is the authoritative verdict on whether
that recipe can actually be replayed.
"""
