# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``inference_suite`` -- resolves a serving framework's base launch and runs one inference round on it.

It builds only on ``hyperloom_common`` and third-party libraries; ``.importlinter`` holds the
contracts that keep it so, and names the modules other packages may import.
"""

from __future__ import annotations

__all__: list[str] = []
