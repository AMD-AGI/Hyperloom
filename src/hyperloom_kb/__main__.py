# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m hyperloom_kb`` runs the Experience service, including from a ``pip install --target`` workspace."""

from hyperloom_kb.http_service import main

if __name__ == "__main__":
    raise SystemExit(main())
