# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Backwards-compatibility shims for inference_optimizer.

Modules in this sub-package implement deprecation-window helpers for
field renames, schema migrations, and other compatibility surfaces
that are expected to shrink over time.

Current members:

* :mod:`inference_optimizer.compat.payload_aliases` — read-only
  alias from the legacy payload field ``extra_sglang_args`` to the
  canonical framework-neutral ``extra_server_args``.
"""
