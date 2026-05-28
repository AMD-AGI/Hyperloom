"""Backwards-compatibility shims for inference_optimizer.

Modules in this sub-package implement deprecation-window helpers for
field renames, schema migrations, and other compatibility surfaces
that are expected to shrink over time. Each helper documents the
release at which the alias was introduced and the target release at
which it will be removed so a future cleanup pass has a clear
hit-list.

Current members:

* :mod:`inference_optimizer.compat.payload_aliases` — read-only
  alias from the legacy payload field ``extra_sglang_args`` to the
  canonical framework-neutral ``extra_server_args``. Introduced in
  Phase 4 of ``atom_plan/``; scheduled for removal in the release
  after Hyperloom ships full atom support.
"""
