"""RayJob workload env credential forwarding (deprecated — use install-oob)."""

from __future__ import annotations

import os
from collections.abc import Mapping


def rayjob_credential_fanout(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return workload env credentials for RayJob (always empty).

    RayJob inference pods no longer receive API keys via workload env.
    OOB/GEAK credentials are delivered by :func:`install_oob_on_pods_best_effort`
    using Ray Dashboard ``runtime_env`` (RayJob) or SSH stdin (Dynamo).

    Args:
        environ: Ignored; kept for backward-compatible call sites.

    Returns:
        dict[str, str]: Always ``{}``.
    """
    _ = os.environ if environ is None else environ
    return {}
