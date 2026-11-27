# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the lane rows a lease store holds."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend


async def lane_holders(locks: "SqliteLeaseBackend") -> dict[str, int]:
    """``{lane: holder_count}`` for every ownership row ``locks`` retains."""
    rows = await locks.db.fetchall("SELECT lane, COUNT(*) AS n FROM leases GROUP BY lane")
    return {row["lane"]: int(row["n"]) for row in rows}
