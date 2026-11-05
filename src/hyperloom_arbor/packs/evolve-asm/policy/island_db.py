"""MAP-Elites island database for evolve-asm.

Each island runs an independent evolutionary search. This module stores
candidates in a quality-diversity grid (feature space: vgpr_bin x barrier_bin
x roofline_bin) and manages cross-island migration via ring topology.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Candidate dataclass
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    id: str
    weakest_us: float                   # primary-shape measured latency (us)
    roofline_frac: float                # fraction of roofline achieved (0-1)
    parent_id: Optional[str]
    generation: int
    island: int
    tier: str                           # e.g. "native", "asm", ...
    label: str                          # human-readable description
    sig: Dict[str, Any]                 # {"vgpr": int, "barriers": int,
                                        #  "spills": int, "mfma": int}
    co_path: str                        # path to compiled .co artifact
    s_path: str                         # path to source .s file
    ts: float = field(default_factory=time.time)
    migrant: bool = False               # True if imported from another island


# ---------------------------------------------------------------------------
# Feature-space helpers
# ---------------------------------------------------------------------------

def _cell_key(vgpr: int, barriers: int, roofline_frac: float) -> Tuple[int, int, int]:
    vgpr_bin    = vgpr // 8
    barrier_bin = barriers // 4
    roofline_bin = int(roofline_frac * 10)
    return (vgpr_bin, barrier_bin, roofline_bin)


def _candidate_cell(c: Candidate) -> Tuple[int, int, int]:
    return _cell_key(
        c.sig.get("vgpr", 0),
        c.sig.get("barriers", 0),
        c.roofline_frac,
    )


# ---------------------------------------------------------------------------
# IslandDB
# ---------------------------------------------------------------------------

class IslandDB:
    """MAP-Elites quality-diversity database for one evolve-asm campaign.

    Parameters
    ----------
    num_islands : int
        Number of parallel island agents.
    db_path : str | Path
        Directory where island_db.json is persisted.
    """

    _MIGRATION_INTERVAL = 10   # trigger migration every N KEEPs
    _MIGRATION_RATE     = 0.10 # fraction of cells to migrate per ring step

    def __init__(self, num_islands: int, db_path: str | Path) -> None:
        self.num_islands = num_islands
        self.db_path     = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._json_file  = self.db_path / "island_db.json"

        # Per-island MAP-Elites grids: island_id -> {cell_key -> Candidate}
        self._grids: List[Dict[Tuple, Candidate]] = [
            {} for _ in range(num_islands)
        ]
        # All KEEPs ever accepted (for leaderboard / top-N queries)
        self._all_keeps: List[Candidate] = []
        # Global keep counter (drives migration cadence)
        self._keep_count: int = 0

        self._load()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, candidate: Candidate) -> bool:
        """Insert or replace a cell with *candidate* if it is fitter.

        Returns True if the candidate was accepted into the grid (i.e. it
        improved or filled its cell), False otherwise.
        """
        island = candidate.island
        if island < 0 or island >= self.num_islands:
            raise ValueError(f"Island index {island} out of range [0, {self.num_islands})")

        cell = _candidate_cell(candidate)
        grid = self._grids[island]
        incumbent = grid.get(cell)

        if incumbent is None or candidate.weakest_us < incumbent.weakest_us:
            grid[cell] = candidate
            return True
        return False

    def record_keep(self, candidate: Candidate) -> None:
        """Record a KEEP decision (accepted by the gate) and update the grid."""
        self.add(candidate)
        self._all_keeps.append(candidate)
        self._keep_count += 1
        self._save()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def best(self) -> Optional[Candidate]:
        """Return the globally fastest (lowest weakest_us) accepted candidate."""
        if not self._all_keeps:
            return None
        return min(self._all_keeps, key=lambda c: c.weakest_us)

    def island_best(self, island: int) -> Optional[Candidate]:
        """Return the best candidate for a specific island."""
        grid = self._grids[island]
        if not grid:
            return None
        return min(grid.values(), key=lambda c: c.weakest_us)

    def sample_parent(self, island: int) -> Optional[Candidate]:
        """Sample a parent candidate for mutation.

        Strategy:
          30 % exploitation -- return the best cell in the island grid.
          70 % exploration  -- return a random occupied cell.
        """
        grid = self._grids[island]
        if not grid:
            return None

        cells = list(grid.values())
        if random.random() < 0.30:
            # exploitation: best in grid
            return min(cells, key=lambda c: c.weakest_us)
        else:
            # exploration: random occupied cell
            return random.choice(cells)

    def population_snapshot(self) -> List[Candidate]:
        """Flat list of all candidates currently occupying any grid cell."""
        out: List[Candidate] = []
        for grid in self._grids:
            out.extend(grid.values())
        return out

    def top_n(self, n: int) -> List[Candidate]:
        """Return the top-N fastest accepted candidates (globally)."""
        sorted_keeps = sorted(self._all_keeps, key=lambda c: c.weakest_us)
        return sorted_keeps[:n]

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def should_migrate(self) -> bool:
        """True if migration should run (every _MIGRATION_INTERVAL KEEPs)."""
        return (
            self._keep_count > 0
            and self._keep_count % self._MIGRATION_INTERVAL == 0
        )

    def migrate(self) -> int:
        """Ring-topology migration: send top cells from island i to island i+1.

        Returns the number of candidates migrated.
        """
        if self.num_islands < 2:
            return 0

        migrated = 0
        for src_id in range(self.num_islands):
            dst_id  = (src_id + 1) % self.num_islands
            src_grid = self._grids[src_id]
            if not src_grid:
                continue

            # Select top fraction of src grid cells by roofline_frac
            n_migrate = max(1, int(len(src_grid) * self._MIGRATION_RATE))
            top_cells = sorted(
                src_grid.values(),
                key=lambda c: c.roofline_frac,
                reverse=True,
            )[:n_migrate]

            for c in top_cells:
                # Clone with migrant flag and reassign to destination island
                clone = Candidate(
                    id            = c.id + f"_mig{dst_id}",
                    weakest_us    = c.weakest_us,
                    roofline_frac = c.roofline_frac,
                    parent_id     = c.id,
                    generation    = c.generation,
                    island        = dst_id,
                    tier          = c.tier,
                    label         = c.label + " [migrant]",
                    sig           = dict(c.sig),
                    co_path       = c.co_path,
                    s_path        = c.s_path,
                    ts            = time.time(),
                    migrant       = True,
                )
                if self.add(clone):
                    migrated += 1

        if migrated:
            self._save()
        return migrated

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def leaderboard(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return a leaderboard list of the top-N candidates as plain dicts."""
        top = self.top_n(n)
        rows = []
        for rank, c in enumerate(top, start=1):
            rows.append({
                "rank":          rank,
                "id":            c.id,
                "island":        c.island,
                "tier":          c.tier,
                "weakest_us":    c.weakest_us,
                "roofline_frac": round(c.roofline_frac, 4),
                "label":         c.label,
                "sig":           c.sig,
                "migrant":       c.migrant,
            })
        return rows

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist the full database to JSON."""
        payload: Dict[str, Any] = {
            "num_islands": self.num_islands,
            "keep_count":  self._keep_count,
            "grids": [],
            "all_keeps": [asdict(c) for c in self._all_keeps],
        }
        for grid in self._grids:
            serialised = {}
            for cell_key, cand in grid.items():
                serialised[str(cell_key)] = asdict(cand)
            payload["grids"].append(serialised)

        tmp = self._json_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._json_file)

    def _load(self) -> None:
        """Restore state from JSON if it exists."""
        if not self._json_file.exists():
            return
        try:
            payload = json.loads(self._json_file.read_text())
        except (json.JSONDecodeError, OSError):
            return

        self._keep_count = payload.get("keep_count", 0)

        raw_keeps = payload.get("all_keeps", [])
        self._all_keeps = [Candidate(**d) for d in raw_keeps]

        raw_grids = payload.get("grids", [])
        for i, raw_grid in enumerate(raw_grids):
            if i >= self.num_islands:
                break
            for _cell_str, cand_dict in raw_grid.items():
                c = Candidate(**cand_dict)
                cell = _candidate_cell(c)
                self._grids[i][cell] = c
