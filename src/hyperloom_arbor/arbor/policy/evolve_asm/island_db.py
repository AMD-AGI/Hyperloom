"""MAP-Elites population database for ASM kernel evolution.

Feature dimensions are kernel-structural (not code length):
  - vgpr_bin:     VGPR count binned by 8 (occupancy cliff proxy)
  - barrier_bin:  s_barrier count binned by 4 (sync-stall proxy)
  - roofline_bin: int(roofline_frac * 10)  (0–10)

A cell holds the fastest known candidate with that structural profile.
Diversity is maintained: a slower candidate in an empty cell is kept.

Migration every MIGRATION_INTERVAL KEEPs: ring topology, top candidates only.
Persisted to {run_dir}/db/island_db.json.
"""
from __future__ import annotations

import json, random, time, uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

MIGRATION_INTERVAL = 10
MIGRATION_RATE     = 0.1


def _feature_coords(sig: dict, roofline_frac: Optional[float]) -> tuple:
    return (
        sig.get('vgpr', 0) // 8,
        sig.get('barriers', 0) // 4,
        int((roofline_frac or 0.0) * 10),
    )


@dataclass
class Candidate:
    id: str
    weakest_us: float
    roofline_frac: Optional[float]
    parent_id: Optional[str]
    generation: int
    island: int
    tier: str
    label: str
    sig: dict = field(default_factory=lambda: {'vgpr': 0, 'barriers': 0, 'spills': 0, 'mfma': 0})
    co_path: str = ''
    s_path: str  = ''
    ts: float    = field(default_factory=time.time)
    migrant: bool = False

    def to_dict(self) -> dict: return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Candidate':
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


class IslandDB:
    def __init__(self, num_islands: int = 4, db_path: Optional[str] = None):
        self.num_islands = num_islands
        self.db_path = Path(db_path) if db_path else None
        self.candidates: dict[str, Candidate] = {}
        self.grids: list[dict[tuple, str]] = [{} for _ in range(num_islands)]
        self.generations: list[int] = [0] * num_islands
        self.last_migration_gen: int = 0
        self.best_id: Optional[str] = None
        if self.db_path and (self.db_path / 'island_db.json').exists():
            self._load()

    def add(self, c: Candidate) -> None:
        island = c.island % self.num_islands
        cell = _feature_coords(c.sig, c.roofline_frac)
        self.candidates[c.id] = c
        existing = self.grids[island].get(cell)
        if existing is None or existing not in self.candidates or \
                c.weakest_us < self.candidates[existing].weakest_us:
            self.grids[island][cell] = c.id
        self._update_best(c)
        if self.db_path:
            self._save()

    def record_keep(self, c: Candidate) -> None:
        self.add(c)
        self.generations[c.island % self.num_islands] += 1

    def best(self) -> Optional[Candidate]:
        if self.best_id and self.best_id in self.candidates:
            return self.candidates[self.best_id]
        if not self.candidates: return None
        b = min(self.candidates.values(), key=lambda c: c.weakest_us)
        self.best_id = b.id; return b

    def island_best(self, island: int) -> Optional[Candidate]:
        ids = list(self.grids[island % self.num_islands].values())
        valid = [self.candidates[i] for i in ids if i in self.candidates]
        return min(valid, key=lambda c: c.weakest_us) if valid else None

    def sample_parent(self, island: int) -> Optional[Candidate]:
        ids = [i for i in self.grids[island % self.num_islands].values()
               if i in self.candidates]
        if not ids: return self.best()
        if random.random() < 0.3:
            return min((self.candidates[i] for i in ids), key=lambda c: c.weakest_us)
        return self.candidates[random.choice(ids)]

    def population_snapshot(self) -> list[dict]:
        return [c.to_dict() for c in
                sorted(self.candidates.values(), key=lambda c: c.weakest_us)]

    def leaderboard(self) -> str:
        best_c = self.best()
        ref = best_c.weakest_us if best_c else 1.0
        lines = ['=== LEADERBOARD ===']
        for c in sorted(self.candidates.values(), key=lambda c: c.weakest_us)[:10]:
            gain = (ref - c.weakest_us) / ref * 100.0 if ref else 0.0
            rf = f"{c.roofline_frac*100:.1f}%" if c.roofline_frac else '?'
            lines.append(f"  {c.weakest_us:.3f}µs  island={c.island} tier={c.tier} "
                         f"rf={rf} {gain:+.1f}%  {c.label[:50]}")
        return '\n'.join(lines)

    def should_migrate(self) -> bool:
        return (max(self.generations) - self.last_migration_gen) >= MIGRATION_INTERVAL

    def migrate(self) -> int:
        if self.num_islands < 2: return 0
        added = 0
        for i in range(self.num_islands):
            src = self.island_best(i)
            if src is None or src.migrant: continue
            for target in [(i+1) % self.num_islands, (i-1) % self.num_islands]:
                target_ids = list(self.grids[target].values())
                if any(abs(self.candidates[t].weakest_us - src.weakest_us) < 0.001
                       for t in target_ids if t in self.candidates):
                    continue
                copy = Candidate(id=str(uuid.uuid4()), weakest_us=src.weakest_us,
                                 roofline_frac=src.roofline_frac, parent_id=src.id,
                                 generation=src.generation, island=target,
                                 tier=src.tier, label=f"{src.label} [migrated from {i}]",
                                 sig=dict(src.sig), co_path=src.co_path,
                                 s_path=src.s_path, migrant=True)
                self.add(copy); added += 1
        self.last_migration_gen = max(self.generations)
        if self.db_path: self._save()
        return added

    def _update_best(self, c: Candidate) -> None:
        if self.best_id is None:
            self.best_id = c.id; return
        cur = self.candidates.get(self.best_id)
        if cur is None or c.weakest_us < cur.weakest_us:
            self.best_id = c.id

    def _save(self) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
        data = {'num_islands': self.num_islands, 'best_id': self.best_id,
                'generations': self.generations,
                'last_migration_gen': self.last_migration_gen,
                'grids': [{str(k): v for k, v in g.items()} for g in self.grids],
                'candidates': {k: v.to_dict() for k, v in self.candidates.items()}}
        (self.db_path / 'island_db.json').write_text(json.dumps(data, indent=2) + '\n')

    def _load(self) -> None:
        data = json.loads((self.db_path / 'island_db.json').read_text())
        self.best_id = data.get('best_id')
        self.generations = data.get('generations', [0] * self.num_islands)
        self.last_migration_gen = data.get('last_migration_gen', 0)
        self.candidates = {k: Candidate.from_dict(v)
                           for k, v in data.get('candidates', {}).items()}
        self.grids = [{tuple(json.loads(k)): v for k, v in g.items()}
                      for g in data.get('grids', [{} for _ in range(self.num_islands)])]
