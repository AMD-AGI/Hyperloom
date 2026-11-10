#!/usr/bin/env python3
"""Generate a deterministic synthetic reference + long-read set for the minimap2 example.

Self-contained on purpose: the example must not depend on a dataset the reader
cannot obtain. Seeded, so every run produces byte-identical input.

    python3 gen_reads.py [REF_MB] [N_READS]     # defaults: 20 Mbp, 8000 reads

Caveat worth repeating from the README: a random reference has none of the
repeat structure of a real genome, so chaining on it is unrealistically easy.
This is for exercising the adapter and the sweep, not for performance claims.
"""

from __future__ import annotations

import random
import sys

REF_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_READS = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

SEED = 1234
ERR = 0.06  # total error rate, roughly ONT-like
LEN_LO, LEN_HI = 4000, 14000

BASES = "ACGT"
rng = random.Random(SEED)

ref = "".join(rng.choice(BASES) for _ in range(REF_MB * 1_000_000))
with open("ref.fa", "w") as fh:
    fh.write(">chrSim\n")
    for i in range(0, len(ref), 60):
        fh.write(ref[i : i + 60] + "\n")

with open("reads.fq", "w") as fh:
    for i in range(N_READS):
        length = rng.randint(LEN_LO, LEN_HI)
        pos = rng.randint(0, len(ref) - length - 1)
        out: list[str] = []
        for base in ref[pos : pos + length]:
            roll = rng.random()
            if roll < ERR * 0.6:
                out.append(rng.choice(BASES))  # substitution
            elif roll < ERR * 0.8:
                continue  # deletion
            elif roll < ERR:
                out.append(base)
                out.append(rng.choice(BASES))  # insertion
            else:
                out.append(base)
        seq = "".join(out)
        fh.write(f"@read{i}_{pos}\n{seq}\n+\n{'I' * len(seq)}\n")

print(f"ref.fa: {len(ref) / 1e6:.1f} Mbp   reads.fq: {N_READS} reads")
