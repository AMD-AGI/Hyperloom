#!/usr/bin/env python3
"""A tiny, dependency-free stand-in for a real HPC benchmark.

It exists to make the ``--framework custom`` contract runnable on any machine,
including one with no GPU: the point of the example is the *contract*, not the
kernel. Replace this file with your own benchmark and keep the two things
Hyperloom reads -- a throughput number and a correctness verdict.

The work is a fixed-size integer reduction, chunked and summed. Two environment
knobs change how it runs without changing *what* it computes:

``EXAMPLE_WORKERS``
    Processes the chunks are spread across. Processes, not threads: the work is
    pure Python, so the GIL would make threads strictly slower and the sweep
    would have nothing to find. More is not automatically better either --
    past some point pool startup and pickling cost more than the parallelism
    buys, and where that point falls depends on the machine. That is exactly
    the kind of thing a sweep should discover rather than a human guess.
``EXAMPLE_CHUNK``
    Items per chunk. Trades scheduling and pickling overhead against
    parallel granularity.

Because the reduction is associative and the chunking is deterministic, the
checksum is identical for every setting. That is the property the quality gate
checks, and it is what makes a speedup trustworthy: faster *and* provably the
same answer.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

#: Total items reduced per iteration. Sized so the serial pass takes about a
#: second on a laptop -- long enough that pool startup is amortized and the
#: parallel knob shows a real effect, short enough that the example is quick.
N_ITEMS = 12_000_000

#: Timed iterations. The slowest is discarded and the rest averaged, so a single
#: scheduler hiccup does not decide which variant wins.
ITERATIONS = 3


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    """Read a positive int knob, clamped, falling back on anything unparseable."""
    try:
        value = int(str(os.environ.get(name, "")).strip() or default)
    except ValueError:
        return default
    return max(low, min(high, value))


def _reduce_chunk(bounds: tuple[int, int]) -> int:
    """Reduce one half-open range. Pure, so the split cannot change the total."""
    start, stop = bounds
    total = 0
    for i in range(start, stop):
        total += (i * 2654435761) % 1_000_003
    return total


def run_once(workers: int, chunk: int) -> tuple[int, float]:
    """Run the reduction once. Returns ``(checksum, elapsed_seconds)``."""
    bounds = [(i, min(i + chunk, N_ITEMS)) for i in range(0, N_ITEMS, chunk)]
    started = time.perf_counter()
    if workers == 1:
        checksum = sum(_reduce_chunk(b) for b in bounds)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            checksum = sum(pool.map(_reduce_chunk, bounds))
    return checksum, time.perf_counter() - started


def main() -> int:
    workers = _env_int("EXAMPLE_WORKERS", 1, low=1, high=64)
    chunk = _env_int("EXAMPLE_CHUNK", 100_000, low=1_000, high=N_ITEMS)

    # Untimed warmup: pays thread-pool creation and any first-touch cost once,
    # so it is not charged to the first timed iteration.
    reference, _ = run_once(workers, chunk)

    samples: list[float] = []
    for _ in range(ITERATIONS):
        checksum, elapsed = run_once(workers, chunk)
        if checksum != reference:
            # Non-determinism in the workload itself: report it rather than
            # average over it, because a speedup measured here means nothing.
            print(
                f"ERROR: checksum drifted within the run ({checksum} != {reference})",
                file=sys.stderr,
            )
            return 1
        samples.append(elapsed)

    # Drop the worst sample, then average. With ITERATIONS=3 that is the mean of
    # the two best -- enough to shrug off one noisy iteration without pretending
    # a single lucky run is the truth.
    kept = sorted(samples)[:-1] or samples
    mean_elapsed = sum(kept) / len(kept)
    throughput = N_ITEMS / mean_elapsed / 1_000.0  # kitems/s

    # The reference is one unchunked, single-process reduction -- the definition
    # of the right answer. It is a property of N_ITEMS alone, so any knob setting
    # that changes it has changed the answer, not just the speed.
    expected = _reduce_chunk((0, N_ITEMS))
    passed = reference == expected

    result = {
        "output_throughput": round(throughput, 4),
        "throughput_unit": "kitems/s (reduction throughput)",
        "mean_latency_ms": round(mean_elapsed * 1_000.0, 3),
        "quality_gate": {
            "passed": passed,
            "checksum": reference,
            "reference_checksum": expected,
            "metric": "exact checksum match against the single-chunk reduction",
        },
        "workload": {
            "name": "example-reduction",
            "runner_type": os.environ.get("RUNNER_TYPE", ""),
            "config": {"workers": workers, "chunk": chunk, "items": N_ITEMS},
        },
    }

    result_dir = Path(os.environ.get("RESULT_DIR") or ".")
    result_dir.mkdir(parents=True, exist_ok=True)
    out = result_dir / f"{os.environ.get('RESULT_FILENAME') or 'inferencex_result'}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        f"[example] workers={workers} chunk={chunk} "
        f"-> {throughput:.2f} kitems/s, gate={'PASS' if passed else 'FAIL'}"
    )
    print(f"[example] wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
