# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Offline generations: whether the playbook's closed loop improves on what it has not seen.

:func:`generations` splits the archived sessions into an evolution set and :data:`HELD_OUT` held-out sessions, at least
one of each model type the archive holds where there are enough. Each generation the LLM reads how the champion
replays on the evolution set, its stop audits and the ledger of every change tried so far
(:func:`generation_prompt`), and proposes one change, as at CLOSE. A change that passes the static check (R0) and the
replay of the evolution set (R1) is crowned; the rest are retired with their evidence, so the next generation reads
them. At the end the last champion and H0 are replayed on the held-out sessions, both ways beyond the data: a
champion better on the evolution set and not on the held-out one has fitted the archive.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom_multibranch.notebook.rsi.archive import SearchTree
from hyperloom_multibranch.notebook.rsi.authoring import PLAYBOOK_SYSTEM_PROMPT, read_change
from hyperloom_multibranch.notebook.rsi.history import MODEL_TYPE_LABEL
from hyperloom_multibranch.notebook.rsi.ladder import Complete, r0, r1
from hyperloom_multibranch.notebook.rsi.playbook import BOUNDS, H0, Crowned, Playbook, PlaybookPool, Retired, changes
from hyperloom_multibranch.notebook.rsi.replay import OPTIMISTIC, PESSIMISTIC, SchedulingComparison, compare
from hyperloom_multibranch.notebook.rsi.stops import stall_study

HELD_OUT = 4
GENERATIONS = 5
#: Runs a session of each replay a generation reads or is judged by; the held-out comparison runs as many.
GENERATION_RUNS = 50


def split(
    trees: Sequence[SearchTree], held_out: int = HELD_OUT, seed: str = ""
) -> tuple[list[SearchTree], list[SearchTree]]:
    """``trees`` as (the evolution set, the ``held_out`` sessions), one session of each model type held out first,
    the rest drawn by ``seed``; the evolution set keeps at least as many as it gives."""

    order = sorted(trees, key=lambda tree: hashlib.sha256(f"{seed}:{tree.session_id}".encode()).hexdigest())
    count = min(held_out, len(order) // 2)
    held: list[SearchTree] = []
    for model_type in sorted({tree.task.labels.get(MODEL_TYPE_LABEL, "") for tree in order}):
        first = next((tree for tree in order if tree.task.labels.get(MODEL_TYPE_LABEL, "") == model_type), None)
        if first is not None and len(held) < count:
            held.append(first)
    held.extend(tree for tree in order if tree not in held and len(held) < count)
    return [tree for tree in order if tree not in held], held


@dataclass(frozen=True)
class Generation:
    """One generation: the playbook proposed, if any, what changed, and what R0 and R1 said; ``crowned`` when it
    became the champion."""

    number: int
    proposed: str
    changed: str
    verdict: str
    crowned: bool


@dataclass(frozen=True)
class GenerationsReport:
    """The generations, the last champion, and the held-out comparison of it and H0 both ways beyond the data."""

    evolution: tuple[str, ...]
    held_out: tuple[str, ...]
    generations: tuple[Generation, ...]
    champion: Playbook
    optimistic: SchedulingComparison
    pessimistic: SchedulingComparison

    def render(self) -> str:
        lines = [
            f"Offline generations: {len(self.evolution)} evolution sessions, held out {', '.join(self.held_out)}.",
            "",
            *(
                f"- generation {item.number}: {item.proposed or 'no change'}"
                + (f" ({item.changed})" if item.changed else "")
                + f": {item.verdict}"
                + ("; crowned" if item.crowned else "")
                for item in self.generations
            ),
            "",
            f"The last champion is {self.champion.label}. On the held-out sessions:",
            "",
            self.optimistic.render(),
            "",
            self.pessimistic.render(),
        ]
        return "\n".join(lines)


def generation_prompt(
    evolution: Sequence[SearchTree], champion: Playbook, pool: PlaybookPool, runs: int, seed: str
) -> str:
    """What a generation's LLM reads: the numbers and their bounds, the champion, how it replays on the evolution
    set, the stop audits of those sessions, and the ledger of the changes tried so far."""

    replayed = compare(evolution, {champion.label: champion}, runs=runs, seed=seed, baseline=champion.label)
    ledger = [
        f"- {item.label} (of {item.parent}, changing {', '.join(changes(pool.named(item.parent) or H0, item))}): "
        f"{item.hypothesis}; aims at {item.measure or 'n/a'}; "
        + next(
            (f"retired: {record.evidence}" for record in pool.retired() if record.label == item.label),
            next(
                (f"crowned: {record.evidence}" for record in pool.crowned() if record.label == item.label), "on trial"
            ),
        )
        for item in pool.playbooks()
    ]
    numbers = [
        f"- {name}: {'default' if getattr(champion.numbers, name) is None else getattr(champion.numbers, name)} "
        f"(bounds {low:g} to {high:g})"
        for name, (low, high) in BOUNDS.items()
    ]
    return "\n".join(
        [
            "You are improving the playbook offline, on archived sessions replayed under it, not on a live session.",
            "The numbers, the champion's value of each and its bounds:",
            *numbers,
            "",
            "The champion:",
            champion.render(),
            "",
            replayed.render(),
            "",
            stall_study(tuple(evolution)).render(),
            "",
            "The evidence ledger:",
            *(ledger or ["- nothing tried yet"]),
        ]
    )


async def generations(
    trees: Sequence[SearchTree],
    complete: Complete,
    pool_root: Path,
    *,
    model: str,
    count: int = GENERATIONS,
    held_out: int = HELD_OUT,
    runs: int = GENERATION_RUNS,
    seed: str = "",
) -> GenerationsReport:
    """``count`` generations of the playbook on ``trees`` less the held-out ones, its pool kept under ``pool_root``,
    then the last champion beside H0 on the held-out sessions."""

    evolution, held = split(trees, held_out, seed)
    pool = PlaybookPool(pool_root)
    done = []
    for number in range(1, count + 1):
        champion = pool.champion()
        answer = await complete(PLAYBOOK_SYSTEM_PROMPT, generation_prompt(evolution, champion, pool, runs, seed))
        proposed, why = read_change(answer, champion, {"model": model, "generation": str(number)}, float(number))
        if proposed is None:
            done.append(Generation(number, "", "", why, False))
            continue
        try:
            proposed = pool.register(proposed)
        except (OSError, ValueError) as exc:
            done.append(Generation(number, proposed.label, "", str(exc), False))
            continue
        checked = r0(proposed, champion)
        replayed = r1(proposed, champion, evolution, runs=runs, seed=seed) if checked.passed else checked
        crowned = checked.passed and replayed.passed and not replayed.skipped
        verdict = f"{checked.rung} {checked.reason}; {replayed.rung} {replayed.reason}"
        if crowned:
            pool.crown(Crowned(proposed.label, float(number), verdict))
        else:
            pool.retire(Retired(proposed.label, float(number), verdict))
        done.append(Generation(number, proposed.label, ", ".join(changes(champion, proposed)), verdict, crowned))
    champion = pool.champion()
    pair = {H0.label: H0, **({} if champion.label == H0.label else {champion.label: champion})}
    optimistic, pessimistic = (
        compare(held, pair, runs=runs, beyond=beyond, seed=seed, baseline=H0.label)
        for beyond in (OPTIMISTIC, PESSIMISTIC)
    )
    return GenerationsReport(
        tuple(tree.session_id for tree in evolution),
        tuple(tree.session_id for tree in held),
        tuple(done),
        champion,
        optimistic,
        pessimistic,
    )


__all__ = [
    "GENERATIONS",
    "GENERATION_RUNS",
    "HELD_OUT",
    "Generation",
    "GenerationsReport",
    "generation_prompt",
    "generations",
    "split",
]
