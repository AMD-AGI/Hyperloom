"""The dossier's parts: their numbers and file names, and which of the dossier plugin's skills writes each.

The dossier on disk does not change with the plugin: ``docs/dossier/<name>/README.md``, the index, and one
``NN-<name>.md`` per part (an update round's ``NN.uK-<name>.md`` beside it). The checker (``check.py``) reads its part
lists from here, and the plugin's skills declare these files as what they ``produce`` (a test holds them to it).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Part:
    number: str
    name: str
    title: str
    writer: str
    gpu: bool = False
    #: Whether a dossier is incomplete without it. Part 9 is not: its writer needs a person's traces and is opt-in, so
    #: a pack that does not run it still ends with a whole dossier.
    required: bool = True

    @property
    def file(self) -> str:
        return f"{self.number}-{self.name.replace(' ', '-')}.md"


PARTS: tuple[Part, ...] = (
    Part("01", "model", "Model architecture", "dossier:discover"),
    Part("02", "system", "Framework and system architecture", "dossier:discover"),
    Part("03", "code", "Code reading", "dossier:discover"),
    Part("04", "target", "Optimization target", "dossier:discover"),
    Part("05", "workload", "Workload, and the CPU work it causes", "dossier:discover"),
    Part("06", "baseline", "Baseline and noise floor", "dossier:measure", gpu=True),
    Part("07", "profile", "Profile: plan, then results", "dossier:measure", gpu=True),
    Part("08", "kernels", "Kernel usage", "dossier:measure", gpu=True),
    Part("09", "reference", "Reference platform (B200)", "dossier:reference", required=False),
    Part("10", "history", "Earlier optimizations", "dossier:discover"),
    Part("11", "directions", "Directions", "dossier:directions"),
)
#: Part 12, the fast bench, may be in progress when the dossier is done.
OPTIONAL: tuple[Part, ...] = (Part("12", "fast bench", "Fast bench", "dossier:measure", gpu=True, required=False),)
#: The parts an update round rewrites (``NN.uK-<name>.md``), as the ``dossier:update`` skill writes them; part 10's
#: round file is its lessons.
ROUND_NAMES = {"03": "code", "06": "baseline", "07": "profile", "10": "lessons", "11": "directions"}
INDEX = "README.md"

BASE_PARTS = {part.number: part.name for part in PARTS}
#: The parts the checker skips when they are absent, and checks when they are there.
OPTIONAL_PARTS = {part.number: part.name for part in (*PARTS, *OPTIONAL) if not part.required}
ROUND_PARTS = dict(ROUND_NAMES)


def written_by(skill: str) -> tuple[Part, ...]:
    """The parts ``skill`` writes."""

    return tuple(part for part in PARTS if part.writer == skill)
