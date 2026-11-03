"""What an iteration is FOR, and what happened to it — two axes, kept apart.

An iteration has exactly two independent properties, and conflating them is
what this module exists to stop:

``INTENT``
    Declared by the agent BEFORE the measurement, and it chooses the gate rule.
    An intent is a claim about what the change is buying, so it is the agent's
    to make and the critic's to audit.

``VERDICT``
    Produced by the gate AFTER the measurement, and it IS the git action.
    ``KEEP`` and ``REVERT`` are the only two things a tree can do; the third,
    ``NOT_MEASURED``, means no action at all.

Previously one word did both jobs — ``claim_class`` held ``A``/``B``/``PROBE``
while ``decision`` held ``KEEP``/``REVERT``/``PROBE``/``INVALID``, so ``PROBE``
appeared in both and the rules for each intent were spread across the gate, the
state machine, the loop, two prompt files and ~20 hand-written lines in every
pack. Adding an intent meant editing all of them and keeping them consistent by
hand. Here, an intent is a row.

WHY ``PROBE`` IS NOT A VERDICT. Its tree action was always identical to
``KEEP`` — it existed only to carry the intent through the verdict, because the
verdict was the only thing recorded. What actually made a probe different is
``books=False``: it may never ratchet ``best_metric``. That is a property of
the intent, and it now lives on the intent.

WHY ``NOT_MEASURED`` IS. A bench that never ran, a server that never came up, a
404 from the wrong wire name, a moved instrument — none of these say anything
about the change. They are not a REVERT, they are the absence of a
measurement: the tree stands, the counters do not move, and the caller retries.
That rule was bought with six lost iterations of the dsv41flash campaign, five
of them the same FP8 o-projection port that was correct the whole time.
"""

from __future__ import annotations

from dataclasses import dataclass

# The git actions, plus the absence of one. There is no fourth.
KEEP = "KEEP"
REVERT = "REVERT"
NOT_MEASURED = "NOT_MEASURED"

VERDICTS = (KEEP, REVERT, NOT_MEASURED)

# Verdicts under which the change stays in the tree. `NOT_MEASURED` is here
# because it means "no action" — the tree is untouched precisely BECAUSE
# nothing was learned about it.
TREE_STANDS = (KEEP, NOT_MEASURED)


@dataclass(frozen=True)
class Intent:
    """One row: the complete rule for iterations declaring this intent.

    ``tolerance_key``
        Which ``objective.measurement`` key bounds how far this intent may
        regress. Naming the key rather than the number is what lets a pack
        widen the band for a noisy validator without the gate knowing.

    ``requires_bar``
        Whether clearing ``threshold_rel`` is required to be KEPT at all. False
        for everything except a metric claim: foundation work and probes ship
        on evidence, so neutral is a keep for them.

    ``books``
        Whether a gain may be claimed as an advance. ``booked`` is what resets
        the escalation counter, so a probe must never set it — otherwise a
        campaign could escape escalation pressure by surveying.

    ``server``
        ``skip_if_readonly`` lets the loop bypass the ~45-minute server run
        when the diff cannot change what the server does. Only a probe earns
        this, and only when its diff is confined to inert paths.

    ``pareto_exempt``
        Whether the second axis may decide this intent's verdict. True only for
        a correctness fix: it carries no performance claim on EITHER axis, so
        it is judged solely against its own tolerance and a secondary
        regression must not refuse it. Note this is not the same as "makes no
        metric claim" — a probe makes none either, yet a probe that regresses
        the tree is still refused, and a frontier trade still applies to it.
    """

    name: str
    aliases: tuple[str, ...]
    tolerance_key: str
    tolerance_default: float
    requires_bar: bool
    books: bool
    server: str
    deliverable: str
    pareto_exempt: bool = False

    @property
    def skips_server_when_readonly(self) -> bool:
        return self.server == "skip_if_readonly"


# How far below best a change may land and still count as neutral. Default 1% —
# just above the 0.79% run-to-run spread measured on MI350X. Packs with a
# noisier validator raise this in objective.measurement.
DEFAULT_NOISE_BAND_PCT = 1.0

# How far a correctness fix may regress before it is refused. Wide on purpose:
# the point of the intent is that the fix ships regardless of a small cost, and
# only a catastrophe — a fix that disables a fast path outright — is worth
# refusing. Without this, the loop reverts fixes for being slow, which is how
# the FP8 o-projection's verification gate was lost while still inert.
DEFAULT_CORRECTNESS_CAP_PCT = 5.0


REGISTRY: tuple[Intent, ...] = (
    Intent(
        name="WIN",
        aliases=("B",),
        tolerance_key="noise_band_pct",
        tolerance_default=DEFAULT_NOISE_BAND_PCT,
        requires_bar=True,
        books=True,
        server="always",
        deliverable=(
            "a metric claim. Gated on the pack's threshold — the strictest "
            "bar, and the default. If the only reason to make the change is "
            "speed, it is a WIN no matter what else is true about it."
        ),
    ),
    Intent(
        name="FIX",
        aliases=("A",),
        tolerance_key="noise_band_pct",
        tolerance_default=DEFAULT_NOISE_BAND_PCT,
        requires_bar=False,
        books=True,
        server="always",
        deliverable=(
            "foundation work, a bug fix, or instrumentation. Ships on "
            "evidence: NEUTRAL IS A KEEP. State the evidence — the defect and "
            "proof it is real, or the step number in a declared multi-step "
            "plan and what the terminal step buys, or what the instrumentation "
            "now makes measurable. A regression past the noise band is still a "
            "REVERT."
        ),
    ),
    Intent(
        name="FIX-CORRECTNESS",
        aliases=("A-CORRECTNESS",),
        tolerance_key="correctness_regression_cap_pct",
        tolerance_default=DEFAULT_CORRECTNESS_CAP_PCT,
        requires_bar=False,
        books=True,
        server="always",
        pareto_exempt=True,
        deliverable=(
            "a correctness fix carrying no performance claim: a wrong result, "
            "a check that can never fail, a guard that guards nothing. Exempt "
            "from the throughput bar, refused only on a catastrophic "
            "regression. The test: if the defect were harmless to performance, "
            "would it still be worth fixing? If no, it is a WIN wearing a "
            "costume."
        ),
    ),
    Intent(
        name="LEARN",
        aliases=("PROBE",),
        tolerance_key="noise_band_pct",
        tolerance_default=DEFAULT_NOISE_BAND_PCT,
        requires_bar=False,
        books=False,
        server="skip_if_readonly",
        # NAME THE PATHS. This said "inert paths" until 2026-09-24, which is
        # abstract where it needs to be operational: the saving is only worth
        # reaching for if the agent knows IN ADVANCE that its probe qualifies,
        # and "inert" leaves it guessing. `READONLY_PROBE_PREFIXES`
        # (claude_loop.py) is the rule that actually decides -- eval/,
        # knowledge/, ledger.jsonl, arbor/ -- and those are harness-level
        # conventions (cf. ALLOWED_PACK_DIRS), not one pack's layout, so naming
        # them here is general as well as true. The pre-split prompt named them
        # and the v2 campaign probed 49% of iterations; do not re-abstract this
        # without measuring what it costs.
        deliverable=(
            "buying knowledge rather than throughput. The deliverable is a "
            "resolved ledger entry, and an iteration that closes none is a "
            "REVERT. A LEARN touching only `eval/` and the ledger SKIPS THE "
            "SERVER RUN ENTIRELY — minutes instead of ~45 — so reach for it "
            "often. The best one kills its own lead."
        ),
    ),
)

# The intent an undeclared or unrecognised iteration gets. WIN is the safe
# default precisely because it is the STRICTEST: an agent that fails to declare
# is held to the metric bar rather than slipping through the evidence-gated
# path. Erring the other way silently applies the easiest rule to work nobody
# vouched for.
DEFAULT_INTENT_NAME = "WIN"

_BY_NAME: dict[str, Intent] = {}
for _i in REGISTRY:
    _BY_NAME[_i.name] = _i
    for _a in _i.aliases:
        _BY_NAME[_a] = _i

# Every spelling the loop's declaration parser should accept, longest first so
# that `FIX-CORRECTNESS` is matched before `FIX` and `A-CORRECTNESS` before `A`.
# Regex alternation is first-match-wins, so this ordering is load-bearing.
ALL_NAMES: tuple[str, ...] = tuple(
    sorted(_BY_NAME, key=lambda n: (-len(n), n))
)


def resolve(name: str | Intent | None) -> Intent:
    """Look up an intent by either vocabulary. Never raises.

    Accepts an ``Intent`` unchanged so callers may pass either, which keeps the
    migration from needing two code paths.
    """
    if isinstance(name, Intent):
        return name
    key = (name or "").strip().upper()
    return _BY_NAME.get(key, _BY_NAME[DEFAULT_INTENT_NAME])


def tolerance(intent: Intent, measurement: dict) -> float:
    """The regression band this intent is held to, per the pack's measurement."""
    value = measurement.get(intent.tolerance_key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return intent.tolerance_default
    return float(value)


# ── Legacy vocabulary ───────────────────────────────────────────────────────
# The old (claim_class, decision) pair is still written to session.json and
# still read by everything downstream — `check_review_consistency.py` counts
# `l["decision"] == "KEEP"` straight out of the file, so promoting a probe to
# KEEP would silently change the KEEP count of four archived campaigns. These
# functions are the whole compatibility surface, and they are exact.

# Rows in `lessons` that are ANNOTATIONS, not judgements. No gate produced
# them and no intent was declared for them:
#   BASELINE  written by record_baseline, the opening measurement
#   ESCALATE  written by apply_escalation, a forced tier move
#   NOTE      written BY HAND by the campaign owner, to correct the record —
#             e.g. "iteration 2's REVERT was a HARNESS failure, not a verdict".
#             Three exist across the archived campaigns.
#   REBASELINE  written by scripts/model-m-campaign/rebase_bar.py, the bar on a new node
#   MAP_ROUND   written by apply_map_round (arbor/directions.py), a map round's event
#   ROUND_END   written by arbor/rounds.py, a round end's request, incomplete A/B or stop
#   ROUND_AB    written by arbor/rounds.py, a confirmed round-end A/B: the bar on its node
#
# They must never be re-derived from an (intent, verdict) pair, because that
# round trip cannot reproduce them: every one of them would come back as a
# plain KEEP and inflate the campaign's KEEP count. Anything reading history
# should treat a row with no `verdict` field as one of these and use its stored
# `decision` verbatim.
HISTORY_ROWS = ("BASELINE", "ESCALATE", "NOTE", "REBASELINE", "MAP_ROUND", "ROUND_END", "ROUND_AB")


def is_history_row(decision: str) -> bool:
    """True for an annotation row that no gate produced."""
    return (decision or "").strip().upper() in HISTORY_ROWS

def legacy_decision(intent: Intent, verdict: str) -> str:
    """The pre-split `decision` string for a given (intent, verdict)."""
    if verdict == NOT_MEASURED:
        return "INVALID"
    if verdict == KEEP and not intent.books:
        # A LEARN that passed. It was recorded as its own decision because the
        # verdict was the only field; keep emitting that so history, the
        # review checker and the eval traces read unchanged.
        return "PROBE"
    return verdict


# What a prompt template writes where the intent table should appear.
INTENTS_PLACEHOLDER = "{{INTENTS}}"

# Where the generated pack-facing copy lives, relative to the repo root.
SHARED_DOC_RELPATH = "packs/_shared/intents.md"


def render_deliverables() -> str:
    """The intent table as markdown, for the agent and the critic.

    Both prompts used to carry a hand-written copy of this, so every intent's
    rule existed in three places — the registry, `claude_loop_prompt.md` and
    `critic_prompt.md` — and stayed consistent only by attention. They now
    interpolate it. A fifth intent reaches both readers by being a fifth row.
    """
    lines = []
    for i in REGISTRY:
        legacy = "`, `".join(i.aliases)
        lines.append(f"- **`{i.name}`** (also accepted: `{legacy}`) — {i.deliverable}")
    return "\n".join(lines)


def fill(text: str) -> str:
    """Substitute the rendered table wherever a template names it.

    Templates keep the placeholder rather than the table so they stay readable
    on their own, and a template that does not use it is returned unchanged.
    """
    return text.replace(INTENTS_PLACEHOLDER, render_deliverables())


def render_shared_doc() -> str:
    """`packs/_shared/intents.md` — the file packs include instead of restating.

    Generated, never hand-edited: `scripts/pack/render_intents.py` writes it and
    `test_intents_doc_in_sync` fails if the checked-in copy has drifted.
    """
    return (
        "<!-- GENERATED by scripts/pack/render_intents.py — do not edit. -->\n"
        "<!-- Source of truth: arbor/measure/intents.py -->\n\n"
        "# Declaring an intent\n\n"
        "Every iteration declares what it is FOR, before the measurement, and "
        "the declaration chooses the bar it is held to:\n\n"
        "```\n"
        "INTENT: WIN\n"
        "```\n\n"
        f"{render_deliverables()}\n\n"
        "Declaring the wrong intent to reach an easier bar is the specific "
        "thing the critic is looking for. An absent or unrecognised "
        f"declaration resolves to `{DEFAULT_INTENT_NAME}` — the strictest bar, "
        "so a formatting slip cannot buy the easiest rule.\n\n"
        "The verdict is the harness's, and it is the git action: "
        f"`{'` / `'.join(VERDICTS)}`. `{NOT_MEASURED}` means no action at all — "
        "the tree stands, the counters do not move, and the iteration is "
        "retried.\n"
    )


def from_legacy(decision: str, claim_class: str = "") -> tuple[Intent, str]:
    """Read an archived lesson back into (intent, verdict).

    The exact inverse of ``legacy_decision`` for the four judgement strings.
    For a ``HISTORY_ROWS`` annotation the verdict is KEEP — the tree stood —
    but the pair does NOT reconstruct the original string; check
    ``is_history_row`` first when that matters.
    """
    decision = (decision or "").strip().upper()
    if decision == "INVALID":
        return resolve(claim_class), NOT_MEASURED
    if decision == "PROBE":
        return resolve("PROBE"), KEEP
    if decision == "REVERT":
        return resolve(claim_class), REVERT
    # KEEP, plus every annotation row: the tree stood.
    return resolve(claim_class), KEEP
