"""The throughput schema the service, CLI and sync tests write Experiences under."""

from __future__ import annotations

from hyperloom_kb.schema import (
    ExperienceDeclaration,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    ObjectiveDeclaration,
)


def throughput_declaration(objective: str = "throughput@v1") -> ExperienceDeclaration:
    """A knob-tuning schema: a model, a grouped baseline and knob, and a keep-or-revert throughput outcome."""

    return ExperienceDeclaration(
        objectives=(ObjectiveDeclaration(objective, "Maximize throughput."),),
        identity=(FieldDeclaration("model", "Model."),),
        baseline=(FieldDeclaration("config", "Baseline.", group=True),),
        change=(
            FieldDeclaration("knob", "Knob.", group=True),
            FieldDeclaration("summary", "What changed.", role=FieldRole.SUMMARY),
        ),
        outcome=(
            FieldDeclaration("decision", "Decision.", role=FieldRole.DECISION, values=("keep", "revert")),
            FieldDeclaration("value", "Throughput.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
        ),
        reflection=(FieldDeclaration("text", "Reflection.", kind=FieldKind.TEXT),),
    )
