# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render workload recipes and packaged optimization knowledge into Ave prompts."""

from __future__ import annotations

from importlib.resources import files

from ave.task import PromptStep, parse_prompts


def _knowledge(topic: str) -> str:
    """Read a named knowledge document from the installed Ave package."""
    if not topic.isidentifier():
        raise ValueError(f"Invalid Ave knowledge topic: {topic!r}")
    document = files("ave").joinpath("knowledge").joinpath(f"{topic}.md").read_text(encoding="utf-8")
    return document.split("-->", 1)[-1].strip()


def build_prompts(recipe: dict, *, context: str = "") -> tuple[PromptStep, ...]:
    """Combine reusable knowledge with ordered, workload-specific requirements.

    A recipe supplies an objective, shared knowledge topics and steps with a name,
    files, knowledge topics and required changes. Notes and exclusions are optional.
    """
    steps = []
    for step in recipe["steps"]:
        topics = dict.fromkeys([*recipe["knowledge"], *step["knowledge"]])
        sections = [
            f"# {step['name'].replace('_', ' ').capitalize()}",
            recipe["objective"],
            "# Applicable knowledge\n\n" + "\n\n".join(_knowledge(topic) for topic in topics),
            "# Required final state\n\n" + "\n".join(f"- {item}" for item in step["required"]),
        ]
        for key, title in (
            ("notes", "Implementation notes"),
            ("avoid", "Out of scope"),
        ):
            if items := step.get(key):
                sections.append(f"# {title}\n\n" + "\n".join(f"- {item}" for item in items))
        if context:
            sections.append(f"# Workload and validation\n\n{context}")
        steps.append({"id": step["name"], "files": step["files"], "prompt": "\n\n".join(sections)})
    return parse_prompts({"steps": steps})
