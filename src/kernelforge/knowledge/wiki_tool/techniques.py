# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A knowledge base's technique vocabulary, and the techniques an Implementer declares its change applies.

The Implementer names them in the knowledge base's own terms when it ends its session, before the change is measured,
so whether a kept change is one the knowledge base covers is read from what was declared, not inferred afterwards.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from kernelforge.config import Config

ALIASES_FILE = "aliases.yaml"
TAGS_FILE = Path("data") / "tags.yaml"
TECHNIQUES_TAG = "TECHNIQUES:"

_MAX_UNLISTED = 5
_UNLISTED_CHARS = 60
_NONE = "none"
# The last line that opens with the tag, as an agent writes it: plain, bulleted or bold.
_LINE = re.compile(r"^[\s>*_`-]*TECHNIQUES:\**\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_DECORATION = re.compile(r"[*`\"'.]+")


def _key(name: str) -> str:
    return " ".join(_DECORATION.sub(" ", name).lower().split())


@dataclass(frozen=True)
class TechniqueVocabulary:
    """The techniques a knowledge base names, and the spellings of each it accepts."""

    terms: tuple[str, ...]
    # Each accepted spelling, lowercased, mapped to the term it names.
    spellings: Mapping[str, str]

    @classmethod
    def load(cls, tags_path: Path) -> TechniqueVocabulary:
        """Read the ``techniques`` axis of a ``tags.yaml``, with the aliases its sibling ``aliases.yaml`` gives them."""
        aliases_path = tags_path.parent / ALIASES_FILE
        try:
            tags = yaml.safe_load(tags_path.read_text(encoding="utf-8")) or {}
            aliases = (yaml.safe_load(aliases_path.read_text(encoding="utf-8")) if aliases_path.is_file() else {}) or {}
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"technique vocabulary {tags_path} is unreadable: {error}") from error
        terms = tuple(str(term) for term in tags.get("techniques") or [] if str(term).strip())
        if not terms:
            raise ValueError(f"technique vocabulary {tags_path} lists no techniques")
        spellings: dict[str, str] = {}
        for term in terms:
            spellings[_key(term)] = term
            spellings[_key(term.replace("-", " "))] = term
            for variant in aliases.get(term) or []:
                spellings.setdefault(_key(str(variant)), term)
        return cls(terms=terms, spellings=spellings)

    @classmethod
    def from_config(cls, config: Config) -> TechniqueVocabulary:
        """The campaign's vocabulary, else its knowledge base checkout's, the one that knowledge base judges coverage
        on; without either, a vocabulary with no terms, which asks for nothing and reads nothing."""
        if config.technique_vocabulary:
            return cls.load(Path(config.technique_vocabulary))
        tags = Path(config.knowledge_wiki_root) / TAGS_FILE
        if config.knowledge_wiki_root and tags.is_file():
            return cls.load(tags)
        return cls(terms=(), spellings={})

    def request(self) -> str:
        """The Implementer prompt's instruction for the line it declares its techniques on."""
        return (
            f"Just before your `PLAN:` line, output one line starting with `{TECHNIQUES_TAG}` that names, "
            "comma-separated, every technique the change now in the file applies, each from this list: "
            f"{', '.join(self.terms)}. Name a technique the list does not have in at most five words of your own, "
            f"and write `{TECHNIQUES_TAG} none` when the change applies none of them."
        )

    def request_suffix(self) -> str:
        """The request, to follow the prompt's instruction for the `PLAN:` line; empty without terms to ask in."""
        return f" {self.request()}" if self.terms else ""

    def labels(self, text: str) -> dict[str, Any] | None:
        """The techniques a session's last ``TECHNIQUES:`` line names, the vocabulary's and its own; ``None`` when it
        wrote no such line or there were no terms to ask in."""
        lines = _LINE.findall(text or "")
        if not self.terms or not lines:
            return None
        listed: list[str] = []
        unlisted: list[str] = []
        for entry in re.split(r"[,;]", lines[-1]):
            key = _key(entry)
            if not key or key == _NONE:
                continue
            term = self.spellings.get(key) or self.spellings.get(key.replace(" ", "-"))
            if term is not None:
                if term not in listed:
                    listed.append(term)
            elif len(unlisted) < _MAX_UNLISTED:
                unlisted.append(" ".join(entry.split())[:_UNLISTED_CHARS])
        return {"listed": listed, "unlisted": unlisted}
