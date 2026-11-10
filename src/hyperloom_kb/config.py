"""Declaration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hyperloom_kb.schema import (
    ExperienceDeclaration,
    SchemaValidationError,
)

PACKAGED_DECLARATION = Path(__file__).with_name("declarations") / "inference-recipe-v1.yaml"


class ConfigurationError(ValueError):
    """Raised for explicit invalid bootstrap configuration."""


def load_declaration(path: str | Path) -> ExperienceDeclaration:
    """Load and validate one YAML or JSON declaration file."""

    declaration_path = Path(path)
    try:
        text = declaration_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read Hyperloom-KB declaration {declaration_path}: {exc}") from exc
    try:
        value: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"cannot parse Hyperloom-KB declaration {declaration_path}: {exc}") from exc
    try:
        return ExperienceDeclaration.from_dict(value)
    except SchemaValidationError as exc:
        raise ConfigurationError(f"invalid Hyperloom-KB declaration {declaration_path}: {exc}") from exc


__all__ = [
    "PACKAGED_DECLARATION",
    "ConfigurationError",
    "load_declaration",
]
