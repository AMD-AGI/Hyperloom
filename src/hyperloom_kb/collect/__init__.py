"""Configuration-driven Experience collection from producer logs."""

from hyperloom_kb.collect.engine import (
    MAX_CHANGE_CONTENT_BYTES,
    MAX_EXPERIENCE_BYTES,
    REPORT_FORMAT,
    CollectedExperience,
    CollectionTarget,
    CollectReport,
    FailedUnit,
    SkippedUnit,
    SourceDocumentError,
    collect,
    load_document,
    write_report,
)
from hyperloom_kb.collect.expressions import BUILTINS, EvaluationError, MappingError
from hyperloom_kb.collect.mapping import (
    MAPPING_FORMAT,
    CollectMapping,
    compile_mapping,
    load_mapping,
)

__all__ = [
    "BUILTINS",
    "MAPPING_FORMAT",
    "MAX_CHANGE_CONTENT_BYTES",
    "MAX_EXPERIENCE_BYTES",
    "REPORT_FORMAT",
    "CollectMapping",
    "CollectReport",
    "CollectedExperience",
    "CollectionTarget",
    "EvaluationError",
    "FailedUnit",
    "MappingError",
    "SkippedUnit",
    "SourceDocumentError",
    "collect",
    "compile_mapping",
    "load_document",
    "load_mapping",
    "write_report",
]
