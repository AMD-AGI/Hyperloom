"""A raw log the tests give an Experience."""

from __future__ import annotations

from pathlib import Path

from hyperloom_kb.files import file_ref
from hyperloom_kb.schema import FileRef

LOG_PATH = Path(__file__).with_name("fixtures") / "attempt.txt"
LOG: FileRef = file_ref(LOG_PATH)
