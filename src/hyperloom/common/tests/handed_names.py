# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a text handed to a specialist names of a server built from its checkout.

A prompt teaches a method and asks for evidence: an option of the server, a word of a kernel knob, an identifier of
its source or an upstream pull request named in one is an answer handed over, not one the session found. The names
are read from the server's own help (the suite's sample of it) and from the rule pack, so no test holds a list of
them.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import inference_suite
from hyperloom.common.outside_facts import kernel_knobs

__all__ = ["identifiers_named", "knob_words_named", "pull_requests_named", "server_options_named"]

_HELP = Path(inference_suite.__file__).parent / "tests" / "fixtures" / "llamacpp" / "llama-server-help.txt"
#: A flag as a text writes it. A one-letter flag is what any command takes, so it names no option of this server.
_FLAG = re.compile(r"(?<![\w-])(?:--[A-Za-z][\w-]*|-[A-Za-z][\w-]+)")
_IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_PULL_REQUEST = re.compile(r"#\d+|\bpull/\d+|\b(?:PR|pull request) #?\d+", re.IGNORECASE)
#: A part of a knob's name shorter than this is too common a word to look for.
_PART_CHARS = 5


def server_options_named(text: str) -> set[str]:
    """The options of the sample server that ``text`` writes as flags, by the spelling it uses."""
    spellings = {
        spelling
        for option in inference_suite.help_options(_HELP.read_text(encoding="utf-8"))
        for spelling in option.spellings
    }
    return set(_FLAG.findall(text)) & spellings


def _knob_words() -> set[str]:
    words: set[str] = set()
    for name, knob in kernel_knobs().items():
        file = PurePosixPath(knob.file)
        words |= {name, file.name, file.stem}
        words |= {part for part in name.split("_") if len(part) >= _PART_CHARS}
        words |= {found.rstrip("_") for found in _IDENTIFIER.findall(f"{knob.anchor} {knob.insert}")}
        words |= {value for _, values in knob.axes for value in values if any(c.isalpha() for c in value)}
    return {word.lower() for word in words}


def knob_words_named(text: str) -> set[str]:
    """The words of the rule pack's kernel knobs that ``text`` holds, in any case: a knob's name and its longer
    parts, its file, the identifiers of the lines it is set by and the values of its axes that are names."""
    lowered = text.lower()
    return {word for word in _knob_words() if word in lowered}


def identifiers_named(text: str) -> set[str]:
    """The identifiers ``text`` holds: words joined by underscores."""
    return set(_IDENTIFIER.findall(text))


def pull_requests_named(text: str) -> set[str]:
    """The pull requests ``text`` names by number."""
    return set(_PULL_REQUEST.findall(text))
