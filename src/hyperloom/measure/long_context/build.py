# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m hyperloom.measure.long_context.build --model DIR --cache DIR --length L``: stage the prompt set for a tokenizer.

Run it on a host that has the model's tokenizer files, before a campaign; the sandbox only reads the entry. ``L`` is the
token length of the prompts: the model's own context limit, at most 128K, and the launch the check runs on must serve ``L``
and 128 tokens more. It prints the entry's key, the SHA-256 of its prompts file and the length.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from hyperloom.measure.long_context.plan import MAX_LENGTH, MIN_LENGTH
from hyperloom.measure.long_context.prompts import PromptSetError
from hyperloom.measure.long_context.staging import stage_entry
from hyperloom.measure.long_context.tasks import build_prompts
from hyperloom.measure.short_prompts.build import huggingface_encoder, tokenizer_hashes
from hyperloom.measure.short_prompts.staging import StagedEntry

__all__ = ["build_entry", "main"]

_DEFAULT_SEED = 20261024


def build_entry(
    model_dir: Path,
    cache: Path,
    *,
    length: int,
    seed: int = _DEFAULT_SEED,
    encode: Callable[[str], Sequence[int]] | None = None,
) -> StagedEntry:
    """Build the prompts for the tokenizer in ``model_dir`` and stage them in ``cache``; ``encode`` replaces the Hugging
    Face tokenizer (a test's)."""
    if isinstance(length, bool) or not MIN_LENGTH <= length <= MAX_LENGTH:
        raise PromptSetError(f"the length must be {MIN_LENGTH} to {MAX_LENGTH} tokens, got {length!r}")
    inputs = {"seed": seed, "length": length, "tokenizer": tokenizer_hashes(model_dir)}
    prompts = build_prompts(encode or huggingface_encoder(model_dir), length=length, seed=seed)
    return stage_entry(cache, prompts, inputs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hyperloom.measure.long_context.build", description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="the model directory (its tokenizer files)")
    parser.add_argument("--cache", type=Path, required=True, help="the runner cache directory to stage into")
    parser.add_argument("--length", type=int, required=True, help="the prompts' length in tokens (at most 128K)")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    arguments = parser.parse_args(argv)
    try:
        entry = build_entry(arguments.model, arguments.cache, length=arguments.length, seed=arguments.seed)
    except (PromptSetError, OSError) as exc:
        print(f"long context: {exc}", file=sys.stderr)
        return 1
    print(f"prompt_set={entry.key}\nprompts_sha256={entry.prompts_sha256}\nlength={arguments.length}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
