# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m hyperloom.measure.long_context.build --model DIR --cache DIR [--served-context N | --length L]``: stage the
prompt set for a tokenizer.

Run it on a host that has the model's tokenizer files, before a campaign; the sandbox only reads the entry. The prompts are
built at the context the check's launch serves: ``N`` (its ``--context-length`` or ``--max-model-len``, else its
``MAX_MODEL_LEN``), no more than the model's own limit (its ``config.json``) and no more than ``LENGTH_BOUND``, less the 128
tokens an answer may have. A set built longer than the launch serves never runs: the check ends incomplete before a server
starts. Without ``N`` the model's limit is used; ``L`` sets the length outright, within the same bound. The entry's key
holds the bound. It prints the entry's key, the SHA-256 of its prompts file and the length.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from hyperloom.inference_optimizer.model_config_utils import load_model_max_position_embeddings
from hyperloom.measure.long_context.plan import LENGTH_BOUND, MIN_LENGTH, REPLY_HEADROOM
from hyperloom.measure.long_context.prompts import PromptSetError
from hyperloom.measure.long_context.staging import stage_entry
from hyperloom.measure.long_context.tasks import build_prompts
from hyperloom.measure.short_prompts.build import huggingface_encoder, tokenizer_hashes
from hyperloom.measure.short_prompts.staging import StagedEntry

__all__ = ["DEFAULT_SEED", "LONGEST", "build_entry", "entry_inputs", "main", "set_length"]

#: The seed the set is built with when none is given; the toolbox finds the default set of a model by it.
DEFAULT_SEED = 20261024
#: The longest prompt a set is built with: ``LENGTH_BOUND`` less the tokens an answer may have.
LONGEST = LENGTH_BOUND - REPLY_HEADROOM


def set_length(model_limit: int | None, served_context: int | None) -> int:
    """The token length to build the prompts at: the context the launch serves, no more than the model's limit and no more
    than ``LENGTH_BOUND``, less the tokens an answer may have.

    Raises:
        PromptSetError: Neither is known, so nothing says how long a prompt the server takes.
    """
    known = [context for context in (model_limit, served_context) if context is not None]
    if not known:
        raise PromptSetError("the model states no context limit and no served context was given; give --served-context")
    return min(*known, LENGTH_BOUND) - REPLY_HEADROOM


def entry_inputs(model_dir: Path, *, length: int, seed: int = DEFAULT_SEED) -> dict[str, object]:
    """What the entry of a set for the tokenizer in ``model_dir`` is keyed by: the seed, the length, the tokenizer files'
    hashes and the bound the set was built under, so a set built under another bound is another entry.

    Raises:
        PromptSetError: The tokenizer files cannot be read.
        OSError: The model directory cannot be read.
    """
    return {"seed": seed, "length": length, "tokenizer": tokenizer_hashes(model_dir), "bound": LENGTH_BOUND}


def build_entry(
    model_dir: Path,
    cache: Path,
    *,
    length: int,
    seed: int = DEFAULT_SEED,
    encode: Callable[[str], Sequence[int]] | None = None,
) -> StagedEntry:
    """Build the prompts for the tokenizer in ``model_dir`` and stage them in ``cache``; ``encode`` replaces the Hugging
    Face tokenizer (a test's)."""
    if isinstance(length, bool) or not MIN_LENGTH <= length <= LONGEST:
        raise PromptSetError(f"the length must be {MIN_LENGTH} to {LONGEST} tokens, got {length!r}")
    inputs = entry_inputs(model_dir, length=length, seed=seed)
    prompts = build_prompts(encode or huggingface_encoder(model_dir), length=length, seed=seed)
    return stage_entry(cache, prompts, inputs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hyperloom.measure.long_context.build", description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="the model directory (its tokenizer files)")
    parser.add_argument("--cache", type=Path, required=True, help="the runner cache directory to stage into")
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--served-context",
        type=int,
        help="the context the check's launch serves, in tokens (default: the model's limit)",
    )
    size.add_argument("--length", type=int, help="the prompts' length in tokens, set outright")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args(argv)
    try:
        length = arguments.length
        if length is None:
            length = set_length(load_model_max_position_embeddings(str(arguments.model)), arguments.served_context)
        entry = build_entry(arguments.model, arguments.cache, length=length, seed=arguments.seed)
    except (PromptSetError, OSError) as exc:
        print(f"long context: {exc}", file=sys.stderr)
        return 1
    print(f"prompt_set={entry.key}\nprompts_sha256={entry.prompts_sha256}\nlength={length}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
