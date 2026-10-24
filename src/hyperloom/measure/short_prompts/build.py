# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m hyperloom.measure.short_prompts.build --model DIR --cache DIR``: stage the prompt set for a tokenizer.

Run it on a host that has the model's tokenizer files, before a campaign; the sandbox only reads the entry. It prints the
entry's key and the SHA-256 of its prompts file, the two values the rule pack's plan names.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from hyperloom.measure.short_prompts.prompts import PromptSetError, build_prompts
from hyperloom.measure.short_prompts.staging import StagedEntry, stage_entry

__all__ = ["build_entry", "main", "tokenizer_hashes"]

#: The files that make a tokenizer what it is, when the model directory has them.
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)
_DEFAULT_SEED = 20261024
_CHUNK = 1 << 20


def tokenizer_hashes(model_dir: Path) -> dict[str, str]:
    """The SHA-256 of each tokenizer file the model directory holds; a directory with none is refused."""
    found = {}
    for name in _TOKENIZER_FILES:
        path = Path(model_dir) / name
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(_CHUNK):
                    digest.update(chunk)
            found[name] = digest.hexdigest()
    if not found:
        raise PromptSetError(f"{model_dir} holds none of the tokenizer files {list(_TOKENIZER_FILES)}")
    return found


def _huggingface_encoder(model_dir: Path) -> Callable[[str], Sequence[int]]:
    try:
        from transformers import AutoTokenizer  # a builder-host dependency; the sandbox never imports this module
    except ImportError as exc:
        raise PromptSetError("the transformers library is needed to read the model's tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    return lambda text: tokenizer.encode(text, add_special_tokens=False)


def build_entry(
    model_dir: Path,
    cache: Path,
    *,
    seed: int = _DEFAULT_SEED,
    encode: Callable[[str], Sequence[int]] | None = None,
) -> StagedEntry:
    """Build the prompts for the tokenizer in ``model_dir`` and stage them in ``cache``; ``encode`` replaces the
    Hugging Face tokenizer (a test's)."""
    inputs = {"seed": seed, "tokenizer": tokenizer_hashes(model_dir)}
    prompts = build_prompts(encode or _huggingface_encoder(model_dir), seed=seed)
    return stage_entry(cache, prompts, inputs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hyperloom.measure.short_prompts.build", description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="the model directory (its tokenizer files)")
    parser.add_argument("--cache", type=Path, required=True, help="the runner cache directory to stage into")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    arguments = parser.parse_args(argv)
    try:
        entry = build_entry(arguments.model, arguments.cache, seed=arguments.seed)
    except (PromptSetError, OSError) as exc:
        print(f"short prompts: {exc}", file=sys.stderr)
        return 1
    print(f"prompt_set={entry.key}\nprompts_sha256={entry.prompts_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
