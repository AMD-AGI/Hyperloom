# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Fix a downloaded model whose tokenizer_config.json declares a tokenizer_class that
transformers cannot resolve (e.g. llm-compressor exports that write
'"tokenizer_class": "TokenizersBackend"'). Sets it to a class that loads tokenizer.json.

Usage: python tools/fix_tokenizer_config.py <model-name-substring> [tokenizer_class]
       (default tokenizer_class: Qwen2Tokenizer)
Operates on the local HuggingFace cache. Run vLLM with HF_HUB_OFFLINE=1 afterwards so the
edit is not overwritten by a re-download.
"""
import glob
import json
import os
import sys

GOOD_DEFAULT = "Qwen2Tokenizer"


def main(argv):
    if not argv:
        print("usage: fix_tokenizer_config.py <model-substring> [tokenizer_class]")
        return 1
    needle = argv[0]
    good = argv[1] if len(argv) > 1 else GOOD_DEFAULT
    base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    cfgs = [p for p in glob.glob(os.path.join(base, "**", "tokenizer_config.json"), recursive=True)
            if needle in p]
    if not cfgs:
        print("no tokenizer_config.json found for", needle, "under", base)
        return 1
    for p in cfgs:
        d = json.load(open(p, encoding="utf-8"))
        tc = d.get("tokenizer_class")
        if tc != good:
            d["tokenizer_class"] = good
            json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"patched {p}: tokenizer_class {tc} -> {good}")
        else:
            print(f"ok: {p} ({tc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
