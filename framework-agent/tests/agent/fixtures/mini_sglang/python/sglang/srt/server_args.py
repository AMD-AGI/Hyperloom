"""Mini sglang server_args fixture for AST scanner tests.

Covers all 3 patterns: argparse + dataclass + pydantic.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerArgs:
    """Stand-in for sglang.srt.server_args.ServerArgs."""

    model_path: str = ""
    tokenizer_path: str = ""
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    schedule_conservativeness: float = 0.3
    enable_chunked_prefill: bool = True
    kv_cache_dtype: str = "auto"


def build_parser() -> ArgumentParser:
    """Stand-in for sglang.srt.server_args.build_parser."""
    parser = ArgumentParser()
    parser.add_argument("--max-running-requests", type=int, default=512,
                        help="Max running requests.")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=128,
                        help="Max batch size for CUDA graph capture.")
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=8192,
        help="Chunked prefill token size.",
    )
    parser.add_argument("--disable-radix-cache", action="store_true",
                        help="Disable the radix prefix cache.")
    parser.add_argument("--kv-cache-dtype", type=str, default="auto",
                        help="KV cache storage dtype.")
    return parser
