"""Mini vllm arg_utils fixture (argparse + dataclass patterns)."""

from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class EngineArgs:
    """Stand-in for vllm.engine.arg_utils.EngineArgs."""

    model: str = ""
    max_model_len: int = 8192
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    quantization: str = "auto"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum context length.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilisation fraction.")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="Maximum number of concurrent sequences.")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Weight quantisation mode.")
    parser.add_argument("--block-size", type=int, default=16)
    return parser
