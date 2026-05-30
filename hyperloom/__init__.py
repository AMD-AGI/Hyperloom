"""Hyperloom — modular LLM inference optimization framework.

Provide a benchmark script and an optional accuracy eval script.
Hyperloom handles the rest: profiling, kernel optimization, agent dispatch,
and iterative improvement with graceful degradation when tools are unavailable.
"""

__version__ = "1.0.0"

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)
