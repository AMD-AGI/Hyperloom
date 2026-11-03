"""In-server correctness checks, one module per check.

ruppell's vLLM plugin imports ``ruppell.checks.<name>`` and calls its
``install()`` in every vLLM process when ``RUPPELL_CHECK=<name>[,<name>...]``
is set, independently of tracing (``RUPPELL_VLLM``).
"""
