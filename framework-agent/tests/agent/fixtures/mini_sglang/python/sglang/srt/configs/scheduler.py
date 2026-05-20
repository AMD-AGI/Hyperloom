"""Mini SchedulerConfig fixture exercising the pydantic pattern."""

from typing import Optional


# Fake BaseModel stand-in so the test fixture imports clean. The
# `ast_scanner` only inspects class-bases textually so the actual
# pydantic import is unnecessary for fixture purposes.
class BaseModel:
    pass


class SchedulerConfig(BaseModel):
    """Stand-in for sglang scheduler config."""

    max_num_batched_tokens: int = 4096
    max_prefill_tokens: int = 16384
    batch_notify_size: int = 16
    enable_dp_attention: bool = False
    stream_interval: Optional[int] = 1
