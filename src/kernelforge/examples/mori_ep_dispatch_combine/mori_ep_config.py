"""Tunable MoRI-EP dispatch/combine launch configuration."""

from __future__ import annotations


def get_ep_launch_config() -> dict:
    """Return MoRI-EP dispatch/combine launch-config overrides."""
    return {
        "dispatch_block_num": 80,
        "dispatch_warp_per_block": 8,
        "combine_block_num": 80,
        "combine_warp_per_block": 8,
        "kernel_type": "IntraNode",
        "combine_zero_copy": False,
    }
