"""The model's weight precision belongs to the campaign, never to a change.

A change of weight precision, down or up, changes what is being optimized: an FP8-weight tree is another model's
entry, whatever it measures. Hyperloom's MB decided so for every host. The precision
of intermediates (KV cache, activations, compute, collectives) may change, with weights as they are and accuracy passing.

How a launch's flags are read, and which of them set the weights' precision, is Hyperloom toolbox's: its module is
copied here unedited as :mod:`arbor.measure.weight_precision` by ``scripts/sync_weight_precision.py``, so Arbor and MB's
acceptance refuse the same launches. This module only fits it to Arbor's call
site, two ``server_flags`` strings. A change in the framework's source that quantizes weights is the agent's rule and the
critic's to catch, not this module's.
"""

from __future__ import annotations

from arbor.measure import weight_precision as _wp
from arbor.measure._hyperloom_shim import LaunchConfig


def changes(before: str, after: str) -> list[str]:
    """Each weight-precision setting ``after``'s flags change from ``before``'s, as ``"flag: old -> new"``, then each
    pair of words ``after`` adds that the frameworks read in two ways, which toolbox refuses too; ``[]`` when neither."""
    old, new = LaunchConfig(server_args=before or ""), LaunchConfig(server_args=after or "")
    return _wp.weight_precision_changes(old, new) + [
        f"{pair}: read one way by vLLM and another by argparse (write --flag.key=value)"
        for pair in _wp.ambiguous_words(old, new)]
