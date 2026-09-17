# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""lm-eval task selection shared by the accuracy gate, the bypass runner and preflight. Stdlib-only.

``MAGPIE_EVAL_TASKS`` picks which lm-eval task the accuracy gate scores. It defaults to :data:`DEFAULT_EVAL_TASKS`;
every other value is the caller's own choice and this module only supplies what the rest of the tree needs to honour
it — the metric keys a result file may carry, and the extra dependency the ``tiny*`` tasks need in order to aggregate
at all.
"""

from __future__ import annotations

# Task passed to ``lm_eval --tasks`` when ``MAGPIE_EVAL_TASKS`` is unset. Changing this changes what every accuracy
# verdict in the tree means, so it is defined once.
DEFAULT_EVAL_TASKS = "gsm8k"

# Per-task metric keys an lm-eval ``results*.json`` may carry, most specific first. ``strict-match`` leads because it
# is the stricter of the two gsm8k filters; ``acc_norm,none`` trails ``acc,none`` because a task that reports both
# means the plain ``acc`` as its headline number.
ACCURACY_METRIC_KEYS: tuple[str, ...] = (
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "acc,none",
    "acc_norm,none",
)

# lm-eval's ``tinyBenchmarks`` tasks (tinyGSM8k, tinyMMLU, ...) score a ~100-item IRT-calibrated anchor set and then
# estimate the full-benchmark number from it. The estimator lives in a separate package that lm_eval's aggregation
# module imports at top level, and that module is loaded while the task YAML is constructed, so a missing install
# aborts the run before a single request is issued rather than producing a degraded score.
TINYBENCHMARKS_MODULE = "tinyBenchmarks"

# Upstream publishes no PyPI distribution (``pip install tinyBenchmarks`` 404s), so the only install is from source.
# git first, then the archive, because the sandbox may not ship a git binary — same fallback order as the pinned
# lm_eval install in ``inference_optimizer.cli.preflight``.
TINYBENCHMARKS_PINNED_REF = "e9a8b1031b0340571beb6c9ca3a27891be09a8fd"
TINYBENCHMARKS_REPO = "github.com/felipemaiapolo/tinyBenchmarks"
TINYBENCHMARKS_PINNED_SPECS: tuple[tuple[str, str], ...] = (
    ("git", f"tinyBenchmarks @ git+https://{TINYBENCHMARKS_REPO}.git@{TINYBENCHMARKS_PINNED_REF}"),
    ("archive", f"tinyBenchmarks @ https://{TINYBENCHMARKS_REPO}/archive/{TINYBENCHMARKS_PINNED_REF}.tar.gz"),
)

# Packages an eval-dependency install must not move, pinned as ``pip -c`` arguments at their installed versions.
# Settled by install.sh (or the image) and load-bearing elsewhere in the stack: pandas for rocprof-compute's CSV
# converter, torch/triton for the ROCm build PyPI has no equivalent of, numpy because both pin against it. scipy is
# here because tinyBenchmarks declares numpy/scipy/requests unpinned: on an image without scipy, resolving it is what
# would drag numpy along with it.
EVAL_INSTALL_FROZEN_DEPS = ("torch", "pandas", "numpy", "scipy", "triton")


def split_eval_tasks(tasks: str | None) -> tuple[str, ...]:
    """Split an ``lm_eval --tasks`` value into its individual task names."""
    return tuple(part.strip() for part in str(tasks or "").split(",") if part.strip())


def eval_tasks_need_tinybenchmarks(tasks: str | None) -> bool:
    """Report whether *tasks* selects at least one lm-eval ``tiny*`` task.

    Every task in upstream's ``tinyBenchmarks`` group — the individual tasks and the group itself — is named with a
    ``tiny`` prefix, and they are the only tasks whose aggregation imports :data:`TINYBENCHMARKS_MODULE`.
    """
    return any(name.lower().startswith("tiny") for name in split_eval_tasks(tasks))
