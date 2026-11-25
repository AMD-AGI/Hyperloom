# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m hyperloom.measure.runner.native.build_check``: whether a specialist's tree of a framework whose server
the session builds from source compiles, built as integration builds it but into a store of the session's checks
(:func:`~hyperloom.measure.runner.native.session_builds.check_native_tree`), with nothing measured.

It prints the build as JSON (``status``, ``key``, ``reused``, ``detail``: the compile's failure or the machine's reason)
and exits 0 when the tree built, 1 when it did not compile or changes what no build may take, and 2 when the machine
could not build it or the session builds no server from source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hyperloom.inference_optimizer.session.paths import session_dir
from hyperloom.inference_optimizer.native_builds import NativeBuildsRefused
from hyperloom.measure.runner.native.session_builds import check_native_tree
from hyperloom.orchestrator.actions.executors.candidate_builds import TreeBuildStatus

_EXIT = {TreeBuildStatus.BUILT: 0, TreeBuildStatus.COMPILE_ERROR: 1, TreeBuildStatus.INFRA: 2}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hyperloom.measure.runner.native.build_check", description=__doc__)
    parser.add_argument("--tree", type=Path, required=True, help="the framework checkout to build (your worktree)")
    parser.add_argument("--session", type=Path, default=None, help="the session directory (this run's when omitted)")
    parser.add_argument(
        "--add", action="append", default=[], help="a new, untracked file of the tree to build too (repeatable)"
    )
    args = parser.parse_args(argv)
    try:
        build = check_native_tree(args.session or session_dir(), args.tree, args.add)
    except NativeBuildsRefused as refused:
        print(json.dumps({"status": TreeBuildStatus.INFRA.value, "key": "", "reused": False, "detail": str(refused)}))
        return _EXIT[TreeBuildStatus.INFRA]
    print(json.dumps(build.to_record()))
    return _EXIT[build.status]


if __name__ == "__main__":
    sys.exit(main())
