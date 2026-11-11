# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The staged InferenceX benchmark client as one file, for a harness that runs its client as ``python <file>``.

The staged client is a module of the ``infx`` package and imports its siblings relatively, so it cannot run as a file.
This script runs it as a module from the staged root named in ``INFERENCE_SUITE_CLIENT_ROOT``, passing its arguments
through. It imports nothing from this package: the harness runs it under its own interpreter, which need not have
``inference_suite``; ``test_client_script.py`` holds its names equal to ``CLIENT_ROOT_ENV`` and ``CLIENT_MODULE``.
"""

import os
import runpy
import sys

sys.path.insert(0, os.environ["INFERENCE_SUITE_CLIENT_ROOT"])
runpy.run_module("infx.bench_serving.benchmark_serving", run_name="__main__", alter_sys=True)
