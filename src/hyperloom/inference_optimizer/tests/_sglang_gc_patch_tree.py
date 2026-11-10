# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A minimal SGLang decode CUDA-graph runner and the TraceLens GC patch that applies to it."""

from __future__ import annotations

from pathlib import Path

RUNNER_REL = Path("python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py")
RUNNER_SOURCE = """\
def capture():
    return 1


def done():
    return 0
"""
RUNNER_PATCH = """\
diff --git a/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py b/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
--- a/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
+++ b/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
@@ -1,2 +1,5 @@
+def _set_profile_trace_tag(self):
+    return "_profile_runner_name"
+
 def capture():
     return 1
"""
