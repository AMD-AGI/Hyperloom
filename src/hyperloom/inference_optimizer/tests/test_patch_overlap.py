# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A proposed patch that repeats an already-measured one is recognised by its diff, not by its name."""

from __future__ import annotations

from hyperloom.orchestrator.framework import patch_overlap as po
from hyperloom.orchestrator.state.shared_state import SharedState

# Two specialists porting one upstream change: same added lines, different file prefix, context and ordering.
_PORT_A = """\
diff --git a/python/sglang/srt/entrypoints/serving_chat.py b/python/sglang/srt/entrypoints/serving_chat.py
--- a/python/sglang/srt/entrypoints/serving_chat.py
+++ b/python/sglang/srt/entrypoints/serving_chat.py
@@ -10,6 +10,9 @@ class ServingChat:
     def encode(self, prompt):
-        return self.tokenizer.encode(prompt)
+        loop = asyncio.get_running_loop()
+        return await loop.run_in_executor(self._pool, self.tokenizer.encode, prompt)
"""

_PORT_B = """\
--- a/sglang/srt/entrypoints/serving_chat.py
+++ b/sglang/srt/entrypoints/serving_chat.py
@@ -88,7 +88,10 @@ def something_else():
     def encode(self, prompt):
-        return self.tokenizer.encode(prompt)
+        loop = asyncio.get_running_loop()
+        return await loop.run_in_executor(self._pool, self.tokenizer.encode, prompt)
     # trailing context differs
"""

_FOLLOW_UP = """\
--- a/python/sglang/srt/entrypoints/serving_chat.py
+++ b/python/sglang/srt/entrypoints/serving_chat.py
@@ -20,3 +20,4 @@ class ServingChat:
     def shutdown(self):
+        self._pool.shutdown(wait=False)
"""


def test_two_ports_of_one_change_overlap_despite_different_names_and_paths():
    a = po.changed_line_keys([_PORT_A])
    b = po.changed_line_keys([_PORT_B])

    assert po.overlap_ratio(a, b) >= 0.5


def test_the_same_code_line_under_different_comments_is_one_change():
    # Three GLM-5.3 specialists wrote this one-line fix, each with its own explanatory comment.
    first = '+++ b/m.py\n+    # the draft arch is rewritten\n+    fused = "NextN"\n'
    second = '+++ b/m.py\n+    # compare against the\n+    # rewritten name\n+    fused = "NextN"\n'
    kernel_a = "+++ b/k.cu\n+// tile for small M\n+  run<64>(x);\n"
    kernel_b = "+++ b/k.cu\n+/* small-M tile\n+ * chosen by sweep */\n+  run<64>(x);\n"

    assert po.changed_line_keys([first]) == po.changed_line_keys([second])
    assert po.changed_line_keys([kernel_a]) == po.changed_line_keys([kernel_b])


def test_a_preprocessor_line_is_code_outside_python():
    tile_64 = "+++ b/k.cu\n+#define TILE_M 64\n+  run<TILE_M>(x);\n"
    tile_128 = "+++ b/k.cu\n+#define TILE_M 128\n+  run<TILE_M>(x);\n"

    assert po.overlap_ratio(po.changed_line_keys([tile_64]), po.changed_line_keys([tile_128])) == 1 / 3


def test_an_edit_or_a_revert_of_a_landed_line_is_not_a_repeat_of_it():
    landed = po.changed_line_keys(['+++ b/m.py\n+arch = "A"\n'])
    edit = po.changed_line_keys(['+++ b/m.py\n-arch = "A"\n+arch = "B"\n'])
    revert = po.changed_line_keys(['+++ b/m.py\n-arch = "A"\n'])

    assert (po.overlap_ratio(edit, landed), po.overlap_ratio(revert, landed)) == (0.0, 0.0)


def test_a_follow_up_to_the_same_file_does_not_look_like_a_repeat():
    measured = po.changed_line_keys([_PORT_A])
    follow_up = po.changed_line_keys([_FOLLOW_UP])

    assert po.overlap_ratio(follow_up, measured) < 0.5
    # Same file: a files-only identity would have called this a duplicate.
    assert set(po.touched_files([_PORT_A])) & set(po.touched_files([_FOLLOW_UP]))


def test_touched_files_drop_the_diff_prefix_and_dev_null():
    assert po.touched_files([_PORT_A]) == ["python/sglang/srt/entrypoints/serving_chat.py"]
    assert po.touched_files(["+++ /dev/null\n"]) == []


def test_overlapping_measured_ranks_strongest_first_and_hides_the_keys():
    measured = [
        {"task_id": "t1", "patch_name": "port-a", "line_keys": po.changed_line_keys([_PORT_A]), "outcome": "adopted"},
        {"task_id": "t2", "patch_name": "unrelated", "line_keys": po.changed_line_keys([_FOLLOW_UP])},
    ]

    hits = po.overlapping_measured(po.changed_line_keys([_PORT_B]), measured, min_ratio=0.5)

    assert [h["task_id"] for h in hits] == ["t1"]
    assert hits[0]["overlap_ratio"] >= 0.5
    assert hits[0]["patch_name"] == "port-a" and hits[0]["outcome"] == "adopted"
    assert "line_keys" not in hits[0]


def test_nothing_measured_or_nothing_changed_reports_no_overlap():
    assert po.overlapping_measured([], [{"line_keys": ["a"]}], min_ratio=0.5) == []
    assert po.overlapping_measured(["a"], [], min_ratio=0.5) == []
    assert po.overlap_ratio([], ["a"]) == 0.0


def test_patch_set_identity_reads_the_files_and_skips_unreadable_ones(tmp_path):
    patch = tmp_path / "a.patch"
    patch.write_text(_PORT_A, encoding="utf-8")

    files, keys = po.patch_set_identity([str(patch), str(tmp_path / "absent.patch"), ""])

    assert files == ["python/sglang/srt/entrypoints/serving_chat.py"]
    assert keys == po.changed_line_keys([_PORT_A])
    assert po.patch_set_identity([]) == ([], [])


def test_the_ledger_keeps_the_newest_entries_and_drops_empty_ones():
    state = SharedState(session_id="s")

    state.record_measured_patch_set(
        task_id="t0", patch_name="empty", identity={"files": [], "line_keys": []}, outcome="reverted"
    )
    assert state.measured_patch_sets == []

    for i in range(po.MAX_MEASURED + 3):
        state.record_measured_patch_set(
            task_id=f"t{i}", patch_name="p", identity={"files": ["f"], "line_keys": [f"k{i}"]}, outcome="adopted"
        )

    assert len(state.measured_patch_sets) == po.MAX_MEASURED
    assert state.measured_patch_sets[-1]["task_id"] == f"t{po.MAX_MEASURED + 2}"
    assert state.measured_patch_sets[0]["ts"]
