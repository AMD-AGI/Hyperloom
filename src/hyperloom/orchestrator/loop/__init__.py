# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator reactor loop: facade + extracted collaborator objects.

``coordinator.py`` holds the thin facade (``__init__``/``tick``/``run``/
``_reactor_pass`` + lazy collaborator properties); the method clusters it
delegates to live in the sibling modules here (``conversation``,
``maintenance``, ``dispatcher``, ``proposals``, ``writeback``,
``intent_router``). ``writeback`` also owns the folded result-recording and
resume-reconcile clusters; ``dispatcher`` owns the folded gating and inline
action clusters; ``conversation`` owns the folded advisory blocks.
"""
