# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator reactor loop: facade + extracted collaborator objects.

``coordinator.py`` holds the thin facade (``__init__``/``tick``/``run``/
``_reactor_pass`` + lazy collaborator properties); the method clusters it
delegates to live in the sibling modules here (``conversation``, ``resume``,
``maintenance``, ``dispatcher``, ``proposals``, ``writeback``, ``gating``,
``advisory``, ``intent_router``, ``result_recorder``).
"""
