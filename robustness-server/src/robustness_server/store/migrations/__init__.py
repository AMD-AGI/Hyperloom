"""Embedded SQL migrations.

Files in this package are loaded via ``importlib.resources`` and applied
in lexical order during database startup. Add a new file named
``NNNN_description.sql`` (zero-padded ordinal) — the runner sorts by
filename, so prefixes guarantee deterministic order.
"""
