# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``hyperloom.common`` — zero-dependency shared library.

Target home for the deduplicated, first-party-dependency-free building blocks
(``io`` / ``env`` / ``payload_aliases`` /
``jsonio`` / ``timeutil`` / ``gain_math`` / ``paths`` / ``subprocess_bridge`` /
``llm`` …). Constraint: this package may only import the stdlib (plus ``httpx``
for the ``llm`` submodule) and must never import ``orchestrator`` /
``inference_optimizer`` / ``agents`` — keeping the dependency graph acyclic.

Phase P2.0 intentionally ships an empty scaffold: extraction happens in P2.1.

No ``paths.py`` module (re-investigated post-P2.8): the extraction plan lists
``paths.py`` (common path primitives, "replaces the 2 ``_paths`` copies") as
an extraction candidate, but a from-scratch re-audit found the "2 copies" are
NOT a safe-to-merge in-package duplication:

* ``hyperloom.inference_optimizer.session.paths.workspace_root`` is the
  real, feature-complete implementation (``Path``-typed, many sibling
  helpers: ``session_dir``/``make_session_dir``/``asset_root``/
  ``open_source_root``/etc.).
* ``hyperloom.agents.kernel.tools._paths.workspace_root`` is a deliberately
  independent, stdlib-only, ``str``-typed mirror of just that one function.
  It is NOT imported as a package-relative sibling (``from ._paths import``)
  — every caller uses the bare module name ``from _paths import
  workspace_root``, because ``agents/kernel/tools/*.py`` are standalone
  scripts invoked as ``python3 <root>/tools/<tool>.py`` subprocesses (see
  ``HYPERLOOM_KERNEL_AGENT_ROOT`` in
  ``hyperloom.orchestrator.kernel.request_handlers``) and some of their
  code paths run inside Ray workers (``tools/backends/``) that do not
  inherit the driver's ``sys.path`` — the same constraint already documented
  for ``tools/_payload_aliases.py`` (standalone-script anti-cycle rule). Adding a
  ``hyperloom.common`` (or even a first-party ``hyperloom.*``) import to
  ``tools/_paths.py`` would break that contract.

No other in-package (non-standalone-script) path-primitive duplication was
found by a full-repo grep (``def workspace_root``, ``def open_source_root``,
``def asset_root``, etc.): every other package's path helpers
(``agents/framework/kb.py:_resolve_kb_root``,
``agents/robustness/config.py:_discover_session_dir``,
``agents/quantization/driver/runner.py:resolve_skill_path``, ``ci/``'s
``get_nfs_root``/``_safe_local_path``) resolve genuinely different,
package-specific roots — not copies of the same primitive. Creating a
``common/paths.py`` here would either (a) re-export a single implementation
with no second caller (a "fake extraction" with zero dedup benefit), or
(b) force the standalone kernel-agent tools to depend on ``hyperloom.common``
and break their remote/no-import contract. Neither is warranted, so this
module intentionally does not exist; see ``tools/_paths.py`` for the
mirrored implementation's own rationale.
"""

from __future__ import annotations

__all__: list[str] = []
