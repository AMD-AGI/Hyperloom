---
myst:
    html_meta:
        "description": "Merge smoke variable for Hyperloom: controls whether Python import smoke-testing runs on patched files before a test is given a sandbox or GPU cards."
        "keywords": "Hyperloom, environment variables, merge smoke, HYPERLOOM_MERGE_SMOKE, byte-compile, import check, patched module, multi-branch"
---
# Merge smoke

Before a test that carries patches is given a sandbox or cards, the multi-branch toolbox (hosted and local) byte-compiles
every Python file the test's patch stack changes and imports each patched component on CPU, with every GPU hidden and the
point's server variables set. A file that does not compile, or, while the image's own copy imports, an `ImportError`
(not `ModuleNotFoundError`) for a name a patched module lacks or an `AttributeError` raised on a patched module, fails
the test as the candidate's (`merge_smoke`), whatever the reason the name is missing (a patched module that defines a
name only under a device check is judged too, since such a stack fails on any other device as well); any other import
failure (a `NameError`, a missing module, a name or attribute an unpatched module or value lacks) is logged and the test
is measured as usual. A smoke that cannot run is the
infrastructure failure `smoke_not_run`, retried once.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_MERGE_SMOKE` | Unset (on) | Read from the toolbox host's environment when the toolbox is built. Unset, empty or `on` (any case, surrounding spaces ignored) keeps the merge smoke on; `off` turns it off, so every test goes straight to its sandbox. Any other value is refused: building the toolbox fails with a `ValueError`. Operator configuration. |
