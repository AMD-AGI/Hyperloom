---
myst:
    html_meta:
        "description": "Plug selection variables for Hyperloom: manifest path and overrides path for choosing plug socket implementations such as the events (progress reporting) socket."
        "keywords": "Hyperloom, environment variables, plug selection, HYPERLOOM_MANIFEST, HYPERLOOM_MANIFEST_OVERRIDES, plug profile, events socket, --plug"
---
# Plug selection

Which implementation serves each plug socket (today only `events`, progress reporting) is chosen by layering, lowest
first: the built-in `local` profile, the `plugs.resolved.json` of the session being resumed, `--plug-profile`,
the manifest (`--manifest` or `HYPERLOOM_MANIFEST`), `HYPERLOOM_MANIFEST_OVERRIDES`, then `--plug` and `--plug-config`.
With none of these set, a run uses the `local` profile, writes nothing to the session and prints nothing. See
[Select plug implementations](../how-to/optimize.md#select-plug-implementations).

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_MANIFEST` | Unset (the `local` profile) | Path of a JSON plug manifest (`"schema": "hyperloom.plugs/v1"`) laid over the profile. `--manifest` wins over it. An exported value changes which implementations every run in that shell uses, and a selection is recorded in the session and re-applied on resume. |
| `HYPERLOOM_MANIFEST_OVERRIDES` | Unset (no overrides) | Path of a JSON file of the same shape laid over the manifest; only `--plug` and `--plug-config` win over it. Meant for a host that must adjust a run without editing the manifest. |

A path that does not exist, a malformed file, an unknown socket or implementation, or a `${VAR}` that is unset and has no
default stops the run with exit code 2 before any session is opened or changed.
