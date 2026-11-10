# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The plug kernel: sockets in the product, the implementations that fill them, and the host that starts them."""

from hyperloom.plugs.builtin import builtin_registry
from hyperloom.plugs.errors import ManifestError, PlugError
from hyperloom.plugs.host import PlugHost, activated, active
from hyperloom.plugs.manifest import ManifestSources, ResolvedManifest, resolve_manifest
from hyperloom.plugs.registry import PlugRegistry
from hyperloom.plugs.socket import Implementation, PlugContext, Socket

__all__ = [
    "Implementation",
    "ManifestError",
    "ManifestSources",
    "PlugContext",
    "PlugError",
    "PlugHost",
    "PlugRegistry",
    "ResolvedManifest",
    "Socket",
    "activated",
    "active",
    "builtin_registry",
    "resolve_manifest",
]
