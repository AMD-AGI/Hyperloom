"""Console entry point.

``python -m robustness_server`` and the ``robustness-server`` script
defined in ``pyproject.toml`` both land here. Production deployments
typically use ``uvicorn`` directly with this module's ``app`` factory.
"""

from __future__ import annotations

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "robustness_server.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
