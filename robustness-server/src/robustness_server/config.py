"""Runtime configuration.

Unlike ``robustness-agent`` which has to discover its peer services at
runtime (it ships inside Claw v2 hands sandboxes that disallow extra env
vars), ``robustness-server`` is its own deployment with explicit
config — environment variables prefixed ``ROBUSTNESS_SERVER_`` are the
canonical source.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server-wide settings.

    All fields are required to have a value at startup. Defaults are
    only set for things that are safe to default in development. Helm
    deployments must override ``database_url``, ``nats_servers``, and
    ``robust_api_url`` explicitly.
    """

    model_config = SettingsConfigDict(
        env_prefix="ROBUSTNESS_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field("0.0.0.0", description="Bind host")
    port: int = Field(8080, description="Bind port")
    log_level: str = Field("INFO", description="Log level")

    database_url: str = Field(
        "postgresql://robustness:robustness@localhost:5432/robustness",
        description="PostgreSQL DSN. Schema lives under `hyperloom_robustness`.",
    )
    database_schema: str = Field(
        "hyperloom_robustness",
        description="Schema namespace; auto-created on startup.",
    )
    database_pool_min: int = Field(2, ge=0)
    database_pool_max: int = Field(10, ge=1)
    apply_migrations_on_start: bool = Field(
        True,
        description="Apply embedded SQL migrations during startup.",
    )

    nats_servers: list[str] = Field(
        default_factory=lambda: ["nats://nats:4222"],
        description="NATS JetStream cluster endpoints.",
    )
    nats_stream: str = Field(
        "PRIMUS_CLAW_EVENTS",
        description="JetStream stream name (created by Claw, we only consume).",
    )
    nats_durable_name: str = Field(
        "hyperloom-robustness-server",
        description="Durable consumer name for replay-safe restarts.",
    )
    nats_subject_filter: str = Field(
        "events.>",
        description="Subject filter for the durable consumer.",
    )
    nats_kv_bucket: str = Field(
        "BRAIN_REGISTRY",
        description="KV bucket holding session→brain pod assignments.",
    )

    robust_api_url: str = Field(
        "http://robust-api.core42.svc.cluster.local",
        description="Base URL of Primus-Robust robust-api (used to query /api/v1/pod-metrics/...).",
    )
    robust_api_timeout_seconds: float = Field(10.0, gt=0)

    workload_reconcile_interval_seconds: float = Field(
        60.0,
        gt=0,
        description=(
            "Tick rate of the workload reconciler that polls Robust's "
            "/api/v1/workloads to recover any (session, pod) pairs that "
            "were not delivered through NATS events."
        ),
    )

    # Claw labels we use to identify Claw-managed sandboxes when
    # walking Robust's workload list. Kept overridable to support
    # label-name evolution without a code change.
    claw_session_label: str = Field("primus-claw/session-id")
    claw_component_label: str = Field("primus-claw/component")
    claw_role_label: str = Field("primus-claw/role")

    # Toggle individual background collaborators. Defaulting them on
    # matches the production deployment shape; tests typically flip
    # them off so the HTTP surface boots without external brokers.
    enable_nats_consumer: bool = Field(True)
    enable_kv_watcher: bool = Field(True)
    enable_workload_reconciler: bool = Field(True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""

    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_for_test() -> None:
    """Clear the cache; tests use this between fixtures."""

    global _settings
    _settings = None
