-- Hyperloom robustness-server initial schema.
--
-- All tables live under the configured schema (default
-- `hyperloom_robustness`). This file is applied verbatim by the
-- migration runner; statements are idempotent so reapplying is safe.
--
-- Per repo conventions: only base tables and indexes; no foreign keys,
-- views, functions, or triggers.

CREATE SCHEMA IF NOT EXISTS hyperloom_robustness;

SET search_path TO hyperloom_robustness;

-- One row per Claw session as observed via NATS events. `t_end` is
-- nullable for long-running sessions; the reconciler / event consumer
-- updates it when a terminal event arrives.
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT        NOT NULL,
    user_id             TEXT,
    plugin_id           TEXT,
    t_start             TIMESTAMPTZ NOT NULL,
    t_end               TIMESTAMPTZ,
    final_state         TEXT,
    last_event_at       TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (session_id)
);

CREATE INDEX IF NOT EXISTS sessions_t_start_idx
    ON sessions (t_start DESC);
CREATE INDEX IF NOT EXISTS sessions_user_idx
    ON sessions (user_id, t_start DESC);

-- Session ↔ pod time-window mapping. One pod (brain or hands sandbox)
-- can be assigned to a session for a [t_start, t_end] window. Brain
-- pods may have many entries (multi-session reuse); hands pods
-- typically have one.
CREATE TABLE IF NOT EXISTS session_pod_assignment (
    assignment_id       BIGSERIAL   NOT NULL,
    session_id          TEXT        NOT NULL,
    pod_namespace       TEXT        NOT NULL,
    pod_name            TEXT        NOT NULL,
    pod_uid             TEXT,
    role                TEXT        NOT NULL,         -- brain | hands_gpu | hands_cpu
    source              TEXT        NOT NULL,         -- nats_kv | nats_event | workload_reconcile
    t_start             TIMESTAMPTZ NOT NULL,
    t_end               TIMESTAMPTZ,
    last_seen_at        TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (assignment_id)
);

CREATE INDEX IF NOT EXISTS session_pod_assignment_session_idx
    ON session_pod_assignment (session_id, t_start DESC);
CREATE INDEX IF NOT EXISTS session_pod_assignment_pod_idx
    ON session_pod_assignment (pod_namespace, pod_name, t_start DESC);
CREATE INDEX IF NOT EXISTS session_pod_assignment_open_idx
    ON session_pod_assignment (session_id)
    WHERE t_end IS NULL;

-- Raw NATS events kept for audit / late mapping. Body is the JSON
-- envelope as published by Claw NatsEmitter; a thin set of canonical
-- fields is denormalised into columns for efficient filtering.
CREATE TABLE IF NOT EXISTS session_events (
    event_id            BIGSERIAL   NOT NULL,
    session_id          TEXT        NOT NULL,
    event_type          TEXT        NOT NULL,
    subject             TEXT        NOT NULL,
    occurred_at         TIMESTAMPTZ NOT NULL,
    received_at         TIMESTAMPTZ NOT NULL,
    pod_name            TEXT,
    plugin_id           TEXT,
    body                JSONB       NOT NULL,
    PRIMARY KEY (event_id)
);

CREATE INDEX IF NOT EXISTS session_events_session_time_idx
    ON session_events (session_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS session_events_type_idx
    ON session_events (event_type, occurred_at DESC);
