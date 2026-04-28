#!/usr/bin/env bash
# scripts/monitor.sh — Inference Optimizer cluster health snapshot.
#
# Usage:
#   bash scripts/monitor.sh                           # one-shot snapshot
#   bash scripts/monitor.sh --watch 5                 # refresh every 5s
#   bash scripts/monitor.sh --per-agent               # per-agent cursor lag
#   bash scripts/monitor.sh --per-lane                # active lease holders
#   bash scripts/monitor.sh --top-events 50           # last 50 events
#
# Env:
#   INFERENCE_OPTIMIZER_DB_PATH   — full path to conductor.db
#   SESSION_DIR                   — fallback (uses $SESSION_DIR/storage/conductor.db)
#
# Exits 0 when everything looks healthy, non-zero otherwise.
set -uo pipefail

DB="${INFERENCE_OPTIMIZER_DB_PATH:-${SESSION_DIR:?SESSION_DIR or INFERENCE_OPTIMIZER_DB_PATH must be set}/storage/conductor.db}"
WATCH_INTERVAL=""
PER_AGENT=0
PER_LANE=0
TOP_EVENTS=0
LAG_THRESHOLD=10

while [ $# -gt 0 ]; do
  case "$1" in
    --watch)         WATCH_INTERVAL="$2"; shift 2 ;;
    --per-agent)     PER_AGENT=1; shift ;;
    --per-lane)      PER_LANE=1; shift ;;
    --top-events)    TOP_EVENTS="$2"; shift 2 ;;
    --lag-threshold) LAG_THRESHOLD="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,17p' "$0" | sed 's/^# //'
      exit 0
      ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

snapshot() {
  if [ ! -f "$DB" ]; then
    echo "DB not found: $DB" >&2
    return 3
  fi
  echo "== $(date -u +%FT%TZ)  db=$DB"
  sqlite3 "$DB" <<'SQL'
.headers on
.mode column
SELECT 'events'                         AS what, COUNT(*) AS n, MAX(ts) AS latest FROM events;
SELECT 'in-flight tasks'                AS what, COUNT(*) AS n
       FROM tasks WHERE state IN ('queued','running');
SELECT 'active leases'                  AS what, COUNT(*) AS n
       FROM leases WHERE expires_at > datetime('now');
SELECT 'cursors lag (events behind)'    AS what,
       MAX(seq) - MIN(last_processed_seq) AS lag
       FROM events, cursors;
SQL
  if [ "$PER_AGENT" = "1" ]; then
    echo "-- per-agent cursor lag --"
    sqlite3 "$DB" <<'SQL'
.headers on
.mode column
SELECT c.agent       AS agent,
       c.last_processed_seq AS cursor_seq,
       (SELECT MAX(seq) FROM events) - c.last_processed_seq AS lag
       FROM cursors AS c
       ORDER BY lag DESC;
SQL
  fi
  if [ "$PER_LANE" = "1" ]; then
    echo "-- active leases --"
    sqlite3 "$DB" <<'SQL'
.headers on
.mode column
SELECT lane, holder_id, action, expires_at FROM leases
 WHERE expires_at > datetime('now')
 ORDER BY expires_at DESC;
SQL
  fi
  if [ "$TOP_EVENTS" -gt 0 ]; then
    echo "-- last $TOP_EVENTS events --"
    sqlite3 "$DB" <<SQL
.headers on
.mode column
SELECT seq, from_agent, to_agent, topic, ts FROM events
 ORDER BY seq DESC LIMIT $TOP_EVENTS;
SQL
  fi

  # Health gate (used by exit code).
  health=$(sqlite3 "$DB" <<'SQL'
SELECT
  CASE
    WHEN ((SELECT MAX(seq) FROM events) - (SELECT COALESCE(MIN(last_processed_seq), 0) FROM cursors)) > $((LAG_THRESHOLD)) THEN 'lag'
    WHEN EXISTS (SELECT 1 FROM tasks WHERE state='running' AND updated_at < datetime('now','-30 minute')) THEN 'zombie'
    ELSE 'ok'
  END;
SQL
)
  if [ "$health" != "ok" ]; then
    echo "STATUS=$health" >&2
    return 1
  fi
  return 0
}

if [ -z "$WATCH_INTERVAL" ]; then
  snapshot
  exit $?
else
  while true; do
    snapshot || true
    sleep "$WATCH_INTERVAL"
  done
fi
