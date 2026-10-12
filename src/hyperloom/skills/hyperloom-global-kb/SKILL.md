---
name: hyperloom-global-kb
description: Deploys or restarts the shared global Experience KB that Hyperloom workspaces push their Experiences to and pull others' from, validates authenticated health, and tells each workspace which .env keys to set. Use when asked to deploy, start, restart, or check a global or team Experience KB.
---

# Deploy a global Experience KB

A global Experience KB is the same Experience service every workspace runs
locally, deployed once on a host that teammates' workspaces can reach. Runs
never read or write it directly: each workspace's local service pushes its own
Experiences to it and pulls it back. It stores every schema its contributors
send.

Run this skill from a Hyperloom install on the host that will serve the global
KB: a `pip install --target .` workspace or a source checkout.

## Resolve inputs

- The URL clients will use. Default to `http://$(hostname -f):8787`; ask only
  when that host name or port is not reachable from the teammates' machines.
- Keep the defaults below unless the user asks for other locations. The state
  directory holds the data and the service token; it survives restarts and
  upgrades.

```bash
export REPO_ROOT="$(pwd -P)"
if [ -d "$REPO_ROOT/hyperloom_kb" ]; then KB_PYTHONPATH="$REPO_ROOT"; else KB_PYTHONPATH="$REPO_ROOT/src"; fi
STATE_DIR="${STATE_DIR:-$HOME/.local/share/hyperloom-global-kb}"
PORT="${PORT:-8787}"
GLOBAL_KB_URL="${GLOBAL_KB_URL:-http://$(hostname -f):$PORT}"
PYTHONPATH="$KB_PYTHONPATH" python3 -c "import hyperloom_kb, yaml" || {
  echo "hyperloom_kb or PyYAML is not importable from $KB_PYTHONPATH" >&2
  return 1 2>/dev/null || exit 1
}
```

The global KB needs no LLM gateway: reads happen in each workspace's local
service, so its log reporting that reads are unavailable is expected.

## Service token

Generate the token once and keep it in a `0600` file. Never print it, and do
not ask the user to paste it into chat.

```bash
umask 077
mkdir -p "$STATE_DIR"
if ! grep -q '^HYPERLOOM_KB_TOKEN=' "$STATE_DIR/service.env" 2>/dev/null; then
  python3 -c 'import secrets; print("HYPERLOOM_KB_TOKEN=" + secrets.token_urlsafe(32))' >"$STATE_DIR/service.env"
fi
chmod 600 "$STATE_DIR/service.env"
```

## Start or restart

The service gets a clean environment, so no LLM key or proxy setting of this
shell leaks into it. Restarting keeps every stored Experience.

```bash
PID_FILE="$STATE_DIR/service.pid"
LOG_FILE="$STATE_DIR/service.log"
if [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  kill "$(cat "$PID_FILE")"
  for _ in $(seq 1 10); do kill -0 "$(cat "$PID_FILE")" 2>/dev/null || break; sleep 1; done
fi

set -a
. "$STATE_DIR/service.env"
set +a
nohup env -i PATH="$PATH" HOME="$HOME" PYTHONPATH="$KB_PYTHONPATH" HYPERLOOM_KB_TOKEN="$HYPERLOOM_KB_TOKEN" \
  python3 -m hyperloom_kb --host 0.0.0.0 --port "$PORT" --home "$STATE_DIR/data" \
  >>"$LOG_FILE" 2>&1 </dev/null &
echo "$!" >"$PID_FILE"
```

## Validate

```bash
for _ in $(seq 1 30); do
  curl --noproxy '*' -fs -H "Authorization: Bearer $HYPERLOOM_KB_TOKEN" "$GLOBAL_KB_URL/health" >/dev/null && break
  sleep 1
done
curl --noproxy '*' -fsS -H "Authorization: Bearer $HYPERLOOM_KB_TOKEN" "$GLOBAL_KB_URL/health"
test "$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' "$GLOBAL_KB_URL/health")" = 401
```

The first request must report `"status":"ok"`, with `experience_count` and the
per-schema counts in `schemas`; the second proves an unauthenticated request is
rejected. On failure, show the last relevant lines of `$LOG_FILE` without any
credential.

## Report

Report the URL, the PID and log files, and that the token is the
`HYPERLOOM_KB_TOKEN` value in `$STATE_DIR/service.env`, to be handed to
teammates through the user's usual secret channel. Each workspace that joins
adds these keys to its own `.env` by editing the file directly:

```bash
HYPERLOOM_GLOBAL_KB_URL=<the URL above>
HYPERLOOM_GLOBAL_KB_TOKEN=<HYPERLOOM_KB_TOKEN from service.env>
# Optional: push after every run instead of only on request.
HYPERLOOM_KB_AUTO_PUSH=1
```

The workspace's next optimize launch, or
`python -m hyperloom.inference_optimizer.experience_kb_service ensure`, restarts
its local service with them. Push and pull use the service as it runs and never
restart it, so run `ensure` before the first push or pull after adding the keys.

The service speaks plain HTTP. When teammates reach it across an untrusted
network, put a TLS-terminating proxy in front of it and give them the proxy's
`https://` URL instead.
