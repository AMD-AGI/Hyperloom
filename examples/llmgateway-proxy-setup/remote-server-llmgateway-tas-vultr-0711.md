# Claude Code on Remote Servers (via AMD LLM Gateway)

Remote servers cannot reach `llm-api.amd.com` directly. This guide sets up a local proxy on your machine that tunnels through SSH to the remote server.

**Architecture:** Remote Claude Code → SSH reverse tunnel → Local proxy → AMD LLM Gateway

---

## Prerequisites

- Claude Code access via AMD gateway (subscription key)
- SSH access to the remote server (`root@<REMOTE_SERVER_IP>`)
- Python 3 on your local machine (WSL or Mac/Linux)

---

## Step 1 — Configure Claude Code on your local machine

Your local machine can reach the gateway directly, so point Claude Code straight at it — no proxy needed for local use. Add this to your local `~/.bashrc`:

```bash
cat >> ~/.bashrc << 'EOF'

# AMD Personal LLM Gateway - Claude CLI
export LLM_GATEWAY_KEY="your_llm_gateway_key"
export ANTHROPIC_BASE_URL="https://llm-api.amd.com/Anthropic"
export ANTHROPIC_API_KEY="dummy"
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: $LLM_GATEWAY_KEY"
EOF
source ~/.bashrc
```

Then confirm Claude works locally (if `claude` isn't installed yet, install it the same way as on the remote — see Step 4):

```bash
claude
```

This uses the **direct** gateway URL, which only works from the local machine — the remote server can't reach it, which is what the proxy and tunnel in the next steps are for. Replace `your_llm_gateway_key` with your AMD LLM gateway subscription key, and keep `~/.bashrc` private once it holds your real key.

---

## Step 2 — Local proxy script

Save this as `~/llm-proxy.py` on your **local machine**:

```python
#!/usr/bin/env python3
"""
Local reverse proxy: forwards requests to AMD LLM gateway.
Usage: python3 llm-proxy.py [port]
"""
import sys
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

TARGET = "https://llm-api.amd.com/Anthropic"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080


class ProxyHandler(BaseHTTPRequestHandler):
    def do_request(self):
        url = TARGET + self.path
        body = None
        length = self.headers.get("Content-Length")
        if length:
            body = self.rfile.read(int(length))

        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        headers["Host"] = "llm-api.amd.com"

        req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding",):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding",):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(e.read())

    def do_GET(self): self.do_request()
    def do_POST(self): self.do_request()
    def do_PUT(self): self.do_request()
    def do_DELETE(self): self.do_request()
    def do_PATCH(self): self.do_request()

    def log_message(self, fmt, *args):
        print(f"[proxy] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), ProxyHandler)
    print(f"[proxy] Listening on localhost:{PORT} → {TARGET}")
    server.serve_forever()
```

---

## Step 3 — SSH config (local machine)

Assume your remote server's IP is `<REMOTE_SERVER_IP>` and you log in as `root` (change these to your own). There is no jumpbox, so connect directly. Add this host entry to `~/.ssh/config`, including the reverse tunnel line:

```
Host llm-remote
    HostName <REMOTE_SERVER_IP>               # <-- your remote server IP
    User root                           # <-- your remote login user
    StrictHostKeyChecking no
    RemoteForward 8080 localhost:8080   # <-- tunnels remote:8080 → local:8080
```

`RemoteForward 8080 localhost:8080` makes the remote's `localhost:8080` forward back to the proxy running on your local machine. Add a `RemoteForward` line to every remote host you want to use Claude Code on.

**If you'll run Claude Code inside a container on the remote server**, start that container with `--network host` so it shares the host's network and reaches the tunnel at `localhost:8080` directly — exactly like the host, with no extra config. For example (note the added `--network host`):

```bash
docker run -d \
  --name hyperloom-local \
  --network host \
  --shm-size 64g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v /mnt:/mnt \
  docker.io/primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix \
  tail -f /dev/null
```

---

## Step 4 — Install Claude Code on the remote (host or container)

Run this **on the remote host** (after `ssh llm-remote`), and/or **inside the container** (after `docker exec -it hyperloom-local bash`) if you want Claude Code there too. The install is identical in both:

```bash
# Install nvm (no root needed)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc

# Install Node.js and Claude Code
nvm install 22
npm install -g @anthropic-ai/claude-code
```

---

## Step 5 — Configure env vars on the remote (host or container)

Add the same env **on the remote host** and/or **inside the container** — with `--network host` the container uses `localhost:8080` just like the host, so the config is identical:

```bash
cat >> ~/.bashrc << 'EOF'

# Claude Code via local proxy tunnel
export ANTHROPIC_API_KEY=dummy
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: your_llm_gateway_key"
export ANTHROPIC_MODEL=Claude-Opus-4.8
EOF
source ~/.bashrc
```

Replace `your_llm_gateway_key` with your AMD LLM gateway subscription key. Keep this file private once it holds your real key — do not commit or share it.

---

## Daily usage

Every time you want to use Claude Code on the remote:

1. **Local terminal** — start the proxy and keep it running:
   ```bash
   python3 ~/llm-proxy.py 8080
   ```

2. **New terminal (on the same local machine)** — SSH in using the `llm-remote` alias. This is what opens the reverse tunnel, so you must use the alias (not `ssh root@<REMOTE_SERVER_IP>`) and run it from the machine where the proxy is running. Keep this session open:
   ```bash
   ssh llm-remote
   ```

3. **On the remote** — run Claude Code, either on the host or inside the container:
   ```bash
   claude
   # or, inside the container:
   docker exec -it hyperloom-local bash
   claude
   ```

The tunnel only exists while an `ssh llm-remote` session is alive. If you'd rather open the remote shell separately, keep one dedicated tunnel running on the local machine instead (run it once; it stays in the background):

```bash
ssh -N -f -o ServerAliveInterval=30 llm-remote
```

With that running, any Claude session on the remote (host or container) will reach `localhost:8080`.

---

## Troubleshooting

**`Unable to connect to API (ConnectionRefused)` on the remote** — the remote's `localhost:8080` has nothing behind it: the reverse tunnel isn't up in the session you're using (or the local proxy died). The most common cause is connecting without the tunnel — e.g. `ssh root@<REMOTE_SERVER_IP>` instead of `ssh llm-remote`, or from a machine that isn't running the proxy. Fix it by connecting with `ssh llm-remote` from the local machine, or by starting a dedicated tunnel on the local machine that stays up regardless of how you open the remote shell:

```bash
ssh -N -f -o ServerAliveInterval=30 llm-remote
```

To confirm the tunnel is live, run `ss -ltn | grep 8080` on the remote — you should see a listener on `127.0.0.1:8080`. Also make sure `python3 ~/llm-proxy.py 8080` is still running on the local machine.

**`Permission denied` when installing packages** — use `nvm` (Step 4) instead of `apt`, which requires root.

**Reverse tunnel port already in use on the remote** — a stale SSH session may be holding port 8080. Close old sessions, or run everything on a different port (`python3 ~/llm-proxy.py 8081`, `RemoteForward 8081 localhost:8081`, and `ANTHROPIC_BASE_URL=http://localhost:8081`).

**Container can't reach the API** — the container must be started with `--network host` (Step 3) so it shares the host's `localhost:8080`. If so, this is the same as the first entry above: make sure the reverse tunnel and local proxy are up (`ss -ltn | grep 8080` on the host should show `127.0.0.1:8080`). A bridge-network container cannot see the host's loopback and will fail here.
