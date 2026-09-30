"""Read-only client for the Hyperloom Pulse API, standard library only.

This machine has no route to the Pulse host, so every call goes through a
local SOCKS5 tunnel (remote DNS, no auth). Only GET is ever issued; the API
has no write surface and this client does not invent one.

Archives are addressed by ``source_session_id`` (``<model>_<TS>_<hash>``).
The claw UUID that the 2026-08 fetch used as the archive key now answers
``404 no_archive`` for archives that are present under their name.

Failure taxonomy, carried over from the 2026-08 fetch:

* a transport failure, 408, 429 or 5xx is *deferred*: retried with backoff,
  and if it still fails it is logged as deferred, never as a verdict;
* ``404 no_archive`` and a 404 on a single file are verdicts and are never
  retried;
* 401/403 stops the run, because every later call would fail the same way;
* a run of consecutive transport failures pauses every worker until a probe
  gets through, instead of spending the target list on timeouts.

Subcommands::

    python pulse.py facets --last-days 180
    python pulse.py enum   --last-days 180 --out ../00_enum
    python pulse.py census --targets names.txt --out ../01_census/ls.jsonl.gz
    python pulse.py fetch  --plan plan.jsonl --out ../02_bundles
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import http.client
import json
import os
import random
import shutil
import socket
import ssl
import sys
import threading
import time
import urllib.parse
from pathlib import Path

DEFAULT_BASE = "https://global.primus-safe.amd.com/hyperloom/api"
PAGE = 200


class Deferred(Exception):
    """A call that did not get an answer. Retry later; it says nothing about the archive."""


class AuthError(Exception):
    """The key was refused. Nothing after this can succeed."""


class Aborted(Exception):
    """The service stayed unreachable past the wait budget."""


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise OSError("SOCKS5 proxy closed the connection mid-handshake")
        buf += chunk
    return buf


def _socks5_connect(sock: socket.socket, host: str, port: int) -> None:
    sock.sendall(b"\x05\x01\x00")
    if _recv_exact(sock, 2) != b"\x05\x00":
        raise OSError("SOCKS5 proxy refused the no-auth method")
    name = host.encode("idna")
    sock.sendall(b"\x05\x01\x00\x03" + bytes([len(name)]) + name + port.to_bytes(2, "big"))
    head = _recv_exact(sock, 4)
    if head[1] != 0:
        raise OSError(f"SOCKS5 CONNECT to {host}:{port} failed, reply code {head[1]}")
    atyp = head[3]
    if atyp == 1:
        _recv_exact(sock, 4)
    elif atyp == 4:
        _recv_exact(sock, 16)
    elif atyp == 3:
        _recv_exact(sock, _recv_exact(sock, 1)[0])
    _recv_exact(sock, 2)


class _SocksHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, socks: tuple[str, int] | None, timeout: float):
        super().__init__(host, 443, timeout=timeout, context=ssl.create_default_context())
        self._socks = socks

    def connect(self) -> None:
        if self._socks is None:
            super().connect()
            return
        sock = socket.create_connection(self._socks, timeout=self.timeout)
        try:
            _socks5_connect(sock, self.host, self.port)
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


class Outage:
    """Pause every worker after a run of transport failures, resume when a probe succeeds."""

    def __init__(self, probe, threshold: int = 10, service_wait: float = 1200.0):
        self._probe = probe
        self._threshold = threshold
        self._service_wait = service_wait
        self._lock = threading.Lock()
        self._consecutive = 0
        self._up = threading.Event()
        self._up.set()
        self.aborted = False
        self.pauses = 0

    def ok(self) -> None:
        with self._lock:
            self._consecutive = 0

    def failed(self, reason: str = "") -> None:
        with self._lock:
            self._consecutive += 1
            self.failures = getattr(self, "failures", 0) + 1
            if self.failures <= 20 or self.failures % 100 == 0:
                sys.stderr.write(f"[fail #{self.failures}] {reason[:200]}\n")
            trip = self._consecutive >= self._threshold and self._up.is_set()
            if trip:
                self._up.clear()
                self.pauses += 1
        if trip:
            threading.Thread(target=self._watch, daemon=True).start()

    def wait(self) -> None:
        self._up.wait()
        if self.aborted:
            raise Aborted("Pulse API stayed unreachable past the service wait")

    def _watch(self) -> None:
        start = time.time()
        sys.stderr.write(f"[outage] {self._threshold} consecutive failures; pausing workers\n")
        while time.time() - start < self._service_wait:
            time.sleep(30)
            try:
                self._probe()
            except Exception:  # noqa: BLE001 - any failed probe only means the service is still down
                continue
            with self._lock:
                self._consecutive = 0
            sys.stderr.write(f"[outage] service back after {time.time() - start:.0f}s\n")
            self._up.set()
            return
        self.aborted = True
        self._up.set()


class Pulse:
    def __init__(
        self, base: str | None = None, key: str | None = None, socks: str | None = None, timeout: float = 120.0
    ):
        base = base or os.environ.get("PULSE_API_BASE") or DEFAULT_BASE
        self.key = key or os.environ.get("PULSE_API_KEY") or ""
        if not self.key.startswith("ak-"):
            raise SystemExit("PULSE_API_KEY is not set (expected an ak- key)")
        parsed = urllib.parse.urlsplit(base)
        self.host = parsed.hostname
        self.prefix = parsed.path.rstrip("/")
        socks = os.environ.get("PULSE_SOCKS", "127.0.0.1:1080") if socks is None else socks
        if socks:
            host, _, port = socks.rpartition(":")
            self.socks = (host, int(port))
        else:
            self.socks = None
        self.timeout = timeout
        self._local = threading.local()
        self.outage = Outage(self._probe, threshold=10)

    def _probe(self) -> None:
        """One call that bypasses the outage gate; going through call() would wait on itself."""
        status, _, _ = self._once("/v1/session-diagnostics/facets?last_days=1", None)
        if status != 200:
            raise Deferred(f"probe HTTP {status}")

    def _conn(self) -> _SocksHTTPSConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = _SocksHTTPSConnection(self.host, self.socks, self.timeout)
            self._local.conn = conn
        return conn

    def _drop(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        self._local.conn = None

    def _once(self, path: str, stream_to: Path | None) -> tuple[int, bytes, str]:
        headers = {"Authorization": f"Bearer {self.key}", "User-Agent": "hl-v5-pulse/1"}
        if stream_to is None:
            headers["Accept-Encoding"] = "gzip"
        conn = self._conn()
        try:
            conn.request("GET", self.prefix + path, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            if stream_to is not None and status == 200:
                digest = hashlib.sha256()
                with open(stream_to, "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        digest.update(chunk)
                        fh.write(chunk)
                return status, b"", digest.hexdigest()
            body = resp.read()
            if resp.getheader("Content-Encoding", "") == "gzip":
                body = gzip.decompress(body)
            if resp.getheader("Connection", "").lower() == "close":
                self._drop()
            return status, body, ""
        except (OSError, http.client.HTTPException) as exc:
            self._drop()
            raise Deferred(f"transport: {type(exc).__name__}: {exc}") from None

    def call(self, path: str, stream_to: Path | None = None, attempts: int = 4) -> tuple[int, bytes, str]:
        last = ""
        for attempt in range(attempts):
            self.outage.wait()
            try:
                status, body, digest = self._once(path, stream_to)
            except Deferred as exc:
                last = str(exc)
                self.outage.failed(f"{last} on {path[:120]}")
                time.sleep(min(30.0, 2**attempt) + random.random())
                continue
            if status in (401, 403):
                raise AuthError(f"HTTP {status} on {path}: {body[:200]!r}")
            if status in (408, 429) or status >= 500:
                last = f"HTTP {status}"
                self.outage.failed(f"{last} {body[:120]!r} on {path[:120]}")
                time.sleep(min(30.0, 2**attempt) + random.random())
                continue
            self.outage.ok()
            return status, body, digest
        raise Deferred(last)

    def _json(self, path: str) -> tuple[int, dict]:
        status, body, _ = self.call(path)
        try:
            return status, json.loads(body)
        except ValueError:
            raise Deferred(f"HTTP {status} with a body that is not JSON") from None

    def facets(self, **params) -> dict:
        status, data = self._json("/v1/session-diagnostics/facets?" + urllib.parse.urlencode(params))
        if status != 200:
            raise ValueError(f"facets HTTP {status}: {data}")
        return data.get("data") or {}

    def search(self, offset: int = 0, limit: int = PAGE, **params) -> tuple[list, int]:
        q = dict(params, offset=offset, limit=limit)
        status, data = self._json("/v1/session-diagnostics/search?" + urllib.parse.urlencode(q))
        if status != 200:
            raise ValueError(f"search HTTP {status}: {data}")
        return data.get("results") or [], int((data.get("pagination") or {}).get("total") or 0)

    def resolve(self, ref: str) -> dict:
        status, data = self._json("/v1/session-diagnostics/resolve?" + urllib.parse.urlencode({"id": ref}))
        if status == 404:
            return {"found": False, "input": ref}
        if status != 200:
            raise ValueError(f"resolve HTTP {status}: {data}")
        return data.get("data") or {}

    def ls(self, name: str) -> dict:
        status, data = self._json(f"/v1/session-archives/{urllib.parse.quote(name, safe='')}/files")
        if status == 200:
            return data
        if status == 404:
            data.setdefault("archive_status", "http404")
            return data
        raise ValueError(f"ls HTTP {status} for {name}: {data}")

    def get(self, name: str, path: str, dest: Path) -> tuple[str, str]:
        """Download one file to ``dest`` through ``dest.part``. Returns (verdict, sha256)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        url = f"/v1/session-archives/{urllib.parse.quote(name, safe='')}/files/" + urllib.parse.quote(path, safe="/")
        status, _, digest = self.call(url, stream_to=part)
        if status == 404:
            part.unlink(missing_ok=True)
            return "file_404", ""
        if status != 200:
            part.unlink(missing_ok=True)
            raise ValueError(f"get HTTP {status} for {name}:{path}")
        os.replace(part, dest)
        return "ok", digest


class RowLog:
    """Append-only JSONL, one line per call, flushed as it goes so an abort loses nothing."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------- enumerate

# The search filter accepts these; the facet value `unknown` is a null run_result and
# can only be reached through a cell that does not filter on run_result at all.
RUN_RESULTS = ("success", "failed", "abandoned", "running", "awaiting_terminal")


def row_key(row: dict) -> tuple:
    """Index identity. A claw UUID can have one row per cluster, so it is not a key alone."""
    return (row.get("cluster"), row.get("claw_session_id"))


def _cell_rows(api: Pulse, params: dict, passes: int) -> tuple[dict, int, list]:
    """Page one filter cell `passes` times and union by (cluster, claw_session_id)."""
    union: dict[tuple, dict] = {}
    totals = []
    for _ in range(passes):
        _, total = api.search(offset=0, limit=1, **params)
        totals.append(total)
        offset = 0
        while offset < total:
            rows, _ = api.search(offset=offset, limit=PAGE, **params)
            for row in rows:
                union[row_key(row)] = row
            if not rows:
                break
            offset += PAGE
    return union, max(totals), totals


def cmd_enum(args) -> int:
    api = Pulse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    facets = api.facets(last_days=args.last_days)
    facet_total = sum(x["count"] for x in facets.get("run_result", []))
    clusters = args.clusters.split(",")
    # Three layers of cells, all unioned by claw_session_id: fine cells keep each page
    # set small (pagination drift is worst in large cells), per-cluster cells reach the
    # null run_result the fine cells cannot filter on, and global cells catch any
    # cluster missing from --clusters.
    layers = set(args.layers.split(","))
    cells = []
    for forge in ("true", "false"):
        if "global" in layers:
            cells.append(dict(last_days=args.last_days, forge=forge))
        for c in clusters:
            if "cluster" in layers:
                cells.append(dict(last_days=args.last_days, cluster=c, forge=forge))
            if "fine" in layers:
                for r in RUN_RESULTS:
                    cells.append(dict(last_days=args.last_days, cluster=c, run_result=r, forge=forge))
    union: dict[tuple, dict] = {}
    report = []

    def work(params):
        rows, total, totals = _cell_rows(api, params, args.passes)
        return params, rows, total, totals

    with cf.ThreadPoolExecutor(args.jobs) as pool:
        for params, rows, total, totals in pool.map(work, cells):
            union.update(rows)
            cell = {k: params.get(k, "*") for k in ("cluster", "run_result", "forge")}
            report.append(
                dict(cell=cell, total=total, pass_totals=totals, unique=len(rows), short=max(0, total - len(rows)))
            )
            print(
                f"{cell['cluster']:>12} {cell['run_result']:>17} forge={cell['forge']:<5} "
                f"total={total:>6} unique={len(rows):>6}",
                flush=True,
            )

    top = "*" if "global" in layers else None
    cell_sum = sum(
        r["total"] for r in report if r["cell"]["run_result"] == "*" and (r["cell"]["cluster"] == "*") == (top == "*")
    )
    with open(out / "index_rows.jsonl", "w", encoding="utf-8") as fh:
        for row in union.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        last_days=args.last_days,
        passes=args.passes,
        clusters=clusters,
        facets_total=facet_total,
        facets_run_result={x["value"]: x["count"] for x in facets.get("run_result", [])},
        cell_sum=cell_sum,
        n_unique=len(union),
        cells_short=[r for r in report if r["short"]],
        cells=report,
    )
    (out / "enumeration.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"facets total {facet_total}, global-cell sum {cell_sum}, unique rows {len(union)}")
    if cell_sum != facet_total:
        print(f"WARNING: the top-layer cells add up to {cell_sum}, facets says {facet_total}", file=sys.stderr)
    if len(union) < facet_total:
        print(f"WARNING: {facet_total - len(union)} rows short of the facets total after the union", file=sys.stderr)
    return 0


# ------------------------------------------------------------------- census


def _done_names(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        try:
            for line in fh:
                row = json.loads(line)
                if row.get("status") != "deferred":
                    done.add(row["name"])
        except (EOFError, OSError, ValueError):
            pass
    return done


def cmd_census(args) -> int:
    api = Pulse()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in open(args.targets, encoding="utf-8") if n.strip()]
    done = _done_names(out)
    todo = [n for n in names if n not in done]
    print(f"{len(names)} targets, {len(done)} already listed, {len(todo)} to go", flush=True)
    rows = RowLog(out.with_suffix(".rows.jsonl"))

    def work(name):
        t0 = time.time()
        try:
            data = api.ls(name)
        except Deferred as exc:
            return dict(name=name, status="deferred", error=str(exc), secs=round(time.time() - t0, 2))
        files = [[f["path"], f.get("bytes"), f.get("sha256")] for f in data.get("files") or []]
        return dict(
            name=name,
            status=data.get("archive_status"),
            claw=data.get("claw_session_id"),
            cluster=data.get("cluster_name"),
            s3_prefix=data.get("s3_prefix"),
            file_count=data.get("file_count"),
            total_bytes=data.get("total_bytes"),
            detail=data.get("detail"),
            files=files,
            secs=round(time.time() - t0, 2),
        )

    counts: dict[str, int] = {}
    t_start = time.time()
    with gzip.open(out, "at", encoding="utf-8") as sink, cf.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(work, n) for n in todo]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                row = fut.result()
            except (AuthError, Aborted) as exc:
                print(f"stopping: {exc}", file=sys.stderr)
                for f in futures:
                    f.cancel()
                break
            if row["status"] != "deferred":
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.write({k: v for k, v in row.items() if k != "files"})
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if i % 500 == 0 or i == len(todo):
                rate = i / max(1e-9, time.time() - t_start)
                print(f"[{i}/{len(todo)}] {counts} {rate:.1f}/s pauses={api.outage.pauses}", flush=True)
    rows.close()
    print("done:", counts)
    return 1 if counts.get("deferred") else 0


def _ls_row(api: Pulse, target: str, key: str, resolved: dict | None, t0: float) -> dict:
    data = api.ls(key)
    files = [[f["path"], f.get("bytes"), f.get("sha256")] for f in data.get("files") or []]
    return dict(
        name=target,
        archive_key=key,
        status=data.get("archive_status"),
        claw=data.get("claw_session_id"),
        cluster=data.get("cluster_name"),
        s3_prefix=data.get("s3_prefix"),
        file_count=data.get("file_count"),
        total_bytes=data.get("total_bytes"),
        detail=data.get("detail"),
        resolved=resolved,
        files=files,
        secs=round(time.time() - t0, 2),
    )


def cmd_census_retry(args) -> int:
    """Second pass over names that answered no_archive, following the CLI's addressing rule.

    An archive is keyed on either the source session id or the claw UUID depending on
    which collector wrote it. The CLI resolves any id to the canonical set, asks for the
    manifest at the source session id, and retries once at the claw id. Known claw ids
    from the index and the 2026-08 inventory are tried as well, since one name can map
    to several claw ids across clusters.
    """
    api = Pulse()
    first = Path(args.census)
    out = Path(args.out)
    misses = {}
    with gzip.open(first, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["status"] in ("no_archive", "http404"):
                misses[row["name"]] = row
            else:
                misses.pop(row["name"], None)
    aliases: dict[str, set] = {}
    for line in open(args.index, encoding="utf-8"):
        r = json.loads(line)
        if r.get("session_id"):
            aliases.setdefault(r["session_id"], set()).add(r["claw_session_id"])
    if args.inventory:
        import csv

        for r in csv.DictReader(open(args.inventory, encoding="utf-8-sig")):
            if r.get("session_id"):
                aliases.setdefault(r["session_id"], set()).add(r["session"])
    done = _done_names(out)
    todo = [n for n in misses if n not in done]
    print(f"{len(misses)} names answered no_archive; {len(done)} already retried; {len(todo)} to go", flush=True)
    rows = RowLog(out.with_suffix(".rows.jsonl"))

    def work(name):
        t0 = time.time()
        try:
            res = api.resolve(name)
            keys = []
            for k in (res.get("source_session_id"), res.get("claw_session_id"), *sorted(aliases.get(name, ()))):
                if k and k != name and k not in keys:
                    keys.append(k)
            last = None
            for key in keys:
                row = _ls_row(api, name, key, res, t0)
                if row["status"] not in ("no_archive", "http404"):
                    return row
                last = row
            if last is None:
                return dict(
                    name=name,
                    archive_key=None,
                    status="no_archive",
                    resolved=res,
                    files=[],
                    detail="no alternative key",
                    secs=round(time.time() - t0, 2),
                )
            last["tried"] = keys
            return last
        except Deferred as exc:
            return dict(name=name, status="deferred", error=str(exc), secs=round(time.time() - t0, 2))

    counts: dict[str, int] = {}
    t_start = time.time()
    with gzip.open(out, "at", encoding="utf-8") as sink, cf.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(work, n) for n in todo]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                row = fut.result()
            except (AuthError, Aborted) as exc:
                print(f"stopping: {exc}", file=sys.stderr)
                for f in futures:
                    f.cancel()
                break
            if row["status"] != "deferred":
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.write({k: v for k, v in row.items() if k != "files"})
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if i % 500 == 0 or i == len(todo):
                rate = i / max(1e-9, time.time() - t_start)
                print(f"[{i}/{len(todo)}] {counts} {rate:.1f}/s pauses={api.outage.pauses}", flush=True)
    rows.close()
    print("done:", counts)
    return 1 if counts.get("deferred") else 0


# -------------------------------------------------------------------- fetch


def bundle_dir(root: Path, name: str) -> Path:
    shard = hashlib.sha1(name.encode()).hexdigest()[:2]
    return root / shard / name


def cmd_fetch(args) -> int:
    api = Pulse()
    root = Path(args.out)
    plan = [json.loads(line) for line in open(args.plan, encoding="utf-8") if line.strip()]
    rows = RowLog(Path(args.log))
    todo = []
    for item in plan:
        dest = bundle_dir(root, item["name"]) / item.get("dest", item["path"])
        if dest.exists() and (item.get("bytes") is None or dest.stat().st_size == item["bytes"]):
            continue
        todo.append((item, dest))
    need = sum(item.get("bytes") or 0 for item, _ in todo)
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    print(
        f"{len(plan)} planned files, {len(todo)} to fetch, {need / 1e9:.2f} GB; "
        f"free {free / 1e9:.1f} GB, floor {args.min_free_gb} GB",
        flush=True,
    )
    if free - need < args.min_free_gb * 1e9:
        print("refusing: the fetch would take free space below the floor", file=sys.stderr)
        return 2
    stop = threading.Event()

    def work(pair):
        item, dest = pair
        if stop.is_set():
            return dict(name=item["name"], path=item["path"], status="skipped_stop")
        t0 = time.time()
        try:
            verdict, digest = api.get(item.get("key") or item["name"], item["path"], dest)
        except Deferred as exc:
            return dict(name=item["name"], path=item["path"], status="deferred", error=str(exc))
        row = dict(
            name=item["name"],
            path=item["path"],
            status=verdict,
            bytes=dest.stat().st_size if verdict == "ok" else None,
            secs=round(time.time() - t0, 2),
        )
        if verdict == "ok" and item.get("sha256") and digest != item["sha256"]:
            dest.unlink(missing_ok=True)
            row["status"] = "sha_mismatch"
        return row

    counts: dict[str, int] = {}
    got = 0
    t_start = time.time()
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(work, pair) for pair in todo]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                row = fut.result()
            except (AuthError, Aborted) as exc:
                print(f"stopping: {exc}", file=sys.stderr)
                stop.set()
                for f in futures:
                    f.cancel()
                break
            rows.write(row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            got += row.get("bytes") or 0
            if i % 1000 == 0 or i == len(todo):
                if shutil.disk_usage(root).free < args.min_free_gb * 1e9:
                    print("free space reached the floor; stopping new downloads", file=sys.stderr)
                    stop.set()
                rate = got / max(1e-9, time.time() - t_start) / 1e6
                print(
                    f"[{i}/{len(todo)}] {counts} {got / 1e9:.2f} GB {rate:.1f} MB/s pauses={api.outage.pauses}",
                    flush=True,
                )
    rows.close()
    print("done:", counts)
    return 1 if any(counts.get(k) for k in ("deferred", "sha_mismatch", "skipped_stop")) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("facets")
    p.add_argument("--last-days", type=int, default=180)
    p = sub.add_parser("enum")
    p.add_argument("--last-days", type=int, default=180)
    p.add_argument("--clusters", default="core42,crusoe,crusoe-spur")
    p.add_argument("--layers", default="global,cluster,fine", help="which cell layers to page: global, cluster, fine")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--out", required=True)
    p = sub.add_parser("census")
    p.add_argument("--targets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=16)
    p = sub.add_parser("census-retry")
    p.add_argument("--census", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--inventory", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=16)
    p = sub.add_parser("fetch")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--min-free-gb", type=float, default=60.0)
    args = ap.parse_args(argv)
    if args.cmd == "facets":
        print(json.dumps(Pulse().facets(last_days=args.last_days), ensure_ascii=False, indent=1))
        return 0
    return {"enum": cmd_enum, "census": cmd_census, "census-retry": cmd_census_retry, "fetch": cmd_fetch}[args.cmd](
        args
    )


if __name__ == "__main__":
    raise SystemExit(main())
