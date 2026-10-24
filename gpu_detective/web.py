"""
Local dashboard: a stdlib HTTP server serving one page and a few JSON
endpoints over the history. It binds 127.0.0.1 only, so the history never
leaves this machine (see the spec's privacy guardrails).
"""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from gpu_spooler.gpu import gpu_model

from gpu_detective import analysis as an
from gpu_detective import store

BIND = "127.0.0.1"
PAGE = Path(__file__).with_name("web") / "index.html"
NOW_LOOKBACK = 6 * 3600     # `now` looks this far back to time idle stretches
STALE = 600
MAX_POINTS = 720            # lanes are averaged down to at most this many buckets


def _num(q: dict, key: str, default: float) -> float:
    try:
        return float(q[key][0])
    except (KeyError, ValueError, IndexError):
        return default


def api_now(q: dict) -> dict:
    now = time.time()
    snaps = store.load(store.connect(), now - NOW_LOOKBACK)
    current = [s for s in an.latest(snaps) if s.ts >= now - STALE]
    idle_for = {(i.host, i.gpu.index): i.idle_seconds for i in an.idle_gpus(snaps, 0)}
    hosts = []
    for s in current:
        gpus = []
        for g in s.gpus:
            procs = s.procs_on(g.index)
            gpus.append({
                "index": g.index, "model": gpu_model(g.name), "util": g.utilization_pct,
                "mem_used": g.mem_used_mb, "mem_total": g.mem_total_mb,
                "free": an.is_free(s, g), "idle_for": idle_for.get((s.host, g.index)),
                "procs": [{"user": p.user, "pid": p.pid, "mem": p.mem_mb, "command": p.command,
                           "runtime": s.ts - p.started if p.started is not None else None}
                          for p in procs]})
        hosts.append({"host": s.host, "reachable": s.reachable, "error": s.error, "gpus": gpus})
    return {"ts": max((s.ts for s in current), default=None), "hosts": hosts,
            "lookback": NOW_LOOKBACK, "idle_util": an.IDLE_UTIL_PCT,
            "coverage": an.coverage(current).line()}


def _bucket(times: list[float], series: dict[str, list[tuple[float, float]]]):
    """Average consecutive sweeps into at most MAX_POINTS buckets."""
    if len(times) <= MAX_POINTS:
        return times, series
    step = len(times) / MAX_POINTS
    edges = [round(i * step) for i in range(MAX_POINTS + 1)]
    out_t = [times[a] for a, b in zip(edges, edges[1:]) if b > a]
    out_s = {}
    for u, v in series.items():
        out_s[u] = [(sum(x[0] for x in v[a:b]) / (b - a), sum(x[1] for x in v[a:b]) / (b - a))
                    for a, b in zip(edges, edges[1:]) if b > a]
    return out_t, out_s


def api_lanes(q: dict) -> dict:
    since = _num(q, "since", 86400)
    snaps = store.load(store.connect(), time.time() - since)
    times, lanes = an.user_lanes(snaps)
    times, lanes = _bucket(times, lanes)
    usage = an.user_usage(snaps)
    return {
        "times": times, "max_gap": an.MAX_GAP, "idle_util": an.IDLE_UTIL_PCT,
        "sweeps": len({s.ts for s in snaps}),
        "users": [{"user": u.user, "gpu_hours": round(u.gpu_hours, 2),
                   "idle_hours": round(u.idle_gpu_hours, 2), "peak_gpus": u.peak_gpus,
                   "peak_hosts": u.peak_hosts, "hosts": sorted(u.hosts),
                   "series": [[round(a, 3), round(b, 3)] for a, b in lanes.get(u.user, [])]}
                  for u in usage],
        "coverage": an.coverage(snaps).line()}


def api_history(q: dict) -> dict:
    host = q.get("host", [""])[0]
    gpu = int(_num(q, "gpu", -1))
    since = _num(q, "since", 86400)
    segs = an.gpu_history(store.load(store.connect(), time.time() - since), host,
                          gpu if gpu >= 0 else None)
    return {"segments": [{"gpu": s.gpu, "start": s.start, "end": s.end, "users": s.users,
                          "commands": s.commands, "util": round(s.mean_util, 1)} for s in segs],
            "idle_util": an.IDLE_UTIL_PCT}


def api_fanouts(q: dict) -> dict:
    snaps = store.load(store.connect(), time.time() - _num(q, "since", 86400))
    return {"fanouts": [{"user": f.user, "shape": f.shape, "hosts": f.hosts,
                         "gpus": f.gpu_count, "start": f.first_start, "last_seen": f.last_seen}
                        for f in an.fanouts(snaps)],
            "window": an.FANOUT_WINDOW, "min_hosts": an.FANOUT_MIN_HOSTS}


ROUTES = {"/api/now": api_now, "/api/lanes": api_lanes, "/api/history": api_history,
          "/api/fanouts": api_fanouts}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        fn = ROUTES.get(url.path)
        if fn is None:
            return self._send(404, b'{"error":"not found"}', "application/json")
        try:
            body = json.dumps(fn(parse_qs(url.query))).encode()
            self._send(200, body, "application/json")
        except Exception as e:  # surface the error to the page rather than a dropped socket
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")

    def log_message(self, fmt, *args) -> None:
        pass


def serve(port: int = 8765) -> None:
    try:
        httpd = ThreadingHTTPServer((BIND, port), Handler)
    except OSError as e:
        raise SystemExit(f"detective: can't serve on {BIND}:{port} ({e.strerror}). "
                         f"Another dashboard may be running; try `detective serve -p {port + 1}`.")
    print(f"Dashboard on http://{BIND}:{port} (local only, Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
