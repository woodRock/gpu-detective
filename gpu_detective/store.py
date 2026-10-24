"""
SQLite history of sweeps: ~/.gpu-detective/history.db (override with
GPU_DETECTIVE_DB). One snapshots row per host per sweep, with its GPUs and
GPU processes hanging off it.
"""

import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from gpu_spooler.gpu import GpuInfo

from gpu_detective.collect import HostSnapshot, Proc

STATE_DIR = Path.home() / ".gpu-detective"

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id        INTEGER PRIMARY KEY,
    ts        REAL NOT NULL,
    host      TEXT NOT NULL,
    reachable INTEGER NOT NULL,
    error     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS snapshots_ts ON snapshots(ts);
CREATE INDEX IF NOT EXISTS snapshots_host_ts ON snapshots(host, ts);

CREATE TABLE IF NOT EXISTS gpus (
    snapshot_id  INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    name         TEXT NOT NULL,
    mem_used_mb  INTEGER NOT NULL,
    mem_free_mb  INTEGER NOT NULL,
    mem_total_mb INTEGER NOT NULL,
    util_pct     INTEGER NOT NULL,
    PRIMARY KEY (snapshot_id, idx)
);

CREATE TABLE IF NOT EXISTS procs (
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    gpu_idx     INTEGER NOT NULL,
    pid         INTEGER NOT NULL,
    user        TEXT NOT NULL,
    mem_mb      INTEGER NOT NULL,
    started     REAL,
    command     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS procs_snapshot ON procs(snapshot_id);
CREATE INDEX IF NOT EXISTS procs_user ON procs(user);
"""


def db_path() -> Path:
    env = os.environ.get("GPU_DETECTIVE_DB")
    return Path(env).expanduser() if env else STATE_DIR / "history.db"


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    path = path or db_path()
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.execute("PRAGMA journal_mode=WAL")   # reports can read while `watch` writes
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(SCHEMA)
    return db


def save(db: sqlite3.Connection, snaps: Iterable[HostSnapshot]) -> None:
    with db:
        for s in snaps:
            sid = db.execute(
                "INSERT INTO snapshots (ts, host, reachable, error) VALUES (?, ?, ?, ?)",
                (s.ts, s.host, int(s.reachable), s.error)).lastrowid
            db.executemany(
                "INSERT INTO gpus VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(sid, g.index, g.name, g.mem_used_mb, g.mem_free_mb, g.mem_total_mb,
                  g.utilization_pct) for g in s.gpus])
            db.executemany(
                "INSERT INTO procs VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(sid, p.gpu_index, p.pid, p.user, p.mem_mb, p.started, p.command)
                 for p in s.procs])


def prune(db: sqlite3.Connection, before: float) -> int:
    """Delete snapshots older than `before` (their GPUs and processes cascade). Returns the count."""
    with db:
        n = db.execute("DELETE FROM snapshots WHERE ts < ?", (before,)).rowcount
    return n


def load(db: sqlite3.Connection, since: float = 0.0) -> list[HostSnapshot]:
    """Snapshots with ts >= since, oldest first."""
    snaps: dict[int, HostSnapshot] = {}
    for sid, ts, host, reachable, error in db.execute(
            "SELECT id, ts, host, reachable, error FROM snapshots WHERE ts >= ? "
            "ORDER BY ts, host", (since,)):
        snaps[sid] = HostSnapshot(host=host, ts=ts, reachable=bool(reachable), error=error)
    if not snaps:
        return []
    lo = min(snaps)
    for sid, idx, name, used, free, total, util in db.execute(
            "SELECT * FROM gpus WHERE snapshot_id >= ? ORDER BY snapshot_id, idx", (lo,)):
        if sid in snaps:
            snaps[sid].gpus.append(GpuInfo(index=idx, name=name, mem_used_mb=used,
                                           mem_free_mb=free, mem_total_mb=total,
                                           utilization_pct=util))
    for sid, gpu_idx, pid, user, mem, started, command in db.execute(
            "SELECT * FROM procs WHERE snapshot_id >= ?", (lo,)):
        if sid in snaps:
            snaps[sid].procs.append(Proc(gpu_index=gpu_idx, pid=pid, user=user, mem_mb=mem,
                                         started=started, command=command))
    return list(snaps.values())
