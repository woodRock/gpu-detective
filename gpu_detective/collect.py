"""
One sweep over the cluster.

The remote script is gpu-spooler's nvidia-smi query with a wider `ps`: as well
as each GPU process's user, it asks for the elapsed time and full command line,
so jobs can be followed across sweeps and fan-outs spotted.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from gpu_spooler import gpu as spooler_gpu
from gpu_spooler.config import Host
from gpu_spooler.gpu import GpuInfo
from gpu_spooler.transport import Transport, auth_hint

QUERY_TIMEOUT = 25

_PS_UPSTREAM = 'ps -o pid=,user= -p "$PIDS" 2>/dev/null || true'
# user:32 stops procps truncating long usernames to 7 chars + '+'; BSD ps (a
# local macOS host) doesn't know it, hence the fallback. A pid that exits
# mid-query makes the first ps fail too, so the parser tolerates repeats.
_PS_DETAIL = ('ps -ww -o pid=,user:32=,etime=,args= -p "$PIDS" 2>/dev/null'
              ' || ps -ww -o pid=,user=,etime=,args= -p "$PIDS" 2>/dev/null || true')

if _PS_UPSTREAM not in spooler_gpu.REMOTE_CMD:
    raise ImportError("gpu-spooler's REMOTE_CMD changed: update _PS_UPSTREAM in "
                      "gpu_detective/collect.py")
REMOTE_CMD = spooler_gpu.REMOTE_CMD.replace(_PS_UPSTREAM, _PS_DETAIL)


@dataclass
class Proc:
    gpu_index: int
    pid: int
    user: str
    mem_mb: int
    started: Optional[float]    # epoch seconds; None if ps didn't report the pid
    command: str                # full command line, or nvidia-smi's short name


@dataclass
class HostSnapshot:
    host: str
    ts: float
    reachable: bool
    error: str = ""
    gpus: list[GpuInfo] = field(default_factory=list)
    procs: list[Proc] = field(default_factory=list)

    def procs_on(self, gpu_index: int) -> list[Proc]:
        return [p for p in self.procs if p.gpu_index == gpu_index]


def parse_etime(s: str) -> Optional[int]:
    """ps's etime, [[dd-]hh:]mm:ss, in seconds."""
    m = re.fullmatch(r"(?:(?:(\d+)-)?(\d+):)?(\d+):(\d+)", s)
    if not m:
        return None
    d, h, mi, sec = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + sec


def parse_proc_detail(raw: str) -> dict[int, tuple[int, str]]:
    """pid → (elapsed seconds, command line) from the PROC_INFO section."""
    out: dict[int, tuple[int, str]] = {}
    in_proc = False
    for line in raw.splitlines():
        s = line.rstrip("\r")
        if s == "PROC_INFO":
            in_proc = True
        elif s == "END":
            break
        elif in_proc:
            parts = s.split(None, 3)
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            elapsed = parse_etime(parts[2])
            if elapsed is not None:
                out[int(parts[0])] = (elapsed, parts[3] if len(parts) > 3 else "")
    return out


def parse_snapshot(host: str, ts: float, raw: str) -> HostSnapshot:
    gpus = spooler_gpu.parse_output(raw)
    detail = parse_proc_detail(raw)
    procs = []
    for g in gpus:
        for p in g.processes:
            elapsed, cmd = detail.get(p.pid, (None, ""))
            procs.append(Proc(
                gpu_index=g.index, pid=p.pid, user=p.user, mem_mb=p.mem_used_mb,
                started=ts - elapsed if elapsed is not None else None,
                command=cmd or p.command))
    return HostSnapshot(host=host, ts=ts, reachable=True, gpus=gpus, procs=procs)


def query_host(transport: Transport, host: Host, ts: float) -> HostSnapshot:
    """Mirrors gpu_spooler.gpu.query_host, with the detailed script."""
    rc, out, err = transport.run(host, REMOTE_CMD, timeout=QUERY_TIMEOUT)

    def down(error: str) -> HostSnapshot:
        return HostSnapshot(host=host.name, ts=ts, reachable=False, error=error)

    if out.strip() == "NO_NVIDIA_SMI":
        # A local machine (or a configured CPU host) without GPUs is up, just empty.
        if host.is_local or host.cpu_slots > 0:
            return HostSnapshot(host=host.name, ts=ts, reachable=True)
        return down("nvidia-smi not found")
    if rc != 0 and not out:
        msg = err.strip().splitlines()[-1] if err.strip() else f"ssh exited {rc}"
        hint = auth_hint(err)
        return down(f"{msg} ({hint})" if hint else msg)
    errline = next((l for l in out.splitlines() if l.startswith("ERROR:")), None)
    if errline:
        return down(errline)
    if "GPU_INFO" not in out:
        return down("no nvidia-smi output")
    return parse_snapshot(host.name, ts, out)


def sweep(transport: Transport, hosts: Iterable[Host], workers: int = 16) -> list[HostSnapshot]:
    """Query every host once. All snapshots share one timestamp, so a sweep is one instant."""
    ts = time.time()
    results = transport.fan_out(hosts, lambda h: query_host(transport, h, ts), workers)
    return sorted(results.values(), key=lambda s: s.host)
