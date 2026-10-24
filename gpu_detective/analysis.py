"""
Reports over a window of snapshots. Everything here is a pure function of a
list[HostSnapshot], so it can be tested without a database or a cluster.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from gpu_spooler.gpu import BUSY_MEM_MB, GpuInfo

from gpu_detective.collect import HostSnapshot, Proc

IDLE_UTIL_PCT = 5       # a held GPU below this utilisation counts as idle...
IDLE_MIN_SECONDS = 600  # ...once it has been, in every sweep, for this long (one low reading is noise)
MAX_GAP = 300           # a sample stands for at most this many seconds (collector was down)
FANOUT_WINDOW = 120     # jobs started this close together belong to one fan-out
FANOUT_MIN_HOSTS = 3


def is_free(snap: HostSnapshot, g: GpuInfo) -> bool:
    return not snap.procs_on(g.index) and g.mem_used_mb < BUSY_MEM_MB


def is_idle(g: GpuInfo) -> bool:
    return 0 <= g.utilization_pct < IDLE_UTIL_PCT


def command_shape(cmd: str) -> str:
    """A command with its numbers blanked, so `train.py --seed 3` and `--seed 4` match."""
    return re.sub(r"\d+", "#", " ".join(cmd.split()))


def latest(snaps: list[HostSnapshot]) -> list[HostSnapshot]:
    """The newest snapshot of each host."""
    out: dict[str, HostSnapshot] = {}
    for s in snaps:
        if s.host not in out or s.ts > out[s.host].ts:
            out[s.host] = s
    return sorted(out.values(), key=lambda s: s.host)


def by_host(snaps: list[HostSnapshot]) -> dict[str, list[HostSnapshot]]:
    hosts: dict[str, list[HostSnapshot]] = defaultdict(list)
    for s in snaps:
        hosts[s.host].append(s)
    for v in hosts.values():
        v.sort(key=lambda s: s.ts)
    return hosts


def sample_weights(snaps: list[HostSnapshot]) -> dict[int, float]:
    """
    id(snapshot) → seconds it stands for: the gap to the host's next snapshot,
    capped at MAX_GAP. A host's last snapshot repeats its previous gap.
    """
    w: dict[int, float] = {}
    for series in by_host(snaps).values():
        gaps = [min(b.ts - a.ts, MAX_GAP) for a, b in zip(series, series[1:])]
        if gaps:
            gaps.append(gaps[-1])
        else:
            gaps = [0.0]
        for s, g in zip(series, gaps):
            w[id(s)] = g
    return w


def sustained_idle(snaps: list[HostSnapshot], min_seconds: float = IDLE_MIN_SECONDS) -> set[tuple[int, int]]:
    """
    (id(snapshot), gpu index) for every sample that belongs to a sustained idle
    stretch: the GPU held, under IDLE_UTIL_PCT, by the same users, in every
    sweep for at least min_seconds. A bursty job caught between kernels in the
    odd sweep never qualifies: nvidia-smi's utilisation covers only the last second.
    """
    out: set[tuple[int, int]] = set()
    for series in by_host(snaps).values():
        runs: dict[int, tuple[frozenset, list[HostSnapshot]]] = {}   # gpu → (users, its samples)

        def close(idx: int) -> None:
            _, samples = runs.pop(idx)
            if samples[-1].ts - samples[0].ts >= min_seconds:
                out.update((id(x), idx) for x in samples)

        for s in series:
            idle_now: dict[int, frozenset] = {}
            if s.reachable:
                for g in s.gpus:
                    users = frozenset(p.user for p in s.procs_on(g.index))
                    if users and is_idle(g):
                        idle_now[g.index] = users
            for idx in [i for i in runs if idle_now.get(i) != runs[i][0]]:
                close(idx)
            for idx, users in idle_now.items():
                runs.setdefault(idx, (users, []))[1].append(s)
        for idx in list(runs):
            close(idx)
    return out


# ── users ─────────────────────────────────────────────────────────────────────


@dataclass
class UserUsage:
    user: str
    gpu_seconds: float = 0.0
    idle_gpu_seconds: float = 0.0
    peak_gpus: int = 0
    peak_hosts: int = 0
    hosts: set[str] = field(default_factory=set)

    @property
    def gpu_hours(self) -> float:
        return self.gpu_seconds / 3600

    @property
    def idle_gpu_hours(self) -> float:
        return self.idle_gpu_seconds / 3600


def user_usage(snaps: list[HostSnapshot], idle_min: float = IDLE_MIN_SECONDS) -> list[UserUsage]:
    """Per-user GPU time, idle-but-held time (sustained stretches only) and peak
    spread, heaviest first. A GPU shared by several users is split evenly between them."""
    weights = sample_weights(snaps)
    idle = sustained_idle(snaps, idle_min)
    usage: dict[str, UserUsage] = {}
    sweeps: dict[float, dict[str, tuple[int, set[str]]]] = defaultdict(dict)

    for s in snaps:
        w = weights[id(s)]
        for g in s.gpus:
            users = sorted({p.user for p in s.procs_on(g.index)})
            for u in users:
                uu = usage.setdefault(u, UserUsage(u))
                uu.gpu_seconds += w / len(users)
                if (id(s), g.index) in idle:
                    uu.idle_gpu_seconds += w / len(users)
                uu.hosts.add(s.host)
                n, hosts = sweeps[s.ts].get(u, (0, set()))
                sweeps[s.ts][u] = (n + 1, hosts | {s.host})

    for per_user in sweeps.values():
        for u, (n, hosts) in per_user.items():
            usage[u].peak_gpus = max(usage[u].peak_gpus, n)
            usage[u].peak_hosts = max(usage[u].peak_hosts, len(hosts))
    return sorted(usage.values(), key=lambda u: -u.gpu_seconds)


# ── idle hoarders ─────────────────────────────────────────────────────────────


@dataclass
class IdleGpu:
    host: str
    gpu: GpuInfo
    procs: list[Proc]
    idle_since: float
    as_of: float

    @property
    def idle_seconds(self) -> float:
        return self.as_of - self.idle_since


def idle_gpus(snaps: list[HostSnapshot], min_seconds: float = 1800) -> list[IdleGpu]:
    """GPUs that are held right now and have sat idle, held by the same users,
    for at least min_seconds."""
    out = []
    for series in by_host(snaps).values():
        now = series[-1]
        if not now.reachable:
            continue
        for g in now.gpus:
            procs = now.procs_on(g.index)
            if not procs or not is_idle(g):
                continue
            users = {p.user for p in procs}
            since = now.ts
            for s in reversed(series[:-1]):
                if not s.reachable:
                    break
                prev = next((x for x in s.gpus if x.index == g.index), None)
                if prev is None or not is_idle(prev) or \
                        {p.user for p in s.procs_on(g.index)} != users:
                    break
                since = s.ts
            if now.ts - since >= min_seconds:
                out.append(IdleGpu(now.host, g, procs, since, now.ts))
    return sorted(out, key=lambda i: -i.idle_seconds)


# ── fan-outs ──────────────────────────────────────────────────────────────────


@dataclass
class Job:
    host: str
    pid: int
    user: str
    command: str
    started: float
    gpus: set[int] = field(default_factory=set)
    last_seen: float = 0.0


@dataclass
class FanOut:
    user: str
    shape: str
    jobs: list[Job]

    @property
    def hosts(self) -> list[str]:
        return sorted({j.host for j in self.jobs})

    @property
    def gpu_count(self) -> int:
        return sum(len(j.gpus) for j in self.jobs)

    @property
    def first_start(self) -> float:
        return min(j.started for j in self.jobs)

    @property
    def last_seen(self) -> float:
        return max(j.last_seen for j in self.jobs)


def jobs(snaps: list[HostSnapshot]) -> list[Job]:
    """Distinct GPU processes seen in the window (a process on two GPUs is one job)."""
    seen: dict[tuple, Job] = {}
    for s in snaps:
        for p in s.procs:
            if p.started is None:
                continue
            key = (s.host, p.pid, p.user, command_shape(p.command))
            j = seen.get(key)
            if j is None:
                j = seen[key] = Job(s.host, p.pid, p.user, p.command, p.started)
            j.gpus.add(p.gpu_index)
            j.last_seen = max(j.last_seen, s.ts)
    return list(seen.values())


def fanouts(snaps: list[HostSnapshot], window: float = FANOUT_WINDOW,
            min_hosts: int = FANOUT_MIN_HOSTS) -> list[FanOut]:
    """
    Groups of jobs by one user, with the same command shape, started within
    `window` seconds of each other on at least `min_hosts` hosts: the
    signature of a launcher fanning one experiment out across the cluster.
    """
    groups: dict[tuple[str, str], list[Job]] = defaultdict(list)
    for j in jobs(snaps):
        groups[(j.user, command_shape(j.command))].append(j)

    out = []
    for (user, shape), js in groups.items():
        js.sort(key=lambda j: j.started)
        cluster: list[Job] = []
        for j in js + [None]:
            if j is not None and (not cluster or j.started - cluster[-1].started <= window):
                cluster.append(j)
                continue
            if len({c.host for c in cluster}) >= min_hosts:
                out.append(FanOut(user, shape, cluster))
            cluster = [j] if j is not None else []
    return sorted(out, key=lambda f: (-len(f.hosts), -f.gpu_count))


# ── coverage ──────────────────────────────────────────────────────────────────


@dataclass
class Coverage:
    busy_samples: int = 0       # (snapshot, busy GPU) pairs
    visible_samples: int = 0    # ... of which showed at least one process
    procs: int = 0
    known_procs: int = 0        # ... with a user and a start time
    blind_hosts: set[str] = field(default_factory=set)   # busy GPUs but no process ever seen

    @property
    def pct(self) -> float:
        return 100.0 * self.visible_samples / self.busy_samples if self.busy_samples else 100.0

    def line(self) -> str:
        s = f"Visibility: {self.pct:.0f}% of busy GPU samples showed their process"
        if self.procs and self.known_procs < self.procs:
            s += f"; {self.procs - self.known_procs} process samples had no user or start time"
        if self.blind_hosts:
            s += f"; no processes visible at all on {', '.join(sorted(self.blind_hosts))}"
        return s + "."


def coverage(snaps: list[HostSnapshot]) -> Coverage:
    """How much of the busy GPU time we could attribute to a process: low on a
    host that hides other users' processes."""
    c = Coverage()
    busy_by_host: dict[str, int] = defaultdict(int)
    seen_by_host: dict[str, int] = defaultdict(int)
    for s in snaps:
        for g in s.gpus:
            if is_free(s, g):
                continue
            c.busy_samples += 1
            busy_by_host[s.host] += 1
            if s.procs_on(g.index):
                c.visible_samples += 1
                seen_by_host[s.host] += 1
        c.procs += len(s.procs)
        c.known_procs += sum(1 for p in s.procs if p.user != "?" and p.started is not None)
    c.blind_hosts = {h for h, n in busy_by_host.items() if n and not seen_by_host[h]}
    return c


# ── per-GPU history ───────────────────────────────────────────────────────────


@dataclass
class Segment:
    """A stretch of time one GPU was held by the same processes (or free)."""
    host: str
    gpu: int
    start: float
    end: float
    users: list[str]
    commands: list[str]
    util_sum: float = 0.0
    samples: int = 0

    @property
    def mean_util(self) -> float:
        return self.util_sum / self.samples if self.samples else 0.0

    @property
    def free(self) -> bool:
        return not self.users


def gpu_history(snaps: list[HostSnapshot], host: Optional[str] = None,
                gpu: Optional[int] = None) -> list[Segment]:
    """
    Every GPU's timeline as segments of unchanged occupancy, merged across
    sweeps while the same pids hold it. A segment ends at the next sweep that
    differs (or the last sample + its weight); unreachable sweeps break it.
    """
    weights = sample_weights(snaps)
    out: list[Segment] = []
    for h, series in sorted(by_host(snaps).items()):
        if host is not None and h != host:
            continue
        open_: dict[int, tuple[tuple, Segment]] = {}
        for s in series:
            if not s.reachable:
                open_.clear()
                continue
            for g in s.gpus:
                if gpu is not None and g.index != gpu:
                    continue
                procs = s.procs_on(g.index)
                key = tuple(sorted((p.pid, p.user) for p in procs))
                end = s.ts + weights[id(s)]
                cur = open_.get(g.index)
                if cur and cur[0] == key and s.ts <= cur[1].end + 1:
                    seg = cur[1]
                    seg.end = end
                else:
                    seg = Segment(h, g.index, s.ts, end,
                                  sorted({p.user for p in procs}),
                                  sorted({p.command for p in procs}))
                    out.append(seg)
                    open_[g.index] = (key, seg)
                if g.utilization_pct >= 0:
                    seg.util_sum += g.utilization_pct
                    seg.samples += 1
    return out


# ── per-user lanes ────────────────────────────────────────────────────────────


def user_lanes(snaps: list[HostSnapshot], idle_min: float = IDLE_MIN_SECONDS
               ) -> tuple[list[float], dict[str, list[tuple[float, float]]]]:
    """
    Sweep times, and for each user the GPUs (held, of which idle) they had at
    each sweep. A shared GPU counts fractionally, and idle means a sustained
    stretch, as in user_usage.
    """
    idle = sustained_idle(snaps, idle_min)
    times = sorted({s.ts for s in snaps})
    index = {t: i for i, t in enumerate(times)}
    lanes: dict[str, list[list[float]]] = {}
    for s in snaps:
        i = index[s.ts]
        for g in s.gpus:
            users = sorted({p.user for p in s.procs_on(g.index)})
            for u in users:
                lane = lanes.setdefault(u, [[0.0, 0.0] for _ in times])
                lane[i][0] += 1 / len(users)
                if (id(s), g.index) in idle:
                    lane[i][1] += 1 / len(users)
    return times, {u: [(a, b) for a, b in v] for u, v in lanes.items()}


# ── formatting helpers ────────────────────────────────────────────────────────


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 86400}d{s % 86400 // 3600}h"


def parse_duration(spec: str) -> float:
    """'90s', '30m', '24h', '7d' → seconds."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", spec)
    if not m:
        raise ValueError(f"bad duration '{spec}' (use e.g. 30m, 24h, 7d)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
