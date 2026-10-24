"""
detective: who is using the GPU cluster, and how.

    detective collect             one sweep of every host into the history
    detective watch -i 60         sweep forever
    detective now                 why is each GPU busy, right now
    detective users --since 24h   GPU-hours, idle-held hours and spread per user
    detective idle --min 30m      GPUs held but idle
    detective fanouts             one experiment launched across many hosts
    detective history cuda14:2    who held one GPU (or every GPU on a host), and when
    detective serve               the dashboard, on http://127.0.0.1:8765
    detective prune --keep 30d    drop history older than that
"""

import argparse
import os
import sys
import time
from datetime import datetime

from gpu_spooler.config import ConfigError, load_config
from gpu_spooler.gpu import gpu_model
from gpu_spooler.transport import Transport

from gpu_detective import analysis as an
from gpu_detective import store
from gpu_detective.collect import HostSnapshot, sweep

STALE = 600     # `now` ignores snapshots older than this
PRUNE_EVERY = 3600
KEEP = 30 * 86400.0


def clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%a %H:%M")


def table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)).rstrip())
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip())


def clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def summary(snaps: list[HostSnapshot]) -> str:
    up = [s for s in snaps if s.reachable]
    gpus = sum(len(s.gpus) for s in up)
    busy = sum(1 for s in up for g in s.gpus if not an.is_free(s, g))
    return f"{len(up)}/{len(snaps)} hosts reachable, {busy}/{gpus} GPUs busy"


# ── commands ──────────────────────────────────────────────────────────────────


def redact(snaps: list[HostSnapshot]) -> None:
    """--shapes-only: keep just the program and script names, without their
    directories or any arguments (`/home/x/proj/train.py --data ...` → `python train.py`)."""
    for s in snaps:
        for p in s.procs:
            p.command = " ".join(os.path.basename(t) for t in p.command.split()[:2])


def cmd_collect(args) -> int:
    cfg = load_config()
    transport = Transport(cfg)
    db = store.connect()
    snaps = sweep(transport, cfg.select(args.hosts))
    if args.shapes_only:
        redact(snaps)
    store.save(db, snaps)
    print(f"{clock(snaps[0].ts) if snaps else ''}  {summary(snaps)}")
    for s in snaps:
        if not s.reachable:
            print(f"  {s.host}: {s.error}")
    return 0


def cmd_watch(args) -> int:
    cfg = load_config()
    transport = Transport(cfg)
    hosts = cfg.select(args.hosts)
    db = store.connect()
    print(f"Sweeping {len(hosts)} hosts every {args.interval:.0f}s into {store.db_path()}, "
          f"keeping {an.fmt_duration(args.keep)} (Ctrl-C to stop)")
    last_prune = 0.0
    try:
        while True:
            start = time.time()
            snaps = sweep(transport, hosts)
            if args.shapes_only:
                redact(snaps)
            store.save(db, snaps)
            if start - last_prune >= PRUNE_EVERY:
                store.prune(db, start - args.keep)
                last_prune = start
            print(f"{clock(start)}  {summary(snaps)}", flush=True)
            time.sleep(max(0.0, args.interval - (time.time() - start)))
    except KeyboardInterrupt:
        return 0


def cmd_now(args) -> int:
    db = store.connect()
    snaps = an.latest(store.load(db, time.time() - STALE))
    if not snaps:
        print(f"No sweeps in the last {an.fmt_duration(STALE)}: run `detective collect`.")
        return 1
    rows = []
    for s in snaps:
        if not s.reachable:
            continue
        for g in s.gpus:
            procs = s.procs_on(g.index)
            if not procs and args.busy:
                continue
            util = f"{g.utilization_pct}%" if g.utilization_pct >= 0 else "?"
            mem = f"{g.mem_used_mb / 1024:.0f}/{g.mem_total_mb / 1024:.0f}G"
            base = [s.host, g.index, clip(gpu_model(g.name), 18), util, mem]
            if not procs:
                rows.append(base + ["free" if an.is_free(s, g) else "(no process)", "", ""])
            for i, p in enumerate(procs):
                age = s.ts - p.started if p.started is not None else None
                lead = base if i == 0 else ["", "", "", "", ""]
                flag = "  idle" if an.is_idle(g) else ""
                rows.append(lead + [p.user, an.fmt_duration(age) + flag,
                                    clip(p.command, args.width)])
    table(rows, ["HOST", "GPU", "MODEL", "UTIL", "MEM", "USER", "RUNTIME", "COMMAND"])
    print(f"\n{summary(snaps)}  (as of {clock(max(s.ts for s in snaps))})")
    print(an.coverage(snaps).line())
    for s in snaps:
        if not s.reachable:
            print(f"  {s.host}: {s.error}")
    return 0


def cmd_users(args) -> int:
    since = time.time() - args.since
    snaps = store.load(store.connect(), since)
    usage = an.user_usage(snaps)
    if not usage:
        print("No GPU processes in that window.")
        return 0
    total = sum(u.gpu_seconds for u in usage) or 1
    rows = [[u.user, f"{u.gpu_hours:.1f}", f"{100 * u.gpu_seconds / total:.0f}%",
             f"{u.idle_gpu_hours:.1f}", u.peak_gpus, u.peak_hosts, len(u.hosts)]
            for u in usage[:args.top]]
    table(rows, ["USER", "GPU-H", "SHARE", "IDLE-H", "PEAK GPUS", "PEAK HOSTS", "HOSTS"])
    print(f"\nWindow: {clock(snaps[0].ts)} → {clock(snaps[-1].ts)}, "
          f"{len({s.ts for s in snaps})} sweeps. IDLE-H = held below {an.IDLE_UTIL_PCT}% util.")
    print(an.coverage(snaps).line())
    return 0


def cmd_idle(args) -> int:
    snaps = store.load(store.connect(), time.time() - max(args.min * 2, 86400))
    idle = an.idle_gpus(snaps, args.min)
    if not idle:
        print(f"No GPU has been held idle for {an.fmt_duration(args.min)} or more.")
        return 0
    rows = [[i.host, i.gpu.index, ",".join(sorted({p.user for p in i.procs})),
             an.fmt_duration(i.idle_seconds), f"{i.gpu.mem_used_mb / 1024:.0f}G",
             clip(i.procs[0].command, args.width)] for i in idle]
    table(rows, ["HOST", "GPU", "USER", "IDLE FOR", "VRAM HELD", "COMMAND"])
    print(f"\nIdle = below {an.IDLE_UTIL_PCT}% util in every sweep, same users throughout. "
          "Counts only time this tool was watching.")
    print(an.coverage(an.latest(snaps)).line())
    return 0


def cmd_fanouts(args) -> int:
    snaps = store.load(store.connect(), time.time() - args.since)
    fos = an.fanouts(snaps, window=args.window, min_hosts=args.min_hosts)
    if not fos:
        print("No fan-outs found in that window.")
        return 0
    rows = [[f.user, len(f.hosts), f.gpu_count, clock(f.first_start),
             an.fmt_duration(f.last_seen - f.first_start), clip(f.shape, args.width)]
            for f in fos]
    table(rows, ["USER", "HOSTS", "GPUS", "STARTED", "SEEN FOR", "COMMAND SHAPE"])
    print(f"\nFan-out = one user's jobs with the same command (numbers ignored) started "
          f"within {an.fmt_duration(args.window) if args.window >= 60 else f'{args.window:.0f}s'}"
          f" of each other on {args.min_hosts}+ hosts.")
    print(an.coverage(snaps).line())
    return 0


def cmd_history(args) -> int:
    host, _, gpu = args.target.partition(":")
    if gpu and not gpu.isdigit():
        raise SystemExit(f"detective: bad GPU '{gpu}' (use HOST or HOST:INDEX)")
    snaps = store.load(store.connect(), time.time() - args.since)
    segs = an.gpu_history(snaps, host, int(gpu) if gpu else None)
    if not segs:
        print(f"No history for {args.target} in that window.")
        return 1
    rows = [[f"{s.host}:{s.gpu}", clock(s.start), an.fmt_duration(s.end - s.start),
             ",".join(s.users) or "free", f"{s.mean_util:.0f}%",
             clip("; ".join(s.commands), args.width)]
            for s in sorted(segs, key=lambda s: (s.gpu, s.start)) if args.all or not s.free]
    table(rows, ["GPU", "FROM", "FOR", "USER", "MEAN UTIL", "COMMAND"])
    return 0


def cmd_prune(args) -> int:
    n = store.prune(store.connect(), time.time() - args.keep)
    print(f"Deleted {n} snapshots older than {an.fmt_duration(args.keep)}.")
    return 0


def cmd_serve(args) -> int:
    from gpu_detective import web
    web.serve(args.port)
    return 0


# ── argument parsing ──────────────────────────────────────────────────────────


def duration(spec: str) -> float:
    try:
        return an.parse_duration(spec)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="detective", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def hosts_arg(sp):
        sp.add_argument("-H", "--hosts", metavar="SPEC",
                        help="hosts, groups or tags, as in gpu-spooler (default: all)")

    def width_arg(sp):
        sp.add_argument("-w", "--width", type=int, default=60, help="command column width")

    def shapes_arg(sp):
        sp.add_argument("--shapes-only", action="store_true",
                        help="store only program and script names, not full command lines")

    sp = sub.add_parser("collect", help="sweep every host once into the history")
    hosts_arg(sp)
    shapes_arg(sp)
    sp.set_defaults(fn=cmd_collect)

    sp = sub.add_parser("watch", help="sweep repeatedly")
    hosts_arg(sp)
    shapes_arg(sp)
    sp.add_argument("-i", "--interval", type=duration, default=60.0, help="default 60s")
    sp.add_argument("-k", "--keep", type=duration, default=KEEP,
                    help="delete history older than this, checked hourly (default 30d)")
    sp.set_defaults(fn=cmd_watch)

    sp = sub.add_parser("now", help="every GPU and who is on it, from the latest sweep")
    sp.add_argument("-b", "--busy", action="store_true", help="hide free GPUs")
    width_arg(sp)
    sp.set_defaults(fn=cmd_now)

    sp = sub.add_parser("users", help="GPU-hours and spread per user")
    sp.add_argument("-s", "--since", type=duration, default=86400.0, help="default 24h")
    sp.add_argument("-n", "--top", type=int, default=20)
    sp.set_defaults(fn=cmd_users)

    sp = sub.add_parser("idle", help="GPUs held but idle")
    sp.add_argument("-m", "--min", type=duration, default=1800.0, help="default 30m")
    width_arg(sp)
    sp.set_defaults(fn=cmd_idle)

    sp = sub.add_parser("fanouts", help="experiments launched across many hosts at once")
    sp.add_argument("-s", "--since", type=duration, default=86400.0, help="default 24h")
    sp.add_argument("--window", type=duration, default=float(an.FANOUT_WINDOW),
                    help="max gap between starts (default 2m)")
    sp.add_argument("--min-hosts", type=int, default=an.FANOUT_MIN_HOSTS)
    width_arg(sp)
    sp.set_defaults(fn=cmd_fanouts)

    sp = sub.add_parser("history", help="who held a GPU, and when")
    sp.add_argument("target", metavar="HOST[:GPU]")
    sp.add_argument("-s", "--since", type=duration, default=86400.0, help="default 24h")
    sp.add_argument("-a", "--all", action="store_true", help="include free stretches")
    width_arg(sp)
    sp.set_defaults(fn=cmd_history)

    sp = sub.add_parser("prune", help="delete old history")
    sp.add_argument("-k", "--keep", type=duration, default=KEEP, help="default 30d")
    sp.set_defaults(fn=cmd_prune)

    sp = sub.add_parser("serve", help="local web dashboard")
    sp.add_argument("-p", "--port", type=int, default=8765)
    sp.set_defaults(fn=cmd_serve)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        sys.exit(args.fn(args))
    except ConfigError as e:
        sys.exit(f"detective: {e}")


if __name__ == "__main__":
    main()
