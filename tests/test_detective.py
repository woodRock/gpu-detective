from gpu_spooler.gpu import GpuInfo

from gpu_detective import analysis as an
from gpu_detective import store
from gpu_detective.collect import REMOTE_CMD, HostSnapshot, Proc, parse_etime, parse_snapshot

RAW = """\
GPU_INFO
0, NVIDIA A40, 1, 46067, 46068, 0, Default
1, NVIDIA A40, 18000, 28068, 46068, 99, Default
2, NVIDIA A40, 9000, 37068, 46068, 0, Default
PMON_INFO
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   command
# Idx           #    C/G      %      %      %      %      %      %     MB   name
    0           -      -      -      -      -      -      -      -      -   -
    1        4242      C     99     40      -      -      -      -  17990   python
    2        5151      C      0      0      -      -      -      -   8990   python3
APPS_INFO
1, 4242
2, 5151
PROC_INFO
 4242 alice     1-02:03:04 python train.py --seed 3 --lr 0.001
 5151 averyverylongname    05:00 /usr/bin/python3 -m ipykernel_launcher -f kernel.json
 4242 alice     1-02:03:04 python train.py --seed 3 --lr 0.001
END
"""


def test_remote_cmd_asks_for_detail():
    assert "etime=,args=" in REMOTE_CMD


def test_parse_etime():
    assert parse_etime("05:00") == 300
    assert parse_etime("02:03:04") == 2 * 3600 + 3 * 60 + 4
    assert parse_etime("1-02:03:04") == 86400 + 2 * 3600 + 3 * 60 + 4
    assert parse_etime("bogus") is None


def test_parse_snapshot():
    s = parse_snapshot("cuda14", 1000000.0, RAW.replace("\n", "\r\n"))
    assert [g.index for g in s.gpus] == [0, 1, 2]
    p1, p2 = s.procs_on(1)[0], s.procs_on(2)[0]
    assert (p1.user, p1.mem_mb, p1.command) == ("alice", 17990, "python train.py --seed 3 --lr 0.001")
    assert p1.started == 1000000.0 - (86400 + 7384)
    assert p2.user == "averyverylongname" and p2.command.endswith("kernel.json")


def test_store_roundtrip(tmp_path):
    db = store.connect(tmp_path / "h.db")
    s = parse_snapshot("cuda14", 1000.0, RAW)
    down = HostSnapshot("cuda00", 1000.0, reachable=False, error="timed out")
    store.save(db, [s, down])
    back = {x.host: x for x in store.load(db)}
    assert back["cuda00"].error == "timed out" and not back["cuda00"].reachable
    assert [g.utilization_pct for g in back["cuda14"].gpus] == [0, 99, 0]
    assert sorted((p.pid, p.user) for p in back["cuda14"].procs) == \
        [(4242, "alice"), (5151, "averyverylongname")]
    assert store.load(db, since=1001.0) == []


# ── analysis on synthetic sweeps ──────────────────────────────────────────────


def gpu(idx, util):
    return GpuInfo(index=idx, name="A40", mem_used_mb=9000 if util >= 0 else 0,
                   mem_free_mb=0, mem_total_mb=46068, utilization_pct=util)


def snap(host, ts, gpus, procs):
    return HostSnapshot(host, ts, True, gpus=gpus, procs=procs)


def test_user_usage_splits_shared_gpus_and_counts_idle():
    snaps = []
    for t in (0, 60, 120):
        snaps.append(snap("a", t, [gpu(0, 90), gpu(1, 0)], [
            Proc(0, 1, "alice", 1, 0, "x"), Proc(0, 2, "bob", 1, 0, "y"),
            Proc(1, 3, "bob", 1, 0, "z")]))
        snaps.append(snap("b", t, [gpu(0, 90)], [Proc(0, 4, "bob", 1, 0, "y")]))
    u = {x.user: x for x in an.user_usage(snaps)}
    # 3 samples x 60s each; alice holds half of a:0, bob half of a:0 + a:1 + b:0.
    assert u["alice"].gpu_seconds == 90
    assert u["bob"].gpu_seconds == 90 + 180 + 180
    assert u["bob"].idle_gpu_seconds == 180
    assert (u["bob"].peak_gpus, u["bob"].peak_hosts) == (3, 2)
    assert an.user_usage(snaps)[0].user == "bob"


def test_sample_weights_cap_gaps():
    a, b, c = snap("a", 0, [], []), snap("a", 60, [], []), snap("a", 4000, [], [])
    w = an.sample_weights([a, b, c])
    assert (w[id(a)], w[id(b)], w[id(c)]) == (60, an.MAX_GAP, an.MAX_GAP)


def test_idle_gpus_needs_same_users_throughout():
    def s(t, util, user):
        return snap("a", t, [gpu(0, util)], [Proc(0, 7, user, 9000, 0, "jupyter")])
    series = [s(0, 80, "carol"), s(600, 0, "carol"), s(1200, 0, "carol"), s(2400, 0, "carol")]
    [i] = an.idle_gpus(series, min_seconds=1800)
    assert (i.host, i.idle_since, i.idle_seconds) == ("a", 600, 1800)
    assert an.idle_gpus(series, min_seconds=3600) == []
    series[1] = s(600, 0, "dave")
    assert an.idle_gpus(series, min_seconds=1700) == []


def test_fanouts_group_by_shape_and_start_time():
    snaps = []
    for i, host in enumerate(["c1", "c2", "c3", "c4"]):
        snaps.append(snap(host, 1000, [gpu(0, 99), gpu(1, 99)], [
            Proc(0, 100 + i, "erin", 1, 500 + 10 * i, f"python train.py --seed {i}"),
            # erin's other.py jobs started minutes apart, one by hand per host
            Proc(1, 900 + i, "erin", 1, 600 * i, "python other.py")]))
    [f] = an.fanouts(snaps)
    assert f.user == "erin" and f.hosts == ["c1", "c2", "c3", "c4"]
    assert f.shape == "python train.py --seed #" and f.gpu_count == 4
    assert len(an.fanouts(snaps, window=600)) == 2
    assert an.fanouts(snaps, min_hosts=5) == []


def test_parse_duration_and_format():
    assert an.parse_duration("30m") == 1800 and an.parse_duration("2d") == 172800
    assert an.fmt_duration(59 * 60) == "59m"
    assert an.fmt_duration(3 * 3600 + 5 * 60) == "3h05m"
    assert an.fmt_duration(90000) == "1d1h"


def test_coverage_flags_hosts_hiding_processes():
    snaps = [snap("open", 0, [gpu(0, 90)], [Proc(0, 1, "alice", 1, 0, "x")]),
             snap("blind", 0, [gpu(0, 90), gpu(1, 90)], [])]
    c = an.coverage(snaps)
    assert (c.busy_samples, c.visible_samples, c.blind_hosts) == (3, 1, {"blind"})
    assert "blind" in c.line() and "33%" in c.line()


def test_gpu_history_merges_sweeps_and_breaks_on_change():
    p = lambda pid, user: [Proc(0, pid, user, 1, 0, f"{user}.py")]
    series = [snap("a", 0, [gpu(0, 90)], p(1, "alice")), snap("a", 60, [gpu(0, 70)], p(1, "alice")),
              snap("a", 120, [gpu(0, -1)], []), snap("a", 180, [gpu(0, 0)], p(2, "bob"))]
    segs = an.gpu_history(series, "a", 0)
    assert [(s.start, s.end, s.users) for s in segs] == [
        (0, 120, ["alice"]), (120, 180, []), (180, 240, ["bob"])]
    assert segs[0].mean_util == 80 and segs[0].commands == ["alice.py"]


def test_user_lanes_per_sweep():
    snaps = [snap("a", 0, [gpu(0, 90), gpu(1, 0)], [Proc(0, 1, "alice", 1, 0, "x"),
                                                     Proc(1, 2, "alice", 1, 0, "x")]),
             snap("a", 60, [gpu(0, 90)], [Proc(0, 1, "alice", 1, 0, "x"), Proc(0, 3, "bob", 1, 0, "y")])]
    times, lanes = an.user_lanes(snaps)
    assert times == [0, 60]
    assert lanes["alice"] == [(2, 1), (0.5, 0)] and lanes["bob"] == [(0, 0), (0.5, 0)]


def test_prune(tmp_path):
    db = store.connect(tmp_path / "h.db")
    store.save(db, [parse_snapshot("a", 100.0, RAW), parse_snapshot("a", 200.0, RAW)])
    assert store.prune(db, 150.0) == 1
    assert [s.ts for s in store.load(db)] == [200.0]
    assert db.execute("SELECT COUNT(*) FROM procs").fetchone()[0] == 2


def test_redact_keeps_program_and_script_names():
    from gpu_detective.cli import redact
    s = snap("a", 0, [gpu(0, 90)], [Proc(0, 1, "u", 1, 0, "/usr/bin/python3 /home/u/secret/train.py --data /x")])
    redact([s])
    assert s.procs[0].command == "python3 train.py"
