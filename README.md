# gpu-detective

Who is using the GPU cluster, and how. A history of every GPU process on the cluster, with reports on top of it.

It reuses [gpu-spooler](../gpu-spooler)'s host list, SSH setup and `nvidia-smi` query. It also uses the same open SSH connections, so if `task hosts` works, this works too.

## Install

```bash
uv sync                    # uses the sibling ../gpu-spooler checkout
uv run detective --help
```

## Use

```bash
detective collect              # one sweep of every host into ~/.gpu-detective/history.db
detective watch -i 60          # sweep every minute (run it in tmux/nohup)
detective now                  # every GPU: who is on it, how long, what command
detective now --busy           # hide free GPUs
detective users --since 7d     # GPU-hours, idle-but-held hours, peak GPUs/hosts per user
detective idle --min 30m       # GPUs held at <5% utilisation for 30m+
detective fanouts              # one user's same command started on 3+ hosts within 2m
detective history cuda14:2     # who held one GPU over the last 24h (or `cuda14` for every GPU)
detective serve                # dashboard on http://127.0.0.1:8765 (local only)
detective prune --keep 30d     # drop old history (`watch` does this hourly)
```

Every report ends with a visibility line: the share of busy GPU samples where a process could be seen. Servers that hide other users' processes are named there.

## Dashboard

`detective serve` serves one page on 127.0.0.1 with four tabs:

- **Now:** each server's GPUs with utilisation, user and runtime. GPUs held but idle get a ⚠ badge. Click a GPU for its history.
- **Timeline:** one lane per user showing GPUs held over the window (6h, 24h or 7d), split into working and idle, all on one shared scale.
- **Users:** sortable GPU-hours, share, idle hours, and peak GPUs and hosts.
- **Fan-outs:** experiments launched across many servers at once.

## Privacy options

- `--shapes-only` on `collect`/`watch` stores just the program and script names (`python3 train.py`), with no paths or arguments.
- `watch --keep 30d` (the default) deletes older history every hour.
- The dashboard binds 127.0.0.1 only, and the database is git-ignored.

`-H SPEC` limits `collect`/`watch` to hosts, groups or tags, as in gpu-spooler. `GPU_DETECTIVE_DB` overrides the database path.

## How it works

- **Collector** (`collect.py`): runs gpu-spooler's remote script with a wider `ps`. That adds each GPU process's elapsed time and full command line. All hosts in a sweep share one timestamp.
- **Store** (`store.py`): SQLite with a `snapshots` table (one row per host per sweep), plus `gpus` and `procs` tables linked to it. gpu-spooler's optional `[detective]` table reads this file read-only, so treat the `snapshots`, `gpus` and `procs` columns as a contract with it.
- **Dashboard** (`web.py`, `web/index.html`): a stdlib HTTP server with JSON endpoints and one self-contained page. It loads no external scripts, and every value is HTML-escaped because command lines come from other users.
- **Analysis** (`analysis.py`): pure functions over a list of snapshots.
  - Each sample stands for the time until that host's next sweep, capped at 5 minutes.
  - A GPU shared by several users is split evenly between them.
  - Fan-outs match commands with their numbers blanked out, so `--seed 3` and `--seed 4` count as the same experiment.

It only sees what any cluster user can see with `nvidia-smi` and `ps`. If a host hides other users' processes, their jobs show user `?`.

## Develop

```bash
uv run pytest
../gpu-spooler/dev/fake_cluster.sh up      # local fake cluster
GPU_SPOOLER_CONFIG=/tmp/ts-fake-cluster/hosts.toml uv run detective collect
```
