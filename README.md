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
```

`-H SPEC` limits `collect`/`watch` to hosts, groups or tags, as in gpu-spooler. `GPU_DETECTIVE_DB` overrides the database path.

## How it works

- **Collector** (`collect.py`): runs gpu-spooler's remote script with a wider `ps`. That adds each GPU process's elapsed time and full command line. All hosts in a sweep share one timestamp.
- **Store** (`store.py`): SQLite with a `snapshots` table (one row per host per sweep), plus `gpus` and `procs` tables linked to it.
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
