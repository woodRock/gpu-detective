# gpu-detective

Who is using the GPU cluster, and how. gpu-detective sweeps every GPU server once a minute, keeps a history of every GPU process, and reports on it from the terminal or a local dashboard.

It answers questions a one-off `nvidia-smi` can't:

- **Why is this GPU busy?** Who is on it, running what, and for how long.
- **Who uses the cluster most?** GPU-hours per user, and how many GPUs and servers they hold at once.
- **Which GPUs are held but idle?** Processes sitting under 5% utilisation for 30 minutes or more.
- **Who launched one experiment across many servers?** The same command started on several hosts within minutes.

It's built on [gpu-spooler](https://github.com/woodRock/gpu-spooler) and reuses its host list, SSH setup (including the off-campus jump hosts) and `nvidia-smi` query. If `task hosts` works for you, gpu-detective works too. It's set up for the VUW ECS cluster through gpu-spooler's config.

gpu-spooler has since been rewritten in Rust. Until gpu-detective is ported too, it uses the Python gpu-spooler 0.1 (tag [`v0.1-python`](https://github.com/woodRock/gpu-spooler/tree/v0.1-python)) as a library. The `task` commands this README mentions still work with the Rust gpu-spooler, which reads the same `~/.gpu-spooler/hosts.toml`.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/woodRock/gpu-detective/main/install.sh | bash
```

This installs `detective` with [uv](https://docs.astral.sh/uv/) as an isolated tool in `~/.local/bin`, along with its one dependency, gpu-spooler. It installs uv first if needed (and Python 3.11 if the machine doesn't have it). No sudo. Add `bash -s -- --upgrade` to update to the latest `main`, or `bash -s -- --uninstall` to remove it:

```sh
curl -fsSL https://raw.githubusercontent.com/woodRock/gpu-detective/main/install.sh | bash -s -- --upgrade
```

If you don't use gpu-spooler yet, set up its host config first: `task config init --user <your ECS username>`, which comes with gpu-spooler's install (see [its README](https://github.com/woodRock/gpu-spooler#quick-start)). gpu-detective reads the same `~/.gpu-spooler/hosts.toml`.

For development, from a checkout:

```sh
bash install.sh
```

That installs `detective` editable from the checkout. If a `../gpu-spooler` checkout sits next to it and still has the Python package (`gpu_spooler/`, as at tag `v0.1-python`), that's installed editable too, so edits to either take effect straight away. Otherwise gpu-spooler comes from the `v0.1-python` tag.

## Quick start

```bash
detective watch          # sweep every server every minute (leave it running in tmux)
detective now            # in another terminal: every GPU, who is on it, for how long
detective serve          # dashboard on http://127.0.0.1:8765
```

## Commands

```bash
detective collect              # one sweep of every host into ~/.gpu-detective/history.db
detective watch -i 60          # sweep every minute, deleting history older than 30 days hourly
detective now                  # every GPU: who is on it, how long, what command
detective now --busy           # hide free GPUs
detective users --since 7d     # GPU-hours, idle-but-held hours, peak GPUs/hosts per user
detective idle --min 30m       # GPUs held at <5% utilisation for 30m+
detective fanouts              # one user's same command started on 3+ hosts within 2m
detective history cuda14:2     # who held one GPU over the last 24h (or `cuda14` for every GPU)
detective serve                # dashboard on http://127.0.0.1:8765 (local only)
detective prune --keep 30d     # drop old history now
```

`-H SPEC` limits `collect` and `watch` to hosts, groups or tags, as in gpu-spooler. `GPU_DETECTIVE_DB` overrides the database path. Durations take `90s`, `30m`, `24h` or `7d`.

Every report ends with a visibility line: the share of busy GPU samples where a process could be seen. Servers that hide other users' processes are named there.

## Dashboard

`detective serve` serves one page on 127.0.0.1 with four tabs:

- **Now:** each server's GPUs with utilisation, user and runtime. GPUs held but idle get a ⚠ badge. Click a GPU for its history.
- **Timeline:** one lane per user showing GPUs held over the window (6h, 24h or 7d), split into working and idle, all on one shared scale.
- **Users:** sortable GPU-hours, share, idle hours, and peak GPUs and hosts.
- **Fan-outs:** experiments launched across many servers at once.

It refreshes every minute and follows your system's light or dark mode.

## Where to run `watch`

`watch` works from a laptop, through the jump hosts, the same way gpu-spooler does. Two things matter for a process that runs all day:

- **It only records while it runs.** Sweeps stop while a laptop sleeps, and GPU-hours don't count that time.
- **Jump-host logins run out.** If the entry host asks for a password or code, run `task connect` again after `jump_control_persist` (12 hours by default). Until then, sweeps report the servers as unreachable.

For continuous history, run it in tmux on an always-on ECS machine.

## With gpu-spooler

Add a `[detective]` table to `~/.gpu-spooler/hosts.toml` and gpu-spooler reads this history too:

```toml
[detective]
idle_minutes = 30
```

`task gpus` then marks GPUs held but idle, and `task list` flags your own running tasks whose GPUs have sat idle, since they still count against `max_gpus`. See gpu-spooler's README for details.

## Privacy and fair use

gpu-detective is a fair-share tool, not a way to watch people. It only sees what any logged-in cluster user can already see with `nvidia-smi` and `ps`: usernames, process IDs, start times, command lines, and per-GPU memory and utilisation. Keeping that as history shows more than any single snapshot, so:

- **The history stays on your machine.** The dashboard binds 127.0.0.1 only, and nothing is uploaded.
- **Old history is deleted.** `watch` keeps 30 days by default (`--keep`).
- **Command lines can be trimmed.** `--shapes-only` on `collect` and `watch` stores just the program and script names (`python3 train.py`), with no paths or arguments.
- **It never acts on anyone's jobs.** It only reads.
- **Check before you share.** Talk to the cluster admins before showing anyone else a view of other users' jobs.

## How it works

- **Collector** (`collect.py`): runs gpu-spooler's remote script with a wider `ps`. That adds each GPU process's elapsed time and full command line. All hosts in a sweep share one timestamp.
- **Store** (`store.py`): SQLite with a `snapshots` table (one row per host per sweep), plus `gpus` and `procs` tables linked to it. gpu-spooler's optional `[detective]` table reads this file read-only, so treat the `snapshots`, `gpus` and `procs` columns as a contract with it.
- **Dashboard** (`web.py`, `web/index.html`): a stdlib HTTP server with JSON endpoints and one self-contained page. It loads no external scripts, and every value is HTML-escaped because command lines come from other users.
- **Analysis** (`analysis.py`): pure functions over a list of snapshots.
  - Each sample stands for the time until that host's next sweep, capped at 5 minutes.
  - A GPU shared by several users is split evenly between them.
  - Fan-outs match commands with their numbers blanked out, so `--seed 3` and `--seed 4` count as the same experiment.

## Develop

```bash
uv run pytest
../gpu-spooler/dev/fake_cluster.sh up      # a fake two-server cluster on localhost
GPU_SPOOLER_CONFIG=/tmp/ts-fake-cluster/hosts.toml uv run detective collect
```
