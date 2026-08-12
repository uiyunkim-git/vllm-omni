# Worker Auto-Update (CI/CD)

Central tracks which git commit each worker runs, shows drift, and can roll
workers to the latest committed code — with one manual bootstrap per machine.

## Model

- **Version = git commit SHA.** Every machine is a checkout of the same repo, so
  a commit is a machine-independent identity.
- **Target** = latest commit on `WORKER_BRANCH` (`main`) as seen from central's
  checkout after a `git fetch`. Central has the docker socket but no git, so it
  computes this with a throwaway `alpine/git` container.
- **Worker report**: each heartbeat carries `{commit, subtree, updated_at}` read
  from `worker/data/.worker_version`. A worker that has never self-updated
  reports `unmanaged` (central can't know its exact commit).
- **Drift** = worker commit ≠ target.

## How an update runs

`POST /api/workers/{id}/update` → worker's `POST /api/internal/self_update` →
the worker launches a **detached** updater container (its own image, which has
git + docker + compose) that:

1. `git fetch` + `git checkout -f <branch>` + `git reset --hard origin/<branch>`
2. writes `worker/data/.worker_version`
3. `docker compose -f docker-compose.worker.yml up -d --build worker`

Detached (`docker run -d`, a sibling on the host daemon) so it survives the
worker's own restart. It only rebuilds the **worker agent** container — the
vLLM model containers it manages are separate compose projects and keep serving.

Guards:
- Skips if a deploy/stop is in progress on that worker (returns 409).
- One updater at a time (fixed container name).
- `update_all` and the auto-loop skip workers currently **serving** a deployment
  (their GPUs are in use) and workers reporting `unmanaged`.

> ⚠️ `git reset --hard` **discards uncommitted local changes** — this is the
> point (workers must run committed code), but never point auto-update at a
> machine whose uncommitted work you care about (e.g. a dev box). Commit + push
> first.

## One-time manual bootstrap (per machine)

Workers must be started once by hand with the new env. The self-updater takes
over afterward.

```bash
# In the repo root on the worker machine:
export HOST_GIT_DIR=$(git rev-parse --git-common-dir)   # absolute; needed for submodules/worktrees
export WORKER_BRANCH=main
docker compose -f docker-compose.worker.yml up -d --build worker
```

For central (this machine already done):

```bash
export HOST_GIT_DIR=$(git rev-parse --git-common-dir)
# optional: export WORKER_AUTO_UPDATE=1   # enable the rolling auto-loop
docker compose -f docker-compose.central.yml up -d central
```

`HOST_GIT_DIR` matters when the checkout is a **git submodule or linked
worktree** — then `.git` is a file pointing outside the tree, so the updater
must mount the real git dir. For a plain clone leave it empty. On this machine
it's `/home/uiyunkim/bisl-uiyunkim/.git/modules/production/vllm-omni` (a
submodule), persisted in the gitignored root `.env`.

## Config (env)

| var | where | default | meaning |
|---|---|---|---|
| `HOST_REPO_DIR` | worker, central | `${PWD}` (compose) | host path of the checkout |
| `HOST_GIT_DIR` | worker, central | empty | real git dir for submodule/worktree checkouts |
| `WORKER_BRANCH` | worker, central | `main` | branch workers converge to |
| `WORKER_AUTO_UPDATE` | central | `0` | `1` = rolling auto-update of drifted, idle workers |

## API

- `GET  /api/version` — `{target, branch, auto_update, workers:[{worker_id, commit, up_to_date, drift, updating, …}]}`
- `POST /api/workers/{worker_id}/update[?branch=]` — update one worker (409 if busy)
- `POST /api/workers/update_all[?branch=]` — update every drifted, active, idle worker

UI: **Endpoints** page → "Worker Versions" panel (per-worker commit, drift
badge, per-row Update, "Update all drifted").

## Auto-update loop (opt-in)

With `WORKER_AUTO_UPDATE=1`, central every 60s picks **one** drifted, active,
idle (not serving), non-`unmanaged` worker past a 5-min cooldown and updates it,
re-evaluating each tick — a slow rolling upgrade that never yanks a GPU mid-inference.
