# CI runners for KernelForge

All CI runs on self-hosted runners in the Primus-SaFE **global** cluster. GitHub-hosted
runners are not usable here: the `AMD-BRAIN-Internal` organisation enforces a GitHub IP
allow list, their egress addresses are not on it, and every job dies in
`actions/checkout` before doing any work:

```
remote: The repository owner has an IP allow list enabled, and <ip> is not permitted
        to access this repository.
fatal: unable to access '.../KernelForge/': The requested URL returned error: 403
```

The global cluster egresses from an allow-listed address, so that is where the runners
live. Manifests: [`deploy/ci/global-runners.yaml`](../ci/global-runners.yaml).

## Pools

| Label | Replicas | Used by | Shape |
|---|---|---|---|
| `KernelForge-ci` | 8 | lint, pre-merge, tests-coverage, docs, codeql, reuse, secret-scan, and the e2e `resolve`/`opt-out-status` jobs | installs packages, needs CPU + disk |
| `KernelForge-e2e-ci` | 2 | the e2e job in `ci-e2e.yml` | one API call plus ~45 min of polling; nearly idle |

They are separate so a 45-minute e2e run cannot starve the fast checks.

Every workflow takes its label from a repo variable, so a pool can be repointed without
editing YAML: `KF_CI_RUNNER_LABEL`, `KF_E2E_RUNNER_LABEL`, `KF_E2E_RESOLVE_RUNNER`.

## Why the e2e runner is not on Crusoe

The e2e job does no computation locally — it POSTs one workload to the Crusoe
orchestration API and polls until it finishes, and the GPU work runs on Crusoe either
way. A runner on the Crusoe host would in fact be *broken*: that host's egress address
is not on the allow list (verified: HTTP 403), so it could not read this repository.
The global cluster resolves and reaches `crusoe.primus-safe.amd.com/robust-api`
directly, so dispatching from there is equivalent and it can also talk to GitHub.

## How the GPU node gets the PR source

The allow list also blocks the **GPU pod**: its egress address is not listed, so a clone
there fails with HTTP 403 and the run dies before forge-loop starts. So the runner checks
the tree out itself and streams it onto the GPU side's shared mount over SSH, then
dispatches with `KF_USE_GIT=0` and `KF_SOURCE_DIR` pointing at the staged copy. The
workload builds from the shared mount and never contacts GitHub.

Pushing is the only direction that works. Measured in both directions:

| From → to | TCP handshake | Data flow |
|---|---|---|
| global runner → GPU-side host `:22` | ok | ok (SSH banner arrives) |
| global runner → GPU-side orchestration API | ok | ok (HTTP 200) |
| GPU side → global `:22` / NodePort | ok | **none** — no banner, no reply |

The handshake toward the global cluster is answered by something in the path, not by the
service: SSH never sends its banner, and a CONNECT proxy on the global side saw the
connection but never the request. So the GPU pod can neither pull from the global cluster
nor proxy its clone through it; only a push from the runner side works.

### Configuration

Nothing is hardcoded — staging turns on only when both of these repo secrets are set:

| Secret | Meaning |
|---|---|
| `CI_E2E_STAGE_HOST` | ssh target owning the shared mount, e.g. `<user>@<host>` |
| `CI_E2E_STAGE_ROOT` | staging root on it, e.g. `<shared-mount>/kernelforge-ci/src` |

Optional repo variables: `CI_E2E_STAGE_SSH_KEY` (mount path of the key, defaults to the
path used by `deploy/ci/global-runners.yaml`), `CI_E2E_STAGE_TIMEOUT_S` (default 900) and
`CI_E2E_STAGE_TTL_DAYS` (default 2).

Leave the pair unset and the workload fetches from GitHub as before, which is correct for
any repository the GPU node may read. Set it and a staging failure is **fatal**: falling
back would reproduce the exact 403 this path exists to avoid, and it would look like the
allow list had broken again.

### The push key

Generate it **on** the cluster so the private half never leaves it, and install the public
half on the account that owns the shared mount:

```bash
ssh-keygen -t ed25519 -N '' -f /tmp/k
kubectl -n kernelforge-ci create secret generic kernelforge-crusoe-ssh \
  --from-file=id_ed25519=/tmp/k
cat /tmp/k.pub          # append to <user>@<mount-host>:~/.ssh/authorized_keys
shred -u /tmp/k /tmp/k.pub

# On that host: a staging root the account can write without sudo.
sudo mkdir -p <stage-root> && sudo chown <user>: <stage-root>
```

The volume is `optional: true`, so a cluster missing the secret still gets working
runners rather than a pool stuck in `ContainerCreating`.

The secret and the two repo secrets are one switch, though, and must be set or unset
together. With `CI_E2E_STAGE_HOST` / `CI_E2E_STAGE_ROOT` set but the key absent, e2e
does **not** quietly fall back — every run fails with `no key at <path>`, on purpose:
that combination can only mean a half-finished deployment, and the fetch it would fall
back to is one the GPU node is not allowed to make.

### Safety properties

- The tree ships as `git archive HEAD` — tracked files only, no `.git`, so no persisted
  credentials reach a mount other teams can read. The checkout also runs with
  `persist-credentials: false`.
- Unpacking happens on the **mount owner's account**, not inside the GPU job, so the
  archive is extracted with Python's `data` filter, which rejects absolute paths, `..`
  traversal, links pointing outside the destination, and special files. Plain `tar -x`
  would let a crafted PR write anywhere that account can reach.
- Each run stages under its own `pr_<n>_<sha>_<run-id>_<attempt>` directory. Sharing one
  path would let a retest or a parallel dispatch delete the tree a running pod is still
  copying, which on NFS hangs instead of failing.
- Cleanup has three layers: the dispatch script removes the tree on exit, a run kept for
  triage after a timeout is left alone deliberately, and the remote side reaps anything
  older than `CI_E2E_STAGE_TTL_DAYS` — the last one covers a runner killed with SIGKILL,
  where no trap runs at all.
- With a staged tree the workload gets no `GITHUB_TOKEN`: it never fetches, and
  `template.env` is readable through the orchestration API.

## Deploy

A registration token is valid for one hour and can register any number of runners in
that window. It is read only on a pod's **first** start: `config.sh` writes credentials
into the pod's PVC, so restarts and node reboots reuse them and never need a new token.

```bash
# 1. Mint a registration token (needs repo admin).
TOKEN="$(gh api --method POST \
  repos/AMD-BRAIN-Internal/KernelForge/actions/runners/registration-token --jq .token)"

# 2. Create the namespace and the secret the StatefulSets read.
kubectl create namespace kernelforge-ci
kubectl -n kernelforge-ci create secret generic kernelforge-runner \
  --from-literal=reg-token="$TOKEN"

# 2b. Create the source-staging push key as well, and set CI_E2E_STAGE_HOST /
#     CI_E2E_STAGE_ROOT — see "How the GPU node gets the PR source" above.

# 3. Deploy both pools.
kubectl apply -f deploy/ci/global-runners.yaml

# 4. Watch them register.
kubectl -n kernelforge-ci get pods -w
kubectl -n kernelforge-ci logs kernelforge-ci-runner-0 | tail -20
```

Verify on GitHub under *Settings → Actions → Runners*: 8 `kernelforge-ci-runner-*` and
2 `kernelforge-e2e-runner-*` should be idle.

## Scaling

```bash
kubectl -n kernelforge-ci scale statefulset kernelforge-ci-runner --replicas=12
```

New replicas need a valid token in the secret to register for the first time; refresh it
with `kubectl -n kernelforge-ci create secret generic kernelforge-runner --from-literal=reg-token=<NEW> --dry-run=client -o yaml | kubectl apply -f -`
before scaling up.

## Notes on the runner image

`ghcr.io/actions/actions-runner:2.336.0` is Ubuntu 24.04 and ships `git`, `curl`, `jq`,
`python3`, `tar`, `unzip` and passwordless `sudo`. It has **no** `pip`, `node`, `gcc` or
Docker daemon. Consequences already handled in the workflows:

- Python comes from `actions/setup-python`, which caches into the PVC-backed tool cache.
- `secret-scan.yml` runs the gitleaks binary directly; `docker run` cannot work in a pod
  with no Docker socket.
- If a future job needs a compiler, install it in-job with `sudo apt-get install -y
  build-essential` rather than assuming it is present.

## Do not run a runner with `nohup`

The previous Crusoe runner was started as `nohup ./run.sh &` with no service
supervision. When the host was rebooted for maintenance it never came back, and CI sat
silently queued for three days. The StatefulSet here is supervised by Kubernetes and
restarts on its own; keep it that way.
