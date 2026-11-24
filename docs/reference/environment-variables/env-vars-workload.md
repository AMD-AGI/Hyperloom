---
myst:
    html_meta:
        "description": "Workload parameter environment variables for Hyperloom: model shape, budget goals, GPU power settings, cluster topology, PD disaggregation, and phase toggles."
        "keywords": "Hyperloom, environment variables, workload, ISL, OSL, CONC, TP, EP, GPU power, phase toggles, quantization"
---
# Workload configuration

Set with CLI flags. `ISL` / `OSL` / `CONC` / `TP` / `EP` resolve on a ladder,
and the ladder differs by branch. A fresh launch takes an explicit flag, then
the environment (where an `--extra-env NAME=VALUE` pin lands), then the
default. A resume takes an explicit flag, then the pins in effect for this
resume — its own `--extra-env`, or the launch's persisted pins when it passes
none — then the session's recorded value, then the shell environment, then the
default — so a value left over in the shell cannot displace the shape the
session was measured at. `PRECISION` is not on that ladder: on both branches it
is `--precision`, then `$PRECISION`, then the session's recorded value, then
the checkpoint's own dtype, then the default, so on a resume a `$PRECISION`
left in the shell still outranks the recorded value. `GPU_TYPE` is a fallback
when `--gpu-type` is omitted.

- **Model / workload shape:** `--model`, `--model-class`, `--framework`,
  `--framework-version`, `--precision`, `--tp`, `--ep`, `--isl`, `--osl`,
  `--conc`, `--max-model-len`, `--profile-osl`.
- **Goal / budget:** `--target-gain`, `--target-roofline`, `--max-hours`,
  `--target-tput`, `--compare-against-gpu`. The roofline target composes with
  the others: whichever is met first ends the run.
  `--max-latency-ms` is a constraint on KEEP rather than a target, and applies
  to scriptable frameworks (`xdit`, `custom`) only; the CLI refuses it for a
  serving framework, including on resume. `--max-power-w` is the same kind of
  constraint on the serving GPUs' total mean power, for the serving frameworks
  (refused on a scriptable one, which runs no server to sample), and
  `--max-per-gpu-power-w GPU_ID W` (repeatable) adds a limit for one card; the
  per-GPU limits must add up to no more than the total. That power is sampled
  by Hyperloom over the measured window into each round's `power.json`;
  `HYPERLOOM_GPU_POWER_SAMPLING=0` turns the sampler off (the round then falls
  back to the benchmark report's reading, and either power budget is refused at
  launch, since it could not be measured) and `HYPERLOOM_GPU_POWER_INTERVAL_S`
  sets the cadence of its sysfs reads (default 1 s; below 0.5 s it is raised to
  0.5 s; above 1.5 s it is refused and the round samples nothing, since a late
  read can put two samples two intervals apart and a gap over 3 s invalidates the
  window's energy).
- **GPU power settings:** `--gpu-power-cap-w`, `--gpu-perf-level`,
  `--gpu-sclk-max-mhz`. By default asserted: set them with `amd-smi set` before
  launch; the session refuses to start or resume if a card it uses differs, and
  records what the cards report either way. `--apply-gpu-power-settings` lets
  the session set the cap and perf level itself (never the clock limit) on
  the cards of its integer visible-device mask for the whole session, and
  restore the originals when it ends; it needs root (records under
  `/var/tmp/hyperloom/gpu_power_settings/`).
- **Node power management (MI350 series):** `--node-power-management`
  (`enabled` / `disabled`), `--node-power-limit-w`. Always asserted against
  `amd-smi node -p`, never set by the session: NPM is set on the BMC. The
  session refuses to start or resume if the node differs, and records the
  observed state and limit either way where `amd-smi` reports them. With NPM
  enabled, a session whose visible-device mask leaves any of the node's GPUs
  out is refused; `--allow-shared-npm-node` (on every launch and resume)
  measures anyway and records the cards left out.
- **Cluster topology & multi-node backend:** `--nodes`, `--gpus-per-node`,
  `--gpu-type`, `--mn-backend` (`rayjob` / `infera`), `--server-args` (rayjob).
  Per-pod sizing, the pod image and pod-side env are the provisioning
  platform's inputs, not `optimize` flags — the cluster already exists by the
  time the optimizer runs.
- **PD disaggregation (infera):** `--pd-mode disaggregated`,
  `--pd-prefill-nodes` / `--pd-prefill-tp` / `--pd-prefill-ep` /
  `--pd-prefill-extra-args`, `--pd-decode-nodes` / `--pd-decode-tp` /
  `--pd-decode-ep` / `--pd-decode-extra-args`, `--pd-transfer-backend`,
  `--pd-ib-device`.
- **Phase toggles:** `--enable-roofline` / `--no-enable-roofline`,
  `--enable-conc-sweep` / `--no-enable-conc-sweep`, `--conc-sweep-concs`,
  `--conc-sweep-total-budget-sec` (whole-sweep budget), `--no-framework-agent`, `--no-framework-local-explore`, `--no-kernel`.
- **Agent models:** `--claude-model`, `--codex-model`.
- **Session / resume:** `--resume-from`, `--force-resume`, `--reset-state`,
  `--extend-hours`.
- **Quantization:** `--quantize`, `--quantize-scheme`.
- **Operator-defined specialist:** `--custom-specialist-prompt-file` (focus
  text, max 16 KiB) and `--custom-specialist-description` (one line, max 200
  chars), passed together; unset means no `custom_specialist`. Refused with
  `--no-framework-agent`, `--research-lane-capacity 0` or `--reset-state`.
  `HYPERLOOM_CUSTOM_SPECIALIST_ONLY=1` (settable by `export` or `--extra-env`;
  needs the pair; any other value than a boolean exits 2) makes FRAMEWORK_AGENT run only that specialist and the
  bench of its proposals and patches, then move on to KERNEL_AGENT when
  enabled, else SWEEP. All three are persisted in `state.json`; the env on a
  `--resume-from` turns only mode on, never off. Re-pass both flags on
  `--resume-from` only to replace the definition, or pass `--reset-state`
  alone to clear it.

Run `hyperloom optimize --help` for the exhaustive flag list.
