# What campaigns on a serving benchmark learned to measure

These rules come from earlier serving campaigns, and from what failed in them. Each says what it guards against. When
you study a benchmark, check whether each applies. Where it does, put it in `rules.yaml` or `metric.md` with this
benchmark's own source.

| Rule | What it guards against |
| --- | --- |
| **Golden acceptance is a contract, checked on the live server.** SGLang `SGLANG_SIMULATE_ACC_*` and vLLM `synthetic_acceptance_length` follow the benchmark's golden curve for the configured draft depth. After the run, check the observed acceptance from the replay's median, not the end-of-run gauge. | Natural acceptance looks like a free P90 gain, and the end-of-run gauge can read well below the replay's median. |
| **Draft at shipped precision** (the InferenceX PR checklist). Refuse known draft-precision switches. | Golden acceptance makes a cheaper draft look free; a campaign that built a lower-precision draft head was disqualified. |
| **Score what the chart plots:** raw P90 interactivity and total throughput per GPU, with a cap on the other axis. | A campaign that optimized output throughput adopted a change that lowered raw P90. |
| **Measure noise first** (3 replays), and set bands from it. Move the primary to the stable axis where P90 is noisy. | P90 can be far noisier than throughput, and a change inside the noise was kept. |
| **Three rungs:** a component eval, then a 1200 s screen, then a 3600 s confirm outside the loop. | A short bench overstates gains against the full window, and a full round per candidate is slow. |
| **Refuse bad runs:** a non-zero client exit, or a window under 90% of DURATION. | A client can reject a run and still write an aggregate. |
| **A fixed workload:** the corpus, CONC, the InferenceX client, the chat template, prefix cache on, and DURATION never changed mid-campaign. | P90 is non-stationary within a run, as contexts grow. |
| **The baseline is the patched tree, committed before the session.** Reject empty diffs. | A round spent measuring the baseline under a candidate's name. |
| **No target exit on a proxy window.** | A target reached on a short window stopped a campaign with most of its time left. |
| **Correctness is not covered by throughput.** Add a real-acceptance eval, short and long context, at the 3600 s confirm. | A numerics bug was invisible under synthetic acceptance. |

**Under a host,** the host's client is the only bench, so the pack's own bench and its contract checks do not run.
Each rule then says who checks it under the host: its gate, or nobody.
