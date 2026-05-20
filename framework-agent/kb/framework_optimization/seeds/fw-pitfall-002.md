# fw-pitfall-002 scheduler.py stride changes drop tokens

Framework:
Tags: pitfall, scheduler, accuracy

Adjusting prefill/decode stride or chunk boundaries in `scheduler.py`
(both vllm + sglang) easily drops tokens at chunk seams without
crashing -- the bench reports the same tput but accuracy_gate fails
2-5% below baseline. Tell-tale signal: gsm8k drops ~3% while bench
tput numbers are unchanged.

Mitigation: any scheduler.py patch MUST be paired with a strict
accuracy-drop threshold (`max_accuracy_drop_pct: 0.5`) in
framework_integrate -- design §4.2 default 1% is too lenient for
scheduler patches.
