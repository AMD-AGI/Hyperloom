---
myst:
    html_meta:
        "description": "Measurement card guard variable for Hyperloom: controls what happens when a process outside a test holds one of the test's GPU cards during measurement."
        "keywords": "Hyperloom, environment variables, measurement card guard, HYPERLOOM_CARD_GUARD, GPU cards, foreign process, enforce, observe"
---
# Hyperloom Measurement card guard

What is done about a process outside a test that holds one of the test's GPU cards. The local toolbox reads it once, when
it is built (the `card_guard` option wins over it), applies it to the cards it hands out to tests and leases, and passes
the same value to every measurement runner it starts.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_CARD_GUARD` | `enforce` | `enforce`: a card some process holds is not handed out (the test waits for a free one; the log names the card, the process and the memory it holds, once per holder), and a test whose card a process outside it holds is ended at the first look that finds it (about once a second) and refused as the infrastructure failure `foreign_process_on_card`, which is retried on other cards. `observe`: cards are handed out and tests run to their end; the signed result says `foreign` with the process, the card and when, and carries a warning. Any other value is refused at start-up. Where the cards cannot be read, both modes go on and the result says `unknown` and why. |
