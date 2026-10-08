<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->
# kbmine: estimate uplift without running

Estimate inference uplift for a target **without running a session**, by mining
evidence prior sessions already produced. The import name and CLI are `kbmine`.

This is a standalone tool. It lives under `tools/` and is not part of the
optimizer: nothing in `src/` imports it, it adds no step to the run loop, and it
changes neither the Recipe KB schema nor what a session publishes. It only reads
what the Recipe KB and Pulse already hold.

Two questions, two sources, one report:

| question | source | why that source |
| --- | --- | --- |
| What gain did prior sessions settle on for this scope? | Pulse or Recipe KB | both carry validated gain |
| How much roofline headroom did they actually close? | **Pulse only** | the KB has no roofline, storewide |
| Which parallelism layout won inside a fixed GPU count? | **Recipe KB only** | Pulse carries no accepted server args |
| What did prior sessions accept, and what did they learn? | **Recipe KB only** | Pulse carries no config and no prose |

See [docs/architecture.md](docs/architecture.md) for the component and run-flow
diagrams, and for why the scoping is not optional.

The tool is deliberately dependency-free: every file is Python standard
library, so it runs anywhere a Python does and needs no wheel resolution
inside a restricted network.

## Install

From the Hyperloom checkout:

```bash
python -m pytest tools/kbmine/tests -q   # no network
pip install -e tools/kbmine              # optional; provides the `kbmine` entry point
```

Without installing, run it as `python -m kbmine.cli` from `tools/kbmine`.

## Credentials and TLS

**No token and no service URL is stored in this repository, and none should
ever be committed.** Supply them at runtime through the environment:

```bash
export KB_STORE_URL=https://<kb-store-host>/knowledge-base   # Recipe KB
export PULSE_URL=https://<pulse-host>/<pulse-api-base>         # Pulse, for --pulse-url
export KB_STORE_TOKEN=<read-scoped-token>      # same token reaches the KB and Pulse
export AMD_CA=/path/to/amd_bundle.pem          # only if the AMD root is not installed
```

Ask the KB Store owners for the URLs that apply to you.

`AMD_CA` is not read by this tool. It is a shell variable the examples below
expand into `--ca-bundle "$AMD_CA"`, so the long path is written once. The flag
is the real interface; `KB_STORE_CA_BUNDLE` also works for the KB when a client
is built from the environment.

Both are unnecessary on a machine where the AMD root is installed system-wide,
which is the better setup and takes one command. Check with
`./scripts/provision-amd-ca.sh --check`; see "TLS" below for why the root is
needed at all.

The same value authenticates both services. Use a read-only token scoped to the
`inference` scheme, and rotate it if it has passed through a shell history or a
chat. `.gitignore` blocks `*.pem` and token-shaped filenames as a backstop, but
the real guarantee is that the token only ever lives in the environment.

A token file (`--kb-store-token-file`) and a flag (`--kb-store-token`) are also
accepted, resolved flag then file then environment. Avoid the flag outside CI:
an argument is visible in `ps` and in shell history. The report echoes the
service URL but never the token.

### From a fresh clone

Once the environment above is set, nothing about TLS needs to be arranged by
hand:

```bash
cd tools/kbmine
pip install -e .

./scripts/provision-amd-ca.sh --check      # already trusted? then you are done
./scripts/provision-amd-ca.sh --bootstrap  # otherwise: fetch, verify, install (needs AMD_ROOT_CA_URL)

python -m kbmine.cli --model kimi-k3 --hardware mi355x --tp 8 --conc 1 \
    --isl 8192 --osl 1024
```

`--check` exits 0 on a corporate-imaged machine, where the root is already
present and there is nothing to do. `--bootstrap` covers everything else,
including containers and CI runners. After either, no `--ca-bundle` and no
`SSL_CERT_FILE` is required, and the URL and token are the only things you supply.

The sections below explain what that root is and why it is missing by default;
they are worth reading before overriding any of it, but not required to start.

### TLS: why any of this is needed

A TLS client accepts a server only if it can build a chain from the certificate
the server presents up to a root it *already holds locally*. A stock machine
holds around 121 public roots. Neither service is signed by any of them:

```text
leaf  CN = <service-host>               issued by  CN = AMD-com Issuing CA
      CN = AMD-com Issuing CA           issued by  CN = AMD Corporate Root CA
```

The server sends the leaf and the intermediate and stops. It never sends the
root, and that is correct rather than a misconfiguration: a root arriving over
the wire proves nothing, since an attacker could supply one too. Trust has to
come from something the client already has.

So with only public roots the chain dead-ends at the issuing CA with no trusted
parent, and OpenSSL reports `Verify return code: 20 (unable to get local issuer
certificate)`. Python raises `CERTIFICATE_VERIFY_FAILED`. Neither message means
the certificate is bad; both mean the verifier lacks the one root that would let
it judge. That single root is all that is missing.

Do not work around this by disabling verification. Verification is what protects
the token, and the same token reaches both services: without it, anyone on the
network path can present their own certificate for the hostname and be handed
the credential.

### TLS: provisioning the root

Install the root once and every TLS client on the machine works, including this
tool, with no flag and no environment variable:

```bash
./scripts/provision-amd-ca.sh --check                 # is it already trusted?
./scripts/provision-amd-ca.sh --bootstrap             # fetch from AMD PKI, verify, install
./scripts/provision-amd-ca.sh /path/to/amd_root.pem   # or install a copy you have
./scripts/provision-amd-ca.sh --from-bundle amd_bundle.pem   # or pull it out of a bundle
```

The script pins the fingerprint below and refuses anything else, converts DER to
PEM, picks the anchor directory for the distro family, and finishes by checking
that the endpoint verifies against the *default* store with the TLS environment
variables stripped, so an ambient setting cannot make a failed install look
successful. By hand it is two commands:

```bash
# Debian, Ubuntu
sudo install -m 644 amd_root.pem /usr/local/share/ca-certificates/AMD_Corporate_Root_CA.crt
sudo update-ca-certificates

# RHEL, Fedora, Rocky, Azure Linux
sudo cp amd_root.pem /etc/pki/ca-trust/source/anchors/AMD_Corporate_Root_CA.crt
sudo update-ca-trust
```

On the Debian path the filename must end in `.crt` and the contents must be PEM.
`update-ca-certificates` silently skips a file named `.pem`, and silently skips
DER, so a wrong name or encoding reports `1 added` and changes nothing.

In a container image, bake it in rather than mounting it at run time:

```dockerfile
COPY AMD_Corporate_Root_CA.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates
```

One trap follows a correct system install: `certifi` carries its own root list
and ignores the system store, so `pip` and `requests` still fail inside a
virtualenv even though the system is fixed. This tool uses stdlib `urllib` and
is unaffected; for the rest, set `REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`.
The script warns when it detects this.

For CI runners the root is **not** a secret. It is a public CA certificate, so
it belongs in a repository or organisation *variable*, written out by a setup
step, or better still baked into the runner image. For managed laptops and
workstations this should not be a per-user step at all: the corporate base image
or MDM policy is the right place, and on a properly imaged machine the error
never appears.

### TLS: obtaining the root, and the bootstrap problem

The certificate is deliberately **not** in this repository; `.gitignore` blocks
CA material. What is committed is its identity, which is safe to publish and is
what lets you validate a copy obtained through any channel:

```text
subject=CN = AMD Corporate Root CA      (self-signed)
sha256  74:7D:15:A5:63:89:04:B0:98:EE:1C:8D:04:30:AB:37:E7:A9:F2:D2:D7:A4:C2:C9:CB:CD:7E:89:6E:05:32:C4
expires Aug  6 13:24:37 2032 GMT
```

AMD's PKI publishes the root at the `caIssuers` URL in the issuing CA's
Authority Information Access extension, over plain HTTP. The URL is not written
here; read it from the issuing CA the service presents, and export it:

```bash
openssl s_client -connect "<service-host>:443" -showcerts </dev/null 2>/dev/null \
  | awk '/BEGIN CERT/{n++} n==2' > issuing_ca.pem
openssl x509 -in issuing_ca.pem -noout -ext authorityInfoAccess   # the CA Issuers URI
export AMD_ROOT_CA_URL='<that URI>'     # several, space-separated, are tried in order
```

`--bootstrap` fetches from there and installs only on a fingerprint match. The
unauthenticated transport is not the weakness it looks like, because the pin
above did **not** arrive that way: it came with this repository over verified
GitHub TLS. A tampered download cannot match a SHA-256 chosen in advance, so
the pin, not the connection, is what establishes trust. Skipping the pin check
is the unsafe act, not the HTTP.

What genuinely does not work is trying to obtain the root from the endpoint
itself. `openssl s_client` and `curl -k` against the service host
return the leaf and the intermediate only, because a server never sends its
root, so there is nothing to harvest. And accepting a root offered by the host
you are trying to authenticate would be circular anyway: it lets a network
attacker choose the root you install and then pass every later check silently.

Any other channel is fine as long as you check the fingerprint on arrival: an
IT-managed image, an internal artifact store, or a copy from a machine that
already trusts it.

```bash
openssl x509 -in amd_root.pem -noout -subject -issuer -fingerprint -sha256
```

Subject and issuer must be identical; that is what makes it a root rather than
an intermediate, which cannot complete the chain alone. Note the 2032 expiry:
renewal is a fleet-wide event, not a one-machine fix.

Export from a machine that already trusts it:

```bash
# macOS
security find-certificate -c "AMD Corporate Root CA" -p \
  /Library/Keychains/System.keychain > amd_root.pem

# Windows
certutil -store Root "AMD Corporate Root CA" amd_root.crt
openssl x509 -inform der -in amd_root.crt -out amd_root.pem

# Linux
cp /usr/local/share/ca-certificates/AMD_Corporate_Root_CA.crt amd_root.pem
```

### TLS: `--ca-bundle`, for when you cannot touch the trust store

Where the system store is not writable, point one process at a file instead.
The bundle must hold the public roots *as well as* the AMD one, because
`--ca-bundle` replaces the trust store rather than extending it: a file holding
only the AMD root reaches these two hosts and makes every public host stop
verifying.

```bash
cat amd_root.pem /etc/ssl/certs/ca-certificates.crt > amd_bundle.pem
cat amd_root.pem "$(python3 -m certifi)" > amd_bundle.pem   # non-Debian
```

Pass it explicitly rather than relying on an ambient `SSL_CERT_FILE`. An
exported variable makes a broken setup look healthy, which is exactly how the
missing `--ca-bundle` wiring in `KBStoreClient` went unnoticed: every test
passed because the variable was set in the shell, not because the flag worked.

## Usage

Fleet gain and capture from Pulse:

```bash
python -m kbmine.cli \
    --pulse-url \
    --ca-bundle "$AMD_CA" \
    --start 2026-07-01 --end 2026-08-24 \
    --data-source global --pipeline-tag text-generation \
    --hardware mi355x --max-rows 6000
```

Parallelism layout ranking from the Recipe KB:

```bash
python -m kbmine.cli --hardware mi355x --framework-name sglang --tp 8
```

`--pulse-url` with no value reads `$PULSE_URL`, and the KB path reads
`$KB_STORE_URL`; either flag also takes a URL directly.

Offline, against a saved pool of KB session envelopes:

```bash
python -m kbmine.cli --input prior_sessions.json --tp 8 --isl 1024 --osl 256
```

## Reading the report

* `historical` — p50/p90 validated end-to-end gain across the pool.
* `by_shape` — the same per `tp/conc/isl/osl` bucket. Read this instead of
  `historical` whenever the pool spans shapes.
* `capture` — p50/p90 of the fraction of each session's roofline gap that was
  actually closed. Populated only from Pulse; on the KB path it is reported as
  unmeasured, because an unmeasured capture is not a zero capture.
* `sharding_whatif` — accepted layouts ranked per replay scope, with the vLLM
  and SGLang spellings of tp/dp/ep/pp normalized. KB path only.
* `recipe_knobs` — every accepted flag and env var, not just the sharding
  ones, classified by family and annotated where a value was sized to the
  scope it was measured at. `--max-running-requests 64` measured at
  concurrency 64 is a rule to re-apply, not a value to copy.
* `learnings` — the `what_worked`, `what_failed`, `lessons`, `pitfalls` and
  `remaining_gaps` a session recorded, each tagged with the scope it was
  learned at. These deliberately cross the shape filter, because a cold target
  has no in-scope evidence by definition; items are labelled
  `in_requested_scope: false`, and anything whose text names the concurrency
  you asked about leads the digest.
* `pool_warnings` — raised whenever the pool mixes models, boards, frameworks,
  versions, precisions or shapes. A median across those is not a prior for any
  of them.
* `coverage`, `sessions_scored`, `limitations` — how much the report is
  actually standing on.

A no-run forecast is `baseline + p50_capture x (ceiling - baseline)`. On
mi300x/sglang/fp8 tp1/conc64/isl1024/osl1024 (n=5889, capture on 4939) that
gave 6,428 tok/s/GPU against an observed p50 of 41.9% gain — the forecast
implied 34.4%, so it under-predicted by 7.5 points.

## Worked example: a target at an unseen scope

Asked about Kimi-K3 at concurrency 1, the Recipe KB holds five sessions, all at
`tp8/conc64/isl8192/osl1024`. The gain prior correctly reports nothing: all five
are dropped by the shape filter, and `historical` is `null` rather than a median
borrowed from another concurrency.

The recipe and the prose are where the answer lives. Those five sessions
accepted 15 distinct knobs and 5 env vars — KV cache dtype, mamba pool sizing,
chunked prefill, attention backend, request-pool occupancy — and every one of
their measured wins came from those rather than from sharding, so the layout
ranking alone sees an empty pool. Of the 120 recorded learnings, the one the
digest leads with is a reverted lever whose vendor measurement is *at
concurrency 1*: the persistent aiter MLA decode KV-split budget, 416.7 us to
48.0 us per kernel call and 1.30x on ITL, closed as a true negative at
concurrency 64 because the budget saturates there. Correctly rejected for the
scope it was tested at, and the first thing to reconsider for a target at
concurrency 1.

## Gotchas that are easy to get wrong

Each of these was measured against the live services, not inferred:

1. **`roofline_*_within_pct` on a Pulse row is a fleet aggregate**, not a
   per-session value: 38 distinct values across 50 rows, counts in the
   hundreds, a max above 100%, and a trend spanning days. Per-session capture
   must come from the nested `roofline` object plus the row's two arms.
2. **Pulse throughput is already per-GPU** (`opt_tok_per_s_per_gpu`) while the
   KB stores a total and divides by tp. The projector scales back up so one
   downstream division cannot silently shrink a per-GPU figure by tp.
3. **Pulse rows carry no accepted server args**, so a layout cannot be read out
   of one. Rows are marked `layout_unknown`; calling an unknown layout
   `framework-default` would invent evidence.
4. **The Recipe KB has no roofline at all.** All 165 session documents and 401
   distinct key paths were scanned: no `roofline`, `token_usage`, `platform` or
   `elapsed_minutes` key exists. Its artifacts are 25 patch files totalling
   5.6 MB, and a session archive is `values.json` plus a manifest plus patches.

## Known limits

* Pulse returns rows in insertion order, so a truncated crawl is not a random
  sample — a 6000-row pull came back 98% one CI cohort. Per-identity priors
  need server-side filtering or a full crawl.
* Pulse's server-side filters are inconsistent (`prec` is silently ignored,
  `gpu_type` matches far fewer rows than carry that value), so identity
  narrowing is done client-side where it is verifiable.
* The token appears to see a leaderboard-filtered subset: the summary reports
  `visibility: leaderboard` with 2,378 hidden models and 3,517 hidden rows.
* 3 of 5008 sampled rows show capture above 100%, meaning the analytic ceiling
  was too low. Pulse exposes `roofline_ceiling_exceeded`; this tool does not
  yet surface it.
* Pulse's OpenAPI document is behind SSO, so the parameter contract here was
  reverse-engineered from the Pulse SPA rather than read from a spec.

## Planned

* Surface `roofline_ceiling_exceeded` and exclude those rows from the capture
  prior.
* Server-side filtering once the Pulse parameter contract is confirmed.
* MAIDAS roofline integration. Pulse already computes an analytic ceiling with
  stated provenance (`compute_peak_tflops` from the TraceLens arch JSON,
  `hbm_bw_gbps`, per-op arithmetic intensity), so the first question is where
  MAIDAS *disagrees* with it on the same sessions, not how to wire it in.

## Provenance

The estimator was first proposed inside the optimizer as
[PR #1391](https://github.com/AMD-AGI/Hyperloom/pull/1391), which was closed
because mining fleet evidence is not part of the Hyperloom workflow. This tool
keeps that boundary: it lives under `tools/`, outside `src/`, and reads only.
`kbmine/kb_store_client.py` is a copy of the client vendored under
`src/hyperloom/orchestrator/knowledge/remote_recipe/_vendor/`, with a
`ca_bundle` parameter added so `--ca-bundle` reaches the KB, and without the
producer-side section helpers a read-only tool never calls. It is kept separate
so the tool stays standard-library only and installable on its own.
