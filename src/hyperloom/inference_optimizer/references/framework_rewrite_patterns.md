# Framework-Level Rewrite Patterns

Reference for the `framework_rewrite_specialist` domain: the catalogue of
source-rewrite patterns that pay on an **iterative model pipeline** — a
diffusion or autoregressive rollout that runs the same transformer stack once
per block, per denoising step, per chunk.

The specialist's prompt carries a condensed form of this taxonomy as a prior.
This file is the long form: the before/after code shapes, the cache-key recipe
and the failure modes. Read it when a candidate's category is clear but the
mechanics are not.

## Why this is a distinct domain

A request-serving framework and an iterative pipeline share almost no
optimization surface. Serving wins come from scheduling, batch composition,
KV-cache admission and graph capture. An iterative pipeline has none of those:
there is one request, one batch, and a fixed loop. Its wins come from the loop
structure itself.

Two properties drive everything:

1. **Cost is multiplied.** A single step's work is repeated
   `blocks x steps x chunks` times. At a representative operating point that is
   20 blocks x 50 steps x 8 chunks = 8000 repetitions, so a computation whose
   result does not change across that product is 7999 parts waste.
2. **The loop is synchronous.** Every host round-trip inside it stalls the whole
   pipeline, because there is no other request to overlap with.

Both classes of waste are invisible in a GPU kernel breakdown, which is why the
host probe exists (`assets/host_probe/hl_host_probe.py`) and why its output is
the primary input to this domain.

## The taxonomy

Category ids match the `category` field the evidence aggregator emits and the
`framework_switches` manifest declares, so a candidate can be traced from
measurement to rewrite to accepted lever.

### (a) `memoize_invariant` — memoize a step- or block-invariant computation

**Signal.** A high strict-repeat rate: the function was called again with
argument objects it had already received.

**Before.** A pure function of some per-chunk constant, called once per block.
Here it derives projection matrices from camera geometry, but the shape is what
matters, not the domain — the same pattern covers a mask built from a sequence
length, a schedule derived from a step count, or a table derived from a shape:

```python
def apply_transform(...):
    fn_q, fn_kv, fn_o = _build_transform(head_dim, extrinsics, intrinsics, ...)
```

**After.** The same call behind a keyed cache:

```python
def _build_transform_cached(head_dim, extrinsics, intrinsics, ...):
    if not _env_on("HL_TRANSFORM_CACHE"):
        return _build_transform(head_dim, extrinsics, intrinsics, ...)
    key = (head_dim, _tensor_key(extrinsics), _tensor_key(intrinsics), ...)
    hit = _CACHE.get(key)
    if hit is not None and hit.versions == (extrinsics._version, intrinsics._version):
        return hit.value
    value = _build_transform(head_dim, extrinsics, intrinsics, ...)
    # The source tensors are pinned in the entry; see the cache-key recipe.
    _CACHE[key] = _Entry(value=value, sources=(extrinsics, intrinsics),
                         versions=(extrinsics._version, intrinsics._version))
    return value
```

### (b) `hoist_loop_invariant` — hoist a loop-invariant computation out of the loop

**Signal.** A high *loose*-repeat rate with a low *strict*-repeat rate: the
arguments look like the same value every iteration but arrive as freshly
allocated tensors.

**This is usually an enabler.** On its own it saves only the allocation, which
measures flat. Its value is that it makes the (a) rewrites downstream of it
start hitting: while the inputs are rebuilt every iteration, a cache keyed on
tensor identity has a 0% hit rate no matter how correct it is.

**Before.** A slice and a cast inside the step loop, where the slice bounds are
constant for the whole chunk:

```python
for step in timesteps:
    window = source[:, start:end].to(device).to(dtype)
    ...
```

**After.** Computed once per chunk and reused:

```python
hoisted = (source[:, start:end].to(device).to(dtype), ...)
for step in timesteps:
    window = hoisted[0]
    ...
```

Declare the dependency in the manifest: the hoist's `enables` lists the caches
it unlocks, and each cache's `depends_on` lists the hoist. Without that the
hoist is judged on its standalone number and rejected, and the caches it would
have unlocked are then measured with a permanently cold cache.

### (c) `eliminate_host_round_trip` / `eliminate_host_sync`

**Signal.** An object collective (`all_gather_object` and friends), or an
`.item()` / `.tolist()` / `.cpu()` / `synchronize`, called on the hot path.

Object collectives are the more expensive of the two and the easier to miss: the
payload is often a single integer, so the call looks free, while each one is a
pickle plus a full rendezvous. Multiply by three tensors per attention call, by
blocks, by steps.

**Before.**

```python
seq_lens = [None] * world_size
dist.all_gather_object(seq_lens, input.shape[1], group)
```

**After.** The exchange is deterministic for a given `(local value, world size,
group)`, so do it once:

```python
def _gather_seq_lens(local_len, world_size, group):
    if not _env_on("HL_A2A_SEQLEN_CACHE"):
        return _exchange(local_len, world_size, group)
    key = (local_len, world_size, id(group))
    hit = _CACHE.get(key)
    if hit is None:
        hit = _CACHE[key] = _exchange(local_len, world_size, group)
    return list(hit)
```

Include the process group in the key. Two groups with the same world size are
not interchangeable, and a cache that conflates them returns one group's
topology to the other.

**The implicit syncs.** A device-to-host read does not have to be spelled
`.item()`, and the unspelled ones are the hardest to see, because nothing in the
line looks like a transfer:

| line | what it does |
|---|---|
| `if scalar_tensor == 0:` | `Tensor.__bool__` — a blocking read |
| `float(t)` / `int(t)` | `Tensor.__float__` / `__int__` |
| `some_list[t]` | `Tensor.__index__` |
| `torch.full((n,), t)` with a 0-dim device `t` | converts inside ATen |
| `f"{t}"`, `print(t)` | formats, so reads |

The probe wraps the four dunders, so the first three appear in the evidence as
`torch.Tensor.__bool__` and friends. The fourth does **not**: a tensor passed
where the C++ argument parser wants a `Scalar` is converted inside ATen without
calling any Python method, so it is invisible to any monkeypatch. An absent sync
in the evidence is therefore not proof of absence — read the per-step path for
tensors handed to APIs that take scalars. The fix is to stay on the device
(`t.expand(n)` rather than `torch.full((n,), t)`).

### (d) `fuse_collectives` — fuse adjacent collectives, GEMMs or concatenations

**Signal.** Several distinct call lines in one enclosing function issuing the
same collective with one identical payload shape. (The innermost attribution is
not enough on its own: a collective wrapped in a helper attributes to one line
inside that helper however often it runs, so the enclosing call lines are what
distinguish three adjacent exchanges from one exchange in a loop.)

**Before.**

```python
query = seq_parallel_all_to_all(query, group, scatter_dim=2, gather_dim=1)
key   = seq_parallel_all_to_all(key,   group, scatter_dim=2, gather_dim=1)
value = seq_parallel_all_to_all(value, group, scatter_dim=2, gather_dim=1)
```

**After.** One collective with three times the payload, guarded on the shapes
actually matching:

```python
if _env_on("HL_COMBINE_QKV") and query.shape == key.shape == value.shape:
    fused = seq_parallel_all_to_all(torch.cat([query, key, value], dim=0), group,
                          scatter_dim=2, gather_dim=1)
    query, key, value = torch.chunk(fused, 3, dim=0)
else:
    ...  # original three calls
```

The same shape applies to a chain of `torch.cat` calls building one buffer in
stages, and to several narrow GEMMs sharing an input (concatenate the weights
once, then one wide GEMM and a split).

### (e) `swap_vendor_kernel` — swap an operator implementation for a vendor kernel

Not host-observable; read it off the GPU kernel breakdown. Worth naming here
because it is frequently the single largest lever, and because it interacts with
the rest: a vendor attention kernel changes which host-side costs still matter.

Route the call site through the vendor entry point behind a switch, keep the
original as the fallback, and preserve the surrounding pad/unpad framing exactly.

### (f) `keep_device_resident` — keep a tensor resident on the device

**Signal.** Repeated host-to-device copies from a CPU-resident source.

**Before.** A table built on the host and uploaded at each use:

```python
def build_pos_tables(self, sizes):
    return build_rope_tables(sizes)  # CPU tensors; caller does .to(device)
```

**After.** Built once, cached on the device, keyed by the geometry that
determines it:

```python
def build_pos_tables(self, sizes, device=None):
    key = (tuple(sizes), self.dim_list, self.theta, str(device))
    hit = self._rope_cache.get(key) if _env_on("HL_HTOD_CACHE") else None
    if hit is None:
        hit = tuple(t.to(device) for t in build_rope_tables(sizes))
        if _env_on("HL_HTOD_CACHE"):
            self._rope_cache[key] = hit
    return hit
```

### (g) `drop_noop_glue` — drop no-op glue work

Not host-observable; found by reading the code. An intermediate materialised only
to be consumed by the next line, a broadcast by a factor of one, a cast to the
dtype the tensor already has. Individually tiny, but each one is multiplied by the
loop product.

**Check whether PyTorch already short-circuits it before spending a switch.**
Some of these operations return `self` when they have nothing to do, so
"eliminating" them buys nothing and adds a guard to the hot path for free. Others
allocate unconditionally. The difference is not guessable from the API, and it
decides whether a candidate is worth anything:

| operation | when it is already a no-op | verdict |
|---|---|---|
| `t.to(dtype)` / `t.type_as(x)` | dtype already matches | returns `self` — **nothing to win** |
| `t.to(device)` | already on that device | returns `self` — **nothing to win** |
| `t.contiguous()` | already contiguous | returns `self` — **nothing to win** |
| `t.repeat_interleave(1, dim=d)` | factor is 1 | **allocates a full copy** |
| `einops.repeat(t, ...)` | any expansion | **materialises a copy**, not a view |
| `torch.cat([t], dim=d)` | one-element list | **allocates a full copy** |
| `torch.split(t, full_width, dim=-1)` | single full-width split | returns a view — free |

So a `repeat_interleave` whose factor happens to be 1 on the measured operating
point, or a `torch.cat` over a list that happens to hold one element, is a real
find; a defensive `.contiguous()` next to it is not. Verify the specific call
rather than assuming either way, and state in the manifest evidence which it is.

Guard on the condition and skip:

```python
if not (_env_on("HL_GLUE_FUSE") and hidden.dtype == query.dtype):
    hidden = hidden.to(query.dtype)
```

## Cache-key recipe

Every (a), (c) and (f) rewrite is a cache, and a cache with an incomplete key is
a correctness bug that a throughput benchmark will happily accept.

1. **Key on the complete argument identity.** For a tensor:
   `(data_ptr, shape, dtype, device, _version)`. Plus every scalar that changes
   the result. A key missing one input returns another input's answer.
2. **Pin the source tensors in the entry.** Under a caching allocator a freed
   tensor's address goes straight back to the next allocation, so `data_ptr`
   alone reports a brand-new tensor as a hit. Holding a reference to the keyed
   tensors makes the address unreusable while the entry lives.
3. **Check `_version`** so an in-place mutation invalidates the entry.
4. **Never hash tensor contents.** That forces a device-to-host sync per call
   and costs more than the cache saves — and on a `(c)` rewrite it reintroduces
   exactly the stall being removed.
5. **Bound the cache and size it for the calling pattern.** A small LRU is
   enough. Under classifier-free guidance the positive and negative branches
   alternate, so a single-entry cache thrashes to a 0% hit rate; size for the
   number of interleaved callers.
6. **Include the chunk identity** when caching across a chunk boundary.
   Geometry alone repeats between chunks whose contents differ.

## Switch discipline

Every rewrite is gated by its own environment switch that defaults OFF, so an
environment with no switches set runs the original code path byte-for-byte.

This is enforced, not advisory:

- a **parity leg** runs with every switch unset and must reproduce the baseline
  within its noise band and pass the quality gate, so a rewrite that changes
  behaviour while disabled is rejected;
- accepted switches are registered as **search levers**, so each one's own
  contribution is measured and combinations are searched rather than the
  authored bundle being taken as given.

The payoff is that a patch carrying three rewrites where two help and one
regresses is no longer discarded whole, and every lever arrives with its own
number instead of a position in someone's ordering.

## Pitfalls

- **Graph capture conflicts with lazily populated caches.** The first call
  allocates inside the capture. Do not combine graph capture with these
  rewrites; disable it or populate the caches during warm-up.
- **Switch-name collisions.** A name that matches an upstream variable will be
  honoured by upstream code too. Namespace yours.
- **A cache that never invalidates across workloads.** Fine within one
  benchmark, wrong for a long-lived process. Key by workload geometry so a
  changed shape misses instead of returning a stale answer.
- **Measuring an enabler alone.** Covered above, and it is the most expensive
  mistake available here: it does not merely lose the enabler's own gain, it
  quietly removes the ceiling from everything downstream of it.
