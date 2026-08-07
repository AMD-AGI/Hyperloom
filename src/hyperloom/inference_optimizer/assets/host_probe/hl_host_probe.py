# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-side evidence probe for framework-level rewrite discovery.

Runs *inside* the benchmark process (injected by prepending this directory to
``PYTHONPATH``, so the sibling ``sitecustomize`` imports it at interpreter
start-up). Deliberately standalone: no ``hyperloom`` imports, no third-party
imports beyond an optional ``torch``, because the benchmark process usually
lives in a different virtualenv than the orchestrator.

Why this exists
---------------
The orchestrator's existing profile evidence is a GPU kernel-time breakdown. A
whole class of framework-level inefficiency is invisible there because it never
shows up as a kernel:

* a collective that does a host round-trip to agree on a shape
  (``all_gather_object`` on a Python int) burns wall time in RCCL rendezvous and
  host stalls, not in a nameable kernel;
* a pure function recomputed once per transformer block per denoising step costs
  host time plus a pile of small launches;
* a CPU-resident table re-uploaded on every use costs blocking H2D copies.

This probe measures those three shapes directly.

Two tiers
---------
**Tier 1** (``HYPERLOOM_HOST_PROBE=1``) wraps a fixed, small set of host-stall
and transfer entry points. Cost is proportional to how often those specific
APIs are called, which is negligible against a full benchmark, so tier 1 is
safe to run inside the ordinary profile leg alongside the torch profiler.

**Tier 2** (``HYPERLOOM_HOST_PROBE_DEEP=1``) installs a :func:`sys.setprofile`
hook that counts every call into the framework source roots and fingerprints
its arguments. This is what surfaces memoization and loop-hoisting candidates,
but it inflates host time enough to distort a co-collected torch trace, so it
defaults off and is meant for a dedicated evidence leg.

Argument fingerprints: strict and loose
---------------------------------------
Every sampled call records two fingerprints of its arguments:

* **strict** identifies each tensor as a *live object*, so a repeat means the
  very same tensor was passed again and the callee's result can be memoized
  as-is;
* **loose** keeps only shape, dtype and device, so a repeat means the argument
  looks like the same value in a possibly *freshly allocated* tensor.

A site with a high loose-repeat rate and a low strict-repeat rate is the
signature of a loop-invariant computation whose inputs are rebuilt every
iteration. Memoizing it alone gains nothing (the cache would never hit); the
allocation has to be hoisted out of the loop first, and only then does the
memoization pay. That pairing is otherwise a human insight, and getting it wrong
costs the whole bundle: a greedy accept/reject loop measures the hoist on its
own, sees no gain, and discards the enabler.

Why strict identity is not just ``data_ptr``
--------------------------------------------
Under a caching allocator, a tensor allocated and freed inside a loop gets the
same address back on the next iteration. Keying strict identity on ``data_ptr``
would therefore report a freshly built tensor as "the same argument as last
time" — inverting exactly the distinction above and reclassifying every hoist
candidate as a memoization candidate that would then never hit at runtime. This
is the same address-reuse hazard a hand-written cache has to guard against by
pinning its source tensors.

So strict identity carries a *generation* token instead: the probe holds a weak
reference per observed tensor object and issues a fresh generation whenever an
object address turns out to belong to a new object. Weak references keep this
from pinning device memory, which a probe must never do.

What the probe deliberately does not know
-----------------------------------------
It never reads tensor *contents*. Doing so would require a device-to-host copy
per sampled call, injecting the very host stalls the probe exists to measure. So
"loose repeat" means the arguments were shape-, dtype- and device-compatible, not
that they held equal values. Treat a hoist candidate as a lead to confirm against
the source, not as proof.

Output
------
One JSON file per process at
``$HYPERLOOM_HOST_PROBE_DIR/hl_host_probe_rank<N>_pid<PID>.json``, written at
interpreter exit. Reports are merged by the orchestrator-side aggregator, which
is the only consumer of this schema.

The pid is in the name because a rank is not one process. A launcher, a re-exec,
or any short-lived child inherits ``RANK`` from the environment and installs its
own probe; on a real 8-rank run a 0.8-second helper process with zero recorded
calls wrote last and destroyed rank 0's entire report. Since the aggregator globs
and merges, an extra near-empty file costs nothing, while an overwrite silently
loses a rank.

Environment
-----------
``HYPERLOOM_HOST_PROBE``
    ``1`` to enable tier 1. Anything else leaves the process untouched.
``HYPERLOOM_HOST_PROBE_DEEP``
    ``1`` to additionally enable tier 2.
``HYPERLOOM_HOST_PROBE_DIR``
    Output directory. The probe is inert when unset (nowhere to report).
``HYPERLOOM_HOST_PROBE_ROOTS``
    ``os.pathsep``-separated framework source roots. Call sites are attributed
    to the innermost frame under one of these roots, which is what makes the
    report point at framework code instead of at torch internals.
``HYPERLOOM_HOST_PROBE_MAX_SITES``
    Cap on distinct tracked sites per category (default 512).
``HYPERLOOM_HOST_PROBE_ARG_SAMPLES``
    Per-site cap on argument-fingerprint samples (default 256). Reading
    ``frame.f_locals`` materializes a dict, so sampling is what keeps tier 2
    bounded; a couple hundred calls is already far more than enough to
    establish a repeat rate.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
import weakref


SCHEMA = "hyperloom.host_probe/1"

_DEFAULT_MAX_SITES = 512
_DEFAULT_ARG_SAMPLES = 256

# Deepest stack walk when attributing a wrapped call to framework code. Torch
# puts a handful of frames between the framework and the wrapper; 64 clears that
# with room to spare while keeping the walk bounded on deep recursion.
_MAX_STACK_WALK = 64

# Longest string kept verbatim in a fingerprint. Longer values are truncated so
# a giant prompt string cannot blow up probe memory.
_MAX_STR_FINGERPRINT = 64

# Deepest container recursion when fingerprinting an argument.
_MAX_CONTAINER_DEPTH = 3

# Widest container walked element-wise when fingerprinting. Beyond this only the
# length is recorded, which keeps a long list of tensors from dominating cost.
_MAX_CONTAINER_WIDTH = 8

# Distinct enclosing frames retained per host-API site. Fusion detection needs
# only enough callers to see that more than one adjacent line reaches the same
# site with the same shapes; a hot site called from everywhere is not fusable.
_MAX_CALLERS_PER_SITE = 8

# Tensor objects tracked for strict-identity generations before the table is
# trimmed. The bound exists to stop unbounded growth in a long benchmark; how it
# is enforced decides whether the probe's central distinction survives, so see
# ``_evict_tracked_objects``.
_MAX_TRACKED_OBJECTS = 4096
# Trim down to this rather than to the cap, so eviction is amortised instead of
# running on nearly every insert once the table is full.
_EVICT_TRACKED_OBJECTS_TO = _MAX_TRACKED_OBJECTS // 2


def _env_on(name: str) -> bool:
    """Return True when environment variable ``name`` is set to a truthy token.

    Args:
        name: Environment variable to read.

    Returns:
        True for ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive).
    """
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """Return environment variable ``name`` as an int, or ``default``.

    Args:
        name: Environment variable to read.
        default: Value returned when unset, unparseable, or non-positive.

    Returns:
        The parsed positive int, else ``default``.
    """
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class _SiteStats:
    """Accumulator for one wrapped host-side API call site.

    Attributes:
        count: Number of calls observed.
        wall_s: Summed wall time spent inside the wrapped call.
        nbytes: Summed transferred bytes, when the API makes that knowable.
        shape_sigs: Distinct argument shape signatures seen, capped. Several
            call sites sharing a signature is the collective-fusion signal.
        callers: Distinct enclosing framework frames that reached this site,
            capped. A collective wrapped in a framework helper attributes to a
            single line inside that helper, so the *caller* is the only thing
            that distinguishes three adjacent same-shape collectives (the
            fusable case) from one collective in a loop.
        first_s: Probe-relative timestamp of the first call.
        last_s: Probe-relative timestamp of the last call. Together with
            ``first_s`` this separates one-time set-up work from hot-path work:
            loading a multi-gigabyte checkpoint issues hundreds of host-to-device
            copies, which otherwise outranks a genuine per-step inefficiency on
            call count alone. Set-up calls cluster at the start; hot-path calls
            span the run.
    """

    __slots__ = ("count", "wall_s", "nbytes", "shape_sigs", "callers", "first_s", "last_s")

    def __init__(self) -> None:
        self.count = 0
        self.wall_s = 0.0
        self.nbytes = 0
        self.shape_sigs: set[str] = set()
        self.callers: set[str] = set()
        self.first_s = -1.0
        self.last_s = -1.0

    def record(self, elapsed: float, nbytes: int, shape_sig: str, caller: str, at_s: float) -> None:
        """Fold one observation into the accumulator.

        Args:
            elapsed: Wall seconds spent in the call.
            nbytes: Transferred bytes, or 0 when unknown.
            shape_sig: Shape signature of the call's tensor arguments.
            caller: The enclosing framework frame, or ``""`` when there is none.
            at_s: Probe-relative timestamp of this call.
        """
        self.count += 1
        self.wall_s += elapsed
        self.nbytes += nbytes
        if self.first_s < 0:
            self.first_s = at_s
        self.last_s = at_s
        if shape_sig and len(self.shape_sigs) < _MAX_CONTAINER_WIDTH:
            self.shape_sigs.add(shape_sig)
        if caller and len(self.callers) < _MAX_CALLERS_PER_SITE:
            self.callers.add(caller)


class _CallStats:
    """Accumulator for one framework function observed by the tier-2 hook.

    Attributes:
        count: Number of calls observed.
        wall_s: Summed wall time of outermost invocations (recursive re-entry is
            folded into the outermost frame so time is not double counted).
        depth: Current recursion depth, used to identify the outermost frame.
        started: Start timestamp of the outermost active invocation.
        arg_samples: Number of calls whose arguments were fingerprinted.
        strict_sigs: Distinct strict fingerprints among the samples.
        loose_sigs: Distinct loose fingerprints among the samples.
        first_s: Probe-relative timestamp of the first outermost call.
        last_s: Probe-relative timestamp of the last outermost call. Carries the
            same set-up-versus-hot-path signal as :attr:`_SiteStats.last_s`, and
            tier 2 needs it just as much: a module constructor runs once per
            sub-module during model construction, which looks like a repeated
            argument identity and would otherwise rank as a memoization
            candidate. Recorded on the same timeline as the tier-1 sites so one
            activity end can be derived across both tables.
    """

    __slots__ = (
        "count",
        "wall_s",
        "depth",
        "started",
        "arg_samples",
        "strict_sigs",
        "loose_sigs",
        "first_s",
        "last_s",
    )

    def __init__(self) -> None:
        self.count = 0
        self.wall_s = 0.0
        self.depth = 0
        self.started = 0.0
        self.arg_samples = 0
        self.strict_sigs: set[int] = set()
        self.loose_sigs: set[int] = set()
        self.first_s = -1.0
        self.last_s = -1.0


class HostProbe:
    """Collects host-side rewrite evidence for one process.

    Install with :meth:`install`; the report is written at interpreter exit, or
    on demand via :meth:`write_report` (which tests use instead of relying on
    ``atexit``).
    """

    def __init__(
        self,
        *,
        out_dir: str,
        roots: "tuple[str, ...]",
        deep: bool = False,
        max_sites: int = _DEFAULT_MAX_SITES,
        arg_samples: int = _DEFAULT_ARG_SAMPLES,
    ) -> None:
        """Initialise a probe.

        Args:
            out_dir: Directory the per-rank JSON report is written to.
            roots: Framework source roots used to attribute call sites.
            deep: Enable the tier-2 ``sys.setprofile`` hook.
            max_sites: Cap on distinct tracked sites per category.
            arg_samples: Per-site cap on argument-fingerprint samples.
        """
        self.out_dir = out_dir
        self.roots = tuple(r for r in roots if r)
        self.deep = bool(deep)
        self.max_sites = int(max_sites)
        self.arg_samples = int(arg_samples)

        self._lock = threading.Lock()
        self._host_sites: dict[tuple[str, str], _SiteStats] = {}
        self._calls: dict[str, _CallStats] = {}
        # id(code) -> attributed "file:line:name", or "" when out of scope. Keyed
        # by id because code objects are unhashable-cheap but stable for the
        # process lifetime; the code object is pinned in the value tuple so the
        # id cannot be recycled underneath us.
        self._code_cache: dict[int, tuple[object, str]] = {}
        # id(tensor) -> (weakref, generation). The weakref is what makes a
        # recycled object address detectable; see the module docstring.
        self._object_generations: dict[int, tuple[weakref.ref, int]] = {}
        self._next_generation = 0
        self._host_sites_truncated = False
        self._calls_truncated = False
        self._patched: list[tuple[object, str, object]] = []
        self._installed = False
        # Whether the tier-2 hook is installed right now, versus whether it ever
        # ran. Teardown clears the first but not the second, so a report written
        # after teardown still says the deep data is present.
        self._deep_installed = False
        self._deep_collected = False
        self._started = time.time()
        # Same instant as ``_started``, on the monotonic clock. The tier-2 hook
        # already reads ``perf_counter`` to time an outermost call, so deriving
        # its timestamp from this baseline costs nothing, and both probe tiers
        # end up measuring elapsed seconds from probe init.
        self._perf_started = time.perf_counter()
        self._report_written = False
        self._notes: list[str] = []

    # -- attribution ------------------------------------------------------

    def _under_roots(self, filename: str) -> bool:
        """Return True when ``filename`` lives under one of the framework roots.

        Args:
            filename: Absolute or relative source path from a code object.

        Returns:
            True when the path is inside a configured root. With no roots
            configured every path qualifies, so the probe still reports
            something rather than silently producing an empty file.
        """
        if not self.roots:
            return True
        return any(filename.startswith(root) for root in self.roots)

    def _label_code(self, code: object) -> str:
        """Return the cached ``file:line:name`` label for ``code``, or ``""``.

        Args:
            code: A code object.

        Returns:
            The label when the code object is under a framework root, else an
            empty string. Results are memoized per code object because this
            runs on every observed call.
        """
        key = id(code)
        hit = self._code_cache.get(key)
        if hit is not None:
            return hit[1]
        filename = str(getattr(code, "co_filename", "") or "")
        if not filename or not self._under_roots(filename):
            label = ""
        else:
            display = self._relative(filename)
            lineno = int(getattr(code, "co_firstlineno", 0) or 0)
            name = str(getattr(code, "co_name", "") or "?")
            label = f"{display}:{lineno}:{name}"
        self._code_cache[key] = (code, label)
        return label

    def _relative(self, filename: str) -> str:
        """Shorten ``filename`` to a path relative to its framework root.

        Args:
            filename: Absolute source path.

        Returns:
            The root-relative path when a root matches, else ``filename``.
        """
        for root in self.roots:
            if filename.startswith(root):
                return filename[len(root) :].lstrip("/")
        return filename

    def _caller_context(self) -> "tuple[str, str]":
        """Attribute the current wrapped call to framework frames on the stack.

        Walks outward from the wrapper until a frame's code object is under a
        framework root, so the report names framework code rather than the torch
        internals that sit between, then keeps walking for one more framework
        frame to identify the enclosing call site.

        Returns:
            ``(site, caller)`` where ``site`` is the innermost framework frame as
            ``file:line:name`` and ``caller`` is the next framework frame out
            (``""`` when there is none). ``site`` is ``"<outside-roots>"`` when
            no framework frame is on the stack at all.
        """
        site = ""
        depth = 2
        while depth < _MAX_STACK_WALK:
            try:
                frame = sys._getframe(depth)
            except ValueError:
                break
            code = frame.f_code
            if self._label_code(code):
                display = self._relative(str(code.co_filename))
                label = f"{display}:{frame.f_lineno}:{code.co_name}"
                if not site:
                    site = label
                else:
                    return site, label
            depth += 1
        return (site or "<outside-roots>"), ""

    # -- fingerprinting ---------------------------------------------------

    def _object_generation(self, value: object) -> int:
        """Return a token identifying ``value`` as a distinct live object.

        Two different objects never share a token, even when a caching allocator
        hands the second one the first one's address, because the weak reference
        stored alongside the token reveals that the original object is gone.

        Args:
            value: The object to identify.

        Returns:
            A generation token, or ``-1`` when the object does not support weak
            references (in which case its address is the best available identity
            and the caller falls back to it).
        """
        key = id(value)
        hit = self._object_generations.get(key)
        if hit is not None and hit[0]() is value:
            # Re-insert to mark it as recently seen; dicts keep insertion order,
            # which is what makes eviction below drop the stalest entries.
            del self._object_generations[key]
            self._object_generations[key] = hit
            return hit[1]
        try:
            ref = weakref.ref(value)
        except TypeError:
            return -1
        if len(self._object_generations) >= _MAX_TRACKED_OBJECTS:
            self._evict_tracked_objects()
        self._next_generation += 1
        self._object_generations[key] = (ref, self._next_generation)
        return self._next_generation

    def _evict_tracked_objects(self) -> None:
        """Make room in the generation table without forgetting live objects.

        Clearing the table wholesale costs precisely what the table exists to
        record. A tensor that is still alive and still being passed loses its
        generation, so its next appearance reads as a brand-new object: strict
        repeats disappear while loose repeats do not, and a memoization
        candidate is reported as a hoist candidate. That is the one
        misclassification this probe must not make, and on a long run -- a
        diffusion rollout allocates intermediates continuously -- a wholesale
        clear fires again and again rather than once.

        Dead entries go first. A tensor freed inside the loop can never be
        recognised again whatever we keep, and under allocation churn those are
        most of the table, so this alone is usually enough. Only if it is not do
        the least recently seen live entries go, oldest first.
        """
        for key in [k for k, (ref, _gen) in self._object_generations.items() if ref() is None]:
            del self._object_generations[key]
        overflow = len(self._object_generations) - _EVICT_TRACKED_OBJECTS_TO
        if overflow <= 0:
            return
        for key in list(self._object_generations)[:overflow]:
            del self._object_generations[key]

    def _fingerprint(self, value: object, *, strict: bool, depth: int = 0) -> object:
        """Build a hashable fingerprint of one argument value.

        Args:
            value: The argument to fingerprint.
            strict: When True a tensor is identified as a live object, so only
                the very same tensor fingerprints equal; when False only shape,
                dtype and device are kept, so a freshly allocated tensor of the
                same geometry fingerprints equal.
            depth: Current container recursion depth.

        Returns:
            A hashable fingerprint. Unknown types degrade to their type name,
            which keeps the fingerprint total rather than raising.
        """
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:_MAX_STR_FINGERPRINT]
        if isinstance(value, bytes):
            return ("bytes", len(value))
        shape = getattr(value, "shape", None)
        if shape is not None and hasattr(value, "dtype") and hasattr(value, "device"):
            base: tuple = (
                "T",
                tuple(shape),
                str(getattr(value, "dtype", "")),
                str(getattr(value, "device", "")),
            )
            if not strict:
                return base
            generation = self._object_generation(value)
            if generation < 0:
                # No weak reference available; fall back to the address, which
                # cannot distinguish a recycled allocation but is all there is.
                try:
                    generation = -int(value.data_ptr())
                except Exception:  # noqa: BLE001 - meta/fake tensors have no storage
                    generation = -1
            return base + (generation, int(getattr(value, "_version", -1) or -1))
        if isinstance(value, (tuple, list)) and depth < _MAX_CONTAINER_DEPTH:
            if len(value) > _MAX_CONTAINER_WIDTH:
                return (type(value).__name__, len(value))
            return tuple(self._fingerprint(item, strict=strict, depth=depth + 1) for item in value)
        return type(value).__name__

    def _shape_signature(self, values: "tuple[object, ...]") -> str:
        """Summarise the tensor shapes among ``values`` as a stable string.

        Args:
            values: Positional argument values.

        Returns:
            A compact ``shape@dtype`` signature joined by ``|``, or ``""`` when
            no argument looks like a tensor. Several collectives issued from one
            enclosing function under the same signature is the fusion signal.
        """
        parts: list[str] = []
        for value in values[:_MAX_CONTAINER_WIDTH]:
            shape = getattr(value, "shape", None)
            if shape is None or not hasattr(value, "dtype"):
                continue
            parts.append(f"{tuple(shape)}@{getattr(value, 'dtype', '')}")
        return "|".join(parts)

    # -- tier 1 -----------------------------------------------------------

    def _record_host_call(
        self,
        api: str,
        elapsed: float,
        *,
        nbytes: int = 0,
        shape_sig: str = "",
    ) -> None:
        """Fold one wrapped host-API observation into the tier-1 table.

        Args:
            api: Dotted API name, e.g. ``torch.distributed.all_gather_object``.
            elapsed: Wall seconds spent in the call.
            nbytes: Transferred bytes, or 0 when unknown.
            shape_sig: Shape signature of the call's tensor arguments.
        """
        site, caller = self._caller_context()
        key = (api, site)
        with self._lock:
            stats = self._host_sites.get(key)
            if stats is None:
                if len(self._host_sites) >= self.max_sites:
                    self._host_sites_truncated = True
                    return
                stats = _SiteStats()
                self._host_sites[key] = stats
            stats.record(elapsed, nbytes, shape_sig, caller, time.time() - self._started)

    def _wrap(self, owner: object, attr: str, api: str, *, measure_bytes: bool = False) -> None:
        """Replace ``owner.attr`` with a timing wrapper, if it exists.

        Args:
            owner: Module or type holding the attribute.
            attr: Attribute name to wrap.
            api: Dotted API name recorded in the report.
            measure_bytes: Also attribute ``nbytes`` of the first tensor
                argument to the call.
        """
        try:
            original = getattr(owner, attr)
        except AttributeError:
            return
        if not callable(original) or getattr(original, "_hl_host_probe", False):
            return
        probe = self

        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                nbytes = 0
                if measure_bytes:
                    for value in args:
                        raw = getattr(value, "nbytes", None)
                        if isinstance(raw, int):
                            nbytes = raw
                            break
                try:
                    probe._record_host_call(
                        api,
                        elapsed,
                        nbytes=nbytes,
                        shape_sig=probe._shape_signature(args),
                    )
                except Exception:  # noqa: BLE001 - never break the benchmark
                    pass

        wrapper._hl_host_probe = True  # type: ignore[attr-defined]
        try:
            setattr(owner, attr, wrapper)
        except (AttributeError, TypeError):
            self._notes.append(f"cannot wrap {api} (immutable attribute)")
            return
        self._patched.append((owner, attr, original))

    def _wrap_h2d(self, tensor_type: object) -> None:
        """Wrap ``Tensor.to`` / ``Tensor.cuda`` to count host-to-device copies.

        ``.to`` is one of the hottest methods in any model, so the wrapper takes
        a single-attribute fast path out for tensors that are already on the
        device: only a CPU-resident source can produce an H2D copy, and that is
        the case worth reporting (a table rebuilt on the host and re-uploaded on
        every use).

        Args:
            tensor_type: The ``torch.Tensor`` type to patch.
        """
        probe = self

        for attr, api in (("to", "torch.Tensor.to"), ("cuda", "torch.Tensor.cuda")):
            try:
                original = getattr(tensor_type, attr)
            except AttributeError:
                continue
            if getattr(original, "_hl_host_probe", False):
                continue

            def make(original=original, api=api):  # noqa: ANN001, ANN202
                def wrapper(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
                    if self.is_cuda:
                        return original(self, *args, **kwargs)
                    started = time.perf_counter()
                    result = original(self, *args, **kwargs)
                    if getattr(result, "is_cuda", False):
                        try:
                            probe._record_host_call(
                                api,
                                time.perf_counter() - started,
                                nbytes=int(getattr(self, "nbytes", 0) or 0),
                                shape_sig=probe._shape_signature((self,)),
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    return result

                wrapper._hl_host_probe = True  # type: ignore[attr-defined]
                return wrapper

            wrapper = make()
            try:
                setattr(tensor_type, attr, wrapper)
            except (AttributeError, TypeError):
                self._notes.append(f"cannot wrap {api} (immutable attribute)")
                continue
            self._patched.append((tensor_type, attr, original))

    def _install_tier1(self) -> None:
        """Wrap the host-stall and transfer entry points, if torch is present."""
        try:
            import torch
        except Exception:  # noqa: BLE001 - a torch-less process has nothing to probe
            self._notes.append("torch unavailable; tier 1 inert")
            return

        dist = getattr(torch, "distributed", None)
        if dist is not None:
            # Object collectives pickle through the host, so each one is a
            # host round-trip; tensor collectives are recorded too so a
            # rendezvous can be told apart from real payload movement.
            for attr in (
                "all_gather_object",
                "gather_object",
                "scatter_object_list",
                "broadcast_object_list",
                "all_gather",
                "all_gather_into_tensor",
                "all_to_all",
                "all_to_all_single",
                "all_reduce",
                "reduce_scatter_tensor",
                "broadcast",
                "barrier",
            ):
                self._wrap(dist, attr, f"torch.distributed.{attr}")

        tensor_type = getattr(torch, "Tensor", None)
        if tensor_type is not None:
            for attr in ("item", "tolist", "cpu", "numpy"):
                self._wrap(tensor_type, attr, f"torch.Tensor.{attr}")
            # The implicit conversions. A device-to-host read does not have to be
            # spelled `.item()`: `if scalar_tensor == 0`, `float(t)`, `int(t)` and
            # using a tensor as an index all read device memory back through these
            # dunders, and they are the syncs a reviewer is least likely to spot in
            # the source precisely because nothing in the line looks like a
            # transfer. Note the boundary: a tensor passed where the C++ argument
            # parser wants a Scalar (`torch.full((n,), t)`) converts inside ATen
            # without calling any Python method, so it stays invisible here.
            for attr in ("__float__", "__int__", "__bool__", "__index__"):
                self._wrap(tensor_type, attr, f"torch.Tensor.{attr}")
            self._wrap_h2d(tensor_type)

        cuda = getattr(torch, "cuda", None)
        if cuda is not None:
            self._wrap(cuda, "synchronize", "torch.cuda.synchronize")

    # -- tier 2 -----------------------------------------------------------

    def _profile_hook(self, frame, event: str, _arg) -> None:  # noqa: ANN001
        """``sys.setprofile`` callback counting framework calls and arg repeats.

        Args:
            frame: The frame the event belongs to.
            event: One of the ``sys.setprofile`` event names.
            _arg: Event payload (unused).
        """
        if event != "call" and event != "return":
            return
        try:
            label = self._label_code(frame.f_code)
            if not label:
                return
            stats = self._calls.get(label)
            if stats is None:
                if len(self._calls) >= self.max_sites:
                    self._calls_truncated = True
                    return
                stats = _CallStats()
                self._calls[label] = stats
            if event == "call":
                stats.count += 1
                if stats.depth == 0:
                    stats.started = time.perf_counter()
                    at_s = stats.started - self._perf_started
                    if stats.first_s < 0:
                        stats.first_s = at_s
                    stats.last_s = at_s
                stats.depth += 1
                if stats.arg_samples < self.arg_samples:
                    self._sample_args(frame, stats)
            else:
                stats.depth -= 1
                if stats.depth <= 0:
                    stats.depth = 0
                    if stats.started:
                        stats.wall_s += time.perf_counter() - stats.started
                        stats.started = 0.0
        except Exception:  # noqa: BLE001 - a profile hook must never raise
            return

    def _sample_args(self, frame, stats: _CallStats) -> None:  # noqa: ANN001
        """Fingerprint one call's positional arguments into ``stats``.

        Args:
            frame: The frame whose arguments are sampled.
            stats: Accumulator for the frame's code object.
        """
        code = frame.f_code
        argcount = int(getattr(code, "co_argcount", 0) or 0)
        if argcount <= 0:
            # A zero-argument callable has nothing to key a cache on, so record
            # the sample without a fingerprint rather than inventing one.
            stats.arg_samples += 1
            return
        names = code.co_varnames[:argcount]
        local_vars = frame.f_locals
        values = tuple(local_vars.get(name) for name in names)
        # Drop a bound receiver: `self` differs per module instance but is not
        # what makes a computation repeat, and keeping it would mask the repeat.
        if names and names[0] in ("self", "cls"):
            values = values[1:]
        stats.arg_samples += 1
        try:
            stats.strict_sigs.add(hash(self._fingerprint(values, strict=True)))
            stats.loose_sigs.add(hash(self._fingerprint(values, strict=False)))
        except TypeError:
            # An unhashable argument makes this call unfingerprintable; the
            # sample still counts so the repeat rate stays honest.
            pass

    def _install_tier2(self) -> None:
        """Install the tier-2 profile hook on the current and future threads."""
        existing = sys.getprofile()
        if existing is not None:
            self._notes.append(
                "sys.setprofile was already in use (cProfile or a torch "
                "with_stack profiler); tier 2 skipped to avoid displacing it"
            )
            return
        sys.setprofile(self._profile_hook)
        threading.setprofile(self._profile_hook)
        self._deep_installed = True
        self._deep_collected = True

    # -- lifecycle --------------------------------------------------------

    def install(self) -> "HostProbe":
        """Install the probe and register the exit-time report writer.

        Returns:
            This probe, so callers can chain.
        """
        if self._installed:
            return self
        self._installed = True
        self._install_tier1()
        if self.deep:
            self._install_tier2()
        atexit.register(self.write_report)
        return self

    def uninstall(self) -> None:
        """Restore every patched attribute and remove the tier-2 hook."""
        if self._deep_installed:
            sys.setprofile(None)
            threading.setprofile(None)  # type: ignore[arg-type]
            self._deep_installed = False
        while self._patched:
            owner, attr, original = self._patched.pop()
            try:
                setattr(owner, attr, original)
            except (AttributeError, TypeError):
                pass
        self._installed = False

    def _rank(self) -> int:
        """Return this process's distributed rank from the launcher env.

        Returns:
            The rank from ``RANK`` / ``LOCAL_RANK`` / ``OMPI_COMM_WORLD_RANK``,
            else 0.
        """
        for name in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
            try:
                return int(str(os.environ.get(name, "")).strip())
            except (TypeError, ValueError):
                continue
        return 0

    def _world_size(self) -> int:
        """Return the distributed world size from the launcher env.

        Returns:
            The world size from ``WORLD_SIZE`` / ``OMPI_COMM_WORLD_SIZE``, else 1.
        """
        for name in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE"):
            try:
                return int(str(os.environ.get(name, "")).strip())
            except (TypeError, ValueError):
                continue
        return 1

    def report(self) -> dict:
        """Build this process's evidence report.

        Returns:
            The report dict written to disk; see the module docstring for the
            schema and the aggregator for its interpretation.
        """
        host_calls = [
            {
                "api": api,
                "site": site,
                "count": stats.count,
                "wall_s": round(stats.wall_s, 6),
                "bytes": stats.nbytes,
                "shape_sigs": sorted(stats.shape_sigs),
                "callers": sorted(stats.callers),
                "first_s": round(stats.first_s, 3),
                "last_s": round(stats.last_s, 3),
            }
            for (api, site), stats in self._host_sites.items()
        ]
        host_calls.sort(key=lambda row: row["wall_s"], reverse=True)

        framework_calls = []
        for label, stats in self._calls.items():
            samples = stats.arg_samples
            strict_distinct = len(stats.strict_sigs)
            loose_distinct = len(stats.loose_sigs)
            framework_calls.append(
                {
                    "function": label,
                    "count": stats.count,
                    "wall_s": round(stats.wall_s, 6),
                    "arg_samples": samples,
                    "strict_distinct": strict_distinct,
                    "loose_distinct": loose_distinct,
                    "strict_repeat_rate": _repeat_rate(samples, strict_distinct),
                    "loose_repeat_rate": _repeat_rate(samples, loose_distinct),
                    "first_s": round(stats.first_s, 3),
                    "last_s": round(stats.last_s, 3),
                }
            )
        framework_calls.sort(key=lambda row: row["wall_s"], reverse=True)

        return {
            "schema": SCHEMA,
            "rank": self._rank(),
            "world_size": self._world_size(),
            "pid": os.getpid(),
            "wall_seconds": round(time.time() - self._started, 3),
            "roots": list(self.roots),
            "roots_unset": not self.roots,
            "tier1_enabled": True,
            "tier2_enabled": self._deep_collected,
            "host_calls": host_calls,
            "framework_calls": framework_calls,
            "truncated": {
                "host_calls": self._host_sites_truncated,
                "framework_calls": self._calls_truncated,
            },
            "notes": list(self._notes),
        }

    def write_report(self) -> str:
        """Write this process's report to the output directory.

        Idempotent, because it is registered with ``atexit`` and may also be
        called explicitly. Never raises: a probe that cannot report must not
        turn a good benchmark into a failed one.

        Returns:
            The written path, or ``""`` when nothing was written.
        """
        if self._report_written:
            return ""
        self._report_written = True
        # Stop collecting before serialising so the tier-2 hook cannot observe
        # the report builder itself and mutate the dict mid-iteration.
        if self._deep_installed:
            sys.setprofile(None)
            self._deep_installed = False
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(
                self.out_dir,
                f"hl_host_probe_rank{self._rank()}_pid{os.getpid()}.json",
            )
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.report(), handle, indent=2, sort_keys=True)
            return path
        except Exception as exc:  # noqa: BLE001 - reporting is best-effort
            sys.stderr.write(f"[hl_host_probe] could not write report: {exc!r}\n")
            return ""


def _repeat_rate(samples: int, distinct: int) -> float:
    """Return the fraction of sampled calls that repeated an earlier signature.

    Args:
        samples: Number of fingerprinted calls.
        distinct: Number of distinct fingerprints among them.

    Returns:
        ``1 - distinct/samples`` rounded to four places; 0.0 when there are no
        samples or no fingerprints (an unhashable or zero-argument callable).
    """
    if samples <= 0 or distinct <= 0:
        return 0.0
    return round(max(0.0, 1.0 - (distinct / samples)), 4)


_ACTIVE: "HostProbe | None" = None


def active() -> "HostProbe | None":
    """Return the installed probe, or ``None``.

    Returns:
        The process-wide probe instance when :func:`install_from_env` installed
        one, else ``None``.
    """
    return _ACTIVE


def install_from_env() -> "HostProbe | None":
    """Install a probe from the environment, or do nothing.

    Called by the sibling ``sitecustomize`` at interpreter start-up. Inert
    unless ``HYPERLOOM_HOST_PROBE`` is truthy and ``HYPERLOOM_HOST_PROBE_DIR``
    names an output directory, so an ordinary benchmark that merely has this
    directory on ``PYTHONPATH`` is untouched.

    Returns:
        The installed probe, or ``None`` when the environment did not ask for
        one or installation failed.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        return _ACTIVE
    if not _env_on("HYPERLOOM_HOST_PROBE"):
        return None
    out_dir = str(os.environ.get("HYPERLOOM_HOST_PROBE_DIR", "") or "").strip()
    if not out_dir:
        return None
    roots = tuple(
        os.path.abspath(part) + "/"
        for part in str(os.environ.get("HYPERLOOM_HOST_PROBE_ROOTS", "") or "").split(os.pathsep)
        if part.strip()
    )
    try:
        probe = HostProbe(
            out_dir=out_dir,
            roots=roots,
            deep=_env_on("HYPERLOOM_HOST_PROBE_DEEP"),
            max_sites=_env_int("HYPERLOOM_HOST_PROBE_MAX_SITES", _DEFAULT_MAX_SITES),
            arg_samples=_env_int("HYPERLOOM_HOST_PROBE_ARG_SAMPLES", _DEFAULT_ARG_SAMPLES),
        ).install()
    except Exception as exc:  # noqa: BLE001 - installation must never break the run
        sys.stderr.write(f"[hl_host_probe] install failed: {exc!r}\n")
        return None
    _ACTIVE = probe
    return probe


__all__ = ["SCHEMA", "HostProbe", "active", "install_from_env"]
