# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Single-process `torch.distributed` shim for native Windows + ROCm.

AMD/TheRock PyTorch wheels for Windows are built with USE_DISTRIBUTED=0: the
`torch._C._distributed_c10d` C extension is absent, so `torch.distributed` is gutted
(no Store/Backend/ReduceOp/init_process_group/collectives; the distributed_c10d,
_functional_collectives and _symmetric_memory submodules fail to import). vLLM needs
`torch.distributed` even for single-GPU (it constructs process groups), although every
collective short-circuits when world_size == 1.

This shim injects a minimal, single-process implementation of the surface vLLM uses
directly into the `torch.distributed` module object and pre-registers stub submodules in
sys.modules. Semantics: world_size=1, rank=0, all collectives are identity/no-ops. It does
NOT provide real multi-process communication (that needs RCCL/gloo, unavailable here).

Apply as early as possible (a .pth startup hook calls apply() before any `import vllm`).
"""
import os
import sys
import types
from datetime import timedelta

_SENTINEL = "_VLLM_WIN_SHIM"


def apply() -> None:
    import torch
    import torch.distributed as dist

    if getattr(dist, _SENTINEL, False):
        return  # idempotent

    # ---- core value types ----
    class RedOpType:
        pass

    class ReduceOp:
        SUM = 0
        AVG = 1
        PRODUCT = 2
        MIN = 3
        MAX = 4
        BAND = 5
        BOR = 6
        BXOR = 7
        PREMUL_SUM = 8

        def __init__(self, op=SUM):
            self.op = op

    ReduceOp.RedOpType = RedOpType

    class Backend(str):
        GLOO = "gloo"
        NCCL = "nccl"
        MPI = "mpi"
        UCC = "ucc"
        UNDEFINED = "undefined"

        def __new__(cls, name="undefined"):
            return super().__new__(cls, str(name).lower())

        @classmethod
        def register_backend(cls, name, func=None, extended_api=False, devices=None):
            return None

    class Work:
        def wait(self, *a, **k):
            return True

        def is_completed(self):
            return True

        def is_success(self):
            return True

        def exception(self):
            return None

        def get_future(self):
            fut = torch.futures.Future()
            fut.set_result(None)
            return fut

    # ---- stores (in-memory) ----
    class Store:
        def __init__(self, *a, **k):
            self._data = {}

        def set(self, key, value):
            if isinstance(value, str):
                value = value.encode()
            self._data[key] = value

        def get(self, key):
            return self._data.get(key, b"")

        def add(self, key, amount):
            cur = int(self._data.get(key, b"0") or b"0")
            cur += int(amount)
            self._data[key] = str(cur).encode()
            return cur

        def compare_set(self, key, expected, desired):
            self._data[key] = desired.encode() if isinstance(desired, str) else desired
            return self._data[key]

        def wait(self, *a, **k):
            return None

        def num_keys(self):
            return len(self._data)

        def delete_key(self, key):
            self._data.pop(key, None)
            return True

        def set_timeout(self, *a, **k):
            return None

    class PrefixStore(Store):
        def __init__(self, prefix="", store=None, *a, **k):
            super().__init__()
            self.prefix = prefix
            self.underlying_store = store

    class TCPStore(Store):
        def __init__(self, *a, **k):
            super().__init__()

    class FileStore(Store):
        def __init__(self, *a, **k):
            super().__init__()

    class HashStore(Store):
        pass

    # ---- dummy process group ----
    class _DummyGroup:
        def __init__(self, size=1, name="vllm_win_shim_pg"):
            self._size = size
            self._name = name
            self.world_size = size  # vLLM accesses pg.world_size directly (StatelessPG-style)
            self.rank_in_group = 0

        def size(self):
            return self._size

        def rank(self):
            return 0

        def name(self):
            return self._name

        def __len__(self):
            return self._size

    class GroupMember:
        WORLD = None
        NON_GROUP_MEMBER = -100

    _DEFAULT_GROUP = _DummyGroup(1, "world")
    GroupMember.WORLD = _DEFAULT_GROUP

    class _GroupNS:
        WORLD = None

    _GroupNS.WORLD = _DEFAULT_GROUP

    _state = {"initialized": False, "world_size": 1, "rank": 0}

    # ---- query / lifecycle ----
    def is_available():
        return True

    def is_initialized():
        return _state["initialized"]

    def is_backend_available(backend=None):
        return True

    def is_gloo_available():
        return True

    def is_nccl_available():
        return False

    def is_mpi_available():
        return False

    def is_ucc_available():
        return False

    def is_torchelastic_launched():
        return False

    def init_process_group(*a, **k):
        _state["initialized"] = True
        _state["world_size"] = int(k.get("world_size", 1) or 1)
        _state["rank"] = int(k.get("rank", 0) or 0)
        return None

    def destroy_process_group(group=None):
        if group is None:
            _state["initialized"] = False
        return None

    def get_world_size(group=None):
        if group is not None and hasattr(group, "size"):
            try:
                return group.size()
            except Exception:
                pass
        return _state["world_size"]

    def get_rank(group=None):
        return _state["rank"]

    def get_backend(group=None):
        return Backend.GLOO

    def new_group(ranks=None, *a, **k):
        n = len(ranks) if ranks is not None else _state["world_size"]
        return _DummyGroup(n)

    def new_subgroups(*a, **k):
        return _DummyGroup(1), [_DummyGroup(1)]

    def get_global_rank(group, group_rank):
        return group_rank

    def get_group_rank(group, global_rank):
        return global_rank

    def get_process_group_ranks(group=None):
        return list(range(get_world_size(group)))

    def _get_default_group():
        return _DEFAULT_GROUP

    def get_default_group():
        return _DEFAULT_GROUP

    def barrier(*a, **k):
        return None

    def monitored_barrier(*a, **k):
        return None

    # ---- collectives: identity / no-op (world_size == 1) ----
    def _ret(async_op=False, **_):
        return Work() if async_op else None

    def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
        return Work() if async_op else None

    def reduce(tensor, dst=0, op=ReduceOp.SUM, group=None, async_op=False):
        return Work() if async_op else None

    def broadcast(tensor, src=0, group=None, async_op=False):
        return Work() if async_op else None

    def broadcast_object_list(object_list, src=0, group=None, device=None):
        return object_list

    def all_gather(tensor_list, tensor, group=None, async_op=False):
        if tensor_list:
            tensor_list[0].copy_(tensor)
        return Work() if async_op else None

    def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
        output_tensor.copy_(input_tensor)
        return Work() if async_op else None

    def all_gather_object(object_list, obj, group=None):
        if object_list:
            object_list[0] = obj
        return None

    def gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
        if gather_list:
            gather_list[0].copy_(tensor)
        return Work() if async_op else None

    def gather_object(obj, object_gather_list=None, dst=0, group=None):
        if object_gather_list is not None and len(object_gather_list) > 0:
            object_gather_list[0] = obj
        return None

    def scatter(tensor, scatter_list=None, src=0, group=None, async_op=False):
        if scatter_list:
            tensor.copy_(scatter_list[0])
        return Work() if async_op else None

    def scatter_object_list(scatter_object_output_list, scatter_object_input_list=None,
                            src=0, group=None):
        if scatter_object_output_list and scatter_object_input_list:
            scatter_object_output_list[0] = scatter_object_input_list[0]
        return None

    def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
        if input_list:
            output.copy_(input_list[0])
        return Work() if async_op else None

    def reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False):
        output.copy_(input)
        return Work() if async_op else None

    def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
        for o, i in zip(output_tensor_list, input_tensor_list):
            o.copy_(i)
        return Work() if async_op else None

    def all_to_all_single(output, input, *a, **k):
        output.copy_(input)
        return Work() if k.get("async_op") else None

    def send(tensor, dst=0, group=None, tag=0):
        return None

    def recv(tensor, src=None, group=None, tag=0):
        return 0

    def isend(tensor, dst=0, group=None, tag=0):
        return Work()

    def irecv(tensor, src=None, group=None, tag=0):
        return Work()

    class P2POp:
        def __init__(self, op=None, tensor=None, peer=0, group=None, tag=0):
            self.op = op
            self.tensor = tensor
            self.peer = peer
            self.group = group
            self.tag = tag

    def batch_isend_irecv(p2p_op_list):
        return [Work() for _ in (p2p_op_list or [])]

    # ---- install onto torch.distributed ----
    api = dict(
        ReduceOp=ReduceOp, Backend=Backend, Work=Work,
        Store=Store, PrefixStore=PrefixStore, TCPStore=TCPStore,
        FileStore=FileStore, HashStore=HashStore,
        GroupMember=GroupMember, group=_GroupNS(),
        is_available=is_available, is_initialized=is_initialized,
        is_backend_available=is_backend_available, is_gloo_available=is_gloo_available,
        is_nccl_available=is_nccl_available, is_mpi_available=is_mpi_available,
        is_ucc_available=is_ucc_available, is_torchelastic_launched=is_torchelastic_launched,
        init_process_group=init_process_group, destroy_process_group=destroy_process_group,
        get_world_size=get_world_size, get_rank=get_rank, get_backend=get_backend,
        new_group=new_group, new_subgroups=new_subgroups,
        get_global_rank=get_global_rank, get_group_rank=get_group_rank,
        get_process_group_ranks=get_process_group_ranks,
        _get_default_group=_get_default_group, get_default_group=get_default_group,
        barrier=barrier, monitored_barrier=monitored_barrier,
        all_reduce=all_reduce, reduce=reduce, broadcast=broadcast,
        broadcast_object_list=broadcast_object_list, all_gather=all_gather,
        all_gather_into_tensor=all_gather_into_tensor, all_gather_object=all_gather_object,
        gather=gather, gather_object=gather_object, scatter=scatter,
        scatter_object_list=scatter_object_list, reduce_scatter=reduce_scatter,
        reduce_scatter_tensor=reduce_scatter_tensor, all_to_all=all_to_all,
        all_to_all_single=all_to_all_single, send=send, recv=recv, isend=isend, irecv=irecv,
        P2POp=P2POp, batch_isend_irecv=batch_isend_irecv,
    )
    # Only fill in MISSING symbols. Crucially we do NOT override torch's native
    # is_available() (which returns False on this USE_DISTRIBUTED=0 build): flipping it to
    # True makes torch's own code (e.g. _dynamo -> fsdp -> _shard) try to import the full
    # distributed stack that doesn't exist here. vLLM relies on is_initialized(), which is
    # absent natively, so we still provide that.
    for k, v in api.items():
        if not hasattr(dist, k):
            setattr(dist, k, v)
    # keep real ProcessGroup if present; otherwise use the dummy
    if not hasattr(dist, "ProcessGroup"):
        dist.ProcessGroup = _DummyGroup

    # misc private helpers vLLM imports from distributed_c10d
    dist._get_default_timeout = lambda backend=None: timedelta(seconds=1800)
    dist._unregister_process_group = lambda group_name=None: None
    dist._register_process_group = lambda *a, **k: None
    dist._world = types.SimpleNamespace(default_pg=_DEFAULT_GROUP)

    # ---- stub submodules that import the missing C extension ----
    c10d = types.ModuleType("torch.distributed.distributed_c10d")
    c10d._world = types.SimpleNamespace(default_pg=_DEFAULT_GROUP)
    # Delegate every other name to the (now-populated) torch.distributed module, so any
    # `from torch.distributed.distributed_c10d import X` resolves to our shim symbol.
    c10d.__getattr__ = lambda name: getattr(dist, name)
    sys.modules["torch.distributed.distributed_c10d"] = c10d
    dist.distributed_c10d = c10d

    # vLLM also references the C-extension module `torch._C._distributed_c10d` directly
    # (e.g. type annotations like `store: torch._C._distributed_c10d.Store`). That extension
    # is absent on this build; provide a stand-in that delegates to our shim symbols.
    cmod = types.ModuleType("torch._C._distributed_c10d")
    for _n in ("Store", "PrefixStore", "TCPStore", "FileStore", "HashStore",
               "ProcessGroup", "Backend", "ReduceOp", "Work", "GroupMember"):
        try:
            setattr(cmod, _n, getattr(dist, _n))
        except Exception:
            pass
    # For names we DON'T explicitly provide, raise ModuleNotFoundError (not AttributeError).
    # torch's own optional code (e.g. _dynamo/_inductor -> fsdp -> fake_pg does
    # `from torch._C._distributed_c10d import FakeProcessGroup`) is guarded against the case
    # where this C extension is ABSENT (ModuleNotFoundError) and skips the broken-on-this-build
    # fsdp/_shard subtree. Natively the module is absent -> ModuleNotFoundError. Our stub must
    # reproduce that for unknown names, while still serving vLLM's explicit `...c10d.Store`.
    def _cmod_getattr(name):
        # dunders (e.g. __file__, __spec__) must raise AttributeError so hasattr()-based
        # introspection (inspect.getmodule, etc.) works normally.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise ModuleNotFoundError(
            f"No module named 'torch._C._distributed_c10d.{name}' (vllm-win shim: not provided)"
        )

    cmod.__getattr__ = _cmod_getattr
    sys.modules["torch._C._distributed_c10d"] = cmod
    try:
        torch._C._distributed_c10d = cmod
    except Exception:
        pass

    funcol = types.ModuleType("torch.distributed._functional_collectives")
    funcol.all_reduce = lambda t, *a, **k: t
    funcol.all_gather_tensor = lambda t, *a, **k: t
    funcol.all_gather_into_tensor = lambda t, *a, **k: t
    funcol.reduce_scatter_tensor = lambda t, *a, **k: t
    funcol.all_to_all_single = lambda t, *a, **k: t
    funcol.wait_tensor = lambda t, *a, **k: t
    funcol.all_reduce_coalesced = lambda ts, *a, **k: ts
    sys.modules["torch.distributed._functional_collectives"] = funcol
    dist._functional_collectives = funcol

    symmem = types.ModuleType("torch.distributed._symmetric_memory")
    symmem.is_symm_mem_enabled_for_group = lambda *a, **k: False
    symmem.empty_strided_p2p = lambda *a, **k: None
    sys.modules["torch.distributed._symmetric_memory"] = symmem
    dist._symmetric_memory = symmem

    rendezvous_mod = types.ModuleType("torch.distributed.rendezvous")
    rendezvous_mod.rendezvous = lambda *a, **k: iter([(Store(), 0, 1)])
    sys.modules["torch.distributed.rendezvous"] = rendezvous_mod
    dist.rendezvous = rendezvous_mod

    # DeviceMesh lives in a submodule that imports fine; some torch code expects it at the
    # top level (`from torch.distributed import DeviceMesh`).
    try:
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
        if not hasattr(dist, "DeviceMesh"):
            dist.DeviceMesh = DeviceMesh
        if not hasattr(dist, "init_device_mesh"):
            dist.init_device_mesh = init_device_mesh
    except Exception:
        pass

    # ---- torch.distributed.tensor (DTensor) stub: make it look ABSENT like native ----
    # On this USE_DISTRIBUTED=0 build, `torch.distributed.tensor` is natively unimportable
    # (ModuleNotFoundError), and torch's own code guards it accordingly -- notably torch.fx
    # graph printing does `try: from torch.distributed.tensor._api import DTensor, DTensorSpec
    # except ModuleNotFoundError`, which inductor hits when logging the compiled graph
    # (_log_inference_graph, verbose=True). But our funcol/distributed_c10d stubs let the REAL
    # torch.distributed.tensor.__init__ start importing and then fail with ImportError (e.g.
    # `cannot import name 'AsyncCollectiveTensor'`) -- which is NOT caught -> inductor compile
    # dies. Fix: pre-register a NON-PACKAGE stub for `torch.distributed.tensor` so the real
    # __init__ never runs and `torch.distributed.tensor._api` raises ModuleNotFoundError (caught
    # by torch). We still expose the few top-level names so any direct `from ...tensor import X`
    # works; single-GPU has no real DTensors, so the dummy classes only feed isinstance()->False.
    if "torch.distributed.tensor" not in sys.modules:
        tmod = types.ModuleType("torch.distributed.tensor")
        tmod.DTensor = type("DTensor", (), {})
        tmod.DTensorSpec = type("DTensorSpec", (), {})
        tmod.Shard = type("Shard", (), {})
        tmod.Replicate = type("Replicate", (), {})
        tmod.Partial = type("Partial", (), {})
        tmod.Placement = type("Placement", (), {})
        tmod.distribute_tensor = lambda t, *a, **k: t
        tmod.distribute_module = lambda m, *a, **k: m
        if hasattr(dist, "DeviceMesh"):
            tmod.DeviceMesh = dist.DeviceMesh
        if hasattr(dist, "init_device_mesh"):
            tmod.init_device_mesh = dist.init_device_mesh
        # Intentionally NO __path__: importing `torch.distributed.tensor._api` (or any submodule)
        # raises ModuleNotFoundError ("not a package"), exactly the native-absent behavior that
        # torch.fx's `except ModuleNotFoundError` relies on.
        sys.modules["torch.distributed.tensor"] = tmod
        dist.tensor = tmod

        # ...except transformers >= 5.x does NOT guard it: sharding_utils.py does a bare
        # `from torch.distributed.tensor._utils import compute_local_shape_and_global_offset`,
        # so the "not a package" error would abort model loading. Register that ONE submodule
        # under its full dotted name: importlib resolves sys.modules[full_name] before it asks
        # whether the parent is a package, so `._api` still raises ModuleNotFoundError for
        # torch.fx while this import succeeds.
        umod = types.ModuleType("torch.distributed.tensor._utils")

        def _compute_local_shape_and_global_offset(global_shape, *a, **k):
            # world_size 1: the local shard IS the whole tensor, sitting at offset 0.
            return tuple(global_shape), tuple(0 for _ in global_shape)

        umod.compute_local_shape_and_global_offset = _compute_local_shape_and_global_offset
        if os.environ.get("VLLM_WIN_DTENSOR_UTILS", "1") != "0":  # =0: bisect switch
            sys.modules["torch.distributed.tensor._utils"] = umod
            tmod._utils = umod

    _install_amdsmi_stub(torch)
    _install_uvloop_stub()
    _install_fcntl_stub()
    _install_tokenizer_compat()
    try:
        from . import cops
        cops.install()  # register torch.ops._C.* fused-op fallbacks
    except Exception as e:
        print("vllm-win cops install warning:", repr(e))
    # NOTE: the AWQ-GEMV MPLinearKernel is registered later, from
    # WindowsRocmPlatform.check_and_update_config (importing vllm.platforms here, mid-bootstrap,
    # would re-enter the platform-plugin loader and circular-import this package).

    setattr(dist, _SENTINEL, True)


def _install_tokenizer_compat() -> None:
    """Some community models (e.g. llm-compressor exports) ship a bogus
    tokenizer_config.json with `"tokenizer_class": "TokenizersBackend"`, which transformers
    can't resolve and errors before it ever loads the (perfectly good) fast tokenizer.json.
    Alias that name to PreTrainedTokenizerFast so AutoTokenizer loads tokenizer.json directly.
    Harmless if the class is never referenced."""
    try:
        import transformers
        from transformers import PreTrainedTokenizerFast

        if not hasattr(transformers, "TokenizersBackend"):
            transformers.TokenizersBackend = PreTrainedTokenizerFast
    except Exception:
        pass


def _install_fcntl_stub() -> None:
    """fcntl is POSIX-only; vLLM imports it for advisory file locks. No-op stub is fine for
    single-process use on Windows."""
    if "fcntl" in sys.modules:
        return
    try:
        import fcntl  # noqa: F401
        return
    except Exception:
        pass
    m = types.ModuleType("fcntl")
    m.LOCK_SH, m.LOCK_EX, m.LOCK_NB, m.LOCK_UN = 1, 2, 4, 8
    m.flock = lambda *a, **k: None
    m.lockf = lambda *a, **k: None
    m.fcntl = lambda *a, **k: 0
    m.ioctl = lambda *a, **k: 0
    sys.modules["fcntl"] = m


def _install_uvloop_stub() -> None:
    """uvloop is POSIX-only (no Windows wheel); vLLM imports it for its async event loop.
    Map it onto asyncio so the import succeeds. Offline LLM.generate() does not need a fast
    loop; the async server paths fall back to the default asyncio loop."""
    if "uvloop" in sys.modules:
        return
    try:
        import uvloop  # noqa: F401
        return
    except Exception:
        pass
    import asyncio

    m = types.ModuleType("uvloop")
    m.install = lambda *a, **k: None
    m.new_event_loop = asyncio.new_event_loop
    m.EventLoopPolicy = asyncio.DefaultEventLoopPolicy
    m.Loop = asyncio.AbstractEventLoop

    def _run(coro, *a, **k):
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        return asyncio.run(coro)

    m.run = _run
    sys.modules["uvloop"] = m


def _install_amdsmi_stub(torch) -> None:
    """vLLM's rocm.py uses amdsmi for device introspection; it is Linux-only. Without it,
    the module-load gcnArch query falls into a warning_once() path that triggers a circular
    import during platform resolution. A minimal amdsmi stub (reading the arch/name from
    torch.cuda) keeps the amdsmi path working and avoids that."""
    if "amdsmi" in sys.modules:
        return
    try:
        import amdsmi  # noqa: F401  (real one present -> use it)
        return
    except Exception:
        pass

    def _arch():
        try:
            return torch.cuda.get_device_properties("cuda").gcnArchName.split(":")[0]
        except Exception:
            return "gfx1100"

    def _name():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "AMD GPU"

    m = types.ModuleType("amdsmi")

    class AmdSmiException(Exception):
        pass

    m.AmdSmiException = AmdSmiException
    m.amdsmi_init = lambda *a, **k: None
    m.amdsmi_shut_down = lambda *a, **k: None
    m.amdsmi_get_processor_handles = lambda *a, **k: [0]
    m.amdsmi_get_gpu_asic_info = lambda h, *a, **k: {
        "target_graphics_version": _arch(),
        "device_id": "",
        "market_name": _name(),
    }
    m.amdsmi_get_gpu_device_uuid = lambda h, *a, **k: "gpu-0000-shim"
    m.amdsmi_topo_get_link_type = lambda a, b, *x, **k: {"hops": 1, "type": 2}
    sys.modules["amdsmi"] = m
