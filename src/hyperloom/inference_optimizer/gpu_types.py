# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD GPU-type helpers shared by the CLI and orchestrator runtime."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from hyperloom_common.gpu_identity import (
    AMD_GPU_BOARDS,
    GpuAgent,
    GpuBoard,
    board_named_in,
    board_of_agent,
    gpu_board,
    normalise_device_id,
)


@dataclass(frozen=True)
class BoardPolicy:
    """What Hyperloom does on one board of ``AMD_GPU_BOARDS``."""

    #: Frameworks validated on this board; ``None`` leaves the framework unconstrained.
    validated_frameworks: frozenset[str] | None = None
    #: Whether the targeted-build (compiled / AITER) path has been validated on this board.
    targeted_build_validated: bool = True
    #: Whether the GEAK handoff names the product and ISA instead of leaving them to GEAK's device probe.
    geak_identity_pinned: bool = False
    #: Whether a bare arch (``gcnArchName``, rocminfo) stands for this board; other boards need their product name.
    names_its_arch: bool = False
    #: Whether a launch must confirm this board on the card: every visible GPU shows its ISA and its product name or
    #: PCI ID, and a GPU type that disagrees with it is refused instead of the probe winning.
    confirmed_at_launch: bool = False


_VLLM_ONLY = frozenset({"vllm"})
_VLLM_OR_LLAMACPP = frozenset({"vllm", "llamacpp"})

#: Keyed by ``AMD_GPU_BOARDS``: every board has exactly one policy row.
_BOARD_POLICIES: dict[str, BoardPolicy] = {
    "mi300x": BoardPolicy(names_its_arch=True),
    "mi308x": BoardPolicy(),
    "mi325x": BoardPolicy(),
    # The same gfx950 die as mi355x; gfx950 alone still names mi355x.
    "mi350x": BoardPolicy(),
    "mi355x": BoardPolicy(names_its_arch=True),
    "r9700": BoardPolicy(
        validated_frameworks=_VLLM_OR_LLAMACPP, targeted_build_validated=False, geak_identity_pinned=True
    ),
    "r9600d": BoardPolicy(validated_frameworks=_VLLM_ONLY, targeted_build_validated=False, geak_identity_pinned=True),
    "w7900": BoardPolicy(
        validated_frameworks=_VLLM_ONLY,
        targeted_build_validated=False,
        geak_identity_pinned=True,
        confirmed_at_launch=True,
    ),
    "w7900d": BoardPolicy(validated_frameworks=_VLLM_OR_LLAMACPP, targeted_build_validated=False),
    "radeon8060s": BoardPolicy(),
}

#: A bare arch resolves only to a board marked ``names_its_arch`` (gfx942 to mi300x, gfx950 to mi355x); gfx1100,
#: gfx12 and gfx1151 cover several products, so those resolve through the product name.
_GFX_REFERENCE_BOARDS: dict[str, str] = {
    AMD_GPU_BOARDS[product].gfx: product for product, policy in _BOARD_POLICIES.items() if policy.names_its_arch
}


def _board_policy(gpu_type: str | None) -> BoardPolicy | None:
    return _BOARD_POLICIES.get(str(gpu_type or "").strip().lower())


def targeted_build_validated(gpu_type: str) -> bool:
    """Whether the targeted-build (compiled / AITER) path has been validated on this board."""
    policy = _board_policy(gpu_type)
    return policy.targeted_build_validated if policy else True


def geak_identity_pin(gpu_type: str) -> dict[str, str]:
    """The ``expected_target`` / ``expected_gfx`` pair a GEAK handoff carries for this board, if any."""
    policy = _board_policy(gpu_type)
    board = gpu_board(gpu_type)
    if not (policy and board and policy.geak_identity_pinned):
        return {}
    return {"expected_target": board.product, "expected_gfx": board.gfx}


#: The boards a launch confirms on the card, sorted for a stable ``--help`` listing.
BOARDS_CONFIRMED_AT_LAUNCH = tuple(
    sorted(name for name, policy in _BOARD_POLICIES.items() if policy.confirmed_at_launch)
)


def _confirmed_board(gpu_type: str | None) -> GpuBoard | None:
    """The board ``gpu_type`` names when a launch must confirm it on the card (``confirmed_at_launch``)."""
    policy = _board_policy(gpu_type)
    return gpu_board(gpu_type) if policy and policy.confirmed_at_launch else None


def geak_checks_the_card(gpu_type: str | None) -> bool:
    """Whether GEAK, handed this board's pin, probes the card itself and refuses one its identity helper cannot name.

    True for a pinned board a launch confirms on the card, which no ISA stands for.
    """
    board = _confirmed_board(gpu_type)
    return bool(board and geak_identity_pin(board.product))


def reference_gpu_type_for_gfx(text: str | None) -> str | None:
    """Return the board a bare gfx arch in ``text`` (``gcnArchName``, rocminfo) stands for, or ``None``."""
    lower = str(text or "").lower()
    return next((product for gfx, product in _GFX_REFERENCE_BOARDS.items() if gfx in lower), None)


def parse_gpu_type(text: str | None) -> str | None:
    """Map rocm-smi / gcnArchName output, possibly mixed into other log text, to a supported board name."""
    board = board_named_in(text)
    return board.product if board else reference_gpu_type_for_gfx(text)


def _board_framework_error(gpu_type: str, framework: str) -> str | None:
    """Why ``framework`` has not been validated on board ``gpu_type``, if it has not."""
    policy = _board_policy(gpu_type)
    framework = str(framework or "").strip().lower()
    supported = policy.validated_frameworks if policy else None
    if not framework or not supported or framework in supported:
        return None
    return (
        f"{str(gpu_type).strip().lower()} is currently validated only for {', '.join(sorted(supported))}; "
        f"framework {framework!r} has not been validated on it"
    )


#: ISAs that enable nothing without a registered board's product name or PCI ID: each is shared by boards Hyperloom
#: has not validated.
_PRODUCT_REQUIRED_GFX = frozenset({"gfx1100", "gfx1101", "gfx1102"})

#: Preflight drops ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES from .env, so only the launching shell's value counts.
_MASK_HINT = (
    "If the host has other GPUs, give the container only the /dev/dri render nodes of the GPUs this session should "
    "use, or export ROCR_VISIBLE_DEVICES with their indices in the shell that launches Hyperloom (a value in .env is "
    "ignored)."
)


def _visible_gpu(agent: GpuAgent) -> str:
    """``the visible GPU``, with the nameplate or else the PCI ID rocminfo printed for it."""
    label = repr(agent.marketing_name) if agent.marketing_name else agent.pci_device_id
    return f"the visible GPU {label}" if label else "the visible GPU"


def _pci_list(board: GpuBoard) -> str:
    return " or ".join(f"1002:{device_id}" for device_id in sorted(board.device_ids))


def _confirmed_list(*gpu_types: str) -> str:
    """``w7900 (PCI 1002:7448 or 1002:744a)`` for each board a launch confirms on the card."""
    boards = (_confirmed_board(gpu_type) for gpu_type in gpu_types)
    return " / ".join(f"{board.product} (PCI {_pci_list(board)})" for board in boards if board)


def _bare_isa_error(named: str, board: GpuBoard | None, agent: GpuAgent) -> str:
    gfx = agent.gfx
    supported = ", ".join(b.product for b in AMD_GPU_BOARDS.values() if b.gfx == gfx)
    prefix = f"{named} dispatches to {board.gfx}, but" if board else "no supported product was identified;"
    return (
        f"{prefix} {_visible_gpu(agent)} reports {gfx}, and a bare {gfx} is not product evidence. "
        + (f"Supported {gfx} products: {supported}." if supported else f"No {gfx} product is validated.")
        + f" {_MASK_HINT}"
    )


def _confirmed_board_error(named: str, board: GpuBoard, agent: GpuAgent) -> str | None:
    """Why ``agent`` is not ``board``: each agent needs its ISA and its nameplate or PCI ID, and neither may differ."""
    if agent.gfx != board.gfx:
        error = f"{named} dispatches to {board.gfx}, but {_visible_gpu(agent)} reports {agent.gfx}"
    elif agent.marketing_name and not board.named_in(agent.marketing_name):
        error = f"{named} is {_confirmed_list(board.product)}, but the visible GPU is {agent.marketing_name!r}"
    elif agent.chip_id and board.device_ids and normalise_device_id(agent.chip_id) not in board.device_ids:
        gpu = f"the visible GPU {agent.marketing_name!r}" if agent.marketing_name else "the visible GPU"
        error = f"{named} is PCI {_pci_list(board)}, but {gpu} has PCI ID {agent.pci_device_id}"
    elif not (agent.marketing_name or agent.chip_id):
        error = (
            f"{named} is identified by its nameplate or PCI ID, and rocminfo prints neither for the visible "
            f"{agent.gfx} GPU"
        )
    else:
        return None
    return f"{error}. {_MASK_HINT}"


def probed_gpu_type(agents: Sequence[GpuAgent] | None) -> str:
    """The board named by the first rocminfo agent whose nameplate or PCI ID names one, or ``""``."""
    for agent in agents or ():
        board = board_of_agent(agent)
        if board:
            return board.product
    return ""


def launch_gpu_type(requested: str | None, probed: str | None, *, resumed: bool = False) -> str:
    """The board a launch runs as: a resume's own GPU type, else the probe over the flag (:func:`_resolve_gpu_type`)."""
    requested = str(requested or "").strip().lower()
    return requested if resumed else str(probed or "").strip().lower() or requested


def _agent_error(named: str, product: str, agent: GpuAgent) -> str | None:
    """Why a launch as board ``product`` cannot run with the visible GPU ``agent``, or ``None``."""
    board = gpu_board(product)
    agent_board = board_of_agent(agent)
    # An unregistered gfx110x card is judged only on a launch with no board or on a gfx110x board of another ISA.
    if agent.gfx in _PRODUCT_REQUIRED_GFX and agent_board is None:
        if board is None or (board.gfx in _PRODUCT_REQUIRED_GFX and board.gfx != agent.gfx):
            return _bare_isa_error(named, board, agent)
    confirmed = _confirmed_board(product)
    if confirmed:
        return _confirmed_board_error(named, confirmed, agent)
    if agent_board and agent_board.product != product and _confirmed_board(agent_board.product):
        return (
            f"{_visible_gpu(agent)} is {_confirmed_list(agent_board.product)}, which a launch confirms on the card, "
            f"but {named} is another board. {_MASK_HINT}"
        )
    return None


def product_launch_error(
    gpu_type: str | None,
    agents: Sequence[GpuAgent] | None,
    *,
    framework: str,
    probed: str | None,
    resumed: bool = False,
) -> str | None:
    """Why this launch cannot run on the board, or ``None``; checked before any install, LLM call or session dir.

    ``gpu_type`` is the flag (or env) value, or with ``resumed`` the one the resumed session was created with;
    ``agents`` is what rocminfo shows the session, or ``None`` when it cannot run; ``probed`` is the board the launch's
    own GPUs name (``""`` when none does), or ``None`` when they are on remote nodes and only the flags are checked.
    The board judged is the one the launch runs as (:func:`launch_gpu_type`), whichever probe named it. A board a
    launch confirms on the card is accepted only when every agent shows its ISA and its nameplate or PCI ID, so a local
    launch that rocminfo shows no GPU for is refused, and no visible GPU may be a confirmed board the launch is not.
    """
    requested = str(gpu_type or "").strip().lower()
    remote = probed is None
    conflict = gpu_type_conflict(requested, probed, resumed=resumed)
    if conflict:
        return conflict
    product = launch_gpu_type(requested, probed, resumed=resumed)
    if resumed:
        named = f"the session's GPU type {product}"
    else:
        named = f"--gpu-type {product}" if requested and requested == product else f"the probed board {product}"
    board = gpu_board(product)
    confirmed = _confirmed_board(product)
    if confirmed and not remote and not agents:
        if agents is None:
            seen, fix = "could not run", "Put rocminfo on PATH and give"
        else:
            seen = "lists no GPU"
            fix = "Check that ROCR_VISIBLE_DEVICES in the launching shell does not hide the card, and give"
        return (
            f"{named} is identified by its nameplate or PCI ID, but rocminfo {seen}, so the card is unconfirmed. "
            f"{fix} the session the card's /dev/kfd and /dev/dri nodes."
        )
    for agent in agents or ():
        error = _agent_error(named, product, agent)
        if error:
            return error
    return _board_framework_error(product, framework) if board else None


def require_board_framework(gpu_type: str, framework: str) -> None:
    """Exit at launch when ``framework`` has not been validated on board ``gpu_type``."""
    unsupported = _board_framework_error(gpu_type, framework)
    if unsupported:
        print(f"ERROR: {unsupported}", file=sys.stderr)
        sys.exit(2)


def gpu_type_conflict(user_specified: str | None, probed: str | None, *, resumed: bool = False) -> str | None:
    """Why a GPU type the probe contradicts is refused when either side is a board a launch confirms on the card.

    Every other disagreement keeps the probe-wins rule of :func:`_resolve_gpu_type`. With ``resumed``, the GPU type
    is the one the resumed session was created with, which no flag changes.
    """
    user = str(user_specified or "").strip().lower()
    probe = str(probed or "").strip().lower()
    if not (user and probe) or user == probe:
        return None
    if not (_confirmed_board(user) or _confirmed_board(probe)):
        return None
    if resumed:
        return (
            f"this session was created for {user}, but the probed board is {probe}; a "
            f"{_confirmed_list(user, probe)} session must resume on the board it was created on."
        )
    return (
        f"--gpu-type {user} disagrees with the probed {probe}; a {_confirmed_list(user, probe)} launch must match "
        "the board it runs on. Pass the --gpu-type the probe reports, or run on the board you named."
    )


def require_gpu_type_agreement(user_specified: str | None, probed: str | None) -> None:
    """Exit at launch when :func:`gpu_type_conflict` refuses the GPU type the probe contradicts."""
    conflict = gpu_type_conflict(user_specified, probed)
    if conflict:
        print(f"ERROR: {conflict}", file=sys.stderr)
        sys.exit(2)


def _resolve_gpu_type(
    user_specified: str,
    probed: str,
) -> tuple[str, list[str]]:
    """Resolve effective gpu_type from a user hint and a hardware probe."""
    warnings: list[str] = []
    if probed and user_specified and probed != user_specified:
        warnings.append(
            f"WARN: --gpu-type={user_specified!r} disagrees with probed "
            f"{probed!r}; using probed {probed!r}. The probe wins because "
            f"KB recipe rows must match the actual hardware to keep "
            f"baseline numbers comparable across sessions."
        )
        return probed, warnings
    return (probed or user_specified), warnings


def _autodetect_gpu_type(agents: Sequence[GpuAgent] | None = None) -> str | None:
    """Return a canonical supported board name, or ``None`` if undetectable.

    ``agents`` is what the launch's rocminfo run showed (``cli.launch_gpu.launch_gpu_probe``); this runs no rocminfo
    itself.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        detected = parse_gpu_type(out)
        if detected:
            return detected
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        # rocm-smi missing / slow / not permitted; fall through to the rocminfo agents and the torch probe below
        # (autodetect is best-effort).
        pass
    # rocm-smi prints "Card Series: N/A" when libdrm refuses it; rocminfo still prints the nameplate and PCI ID.
    detected = probed_gpu_type(agents)
    if detected:
        return detected
    try:
        import torch

        return parse_gpu_type(str(torch.cuda.get_device_properties(0).gcnArchName))
    except (ImportError, OSError, RuntimeError, AssertionError, AttributeError):
        return None


def _resolve_amd_gpu_type(explicit: str | None = None, *, probe: bool = True) -> str | None:
    """Resolve the current AMD GPU type, or None when not on AMD/unknown; ``probe=False`` reads the flag and env only."""
    explicit_norm = str(explicit or "").strip().lower()
    if explicit_norm:
        return explicit_norm if explicit_norm in AMD_GPU_BOARDS else None
    env_norm = (os.environ.get("TARGET_GPU_TYPE", "") or os.environ.get("GPU_TYPE", "")).strip().lower()
    if env_norm:
        return env_norm if env_norm in AMD_GPU_BOARDS else None
    if not probe:
        return None
    detected = (_autodetect_gpu_type() or "").strip().lower()
    return detected if detected in AMD_GPU_BOARDS else None


def amd_gpu_board(gpu_type: str | None = None) -> GpuBoard | None:
    """Return the board of an explicit GPU type, ``$TARGET_GPU_TYPE`` / ``$GPU_TYPE``, or the detected GPU."""
    return gpu_board(_resolve_amd_gpu_type(gpu_type))
