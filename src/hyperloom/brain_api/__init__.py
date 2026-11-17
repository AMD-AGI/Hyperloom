# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The brain contract: the two calls a brain answers, the state it reads, the ports it is lent, the eval points an
evaluator may score, and the record of each call. The body owns this package; a brain imports it and nothing else
of the body."""

from hyperloom.brain_api.brain import (
    CALL_SITE_METHODS,
    CONTRACT_VERSION,
    Brain,
    BrainContractError,
    compatible,
    validate_brain,
)
from hyperloom.brain_api.calls import (
    CallSite,
    Decision,
    Proposal,
    ProposeAsk,
    Review,
    ReviewAsk,
    Verdict,
    check_proposal,
    check_review,
)
from hyperloom.brain_api.loader import load_brain
from hyperloom.brain_api.points import EvalPoint, Observer, eval_point, observing, point_of, points_of
from hyperloom.brain_api.ports import AgentReply, AgentRequest, LlmReply, LlmRequest, PortError, Ports
from hyperloom.brain_api.record import MAIN_POINTS, BrainCall, append_call, read_calls
from hyperloom.brain_api.state import (
    Attempt,
    Candidate,
    ConfigDelta,
    Evidence,
    Gap,
    Outcome,
    PatchFile,
    SearchState,
    Workload,
)
from hyperloom.brain_api.wire import WireError, from_wire, json_value, to_wire

__all__ = [
    "CALL_SITE_METHODS",
    "CONTRACT_VERSION",
    "MAIN_POINTS",
    "AgentReply",
    "AgentRequest",
    "Attempt",
    "Brain",
    "BrainCall",
    "BrainContractError",
    "CallSite",
    "Candidate",
    "ConfigDelta",
    "Decision",
    "EvalPoint",
    "Evidence",
    "Gap",
    "LlmReply",
    "LlmRequest",
    "Observer",
    "Outcome",
    "PatchFile",
    "PortError",
    "Ports",
    "Proposal",
    "ProposeAsk",
    "Review",
    "ReviewAsk",
    "SearchState",
    "Verdict",
    "WireError",
    "Workload",
    "append_call",
    "check_proposal",
    "check_review",
    "compatible",
    "eval_point",
    "from_wire",
    "json_value",
    "load_brain",
    "observing",
    "point_of",
    "points_of",
    "read_calls",
    "to_wire",
    "validate_brain",
]
