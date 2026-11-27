# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Saving state.json, validating a resumed state, and the status summary written with it."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from hyperloom_common.coerce import to_str_list
from ..retest_view import write_retest_view
from ..state_file import write_state
from .._shared_state.phase_state import _control_fields

log = logging.getLogger(__name__)


class _PersistenceMixin:
    """Validates a resumed state and saves the session's state.json."""

    # Persistence
    @classmethod
    def state_path(cls, session_dir: Path) -> Path:
        """Return the canonical ``state.json`` path for a session directory."""
        return Path(session_dir) / "state.json"

    @classmethod
    def _validate_resume_fields(cls, raw: Mapping[str, Any]) -> None:
        """Reject damaged crash evidence."""
        if "crash_count" in raw:
            count = raw["crash_count"]
            if not isinstance(count, int) or isinstance(count, bool):
                raise ValueError(f"state.json.crash_count must be int, got {type(count).__name__}")
        if "crash_timestamps" in raw:
            timestamps = raw["crash_timestamps"]
            if not isinstance(timestamps, list):
                raise ValueError(f"state.json.crash_timestamps must be list, got {type(timestamps).__name__}")
            for index, timestamp in enumerate(timestamps):
                if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
                    raise ValueError(
                        f"state.json.crash_timestamps[{index}] must be a number, got {type(timestamp).__name__}"
                    )

    @staticmethod
    def _build_explore_search(
        *,
        existing: Any,
    ) -> dict[str, Any]:
        """Shape the unified ``explore_search`` ledger at load time."""
        from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint as _fp

        existing = existing if isinstance(existing, dict) else {}
        out: dict[str, Any] = dict(existing)
        out.setdefault("tested", {})
        out.setdefault("accepted", [])
        out.setdefault("rejected", [])
        out.setdefault("cursor", len(out.get("tested") or {}))
        out.setdefault("last_round", {})

        # winners_history: normalize persisted rows, sorted by (round_id, ts).
        wh: list[dict[str, Any]] = []
        persisted = existing.get("winners_history")
        for entry in persisted if isinstance(persisted, list) else []:
            if not isinstance(entry, dict):
                continue
            controls = _control_fields(entry)
            fp_val = entry.get("fingerprint") or _fp(
                str(entry.get("extra_server_args") or ""),
                dict(entry.get("extra_envs") or {}),
                **controls,
            )
            wh.append(
                {
                    "round_id": str(entry.get("round_id") or ""),
                    "variant_name": str(entry.get("variant_name") or entry.get("name") or ""),
                    "fingerprint": str(fp_val),
                    "gain_pct": entry.get("gain_pct"),
                    "extra_args": str(entry.get("extra_args") or entry.get("extra_server_args") or ""),
                    "extra_envs": dict(entry.get("extra_envs") or {}),
                    **controls,
                    "provenance": str(entry.get("provenance") or ""),
                    "ts": str(entry.get("ts") or ""),
                }
            )
        wh.sort(key=lambda r: (str(r.get("round_id") or ""), str(r.get("ts") or "")))
        out["winners_history"] = wh

        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialize this state to a plain JSON-compatible dict."""
        return asdict(self)

    def save(self, session_dir: Path) -> None:
        """Atomically write ``state.json`` (tmp file + ``os.replace``).

        Serializes via :meth:`to_dict` and writes to a temp file in the same directory before an atomic rename, so
        concurrent readers never observe a partial blob; also writes the group-readable ``retest_view.json``.

        Charges the elapsed budget first, so every save is a durable record of
        spend and a leg that dies abruptly loses only the time since the last.

        Args:
            session_dir (Path): The session root directory; created if it does not already exist.
        """
        self.charge_elapsed()
        # Backfill scriptable/diffusion (xDiT) ``e2el_mean_ms`` from ``tput`` so current_best has the latency metric.
        self._backfill_scriptable_latency()
        write_state(self.state_path(session_dir), self.to_dict())
        write_retest_view(session_dir, self)
        # Author-time breakdown capture: snapshot state-owned sections into the recorder spool right after persisting.
        self._session_dir = Path(session_dir)
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        instrument.snapshot_state_sections(session_dir, self)
        # Derived artifact: the current best route as a current_setting.sh that --reference-script takes back.
        try:
            cb = self.current_best or {}
            if cb:
                from hyperloom.inference_optimizer import reference_script

                launched = self.current_best_launch_args
                text = reference_script.render_reference_script(
                    framework=str(self.framework or os.environ.get("FRAMEWORK", "sglang")),
                    launch=reference_script.LaunchSettings(
                        server_args=str(cb.get("extra_server_args") or "") if launched is None else launched,
                        envs=dict(cb.get("extra_envs") or {}),
                        overlay_pythonpath=str(cb.get("final_overlay") or ""),
                        unset_envs=to_str_list(cb.get("unset_envs")),
                        remove_args=to_str_list(cb.get("remove_args")),
                        args_mode=str(cb.get("args_mode") or "append"),
                    ),
                    target=reference_script.ServingTarget(
                        model=os.environ.get("MODEL_PATH"),
                        tp=int(self.tp or 0) or None,
                        max_model_len=int(self.max_model_len or 0) or None,
                        gpu_type=str(self.gpu_type or os.environ.get("GPU_TYPE", "")) or None,
                    ),
                )
                (Path(session_dir) / "current_setting.sh").write_text(text, encoding="utf-8")
        except (OSError, ValueError):
            # Never leave a stale launcher or export a subset of the retained
            # configuration. The complete settings remain in durable state.
            try:
                (Path(session_dir) / "current_setting.sh").unlink(missing_ok=True)
            except OSError:
                log.warning("Could not remove stale current_setting.sh", exc_info=True)
            log.warning("current_setting.sh render failed", exc_info=True)
        # Live status mirror: reflect the persisted snapshot into Langfuse for real-time status.
        try:
            from hyperloom.inference_optimizer.trace.langfuse_emitter import record_status as _lf_record_status

            _lf_record_status(session_dir, self._langfuse_status_summary())
        except (TypeError, ValueError):
            log.debug("langfuse status mirror failed", exc_info=True)

    def _backfill_scriptable_latency(self) -> None:
        """Derive ``current_best.e2el_mean_ms`` from ``tput`` for scriptable runs."""
        from hyperloom.inference_optimizer import framework_registry

        fw = str(getattr(self, "framework", "") or "")
        if not framework_registry.is_scriptable(fw):
            return
        cb = self.current_best
        if cb.get("e2el_mean_ms") is not None:
            return
        tput = cb.get("tput")
        if not isinstance(tput, (int, float)) or tput <= 0:
            return
        e2el = framework_registry.primary_metric_value(fw, float(tput))
        if e2el is not None and e2el > 0:
            cb["e2el_mean_ms"] = round(float(e2el), 4)

    def _langfuse_status_summary(self) -> dict[str, Any]:
        """Flatten the state into an OTEL-friendly scalar status snapshot."""
        summary: dict[str, Any] = {
            "phase": self.phase or "",
            "stop_reason": self.stop_reason or "",
            "closing_phase": bool(self.closing_phase),
            "degraded_mode": bool(self.degraded_mode),
            "cumulative_gain_validated": round(
                float(self.cumulative_gain_validated or 0.0),
                2,
            ),
            "baseline_failure_streak": int(self.baseline_failure_streak or 0),
            "macro_cycle": int(self.macro_cycle or 0),
            "session_id": self.session_id or "",
            "model_name": self.model_name or "",
            "framework": self.framework or os.environ.get("FRAMEWORK", ""),
            "kb_hit": str((self.warm_start_context or {}).get("status") or ""),
        }
        from ...phases.machine_state import PHASE_FRAMEWORK_AGENT, phase_cumulative_seconds

        session_elapsed_s = max(0.0, self.elapsed_minutes() * 60.0)
        summary["session_elapsed_s"] = int(round(session_elapsed_s))
        explore_elapsed_s = phase_cumulative_seconds(self, phase=PHASE_FRAMEWORK_AGENT)
        summary["explore_elapsed_s"] = int(round(explore_elapsed_s))
        summary["explore_ratio"] = round(explore_elapsed_s / session_elapsed_s, 4) if session_elapsed_s > 0.0 else 0.0
        tput = self.current_best.get("tput")
        if isinstance(tput, (int, float)) and not isinstance(tput, bool):
            summary["current_best_tput"] = round(float(tput), 1)
        summary.update(self._last_lifecycle_summary())
        return {k: v for k, v in summary.items() if isinstance(v, (str, bool, int, float))}

    def _last_lifecycle_summary(self) -> dict[str, Any]:
        """The newest lifecycle row's sequence, status and phase, for whichever of them it recorded."""
        last = self.lifecycle[-1] if self.lifecycle else {}
        summary: dict[str, Any] = {}
        if isinstance(last, dict):
            if last.get("seq") is not None:
                summary["last_seq"] = last.get("seq")
            if last.get("status"):
                summary["last_lifecycle_status"] = str(last.get("status"))
            if last.get("phase"):
                summary["last_lifecycle_phase"] = str(last.get("phase"))
        return summary
