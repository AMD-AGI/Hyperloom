# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Claiming fusions the framework implements but ships switched OFF.

Repro of the miss: the pipeline treated "vLLM has a compile pass for this chain"
as "vLLM already fuses it" and dropped the candidate. Nearly every PassConfig
fusion flag defaults to None (off), so the fusion never ran and nobody enabled it.
The pass state must be READ from the target install (never hardcoded) and a pass
that exists but is off must surface as an enable-the-switch recipe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from kernelforge.fusion.diagnose import diagnose_from_shares
from kernelforge.fusion.discover import parse_discovered_recipes
from kernelforge.fusion.locate import build_recipes, vllm_compile_pass_state, vllm_pass_config_flag
from kernelforge.fusion.vllm_passes import (
    _PROBE_SRC,
    _VLLM_PASS_PROBE_MARKER,
    PassState,
    TargetRuntime,
    enable_pass_in_source,
    probe_pass_state,
    probe_pass_states,
    resolve_target_runtime,
)

QK_FLAG = "enable_qk_norm_rope_fusion"
QK_SHARES = {"gemm": 0.5, "rmsnorm": 0.12, "rope": 0.06, "add": 0.02}
QK_BODY = "def _normalize_qk(self):\n    q_norm = 1\n    k_norm = 1\n    return self.rotary_emb(q_norm)\n"


def _fake_probe(
    *, present=True, enabled=False, config_file="/fw/vllm/config/compilation.py", source="default", error=""
):
    """Stand-in for the subprocess probe (tests must not import a real vLLM)."""

    def fn(flag: str) -> PassState:
        return PassState(
            flag=flag, present=present, enabled=enabled, config_file=config_file, source=source, error=error
        )

    return fn


def _fake_vllm_tree(tmp_path, model_type: str, body: str):
    mdir = tmp_path / "fw" / "vllm" / "model_executor" / "models"
    mdir.mkdir(parents=True)
    (mdir / f"{model_type}.py").write_text(body, encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": model_type, "hidden_size": 2048, "num_attention_heads": 16}),
        encoding="utf-8",
    )
    return str(model), str(tmp_path / "fw")


class TestPatternRoute:
    def test_proposes_enabling_the_switch_when_it_is_off(self, tmp_path):
        # THE MISS: pass exists but is disabled -> must be proposed, not dropped.
        model, root = _fake_vllm_tree(tmp_path, "qklm", QK_BODY)
        recipes = build_recipes(
            diagnose_from_shares(QK_SHARES, busy_fraction_of_wall=0.21),
            model_path=model,
            framework="vllm",
            framework_root=root,
            pass_probe=_fake_probe(enabled=False, config_file="/fw/vllm/config/compilation.py"),
        )
        cp = [r for r in recipes if r.candidate_kind == "compile_pass"]
        assert len(cp) == 1, f"expected an enable-the-pass recipe, got {[r.pattern_id for r in recipes]}"
        r = cp[0]
        assert r.compile_pass_flag == QK_FLAG
        # Points at the framework's pass config, NOT the model file: that is the
        # file the emitted patch edits.
        assert r.source_file == "/fw/vllm/config/compilation.py"
        assert r.already_satisfied is False
        assert r.to_dict()["compile_pass_flag"] == QK_FLAG

    def test_still_drops_when_the_switch_is_already_on(self, tmp_path):
        # No regression: an ENABLED pass really does make source-level fusion a no-op.
        model, root = _fake_vllm_tree(tmp_path, "qklm", QK_BODY)
        recipes = build_recipes(
            diagnose_from_shares(QK_SHARES, busy_fraction_of_wall=0.21),
            model_path=model,
            framework="vllm",
            framework_root=root,
            pass_probe=_fake_probe(enabled=True),
        )
        assert all(r.pattern_id != "qk_norm_rope" for r in recipes)
        assert all(r.candidate_kind != "compile_pass" for r in recipes)

    def test_state_matrix(self, tmp_path):
        """enabled deletes; disabled claims; absent / undecidable keep authoring.

        Collapsing the last two into "already satisfied" silently deleted work the
        framework is NOT doing for us.
        """
        cases = {
            # (probe kwargs) -> (compile_pass proposed?, qk authoring kept?)
            "enabled": (dict(enabled=True), False, False),
            "disabled": (dict(enabled=False), True, False),
            "absent": (dict(present=False, enabled=None, config_file="", source="absent"), False, True),
            "undecidable-level": (dict(enabled=None, source="level-dynamic"), False, True),
            "probe-error": (dict(enabled=False, error="boom"), False, True),
            # Disabled but the optimization level pins it: flipping the PassConfig
            # default would not take, so it must not be claimed.
            "level-pinned-off": (dict(enabled=False, source="level"), False, True),
        }
        for label, (kw, want_compile_pass, want_authoring) in cases.items():
            model, root = _fake_vllm_tree(tmp_path / label, "qklm", QK_BODY)
            recipes = build_recipes(
                diagnose_from_shares(QK_SHARES, busy_fraction_of_wall=0.21),
                model_path=model,
                framework="vllm",
                framework_root=root,
                pass_probe=_fake_probe(**kw),
            )
            kinds = [r.candidate_kind for r in recipes]
            got_cp = "compile_pass" in kinds
            got_auth = any(r.pattern_id == "qk_norm_rope" for r in recipes)
            assert got_cp is want_compile_pass, f"{label}: compile_pass={kinds}"
            assert got_auth is want_authoring, f"{label}: authoring kept={kinds}"
            if want_authoring:
                qk = next(r for r in recipes if r.pattern_id == "qk_norm_rope")
                assert qk.compile_pass_note, f"{label}: must record why it was not claimed"
                assert qk.already_satisfied is False

    def test_sglang_never_probes_vllm_passes(self, tmp_path):
        # Compile passes are vLLM-only; a probe here would be a bug.
        def boom(flag):
            raise AssertionError(f"must not probe vLLM passes for sglang ({flag})")

        mdir = tmp_path / "fw" / "python" / "sglang" / "srt" / "models"
        mdir.mkdir(parents=True)
        (mdir / "qklm.py").write_text(QK_BODY, encoding="utf-8")
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"model_type": "qklm", "hidden_size": 2048, "num_attention_heads": 16}), encoding="utf-8"
        )
        recipes = build_recipes(
            diagnose_from_shares(QK_SHARES, busy_fraction_of_wall=0.21),
            model_path=str(model),
            framework="sglang",
            framework_root=str(tmp_path / "fw"),
            pass_probe=boom,
        )
        assert any(r.pattern_id == "qk_norm_rope" for r in recipes)


class TestRanking:
    """Enabling the framework's own pass must outrank authoring a kernel.

    Only the top recipe is acted on, and a compile_pass is a one-line deterministic
    flip onto a vendor-tuned kernel, while a new_fusion costs an LLM authoring loop
    plus compile/parity risk. Ranking by trigger share alone would spend the
    expensive path first and leave the free win unclaimed.
    """

    # residual add+rmsnorm (0.34) outranks qk-norm+rope (0.22) on trigger share, so
    # the compile_pass candidate is NOT first unless kind is ranked ahead of share.
    SHARES = {"gemm": 0.4, "add": 0.20, "rmsnorm": 0.14, "rope": 0.08}
    BODY = (
        "class Layer:\n"
        "    def forward(self, hidden_states, residual):\n"
        "        hidden_states = hidden_states + residual\n"
        "        x = self.input_layernorm(hidden_states)\n"
        "        q_norm = self.q_norm(q)\n"
        "        k_norm = self.k_norm(k)\n"
        "        return self.rotary_emb(positions, q_norm, k_norm)\n"
    )

    def _recipes(self, tmp_path):
        model, root = _fake_vllm_tree(tmp_path, "ranklm", self.BODY)
        return build_recipes(
            diagnose_from_shares(self.SHARES, busy_fraction_of_wall=0.21),
            model_path=model,
            framework="vllm",
            framework_root=root,
            pass_probe=_fake_probe(enabled=False),
        )

    def test_compile_pass_outranks_authoring_despite_lower_share(self, tmp_path):
        recipes = self._recipes(tmp_path)
        kinds = [r.candidate_kind for r in recipes]
        assert "new_fusion" in kinds, "test needs a competing authoring candidate"
        assert recipes[0].candidate_kind == "compile_pass", kinds
        authored = next(r for r in recipes if r.candidate_kind == "new_fusion")
        # Ranked first even though it addresses a SMALLER slice of the trace.
        assert recipes[0].trigger_share < authored.trigger_share

    def test_share_order_still_holds_within_a_kind(self, tmp_path):
        recipes = self._recipes(tmp_path)
        authored = [r.trigger_share for r in recipes if r.candidate_kind == "new_fusion"]
        assert authored == sorted(authored, reverse=True)


class TestDiscoveryRoute:
    def _payload(self):
        return json.dumps(
            [
                {
                    "name": "qk_norm_rope_chain",
                    "env_flag": "FUSED_QK",
                    "op_chain": "q_norm/k_norm rmsnorm + rotary_emb",
                    "fusion_math": "fuse qk norm with rope",
                    "priority": 0.9,
                }
            ]
        )

    def test_proposes_enabling_the_switch_when_it_is_off(self):
        recipes = parse_discovered_recipes(
            self._payload(),
            model_type="m",
            framework="vllm",
            source_file="/x.py",
            shapes={},
            pass_probe=_fake_probe(enabled=False),
        )
        assert len(recipes) == 1
        assert recipes[0].candidate_kind == "compile_pass"
        assert recipes[0].compile_pass_flag == QK_FLAG
        assert recipes[0].source_file == "/fw/vllm/config/compilation.py"

    def test_still_drops_when_the_switch_is_already_on(self):
        recipes = parse_discovered_recipes(
            self._payload(),
            model_type="m",
            framework="vllm",
            source_file="/x.py",
            shapes={},
            pass_probe=_fake_probe(enabled=True),
        )
        assert recipes == []

    def test_state_matrix(self):
        """Same matrix as the pattern route: only ENABLED deletes the proposal."""
        payload = self._payload()
        cases = {
            "enabled": (dict(enabled=True), False, False),
            "disabled": (dict(enabled=False), True, False),
            "absent": (dict(present=False, enabled=None, config_file="", source="absent"), False, True),
            "undecidable": (dict(enabled=None, source="level-dynamic"), False, True),
            "level-pinned-off": (dict(enabled=False, source="level"), False, True),
            "probe-error": (dict(enabled=False, error="boom"), False, True),
        }
        for label, (kw, want_cp, want_auth) in cases.items():
            recipes = parse_discovered_recipes(
                payload, model_type="m", framework="vllm", source_file="/x.py", shapes={}, pass_probe=_fake_probe(**kw)
            )
            kinds = [r.candidate_kind for r in recipes]
            assert ("compile_pass" in kinds) is want_cp, f"{label}: {kinds}"
            assert ("new_fusion" in kinds) is want_auth, f"{label}: {kinds}"
            if want_auth:
                assert recipes[0].compile_pass_note, f"{label}: must record why"

    def test_compile_pass_outranks_a_higher_priority_authoring_proposal(self):
        payload = json.dumps(
            [
                {"name": "author_me", "env_flag": "FUSED_X", "op_chain": "scale add combine", "priority": 0.9},
                {
                    "name": "qk_chain",
                    "env_flag": "FUSED_QK",
                    "op_chain": "q_norm/k_norm rmsnorm + rotary_emb",
                    "fusion_math": "fuse qk norm with rope",
                    "priority": 0.2,
                },
            ]
        )
        recipes = parse_discovered_recipes(
            payload,
            model_type="m",
            framework="vllm",
            source_file="/x.py",
            shapes={},
            pass_probe=_fake_probe(enabled=False),
        )
        assert [r.candidate_kind for r in recipes] == ["compile_pass", "new_fusion"]
        assert recipes[0].trigger_share < recipes[1].trigger_share


class TestFlagMapping:
    def test_qk_norm_rope_maps_to_its_pass_config_field(self):
        assert vllm_pass_config_flag("qk_norm_rope") == QK_FLAG
        assert vllm_pass_config_flag("nope") == ""


class TestEnableInSource:
    SRC = (
        "@config\n"
        "class PassConfig:\n"
        "    fuse_norm_quant: bool = None  # type: ignore[assignment]\n"
        "    eliminate_noops: bool = Field(default=True)\n"
        f"    {QK_FLAG}: bool = None  # type: ignore[assignment]\n"
        '    """Enable fused Q/K RMSNorm + RoPE pass."""\n'
    )

    def test_flips_the_disabled_default_and_keeps_the_rest(self, tmp_path):
        p = tmp_path / "compilation.py"
        p.write_text(self.SRC, encoding="utf-8")
        assert enable_pass_in_source(str(p), QK_FLAG) is True
        text = p.read_text(encoding="utf-8")
        assert f"    {QK_FLAG}: bool = True  # type: ignore[assignment]" in text
        # Only the requested flag moves; neighbours are untouched.
        assert "fuse_norm_quant: bool = None" in text
        assert "eliminate_noops: bool = Field(default=True)" in text

    def test_is_idempotent(self, tmp_path):
        p = tmp_path / "compilation.py"
        p.write_text(self.SRC, encoding="utf-8")
        assert enable_pass_in_source(str(p), QK_FLAG) is True
        after_first = p.read_text(encoding="utf-8")
        assert enable_pass_in_source(str(p), QK_FLAG) is False
        assert p.read_text(encoding="utf-8") == after_first

    def test_absent_flag_or_missing_file_is_a_no_op(self, tmp_path):
        p = tmp_path / "compilation.py"
        p.write_text(self.SRC, encoding="utf-8")
        assert enable_pass_in_source(str(p), "not_a_flag") is False
        assert enable_pass_in_source(str(tmp_path / "nope.py"), QK_FLAG) is False
        assert enable_pass_in_source("", QK_FLAG) is False


def _flag_item(enabled, source="default"):
    """One flag's probe verdict; ``enabled=None`` means undeterminable, not off."""
    return {"present": source != "absent", "enabled": enabled, "source": source}


def _probe_stdout(flags: dict, config_file="/site/vllm/config/compilation.py", error=""):
    payload = {"config_file": config_file, "error": error, "level": "O2", "flags": flags}
    return _VLLM_PASS_PROBE_MARKER + json.dumps(payload)


class TestProbe:
    def _run(self, monkeypatch, stdout="", stderr="", rc=0, exc=None, calls=None):
        def fake_run(cmd, **kw):
            if calls is not None:
                calls.append(cmd)
            if exc is not None:
                raise exc
            return subprocess.CompletedProcess(cmd, rc, stdout, stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        probe_pass_states.cache_clear()
        return probe_pass_state(QK_FLAG)

    def test_reads_the_resolved_value_out_of_the_target_install(self, monkeypatch):
        st = self._run(monkeypatch, stdout=f"some vllm warning\n{_probe_stdout({QK_FLAG: _flag_item(False)})}\n")
        assert st.present is True and st.enabled is False
        assert st.config_file == "/site/vllm/config/compilation.py"
        assert st.missed is True

    def test_enabled_pass_is_not_a_miss(self, monkeypatch):
        st = self._run(monkeypatch, stdout=_probe_stdout({QK_FLAG: _flag_item(True, "level")}))
        assert st.enabled is True and st.missed is False

    def test_flag_absent_from_the_install_is_not_a_miss(self, monkeypatch):
        # Nothing to enable: "not present" must not read as "disabled".
        st = self._run(monkeypatch, stdout=_probe_stdout({QK_FLAG: _flag_item(None, "absent")}))
        assert st.present is False and st.enabled is None and st.missed is False

    def test_level_resolved_flag_is_unknown_not_off(self, monkeypatch):
        # vLLM's optimization level resolves this one from the FULL VllmConfig, so
        # the probe cannot tell. Claiming it would invent no-op work.
        st = self._run(monkeypatch, stdout=_probe_stdout({QK_FLAG: _flag_item(None, "level-dynamic")}))
        assert st.present is True and st.enabled is None
        assert st.missed is False and st.source == "level-dynamic"

    def test_unimportable_vllm_yields_unknown_not_a_guess(self, monkeypatch):
        st = self._run(monkeypatch, stdout="", stderr="ModuleNotFoundError: vllm", rc=1)
        assert st.enabled is None and st.missed is False
        assert st.error

    def test_probe_failure_never_raises(self, monkeypatch):
        st = self._run(monkeypatch, exc=OSError("no exec"))
        assert st.enabled is None and st.missed is False

    def test_malformed_output_is_not_mistaken_for_a_verdict(self, monkeypatch):
        st = self._run(monkeypatch, stdout=f"noise line\n{_VLLM_PASS_PROBE_MARKER}{{not json\n")
        assert st.enabled is None and st.missed is False

    def test_no_flag_means_unknown(self):
        assert probe_pass_state("").enabled is None


class TestProbeSourceAgainstFakeVllm:
    """Runs the REAL probe script against a synthetic vLLM.

    The precedence rules live in the subprocess source, so asserting them through
    stubbed stdout would only test the parser. These build a fake ``vllm`` package
    and execute the probe for real.
    """

    def _fake_vllm(self, tmp_path, *, level_module: str = "", pass_config_extra: str = ""):
        pkg = tmp_path / "vllm"
        (pkg / "config").mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "config" / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "config" / "compilation.py").write_text(
            "import dataclasses\n\n\n"
            "@dataclasses.dataclass\n"
            "class PassConfig:\n"
            f"    {QK_FLAG}: bool = None\n"
            "    fuse_norm_quant: bool = None\n"
            f"{pass_config_extra}",
            encoding="utf-8",
        )
        if level_module:
            (pkg / "config" / "vllm.py").write_text(level_module, encoding="utf-8")
        return tmp_path

    def _probe(self, root, flags=(QK_FLAG, "fuse_norm_quant")):
        env = dict(os.environ, PYTHONPATH=str(root))
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SRC, *flags], capture_output=True, text=True, env=env, timeout=120
        )
        payload = json.loads(proc.stdout.split(_VLLM_PASS_PROBE_MARKER)[-1].strip())
        return payload

    def test_no_level_api_falls_back_to_the_pass_config_default(self, tmp_path):
        # CONFIRMED absent: the class default IS the effective value, so this is a
        # sound fallback rather than an unknown.
        payload = self._probe(self._fake_vllm(tmp_path))
        assert payload["error"] == ""
        assert payload["level_api"].startswith("absent")
        assert payload["flags"][QK_FLAG] == {"present": True, "enabled": False, "source": "default"}

    def test_level_pinned_literal_wins_over_the_default(self, tmp_path):
        level = (
            "import dataclasses, enum\n"
            "class OptimizationLevel(enum.IntEnum):\n"
            "    O2 = 2\n"
            "@dataclasses.dataclass\n"
            "class VllmConfig:\n"
            "    optimization_level: OptimizationLevel = OptimizationLevel.O2\n"
            "OPTIMIZATION_LEVEL_TO_CONFIG = {OptimizationLevel.O2: {'compilation_config':\n"
            "    {'pass_config': {'fuse_norm_quant': False}}}}\n"
        )
        payload = self._probe(self._fake_vllm(tmp_path, level_module=level))
        assert payload["error"] == "" and payload["level_api"] == "ok"
        assert payload["flags"]["fuse_norm_quant"]["source"] == "level"
        # Not owned by the level -> the default still decides.
        assert payload["flags"][QK_FLAG]["source"] == "default"

    def test_level_predicate_is_undecidable_not_off(self, tmp_path):
        level = (
            "import dataclasses, enum\n"
            "class OptimizationLevel(enum.IntEnum):\n"
            "    O2 = 2\n"
            "@dataclasses.dataclass\n"
            "class VllmConfig:\n"
            "    optimization_level: OptimizationLevel = OptimizationLevel.O2\n"
            "def _pred(cfg):\n"
            "    return True\n"
            "OPTIMIZATION_LEVEL_TO_CONFIG = {OptimizationLevel.O2: {'compilation_config':\n"
            "    {'pass_config': {'fuse_norm_quant': _pred}}}}\n"
        )
        payload = self._probe(self._fake_vllm(tmp_path, level_module=level))
        item = payload["flags"]["fuse_norm_quant"]
        assert item == {"present": True, "enabled": None, "source": "level-dynamic"}

    def test_broken_level_api_reports_an_error_instead_of_guessing_off(self, tmp_path):
        # The API exists but does not read as expected: every verdict must be voided,
        # otherwise a None attribute reads as False and gets claimed.
        level = (
            "OPTIMIZATION_LEVEL_TO_CONFIG = None\n"  # not a mapping -> raises on .get
            "class VllmConfig:\n"
            "    pass\n"
        )
        payload = self._probe(self._fake_vllm(tmp_path, level_module=level))
        assert "optimization level unreadable" in payload["error"]
        st = PassState(
            flag=QK_FLAG, present=True, enabled=False, source="default", config_file="/x.py", error=payload["error"]
        )
        assert st.missed is False and st.claimable is False

    def test_absent_flag_reports_absent(self, tmp_path):
        payload = self._probe(self._fake_vllm(tmp_path), flags=("not_a_flag",))
        assert payload["flags"]["not_a_flag"] == {"present": False, "enabled": None, "source": "absent"}


class TestProbeIsBatched:
    def test_every_flag_is_read_in_a_single_vllm_import(self, monkeypatch):
        # Importing vLLM is the whole cost, so N flags must not mean N subprocesses.
        calls = []
        flags = {
            QK_FLAG: _flag_item(False),
            "fuse_rope_kvcache": _flag_item(True, "level"),
            "fuse_norm_quant": _flag_item(None, "level-dynamic"),
        }

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, _probe_stdout(flags), "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        probe_pass_states.cache_clear()
        states = probe_pass_states(tuple(flags))
        assert len(calls) == 1
        assert set(calls[0][3:]) == set(flags)  # every flag passed to the one probe
        assert states[QK_FLAG].missed is True
        assert states["fuse_rope_kvcache"].missed is False
        assert states["fuse_norm_quant"].missed is False

    def test_repeated_lookups_reuse_the_one_probe(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, _probe_stdout({QK_FLAG: _flag_item(False)}), "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        probe_pass_states.cache_clear()
        # Two matched patterns asking for the same table must not re-import vLLM.
        assert vllm_compile_pass_state("qk_norm_rope").claimable
        assert not vllm_compile_pass_state("fuse_rope_kvcache").claimable
        assert len(calls) == 1

    def test_duplicate_and_empty_flags_are_collapsed(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, _probe_stdout({QK_FLAG: _flag_item(False)}), "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        probe_pass_states.cache_clear()
        states = probe_pass_states((QK_FLAG, QK_FLAG, ""))
        assert calls[0][3:] == [QK_FLAG]
        assert set(states) == {QK_FLAG}
        probe_pass_states.cache_clear()
        assert probe_pass_states(()) == {}


class TestTargetIdentity:
    """Probe, edit and serving must all address ONE install."""

    def test_config_outside_the_requested_framework_root_fails_closed(self, monkeypatch):
        # The run was told which framework to target; probing/editing a different
        # install must stop the run, not proceed silently.
        def fake_run(cmd, **kw):
            payload = _probe_stdout({QK_FLAG: _flag_item(False)}, config_file="/other/vllm/config/compilation.py")
            return subprocess.CompletedProcess(cmd, 0, payload, "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        probe_pass_states.cache_clear()
        st = probe_pass_states((QK_FLAG,), require_root="/requested/root")[QK_FLAG]
        assert st.enabled is None and st.missed is False and st.claimable is False
        assert "outside the requested framework root" in st.error

    def test_matching_root_is_accepted(self, monkeypatch, tmp_path):
        cfg = tmp_path / "vllm" / "config" / "compilation.py"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("x = 1", encoding="utf-8")

        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(
                cmd, 0, _probe_stdout({QK_FLAG: _flag_item(False)}, config_file=str(cfg)), ""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        probe_pass_states.cache_clear()
        st = probe_pass_states((QK_FLAG,), require_root=str(tmp_path))[QK_FLAG]
        assert st.claimable is True

    def test_probe_state_is_not_shared_between_installs(self, monkeypatch):
        # Cache identity must include the target, or install B inherits A's verdict.
        seen = []

        def fake_run(cmd, **kw):
            seen.append(cmd[0])
            enabled = cmd[0] == "/b/python"
            return subprocess.CompletedProcess(cmd, 0, _probe_stdout({QK_FLAG: _flag_item(enabled)}), "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        probe_pass_states.cache_clear()
        a = probe_pass_states((QK_FLAG,), python="/a/python")[QK_FLAG]
        b = probe_pass_states((QK_FLAG,), python="/b/python")[QK_FLAG]
        assert seen == ["/a/python", "/b/python"]
        assert a.enabled is False and b.enabled is True

    def test_runtime_derives_the_interpreter_from_the_launcher(self, tmp_path):
        launcher = tmp_path / "vllm"
        launcher.write_text(f"#!{sys.executable}\nprint('x')\n", encoding="utf-8")
        rt = resolve_target_runtime("vllm", launcher_exe=str(launcher))
        assert rt.python == sys.executable and not rt.error
        assert rt.launcher_exe == str(launcher)

    def test_unattributable_launcher_is_an_error_not_a_guess(self, tmp_path):
        launcher = tmp_path / "vllm"
        launcher.write_bytes(b"\x7fELF binary launcher")
        rt = resolve_target_runtime("vllm", launcher_exe=str(launcher))
        assert rt.error and not rt.python

    def test_unpinned_runtime_refuses_to_judge(self):
        state = vllm_compile_pass_state("qk_norm_rope", runtime=TargetRuntime(error="no vllm launcher on PATH"))
        assert state is not None and state.enabled is None and state.claimable is False
        assert "no vllm launcher" in state.error


class TestEnableInSourceFailures:
    def test_unwritable_file_is_reported_not_crashed(self, tmp_path, monkeypatch):
        p = tmp_path / "compilation.py"
        p.write_text(TestEnableInSource.SRC, encoding="utf-8")

        def boom(self, *a, **kw):
            raise OSError("read-only file system")

        monkeypatch.setattr("pathlib.Path.write_text", boom)
        assert enable_pass_in_source(str(p), QK_FLAG) is False
