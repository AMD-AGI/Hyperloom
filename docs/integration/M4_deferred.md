# M4 deferred items

Items deferred from the M4 main merge to a follow-up PR per M4 §Roll-back.

## B-1 breakdown v1.1 (deferred)

main's breakdown v1.1 (bbb143d / cd43641 / 9d62e8e / 0e9e78a)
adds ~2000 lines to `breakdown/collectors.py` introducing TraceLens
parsers, Kernel Decision Path collectors, roofline\_v2 collector, and
data\_provenance plumbing. Our branch has v0.8 specialist run tracking
(KB\_design \.5 / M5) that is structurally absent on main. A safe
three-way merge would need careful hand-port of:

* main's new collector functions (collect\_kernel\_profiling,
  collect\_kernel\_decision\_path, collect\_roofline, _probe_*)
* preserving ours's collect\_specialist\_runs +
  collect\_phase\_segments + KB provenance collector
* reconciling schema.py top-level sections (4 new from main vs
  ours specialist\_runs key)

Files: `breakdown/collectors.py` (7169 lines), `breakdown/exporter.py`,
`breakdown/schema.py`, `tests/test_breakdown_smoke.py`.

Resolution: `git checkout --ours` for all 4 files. A dedicated follow-up
PR (M4-FU-1) will port breakdown v1.1 with full tests and KDP coverage.

## B-2 profile validator (partial; follow-up acceptable)

Main commit `7e3c8e9` ("post-profile trace structure validator") adds a
Deval `check_torch_trace.py` parity validator block. The M4 merge of
`profile.py` resolved 7 comment-formatting conflict hunks (multi-node
restart rework drift) but did NOT discover any new code-level
validator block in the conflict regions — main's validator landed in
a separate function (`_validate_torch_trace_structure`) that doesn't
overlap with our N36 chunk-quality gate. If we want the structural
validator, it can be cherry-picked in a follow-up PR once we confirm
it doesn't double-fire alongside N36.

## Removed newly-added tests (drop per MAIN_FEATURES_DROPPED §7)

12 test files main shipped that reference retired v0.6 surfaces or the
deferred B-1 breakdown v1.1 collectors. All are tracked as DROP in the
§7 N-series decision table:

* `test_breakdown_kdp_backend_speedup.py` (deferred B-1)
* `test_breakdown_kernel_decision_path.py` (deferred B-1)
* `test_breakdown_roofline.py` (deferred B-1)
* `test_breakdown_session_image_timing.py` (deferred B-1)
* `test_n19c_gain_driven_gates.py` (N19c — DONE in F3-5 with different
  gain-driven gate, not main's `_cheap_exhausted_epsilon`)
* `test_n20_backends_subset.py` (N20-A DROP, v0.6 backends retired)
* `test_n20_params_subset.py` (N20-A DROP, v0.6 params retired)
* `test_n22_keyword_advisory.py` (N22 DROP infra-only, no wired rule)
* `test_n27_roofline_fallback.py` (N27 — different impl on this branch,
  counter only via 6078012; no `_ROOFLINE_FALLBACK_THRESHOLD_DEFAULT`)
* `test_n28_validate_stack_session_dir.py` (N28 DROP, validate_stack retired)
* `test_n30_cheap_exhausted_deep_boost.py` (N30 DROP, scoreboard retired)
* `test_n32_actionability_tags.py` (N32 DROP, scoreboard retired)

## Removed newly-added tests that target main's text/surface

Tests main added that pin against its own orchestration.md wording, CLI
flag layout, or roofline-snapshot field set — surfaces this branch
diverges on deliberately. Dropping the tests, not their underlying
features:

* `test_agents_flag_combo.py` — main's `--no-kernel/--no-framework`
  combo matrix; our branch keeps the toggles but the matrix invariants
  pin specific subset/disjoint relationships we don't match.
* `test_analysis_md_full_injection.py` — orchestration.md consumption
  guidance text; our orchestration.md ships the F1-5 v0.8 wording.
* `test_format_discovered_flags_layered.py` — depends on
  `_tag_helper` shape main introduced; v0.8 prompt renderer is
  different.
* `test_n12_orchestration_hard_rules.py` — pins specific text
  N12 hard-rule block (F1-5 has the v0.8 form on this branch).
* `test_roofline_snapshot.py` — expects `build_summary_dict` /
  `format_comparison_section` extras we don't render.

## Removed newly-added tests for retired / deferred features (round 2)

* `test_breakdown_coverage.py` / `_data_provenance.py` /
  `_decision_journal.py` / `_kernel_profiling.py` / `_v1_1_extras.py` —
  all rely on the deferred B-1 `breakdown.build` collectors.
* `test_dump_session_breakdown_cli.py` — same.
* `test_n31_roofline_comparison.py` — N31 done differently in F3-3.
* `test_n35_critic_exploration_carve_out.py` — N35 references
  `validate_stack` (retired per §1).
* `test_roofline_sequence_denial.py` — N3/N9 done differently per F3-1.
