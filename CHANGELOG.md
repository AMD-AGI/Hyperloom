# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Add repository governance docs (LICENSE, SECURITY.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md).
- Add documentation guides under `docs/`: `ENV_AND_AUTH.md`, `KB_GUIDE.md`, `CONFIGURATION_REFERENCE.md`, `INTEGRATION_SESSION_BREAKDOWN.md`, `OPERATIONS.md`, `OPERATOR_SCRIPTS.md`, `TROUBLESHOOTING.md`, `UPGRADING.md`.
- Refresh `docs/HOW_THE_OPTIMIZATION_LOOP_WORKS.md` and add `inference_optimizer/README.md` as a package-level entry point.
- README now links to the new guides via a "Learn More" doc index and lists `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`, and `SECURITY.md` in the repo-tree.

## [v0.3] - 2026-05-14
### Added
- Opt-in PMC roofline action gated after `select_kernels`, deriving workload from materialized Magpie config.
- PMC roofline integration tests for Ray-based execution path.

### Fixed
- Enforce PMC roofline GPU work to run inside a Ray-owned worker while preserving local debug escape hatches.
- Resolve PMC roofline GPU spec handling for Ray contexts.

## [v0.2] - 2026-04-22
### Added
- Hardened marathon protocol with deep kernel analysis, KM feed pipeline improvements, micro-benchmarking, and GPU time-share handling.
- Vendor kernel configuration guidance and updated kernel-manager skills/actions (including local-test flow).
- Launcher scripts refinements for orchestrator/kernel manager panes.

[Unreleased]: https://github.com/AMD-AGI/Hyperloom/compare/v0.3...HEAD
[v0.3]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v0.3
[v0.2]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v0.2
