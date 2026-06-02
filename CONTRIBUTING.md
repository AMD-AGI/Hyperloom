# Contributing to Hyperloom

Thank you for helping improve Hyperloom. This guide covers the expected workflow, local setup, and quality checks.

## Pull Request workflow
- Create a feature branch off `main`.
- Keep changes focused and include context in the PR description (problem, approach, test coverage).
- Ensure all checks below pass before requesting review.
- Avoid committing generated artifacts; keep diffs minimal.

## Development setup
- Python 3.10+.
- Create and activate a virtual environment.
- Install dependencies (including test extras):  
  `pip install -e .[test]`
- If you plan to run lint/type checks, install tools:  
  `pip install ruff mypy`

## Testing
- Run the full test suite from the repo root:  
  `pytest`
- To target a directory or file:  
  `pytest inference_optimizer/tests`  
  `pytest inference_optimizer/tests/test_prompt_builder.py -k subset`

## Linting and formatting
- Ruff:  
  `ruff check .`
- Type checks (mypy):  
  `mypy inference_optimizer kernel-agent robustness-agent`
  - Adjust paths if you change package locations.

## Before opening a PR
- [ ] Tests pass (`pytest`).
- [ ] Lint clean (`ruff check .`).
- [ ] Type checks clean (`mypy ...`).
- [ ] No unwanted files (build artifacts, large logs, credentials).

## Security
- Do not include secrets in code or logs.
- Report vulnerabilities privately as described in `SECURITY.md`.
