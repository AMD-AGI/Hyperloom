# Docstring Style Guide (Hyperloom docstring pass)

You are adding documentation to Python files. **DO NOT change any logic** —
only insert or expand docstrings. Do not modify code, signatures, imports, or
comments. Preserve exact indentation. Do not reflow existing code.

## What to document

Add a clean Google-style docstring to:
- the **module** (top of file) if it lacks one,
- **every class**,
- **every function/method** (including nested helpers, properties,
  classmethods, staticmethods),

that is missing a docstring OR has an incomplete one.

A docstring is **incomplete** if any of these hold:
- it is missing or empty,
- the callable takes parameters (other than `self`/`cls`) but there is no
  `Args:` section,
- the callable returns a value but there is no `Returns:` (or `Yields:` for
  generators) section,
- it is a trivial one-liner that does not explain what the callable does.

## Google-style format

```
One-line summary in plain language.

Optional extended description with any important behavior, side effects,
or constraints.

Args:
    name (type): What it is. Use the real annotation when present; otherwise
        infer a sensible type. Omit self/cls.

Returns:
    type: What is returned. Use ``Yields:`` for generators.

Raises:
    ExceptionType: When/why it is raised (only if explicitly raised).
```

For **classes / dataclasses / TypedDicts**: describe the class purpose and add
an `Attributes:` section listing the fields with their types when discernible
(from `__init__` or class-level annotations).

## Rules

- Read each body so the description reflects **actual** behavior; never invent.
- If a docstring already exists but is incomplete, **expand it in place** and
  keep the accurate parts of the original summary.
- Keep line length <= 100 chars where practical.
- Only insert/expand docstrings.

## Verify before finishing

For every file you touched, run:

```
python -m py_compile "<file>"
```

It must exit 0. Then run the repo's checker on your assigned files:

```
python "C:\Users\troosta\Hyperloom\_audit_artifacts\check_files.py" <file1> <file2> ...
```

It prints remaining missing/incomplete counts per file. Drive every assigned
file to **0 missing**. Do not commit. Touch only your assigned files.
