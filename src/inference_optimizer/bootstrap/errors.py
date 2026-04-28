"""Bootstrap exceptions."""
from __future__ import annotations


class BootstrapError(RuntimeError):
    """Base class for any bootstrap failure."""


class MissingDependency(BootstrapError):
    """A required binary is not installed and ``auto_install=False``.

    The message includes copy-pasteable install instructions.
    """

    def __init__(self, message: str, *, missing: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.missing = tuple(missing)


class InstallFailed(BootstrapError):
    """``auto_install=True`` was requested but the install step failed."""

    def __init__(self, message: str, *, step: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.step = step
        self.cause = cause


class UnsupportedPlatform(BootstrapError):
    """We don't know how to install Node/Claude on this OS+arch."""
