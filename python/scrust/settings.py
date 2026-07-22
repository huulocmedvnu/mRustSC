"""Runtime settings, mirroring `scanpy.settings`. Owned by feat/accessors.

Deliberately stdlib-only and side-effect free: importing `scrust` must not
install a logging handler, probe for a GPU, or touch process-wide state beyond
creating the singleton at the bottom of this module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import IO

__all__ = ["Settings", "Verbosity", "settings"]


class Verbosity(IntEnum):
    """How much the library says while it works, as `scanpy.Verbosity`."""

    error = 0
    warning = 1
    info = 2
    hint = 3
    debug = 4


# The names `scrust_core::DeviceKind::parse` accepts; keep the two in step.
DEVICES = ("auto", "cpu", "gpu", "metal")

# Line prefixes matching scanpy's log formatter, so output reads the same.
_PREFIXES = {
    Verbosity.error: "ERROR: ",
    Verbosity.warning: "WARNING: ",
    Verbosity.info: "",
    Verbosity.hint: "--> ",
    Verbosity.debug: "    ",
}


def _as_verbosity(value: Verbosity | str | int) -> Verbosity:
    """Accept a `Verbosity`, its name or its level, as `scanpy.settings.verbosity` does."""
    try:
        return Verbosity[value.lower()] if isinstance(value, str) else Verbosity(value)
    except (KeyError, ValueError):
        names = [level.name for level in Verbosity]
        raise ValueError(
            f"cannot set verbosity to {value!r}; accepted values are {names} or 0-{max(Verbosity)}"
        ) from None


def _as_device(value: str) -> str:
    """Reject an unknown device here rather than several calls later inside Rust."""
    if value not in DEVICES:
        raise ValueError(f"device must be one of {list(DEVICES)}, got {value!r}")
    return value


# Assignment to these attributes is validated; everything else is stored as given.
_COERCIONS = {"verbosity": _as_verbosity, "device": _as_device}


@dataclass
class Settings:
    """Process-wide defaults.

    Mirrors the parts of `scanpy.settings` that change behaviour rather than
    plotting: chatter, the default device, and the memory ceiling that the
    chunked paths size their blocks against.
    """

    verbosity: Verbosity = Verbosity.warning
    """How much progress reporting `log` lets through."""

    device: str = "auto"
    """Device an algorithm runs on when its caller does not name one."""

    max_memory_gb: float = 4.0
    """Budget the chunked paths size a row block against."""

    n_jobs: int = 0
    """CPU threads the core may use; `0` leaves the choice to the core."""

    chunk_size: int = 0
    """Rows per streamed block; `0` derives one from `max_memory_gb`."""

    def __setattr__(self, name: str, value: object) -> None:
        """Validate on assignment, so a bad value fails where it was written."""
        coerce = _COERCIONS.get(name)
        super().__setattr__(name, value if coerce is None else coerce(value))

    def resolve_device(self, device: str | None = None) -> str:
        """The device to run on: the caller's choice, or this default when they gave none."""
        return self.device if device is None else _as_device(device)

    def log(
        self,
        message: str,
        *,
        level: Verbosity | str | int = Verbosity.info,
        file: IO[str] | None = None,
    ) -> bool:
        """Report `message` if `verbosity` reaches `level`, and say whether it did.

        Progress reporting goes through here rather than through `logging` so
        that importing scrust never reconfigures a logger the caller owns.
        """
        level = _as_verbosity(level)
        if self.verbosity < level:
            return False
        print(f"{_PREFIXES[level]}{message}", file=sys.stderr if file is None else file)
        return True


settings = Settings()
