"""Runtime settings, mirroring `scanpy.settings`. Owned by feat/settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Settings:
    """Process-wide defaults.

    Mirrors the parts of `scanpy.settings` that change behaviour rather than
    plotting: chatter, the default device, and the memory ceiling that the
    chunked paths size their blocks against.
    """

    verbosity: int = 1
    device: str = "auto"
    max_memory_gb: float = 4.0
    n_jobs: int = 0
    chunk_size: int = 0
    _observers: list[object] = field(default_factory=list, repr=False)


settings = Settings()
