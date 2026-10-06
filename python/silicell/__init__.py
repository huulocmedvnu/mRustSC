"""Single-cell analysis with a Rust core running on the Apple GPU.

The API mirrors scanpy: `pp` for preprocessing, `tl` for tools, `metrics` and
`get` for the accessors. Functions take an `AnnData` and write their results
into the slots scanpy uses, so existing code and plotting keep working.
"""

from silicell import get, metrics, pp, tl
from silicell._silicell import gpu_available as _gpu_available_native
from silicell.settings import settings

__version__ = "0.3.0"


def gpu_available() -> bool:
    """True when a usable Metal device exists and the session has not been pinned to the CPU.

    `silicell.settings.device = "cpu"` (or `SILICELL_DEVICE=cpu` in the environment) makes this
    False, so code that branches on it stays off the GPU together with the API.
    """
    return settings.device != "cpu" and bool(_gpu_available_native())


__all__ = ["__version__", "get", "gpu_available", "metrics", "pl", "pp", "settings", "tl"]


def __getattr__(name: str):
    """Load `silicell.pl` on first access, so `import silicell` stays free of matplotlib."""
    if name == "pl":
        import silicell.pl as pl

        return pl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
