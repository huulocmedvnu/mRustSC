"""Single-cell analysis with a Rust core running on the Apple GPU.

The API mirrors scanpy: `pp` for preprocessing, `tl` for tools, `metrics` and
`get` for the accessors. Functions take an `AnnData` and write their results
into the slots scanpy uses, so existing code and plotting keep working.
"""

from scrust import get, metrics, pp, tl
from scrust._scrust import gpu_available
from scrust.settings import settings

__version__ = "0.2.0"

__all__ = ["__version__", "get", "gpu_available", "metrics", "pp", "settings", "tl"]
