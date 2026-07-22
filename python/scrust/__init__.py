"""Single-cell analysis with a Rust core running on the Apple GPU.

The API mirrors scanpy: `pp` for preprocessing, `tl` for tools. Functions take
an `AnnData` and write their results into the slots scanpy uses, so existing
code and plotting keep working.
"""

from scrust import pp, tl
from scrust._scrust import gpu_available

__version__ = "0.1.0"

__all__ = ["__version__", "gpu_available", "pp", "tl"]
