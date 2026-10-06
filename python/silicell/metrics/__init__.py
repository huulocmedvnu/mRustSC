"""Metrics, mirroring `scanpy.metrics`."""

from silicell.metrics._autocorrelation import gearys_c, morans_i
from silicell.metrics._compare import confusion_matrix, modularity

__all__ = ["confusion_matrix", "gearys_c", "modularity", "morans_i"]
