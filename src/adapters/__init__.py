# Subsystem 2: Architecture-Agnostic Recommender Model Adapters
from src.adapters.base import BaseRecommenderAdapter
from src.adapters.cornac_adapters import (
    CornacMFAdapter,
    CornacBPRAdapter,
    CornacPMFAdapter,
    CornacUserKNNAdapter,
    CornacMostPopAdapter,
)
try:
    from src.adapters.lightgcn_adapter import LightGCNAdapter
except ImportError:
    LightGCNAdapter = None

__all__ = [
    "BaseRecommenderAdapter",
    "CornacMFAdapter",
    "CornacBPRAdapter",
    "CornacPMFAdapter",
    "CornacUserKNNAdapter",
    "CornacMostPopAdapter",
    "LightGCNAdapter",
]

