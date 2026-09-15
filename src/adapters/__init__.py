# Subsystem 2: Architecture-Agnostic Recommender Model Adapters
from src.adapters.base import BaseRecommenderAdapter
from src.adapters.cornac_adapters import (
    CornacMFAdapter,
    CornacBPRAdapter,
    CornacPMFAdapter,
    CornacUserKNNAdapter,
    CornacMostPopAdapter,
)

__all__ = [
    "BaseRecommenderAdapter",
    "CornacMFAdapter",
    "CornacBPRAdapter",
    "CornacPMFAdapter",
    "CornacUserKNNAdapter",
    "CornacMostPopAdapter",
]
