import random
import numpy as np

def set_deterministic_seed(seed: int) -> None:
    """
    Sets deterministic seed state across Python, NumPy, and PyTorch (if available).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
