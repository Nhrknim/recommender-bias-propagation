import numpy as np
import scipy.sparse as sp
from typing import Tuple

def merge_and_deduplicate(
    D_t: sp.csr_matrix, 
    clicks_coo: Tuple[np.ndarray, np.ndarray]
) -> sp.csr_matrix:
    """
    Merges newly simulated interaction clicks into the accumulating 
    sparse interaction matrix D_t and strictly binarizes non-zero values.

    Args:
        D_t: Current sparse binary interaction matrix.
            Shape: (|U|, |I|)
        clicks_coo: Tuple of 1D arrays (user_indices, item_indices).
            Shape: (2, N_clicks) where N_clicks is count of new interaction pairs.

    Returns:
        Updated sparse binary interaction matrix D_{t+1}.
            Shape: (|U|, |I|)
    """
    u_idx, i_idx = clicks_coo
    n_users, n_items = D_t.shape
    
    if len(u_idx) == 0:
        return D_t.copy()
        
    # Construct COO sparse matrix for new clicks
    delta_coo = sp.coo_matrix(
        (np.ones_like(u_idx, dtype=np.float32), (u_idx, i_idx)), 
        shape=(n_users, n_items)
    )
    
    # Merge matrices via addition and convert to CSR layout
    D_next = (D_t + delta_coo).tocsr()
    
    # Deduplicate: Force all non-zero entries strictly to 1.0 (binary implicit signal)
    D_next.data[:] = 1.0
    
    # Boundary & Invariant Assertions
    assert not np.isnan(D_next.data).any(), "NaN detected in interaction matrix buffer"
    assert np.all(D_next.data == 1.0), "Non-binary interaction signal detected in buffer"
    
    return D_next
