import unittest
import numpy as np
import scipy.sparse as sp
from src.data.buffer import merge_and_deduplicate

class TestBuffer(unittest.TestCase):

    def test_merge_and_deduplicate_basic(self):
        """Test merging new clicks into a sparse matrix."""
        # 3 users, 4 items matrix with one initial interaction: user 0 watched item 1
        D_0 = sp.csr_matrix(([1.0], ([0], [1])), shape=(3, 4), dtype=np.float32)
        
        # New clicks: user 1 watched item 2, user 2 watched item 3
        u_clicks = np.array([1, 2], dtype=np.int32)
        i_clicks = np.array([2, 3], dtype=np.int32)
        
        D_1 = merge_and_deduplicate(D_0, (u_clicks, i_clicks))
        
        self.assertEqual(D_1.shape, (3, 4))
        self.assertEqual(D_1.nnz, 3)
        self.assertEqual(D_1[0, 1], 1.0)
        self.assertEqual(D_1[1, 2], 1.0)
        self.assertEqual(D_1[2, 3], 1.0)

    def test_merge_and_deduplicate_deduplication(self):
        """Test that duplicate interactions are strictly binarized to 1.0."""
        D_0 = sp.csr_matrix(([1.0], ([0], [1])), shape=(3, 4), dtype=np.float32)
        
        # Duplicate click: user 0 watches item 1 AGAIN, plus user 0 watches item 1 a third time
        u_clicks = np.array([0, 0], dtype=np.int32)
        i_clicks = np.array([1, 1], dtype=np.int32)
        
        D_1 = merge_and_deduplicate(D_0, (u_clicks, i_clicks))
        
        self.assertEqual(D_1.shape, (3, 4))
        self.assertEqual(D_1.nnz, 1)
        self.assertEqual(D_1[0, 1], 1.0)  # Must be 1.0, NOT 3.0!

    def test_merge_and_deduplicate_empty(self):
        """Test handling of empty click arrays."""
        D_0 = sp.csr_matrix(([1.0], ([0], [1])), shape=(3, 4), dtype=np.float32)
        
        u_clicks = np.array([], dtype=np.int32)
        i_clicks = np.array([], dtype=np.int32)
        
        D_1 = merge_and_deduplicate(D_0, (u_clicks, i_clicks))
        
        self.assertEqual(D_1.shape, (3, 4))
        self.assertEqual(D_1.nnz, 1)
        self.assertEqual(D_1[0, 1], 1.0)

if __name__ == "__main__":
    unittest.main()
