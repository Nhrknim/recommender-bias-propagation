import unittest
import os
import tempfile
import numpy as np
import pandas as pd
import scipy.sparse as sp
from src.data.dataset import parse_and_split_dataframe, load_dataset

class TestDataset(unittest.TestCase):

    def setUp(self):
        """Create a synthetic ratings DataFrame with raw non-contiguous IDs."""
        # Raw user IDs: [10, 20] (Gaps in IDs!)
        # Raw item IDs: [100, 200, 300, 400, 500] (Gaps in IDs!)
        data = [
            # User 10 (chronological ratings)
            {"user_id": 10, "item_id": 100, "rating": 5.0, "timestamp": 1000},
            {"user_id": 10, "item_id": 200, "rating": 4.0, "timestamp": 1001},
            {"user_id": 10, "item_id": 300, "rating": 5.0, "timestamp": 1002},
            {"user_id": 10, "item_id": 400, "rating": 5.0, "timestamp": 1003},
            {"user_id": 10, "item_id": 500, "rating": 4.0, "timestamp": 1004}, # 5 interactions -> 80% train = 4, 20% test = 1
            
            # User 20
            {"user_id": 20, "item_id": 200, "rating": 4.0, "timestamp": 1000},
            {"user_id": 20, "item_id": 300, "rating": 2.0, "timestamp": 1001}, # Rating 2.0 -> Filtered out!
            {"user_id": 20, "item_id": 400, "rating": 5.0, "timestamp": 1002},
            {"user_id": 20, "item_id": 500, "rating": 4.0, "timestamp": 1003},
            {"user_id": 20, "item_id": 100, "rating": 5.0, "timestamp": 1004}, # 4 interactions -> 80% train = 3, 20% test = 1
        ]
        self.df = pd.DataFrame(data)

    def test_continuous_zero_indexing(self):
        """Test mapping raw non-contiguous IDs into continuous 0-indexed codes."""
        train_mat, test_mat, user2code, item2code = parse_and_split_dataframe(
            self.df, threshold=4.0, split_ratio=0.8
        )
        
        # 2 unique users (10, 20) -> codes [0, 1]
        self.assertEqual(len(user2code), 2)
        self.assertIn(10, user2code)
        self.assertIn(20, user2code)
        self.assertEqual(set(user2code.values()), {0, 1})
        
        # 5 unique items (100, 200, 300, 400, 500) -> codes [0, 1, 2, 3, 4]
        self.assertEqual(len(item2code), 5)
        self.assertEqual(set(item2code.values()), {0, 1, 2, 3, 4})
        
        # Shape of matrices must strictly match continuous count (|U|, |I|) = (2, 5)
        self.assertEqual(train_mat.shape, (2, 5))
        self.assertEqual(test_mat.shape, (2, 5))

    def test_implicit_filtering_and_split(self):
        """Test implicit binarization filtering (>= 4.0) and 80/20 per-user split."""
        train_mat, test_mat, user2code, item2code = parse_and_split_dataframe(
            self.df, threshold=4.0, split_ratio=0.8
        )
        
        # User 10 had 5 positive ratings -> 4 in train, 1 in test
        # User 20 had 4 positive ratings (1 filtered out) -> 3 in train, 1 in test
        self.assertEqual(train_mat[user2code[10]].nnz, 4)
        self.assertEqual(test_mat[user2code[10]].nnz, 1)
        
        self.assertEqual(train_mat[user2code[20]].nnz, 3)
        self.assertEqual(test_mat[user2code[20]].nnz, 1)
        
        # Check binary integrity (all non-zero values must be 1.0)
        np.testing.assert_array_equal(train_mat.data, 1.0)
        np.testing.assert_array_equal(test_mat.data, 1.0)

    def test_dataset_agnostic_custom_columns(self):
        """Test parse_and_split_dataframe with custom dataset column headers."""
        custom_df = self.df.rename(columns={
            "user_id": "customer_id",
            "item_id": "product_id",
            "rating": "stars",
            "timestamp": "time_sec"
        })
        train_mat, test_mat, u2c, i2c = parse_and_split_dataframe(
            custom_df,
            threshold=4.0,
            split_ratio=0.8,
            user_col="customer_id",
            item_col="product_id",
            rating_col="stars",
            timestamp_col="time_sec"
        )
        self.assertEqual(train_mat.shape, (2, 5))
        self.assertEqual(len(u2c), 2)
        self.assertEqual(len(i2c), 5)

    def test_load_dataset_file_ingestion(self):
        """Test generic load_dataset function with a temporary TSV file."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".tsv") as tmp:
            tmp.write("userId\tmovieId\tscore\ttime\n")
            tmp.write("u1\tm1\t5.0\t100\n")
            tmp.write("u1\tm2\t4.0\t101\n")
            tmp.write("u1\tm3\t4.0\t102\n")
            tmp.write("u1\tm4\t5.0\t103\n")
            tmp.write("u1\tm5\t4.0\t104\n")
            tmp.write("u2\tm1\t4.0\t100\n")
            tmp.write("u2\tm2\t2.0\t101\n")
            tmp.write("u2\tm3\t4.0\t102\n")
            tmp.write("u2\tm4\t4.0\t103\n")
            tmp.write("u2\tm5\t5.0\t104\n")
            tmp_path = tmp.name

        try:
            train_mat, test_mat, u2c, i2c = load_dataset(
                file_path=tmp_path,
                sep="\t",
                threshold=4.0,
                split_ratio=0.8,
                col_map={"userId": "user_id", "movieId": "item_id", "score": "rating", "time": "timestamp"}
            )
            self.assertEqual(train_mat.shape, (2, 5))
            self.assertIn("u1", u2c)
            self.assertIn("m1", i2c)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()

