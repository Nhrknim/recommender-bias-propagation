import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple, Dict, Optional, Any

def parse_and_split_dataframe(
    df: pd.DataFrame, 
    threshold: float = 4.0, 
    split_ratio: float = 0.8,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    timestamp_col: str = "timestamp"
) -> Tuple[sp.csr_matrix, sp.csr_matrix, Dict[Any, int], Dict[Any, int]]:
    """
    Core dataset-agnostic processing pipeline: performs implicit binarization,
    chronological per-user sorting, continuous zero-indexing, and
    constructs per-user train/test split sparse matrices.

    Args:
        df: DataFrame containing user, item, rating, and timestamp columns.
        threshold: Minimum rating score to treat as positive implicit feedback (1.0).
        split_ratio: Ratio of historical user interactions assigned to training matrix.
        user_col: Column name representing user identifiers.
        item_col: Column name representing item identifiers.
        rating_col: Column name representing rating/interaction scores.
        timestamp_col: Column name representing interaction timestamps.

    Returns:
        train_matrix: CSR matrix of shape (|U|, |I|) for baseline training D_0.
        test_matrix: CSR matrix of shape (|U|, |I|) for held-out evaluation.
        user2code: Dict mapping raw user_id -> continuous user_code [0..|U|-1].
        item2code: Dict mapping raw item_id -> continuous item_code [0..|I|-1].
    """
    for col in [user_col, item_col, rating_col, timestamp_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}")

    # 1. Implicit Feedback Binarization (Keep rating >= threshold)
    liked_df = df[df[rating_col] >= threshold].copy()
    
    if len(liked_df) == 0:
        raise ValueError(f"No interactions found satisfying rating ({rating_col}) >= {threshold}")
    
    # 2. Chronological Per-User Sorting (Prevent temporal data leakage)
    liked_df = liked_df.sort_values(by=[user_col, timestamp_col])
    
    # 3. Continuous Zero-Indexing Mapping
    user_categories = pd.Categorical(liked_df[user_col])
    item_categories = pd.Categorical(liked_df[item_col])
    
    liked_df["user_code"] = user_categories.codes
    liked_df["item_code"] = item_categories.codes
    
    num_users = len(user_categories.categories)
    num_items = len(item_categories.categories)
    
    # Mappings from raw ID to continuous zero-indexed code
    user2code = {raw_id: code for code, raw_id in enumerate(user_categories.categories)}
    item2code = {raw_id: code for code, raw_id in enumerate(item_categories.categories)}
    
    # 4. Per-User Train/Test Split
    train_rows, train_cols = [], []
    test_rows, test_cols = [], []
    
    for user_code, group in liked_df.groupby("user_code"):
        n_interactions = len(group)
        split_idx = int(n_interactions * split_ratio)
        
        user_items = group["item_code"].values
        
        # Train slice (first split_ratio fraction)
        train_rows.extend([user_code] * split_idx)
        train_cols.extend(user_items[:split_idx])
        
        # Test slice (remaining fraction)
        test_rows.extend([user_code] * (n_interactions - split_idx))
        test_cols.extend(user_items[split_idx:])
        
    # Construct COO sparse matrices and convert to CSR layout with locked shapes
    train_matrix = sp.csr_matrix(
        (np.ones(len(train_rows), dtype=np.float32), (train_rows, train_cols)),
        shape=(num_users, num_items)
    )
    test_matrix = sp.csr_matrix(
        (np.ones(len(test_rows), dtype=np.float32), (test_rows, test_cols)),
        shape=(num_users, num_items)
    )
    
    # Boundary & Alignment Assertions
    assert train_matrix.shape == test_matrix.shape, "Train and Test matrix dimensions misaligned"
    assert not np.isnan(train_matrix.data).any(), "NaN detected in train matrix"
    assert not np.isnan(test_matrix.data).any(), "NaN detected in test matrix"
    
    return train_matrix, test_matrix, user2code, item2code

def load_dataset(
    file_path: str,
    sep: str = ",",
    threshold: float = 4.0,
    split_ratio: float = 0.8,
    col_map: Optional[Dict[str, str]] = None,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    timestamp_col: str = "timestamp",
    **read_csv_kwargs
) -> Tuple[sp.csr_matrix, sp.csr_matrix, Dict[Any, int], Dict[Any, int]]:
    """
    Generic dataset loader capable of ingesting CSV, TSV, DAT files from any dataset.

    Args:
        file_path: Path to the interaction dataset file.
        sep: Delimiter character (e.g. ',', '\t', '::').
        threshold: Minimum rating score to treat as positive implicit feedback.
        split_ratio: Ratio of historical interactions allocated to training matrix.
        col_map: Optional dict mapping dataset column names to standard names
                 (e.g., {'userId': 'user_id', 'movieId': 'item_id'}).
        user_col: Final column name for user IDs.
        item_col: Final column name for item IDs.
        rating_col: Final column name for ratings.
        timestamp_col: Final column name for timestamps.
        **read_csv_kwargs: Additional keyword arguments passed to pd.read_csv.

    Returns:
        train_matrix: CSR matrix of shape (|U|, |I|) for baseline training D_0.
        test_matrix: CSR matrix of shape (|U|, |I|) for held-out evaluation.
        user2code: Dict mapping raw user_id -> continuous user_code.
        item2code: Dict mapping raw item_id -> continuous item_code.
    """
    df = pd.read_csv(file_path, sep=sep, **read_csv_kwargs)
    if col_map:
        df = df.rename(columns=col_map)
    
    return parse_and_split_dataframe(
        df,
        threshold=threshold,
        split_ratio=split_ratio,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        timestamp_col=timestamp_col
    )

def load_movielens_1m(
    ratings_path: str = "ml-1m/ratings.dat", 
    threshold: float = 4.0, 
    split_ratio: float = 0.8
) -> Tuple[sp.csr_matrix, sp.csr_matrix, Dict[Any, int], Dict[Any, int]]:
    """
    Ingests MovieLens-1M dataset from raw DAT file and runs processing pipeline.
    Convenience wrapper around generic load_dataset().
    """
    return load_dataset(
        file_path=ratings_path,
        sep="::",
        threshold=threshold,
        split_ratio=split_ratio,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1"
    )

def load_movielens_genres(
    movies_path: str = "ml-1m/movies.dat", 
    item2code: Optional[Dict[Any, int]] = None
) -> np.ndarray:
    """
    Parses MovieLens-1M genre metadata into a binary 2D array of shape (|I|, C)
    aligned to global continuous item codes.

    Args:
        movies_path: Path to ml-1m/movies.dat file.
        item2code: Dict mapping raw item_id -> continuous item_code.

    Returns:
        genre_matrix: 2D float32 array of shape (|I|, C) containing genre indicator probabilities.
    """
    df_movies = pd.read_csv(
        movies_path,
        sep="::",
        names=["item_id", "title", "genres"],
        engine="python",
        encoding="latin-1"
    )

    all_genres = set()
    for genre_str in df_movies["genres"]:
        for g in str(genre_str).split("|"):
            all_genres.add(g)

    genre_list = sorted(list(all_genres))
    genre2idx = {g: idx for idx, g in enumerate(genre_list)}
    n_genres = len(genre_list)

    n_items = len(item2code) if item2code else len(df_movies)
    genre_matrix = np.zeros((n_items, n_genres), dtype=np.float32)

    for _, row in df_movies.iterrows():
        raw_id = row["item_id"]
        if item2code and raw_id not in item2code:
            continue
            
        code = item2code[raw_id] if item2code else raw_id
        for g in str(row["genres"]).split("|"):
            if g in genre2idx:
                genre_matrix[code, genre2idx[g]] = 1.0

    # Normalize rows so each item's genre vector sums to 1.0
    row_sums = np.sum(genre_matrix, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    genre_matrix = genre_matrix / row_sums

    return genre_matrix


