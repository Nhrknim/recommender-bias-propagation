# Recommender Bias Propagation

## Data

This project uses the public MovieLens 1M dataset from GroupLens:

https://files.grouplens.org/datasets/movielens/ml-1m.zip

Large dataset artifacts are intentionally not committed:

- `ml-1m.zip`
- `ml-1m/`

To reproduce the local data files, run `phase_1_data_prep.ipynb` from the repository root. The notebook downloads `ml-1m.zip` if neither the zip file nor the extracted `ml-1m/` directory exists, then extracts it before loading the ratings data.

Expected extracted files include:

- `ml-1m/ratings.dat`
- `ml-1m/movies.dat`
- `ml-1m/users.dat`
- `ml-1m/README`
