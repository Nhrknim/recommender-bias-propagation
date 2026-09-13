# Dynamic bias propagation simulator

1. We support implicit feedback for now (0/1).
2. We use in ./src/data/buffer different matrix representation methods:
   - COO: Coordinate Format
   - CSR: Compressed Sparse Row
   For more details on sparse matrix formats, see: https://docs.scipy.org/doc/scipy/reference/sparse.html#sparse-matrix-formats
   