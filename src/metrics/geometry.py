import numpy as np
import scipy.linalg
from typing import Tuple

def compute_reflection_corrected_procrustes(V_0: np.ndarray, V_t: np.ndarray) -> np.ndarray:
    """
    Solves Orthogonal Procrustes alignment R* = V_svd^T U^T subject to det(R*) = +1
    to guarantee proper rigid rotation without improper reflection.

    Args:
        V_0: Baseline ground-truth item factor matrix, shape (|I|, d).
        V_t: Generation t item factor matrix, shape (|I|, d).

    Returns:
        V_t_aligned: Rotated item factor matrix V_t * R* of shape (|I|, d).
    """
    assert V_0.shape == V_t.shape, f"Matrices shape mismatch: V_0 {V_0.shape} vs V_t {V_t.shape}"

    # Cross-covariance matrix M = V_0^T * V_t
    M = V_0.T @ V_t

    # SVD decomposition M = U * Sigma * Vt_svd
    U, _, Vt_svd = np.linalg.svd(M)

    # Candidate rotation matrix R* = (U * Vt_svd)^T = Vt_svd^T * U^T
    V_svd_T = Vt_svd.T
    U_T = U.T
    R_star = V_svd_T @ U_T

    # Reflection Correction: enforce det(R*) = +1 using clean scipy.linalg.det
    det_val = float(scipy.linalg.det(R_star.astype(np.float64)))
    if det_val < 0:
        V_svd_T_corrected = V_svd_T.copy()
        V_svd_T_corrected[:, -1] *= -1.0
        R_star = V_svd_T_corrected @ U_T
        det_val = float(scipy.linalg.det(R_star.astype(np.float64)))

    assert det_val > 0, f"Determinant of R* must be +1.0 (proper rotation), got {det_val}"

    V_t_aligned = V_t @ R_star
    assert V_t_aligned.shape == V_0.shape, "Aligned matrix shape misaligned"
    assert not np.isnan(V_t_aligned).any(), "NaN detected in Procrustes aligned matrix"

    return V_t_aligned.astype(np.float32)



def compute_relative_norm_shift(
    V_0: np.ndarray, 
    V_t: np.ndarray, 
    epsilon: float = 1e-9
) -> np.ndarray:
    """
    Computes Smooth Relative L2 Vector Norm Shift (R_norm).

    R_norm(i, t) = ( ||v_i^{(t)}||_2 + epsilon ) / ( ||v_i^{(0)}||_2 + epsilon )

    Args:
        V_0: Baseline item factor matrix, shape (|I|, d).
        V_t: Generation t item factor matrix, shape (|I|, d).
        epsilon: Numerical cushion to prevent zero division.

    Returns:
        r_norm: 1D array of shape (|I|,) with relative norm shift values per item.
    """
    norms_0 = np.linalg.norm(V_0, axis=1)
    norms_t = np.linalg.norm(V_t, axis=1)

    r_norm = (norms_t + epsilon) / (norms_0 + epsilon)
    assert not np.isnan(r_norm).any(), "NaN detected in relative norm shift"

    return r_norm.astype(np.float32)


def compute_mag_ang_decomposition(
    V_0: np.ndarray, 
    V_t_aligned: np.ndarray, 
    epsilon: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes total squared Euclidean distance into magnitude shift and angular drift:

    ||v_0 - v_t_aligned||^2 = (||v_0|| - ||v_t||)^2 + 2 ||v_0|| ||v_t|| (1 - cos(theta))

    Args:
        V_0: Baseline item factor matrix, shape (|I|, d).
        V_t_aligned: Procrustes aligned item factor matrix, shape (|I|, d).
        epsilon: Numerical cushion.

    Returns:
        delta_mag: 1D array of magnitude shift values per item, shape (|I|,).
        delta_ang: 1D array of angular rotation shift values per item, shape (|I|,).
    """
    norms_0 = np.linalg.norm(V_0, axis=1)
    norms_t = np.linalg.norm(V_t_aligned, axis=1)

    # Magnitude shift component
    delta_mag = (norms_0 - norms_t) ** 2.0

    # Angular shift component
    dot_products = np.sum(V_0 * V_t_aligned, axis=1)
    cos_theta = np.clip(dot_products / ((norms_0 * norms_t) + epsilon), -1.0, 1.0)
    delta_ang = 2.0 * norms_0 * norms_t * (1.0 - cos_theta)

    assert not np.isnan(delta_mag).any(), "NaN in delta_mag"
    assert not np.isnan(delta_ang).any(), "NaN in delta_ang"

    return delta_mag.astype(np.float32), delta_ang.astype(np.float32)


def compute_effective_rank(V_t: np.ndarray, epsilon: float = 1e-9) -> float:
    """
    Computes Effective Rank S_eff via spectral entropy over singular values of V_t:

    sigma_tilde_k = sigma_k / sum_j sigma_j
    S_eff(t) = exp( - sum_{k=1}^d sigma_tilde_k * ln(sigma_tilde_k) )

    Args:
        V_t: Item factor matrix of shape (|I|, d).
        epsilon: Numerical cushion for log.

    Returns:
        s_eff: Continuous effective rank value in range [1.0, d].
    """
    # Compute singular values
    singular_vals = np.linalg.svd(V_t, compute_uv=False)
    total_singular_sum = np.sum(singular_vals)

    if total_singular_sum == 0.0:
        return 1.0

    # Normalized singular value distribution
    p_k = singular_vals / total_singular_sum
    p_k_smooth = p_k + epsilon

    # Spectral entropy
    entropy = -np.sum(p_k_smooth * np.log(p_k_smooth))
    s_eff = float(np.exp(entropy))

    dim = V_t.shape[1]
    s_eff = float(np.clip(s_eff, 1.0, float(dim)))

    assert 1.0 <= s_eff <= float(dim) + 1e-3, f"Effective rank out of bounds [1, {dim}]: {s_eff}"
    return s_eff


def compute_drift_metrics_32d(
    V_0: np.ndarray, 
    V_t_aligned: np.ndarray, 
    epsilon: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes direct 32-D quantitative geometry metrics per item between V_0 and V_t_aligned:
    1. Euclidean drift: ||v_i^{(t)\text{aligned}} - v_i^{(0)}||_2
    2. Cosine drift: 1 - cos(v_i^{(t)\text{aligned}}, v_i^{(0)})
    3. Generation t vector L2 norms: ||v_i^{(t)}||_2
    4. Baseline vector L2 norms: ||v_i^{(0)}||_2

    Args:
        V_0: Baseline item factor matrix of shape (|I|, d).
        V_t_aligned: Procrustes aligned item factor matrix of shape (|I|, d).
        epsilon: Numerical stability cushion.

    Returns:
        euclidean_drift: 1D array of shape (|I|,) with L2 displacement distances.
        cosine_drift: 1D array of shape (|I|,) with angular cosine drift in [0, 2].
        norms_t: 1D array of shape (|I|,) with generation t vector norms.
        norms_0: 1D array of shape (|I|,) with baseline vector norms.
    """
    assert V_0.shape == V_t_aligned.shape, f"Shape mismatch: {V_0.shape} vs {V_t_aligned.shape}"

    # 1. Euclidean Drift
    euclidean_drift = np.linalg.norm(V_t_aligned - V_0, axis=1)

    # 2. Vector Norms
    norms_0 = np.linalg.norm(V_0, axis=1)
    norms_t = np.linalg.norm(V_t_aligned, axis=1)

    # 3. Cosine Drift
    dot_products = np.sum(V_0 * V_t_aligned, axis=1)
    denom = (norms_0 * norms_t) + epsilon
    cos_sim = np.clip(dot_products / denom, -1.0, 1.0)
    cosine_drift = 1.0 - cos_sim

    assert not np.isnan(euclidean_drift).any(), "NaN in euclidean_drift"
    assert not np.isnan(cosine_drift).any(), "NaN in cosine_drift"
    assert not np.isnan(norms_t).any(), "NaN in norms_t"

    return (
        euclidean_drift.astype(np.float32),
        cosine_drift.astype(np.float32),
        norms_t.astype(np.float32),
        norms_0.astype(np.float32)
    )

