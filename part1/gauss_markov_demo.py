from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def _ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_bias = _add_bias(X)
    xtx = X_bias.T @ X_bias
    xty = X_bias.T @ y
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx) @ xty


def _build_unbiased_perturbation_vector(X_bias: np.ndarray) -> np.ndarray:
    """
    Build a vector v in Null(X_bias^T), i.e., orthogonal to all columns of X_bias.
    """
    u, s, _ = np.linalg.svd(X_bias, full_matrices=True)
    rank = int(np.sum(s > 1e-12))
    if rank >= X_bias.shape[0]:
        raise ValueError("No non-trivial null space for X_bias^T; choose n_obs > p + 1")
    v = u[:, rank]
    return v / np.linalg.norm(v)


def monte_carlo_gauss_markov(
    n_sim: int = 1000,
    n_obs: int = 100,
    true_beta: tuple[float, float, float] = (2.0, -1.5, 0.8),
    true_sigma: float = 1.0,
    alt_scale: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Monte Carlo demo for Gauss-Markov:
    - OLS is approximately unbiased: E[beta_hat] ~= beta
    - OLS has lower/equal variance than another linear unbiased estimator.
    """
    if n_sim <= 0 or n_obs <= 0:
        raise ValueError("n_sim and n_obs must be positive")
    if true_sigma < 0:
        raise ValueError("true_sigma must be non-negative")

    true_beta_np = np.asarray(true_beta, dtype=float)
    if true_beta_np.shape != (3,):
        raise ValueError("true_beta must contain 3 values: intercept, b1, b2")

    rng = np.random.default_rng(random_state)
    X_fixed = rng.normal(0.0, 1.0, size=(n_obs, 2))
    X_bias_fixed = _add_bias(X_fixed)

    beta_ols = np.zeros((n_sim, 3), dtype=float)
    beta_alt = np.zeros((n_sim, 3), dtype=float)
    v_null = _build_unbiased_perturbation_vector(X_bias_fixed)
    e0 = np.array([1.0, 0.0, 0.0], dtype=float)

    for sim in range(n_sim):
        eps = rng.normal(0.0, true_sigma, size=n_obs)
        y_sim = X_bias_fixed @ true_beta_np + eps

        beta_ols[sim] = _ols_beta(X_fixed, y_sim)
        # beta_alt = beta_ols + alt_scale * e0 * (v_null^T y)
        # This remains linear unbiased because v_null^T X_bias = 0.
        beta_alt[sim] = beta_ols[sim] + alt_scale * e0 * float(v_null @ y_sim)

    ols_mean = beta_ols.mean(axis=0)
    ols_var = beta_ols.var(axis=0, ddof=1)
    alt_mean = beta_alt.mean(axis=0)
    alt_var = beta_alt.var(axis=0, ddof=1)

    summary_df = pd.DataFrame(
        {
            "true_beta": true_beta_np,
            "ols_mean": ols_mean,
            "ols_var": ols_var,
            "alt_mean": alt_mean,
            "alt_var": alt_var,
        },
        index=["intercept", "x1", "x2"],
    )
    summary_df["ols_bias"] = summary_df["ols_mean"] - summary_df["true_beta"]
    summary_df["alt_bias"] = summary_df["alt_mean"] - summary_df["true_beta"]

    return {
        "X_fixed": X_fixed,
        "true_beta": true_beta_np,
        "beta_ols_arr": beta_ols,
        "beta_alt_arr": beta_alt,
        "summary": summary_df,
    }


def plot_beta_histograms(
    beta_samples: np.ndarray,
    true_beta: np.ndarray | list[float] | tuple[float, ...],
    bins: int = 30,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one histogram per coefficient with a vertical line at true_beta.
    """
    beta_arr = np.asarray(beta_samples, dtype=float)
    true_beta_arr = np.asarray(true_beta, dtype=float)
    if beta_arr.ndim != 2:
        raise ValueError("beta_samples must have shape (n_sim, n_coef)")
    if true_beta_arr.ndim != 1 or true_beta_arr.shape[0] != beta_arr.shape[1]:
        raise ValueError("true_beta must have length equal to number of coefficients")

    n_coef = beta_arr.shape[1]
    fig, axes = plt.subplots(1, n_coef, figsize=(5 * n_coef, 4))
    if n_coef == 1:
        axes = np.array([axes])

    coef_names = ["intercept"] + [f"x{i}" for i in range(1, n_coef)]
    for j, ax in enumerate(axes):
        ax.hist(beta_arr[:, j], bins=bins, alpha=0.75, edgecolor="black", label="beta_hat")
        ax.axvline(true_beta_arr[j], color="red", linestyle="--", linewidth=2, label="true_beta")
        ax.set_title(f"Distribution of {coef_names[j]}")
        ax.set_xlabel("Estimated value")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    return fig, axes
