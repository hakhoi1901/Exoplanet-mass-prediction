from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def _as_2d_float_array(X: list[list[float]] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array-like")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column")
    return arr


def _as_1d_float_array(y: list[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must have at least one element")
    return arr


def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def _solve_stable(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def _ols_fit_numpy(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """
    Internal OLS helper used by vif.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    X_bias = _add_bias(X)
    n, p1 = X_bias.shape
    if n <= p1:
        raise ValueError("Need n > p + 1 to estimate OLS variance")

    xtx = X_bias.T @ X_bias
    xty = X_bias.T @ y
    beta_hat = _solve_stable(xtx, xty)
    y_hat = X_bias @ beta_hat
    residuals = y - y_hat
    rss = float(np.sum(residuals**2))
    sigma2_hat = rss / (n - p1)

    return {
        "beta_hat": beta_hat,
        "y_hat": y_hat,
        "residuals": residuals,
        "sigma2_hat": sigma2_hat,
    }


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_bar = float(np.mean(y_true))
    rss = float(np.sum((y_true - y_pred) ** 2))
    tss = float(np.sum((y_true - y_bar) ** 2))
    if np.isclose(tss, 0.0):
        return 1.0 if np.isclose(rss, 0.0) else 0.0
    return 1.0 - rss / tss


def coef_inference(
    X: list[list[float]] | np.ndarray,
    y: list[float] | np.ndarray,
    beta_hat: list[float] | np.ndarray,
    sigma2: float,
) -> pd.DataFrame:
    """
    Compute coefficient inference statistics for an OLS model.

    Returns a DataFrame with columns:
    coef, std_err, t_stat, p_value, ci_lower, ci_upper
    """
    X_np = _as_2d_float_array(X, "X")
    y_np = _as_1d_float_array(y, "y")
    beta_np = _as_1d_float_array(beta_hat, "beta_hat")

    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if beta_np.shape[0] != X_np.shape[1] + 1:
        raise ValueError("beta_hat must have shape (p + 1,)")
    if sigma2 < 0:
        raise ValueError("sigma2 must be non-negative")

    X_bias = _add_bias(X_np)
    n, p1 = X_bias.shape
    df = n - p1
    if df <= 0:
        raise ValueError("Need n > p + 1 for coefficient inference")

    xtx = X_bias.T @ X_bias
    xtx_inv = np.linalg.pinv(xtx)
    cov_beta = sigma2 * xtx_inv

    std_err = np.sqrt(np.clip(np.diag(cov_beta), a_min=0.0, a_max=None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.divide(
            beta_np,
            std_err,
            out=np.full_like(beta_np, np.nan, dtype=float),
            where=std_err > 0,
        )
    p_value = 2.0 * t.sf(np.abs(t_stat), df=df)
    t_crit = t.ppf(0.975, df=df)
    ci_lower = beta_np - t_crit * std_err
    ci_upper = beta_np + t_crit * std_err

    feature_names = ["intercept"] + [f"x{i}" for i in range(1, X_np.shape[1] + 1)]
    result = pd.DataFrame(
        {
            "coef": beta_np,
            "std_err": std_err,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
        index=feature_names,
    )
    return result


def vif(X: list[list[float]] | np.ndarray) -> dict[str, float]:
    """
    Compute Variance Inflation Factor (VIF) for each feature in X.
    """
    X_np = _as_2d_float_array(X, "X")
    n, p = X_np.shape
    if p < 2:
        raise ValueError("VIF requires at least 2 features")
    if n <= p:
        raise ValueError("Need n > p for stable VIF estimation")

    vifs: dict[str, float] = {}
    for j in range(p):
        x_j = X_np[:, j]
        X_others = np.delete(X_np, j, axis=1)
        result_j = _ols_fit_numpy(X_others, x_j)
        r2_j = _r2_score(x_j, result_j["y_hat"])

        if np.isclose(1.0 - r2_j, 0.0):
            vif_j = float("inf")
        else:
            vif_j = float(1.0 / (1.0 - r2_j))
        vifs[f"x{j + 1}"] = vif_j

    return vifs
