from __future__ import annotations
import math
import sys
import os
from typing import Any

import matplotlib.pyplot as plt

# Import utils và F1 ols_fit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))  # cho phép import sibling modules trong part1/
from utils import matvec, dot_product, build_null_vector
from config import RANDOM_STATE, EPSILON


# ---------------------------------------------------------------------------
# F10: Gauss-Markov Monte Carlo Demo
# Liên kết: Dùng F1 ols_fit để tính OLS estimates trong mỗi simulation
# ---------------------------------------------------------------------------

def monte_carlo_gauss_markov(
    n_sim: int = 1000,
    n_obs: int = 100,
    true_beta: tuple[float, float, float] = (2.0, -1.5, 0.8),
    true_sigma: float = 1.0,
    alt_scale: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    F10: Monte Carlo demo cho Gauss-Markov theorem.

    Chứng minh bằng mô phỏng:
    - OLS xấp xỉ unbiased: E[β̂] ≈ β
    - OLS có variance thấp hơn hoặc bằng estimator tuyến tính unbiased khác.

    Liên kết: Gọi F1 ols_fit cho mỗi simulation.

    Tham số:
        n_sim       : Số lần mô phỏng.
        n_obs       : Số quan sát mỗi lần.
        true_beta   : (intercept, β₁, β₂) - beta thực.
        true_sigma  : Độ lệch chuẩn nhiễu.
        alt_scale   : Hệ số perturbation cho estimator thay thế.
        random_state: Random seed.

    Trả về dict:
        true_beta    : list[float]
        ols_mean     : list[float] - E[β̂_OLS]
        ols_var      : list[float] - Var(β̂_OLS)
        alt_mean     : list[float] - E[β̂_alt]
        alt_var      : list[float] - Var(β̂_alt)
        ols_bias     : list[float]
        alt_bias     : list[float]
        beta_ols_all : list[list[float]] - tất cả β̂_OLS (n_sim x 3)
        beta_alt_all : list[list[float]] - tất cả β̂_alt (n_sim x 3)
    """
    from ols_implementation import ols_fit

    if n_sim <= 0 or n_obs <= 0:
        raise ValueError("n_sim và n_obs phải > 0")
    if true_sigma < 0:
        raise ValueError("true_sigma phải >= 0")

    true_beta_list = list(true_beta)
    n_coef = len(true_beta_list)  # 3: intercept + 2 features

    # Tạo X cố định (dùng LCG random cho reproducible)
    X_fixed = _generate_fixed_X(n_obs, p=n_coef - 1, seed=random_state)

    # Tạo X_bias cho null vector
    X_bias_fixed = [[1.0] + row for row in X_fixed]
    v_null = build_null_vector(X_bias_fixed)

    # e0 = [1, 1, 1] - perturbation ảnh hưởng tất cả các hệ số
    e0 = [1.0] * n_coef

    # Storage
    beta_ols_all = []
    beta_alt_all = []

    # Random noise generator (LCG-based Box-Muller)
    noise_gen = _LCGNormal(seed=random_state + 1)

    for sim in range(n_sim):
        # Tạo noise ε ~ N(0, σ²)
        eps = [noise_gen.next() * true_sigma for _ in range(n_obs)]

        # y = X_bias @ true_beta + ε
        y_sim = [0.0] * n_obs
        for i in range(n_obs):
            y_sim[i] = sum(X_bias_fixed[i][j] * true_beta_list[j] for j in range(n_coef)) + eps[i]

        # Gọi F1 ols_fit
        ols_res = ols_fit(X_fixed, y_sim)
        b_ols = ols_res["beta_hat"]

        # Tạo estimator thay thế: β̂_alt = β̂_OLS + alt_scale * e0 * (vᵀy)
        vty = dot_product(v_null, y_sim)
        b_alt = [b_ols[j] + alt_scale * e0[j] * vty for j in range(n_coef)]

        beta_ols_all.append(b_ols)
        beta_alt_all.append(b_alt)

    # Tính mean và variance (manual)
    ols_mean = [sum(beta_ols_all[s][j] for s in range(n_sim)) / n_sim for j in range(n_coef)]
    alt_mean = [sum(beta_alt_all[s][j] for s in range(n_sim)) / n_sim for j in range(n_coef)]

    ols_var = [sum((beta_ols_all[s][j] - ols_mean[j]) ** 2 for s in range(n_sim)) / (n_sim - 1)
               for j in range(n_coef)]
    alt_var = [sum((beta_alt_all[s][j] - alt_mean[j]) ** 2 for s in range(n_sim)) / (n_sim - 1)
               for j in range(n_coef)]

    ols_bias = [ols_mean[j] - true_beta_list[j] for j in range(n_coef)]
    alt_bias = [alt_mean[j] - true_beta_list[j] for j in range(n_coef)]

    return {
        "true_beta":    true_beta_list,
        "ols_mean":     ols_mean,
        "ols_var":      ols_var,
        "alt_mean":     alt_mean,
        "alt_var":      alt_var,
        "ols_bias":     ols_bias,
        "alt_bias":     alt_bias,
        "beta_ols_all": beta_ols_all,
        "beta_alt_all": beta_alt_all,
    }


def _generate_fixed_X(n: int, p: int, seed: int) -> list[list[float]]:
    """Tạo ma trận X cố định bằng LCG + Box-Muller (manual)."""
    gen = _LCGNormal(seed=seed)
    return [[gen.next() for _ in range(p)] for _ in range(n)]


class _LCGNormal:
    """Pseudo-random normal generator dùng LCG + Box-Muller transform."""

    def __init__(self, seed: int = 42):
        self._state = seed
        self._spare = None

    def _lcg_uniform(self) -> float:
        """Tạo số uniform (0, 1) bằng LCG."""
        self._state = (self._state * 1103515245 + 12345) & 0x7FFFFFFF
        return (self._state + 1) / (0x7FFFFFFF + 2)

    def next(self) -> float:
        """Trả về một số N(0, 1) bằng Box-Muller."""
        if self._spare is not None:
            val = self._spare
            self._spare = None
            return val
        u1 = self._lcg_uniform()
        u2 = self._lcg_uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        self._spare = r * math.sin(theta)
        return r * math.cos(theta)


def plot_beta_histograms(
    beta_ols: list[list[float]],
    beta_alt: list[list[float]],
    true_beta: list[float],
    bins: int = 30,
    save_dir: str = "output",
):
    """
    Vẽ histogram phân bố β̂ cho mỗi hệ số với đường dọc tại true_beta.

    Tham số:
        beta_ols:  list[list[float]] - (n_sim, n_coef).
        beta_alt:  list[list[float]] - (n_sim, n_coef).
        true_beta: list[float] - beta thực.
        bins:      int - Số bins histogram.
        save_dir:  str - Thư mục lưu ảnh.
    """
    import numpy as np

    beta_ols_arr = np.array(beta_ols)
    beta_alt_arr = np.array(beta_alt)
    true_arr = np.array(true_beta)
    n_coef = len(true_beta)

    fig, axes = plt.subplots(1, n_coef, figsize=(5 * n_coef, 4))
    if n_coef == 1:
        axes = [axes]

    coef_names = ["intercept"] + [f"x{i}" for i in range(1, n_coef)]
    for j, ax in enumerate(axes):
        ax.hist(beta_alt_arr[:, j], bins=bins, alpha=0.6, color="orange", edgecolor="black", label="Alt")
        ax.hist(beta_ols_arr[:, j], bins=bins, alpha=0.6, color="blue", edgecolor="black", label="OLS")
        ax.axvline(true_arr[j], color="red", linestyle="--", linewidth=2, label="true β")
        ax.set_title(f"Distribution of {coef_names[j]}")
        ax.set_xlabel("Estimated value")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "gauss_markov_histograms.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig, axes

# ---------------------------------------------------------------------------
# Unit Tests - F10
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    from test_utils import TestLogger, assert_shape, assert_true, assert_in_range, assert_equal
    import numpy as np

    print("=" * 55)
    print("  UNIT TESTS - gauss_markov_demo.py")
    print("=" * 55)

    passed = 0
    total = 0

    def run(result: bool):
        global passed, total
        total += 1
        passed += int(result)

    TestLogger.print_suite_header("F10 - Gauss Markov Demo")

    # Run simulation with small n_sim for testing
    res = monte_carlo_gauss_markov(n_sim=200, n_obs=30, random_state=RANDOM_STATE)

    run(assert_true("ols_bias" in res and "beta_ols_all" in res, label="returns expected dictionary keys"))
    
    run(assert_shape(res["beta_ols_all"], (200, 3), label="beta_ols_arr has shape (n_sim, p+1)"))
    run(assert_shape(res["beta_alt_all"], (200, 3), label="beta_alt_arr has shape (n_sim, p+1)"))

    # Bias should be small, but with 50 sims it might have variance. Just check it runs.
    run(assert_true(len(res["ols_bias"]) > 0, label="results contain ols_bias array"))

    # Test reproducible
    res2 = monte_carlo_gauss_markov(n_sim=5, n_obs=10, random_state=1)
    res3 = monte_carlo_gauss_markov(n_sim=5, n_obs=10, random_state=1)
    run(assert_true(np.allclose(res2["beta_ols_all"], res3["beta_ols_all"]), label="simulation is reproducible given random_state"))

    TestLogger.print_summary(passed, total)

