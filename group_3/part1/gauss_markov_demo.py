from __future__ import annotations
import math
import sys
import os
from typing import Any

import matplotlib.pyplot as plt
PART1_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))

# Import utils và F1 ols_fit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from utils import matvec, dot_product, build_null_vector
from config import RANDOM_STATE, EPSILON


# ---------------------------------------------------------------------------
# F10: Gauss-Markov Monte Carlo Demo
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
    - OLS xấp xỉ unbiased: E[beta_hat] ~= beta
    - OLS có variance thấp hơn hoặc bằng estimator tuyến tính unbiased khác.

    Liên kết: Gọi F1 ols_fit cho mỗi simulation.

    Tham số:
        n_sim       : Số lần mô phỏng.
        n_obs       : Số quan sát mỗi lần.
        true_beta   : (intercept, beta_1, beta_2) - beta thực.
        true_sigma  : Độ lệch chuẩn nhiễu.
        alt_scale   : Hệ số perturbation cho estimator thay thế.
        random_state: Random seed.

    Trả về dict:
        true_beta    : list[float]
        ols_mean     : list[float] - E[beta_hat_OLS]
        ols_var      : list[float] - Var(beta_hat_OLS)
        alt_mean     : list[float] - E[beta_hat_alt]
        alt_var      : list[float] - Var(beta_hat_alt)
        ols_bias     : list[float]
        alt_bias     : list[float]
        beta_ols_all : list[list[float]] - tất cả beta_hat_OLS (n_sim x 3)
        beta_alt_all : list[list[float]] - tất cả beta_hat_alt (n_sim x 3)
    """
    from part1.ols_implementation import ols_fit

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
        # Tạo noise epsilon ~ N(0, sigma^2)
        eps = [noise_gen.next() * true_sigma for _ in range(n_obs)]

        # y = X_bias @ true_beta + epsilon
        y_sim = [0.0] * n_obs
        for i in range(n_obs):
            y_sim[i] = sum(X_bias_fixed[i][j] * true_beta_list[j] for j in range(n_coef)) + eps[i]

        # Gọi F1 ols_fit
        ols_res = ols_fit(X_fixed, y_sim)
        b_ols = ols_res["beta_hat"]

        # Tạo estimator thay thế: beta_hat_alt = beta_hat_OLS + alt_scale * e0 * (v^T y)
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
    save_dir: str = PART1_OUTPUT_DIR,
):
    """
    Vẽ histogram phân bố beta_hat cho mỗi hệ số với đường dọc tại true_beta.

    Tham số:
        beta_ols:  list[list[float]] - (n_sim, n_coef).
        beta_alt:  list[list[float]] - (n_sim, n_coef).
        true_beta: list[float] - beta thực.
        bins:      int - Số bins histogram.
        save_dir:  str - Thư mục lưu ảnh.
    """
    n_coef = len(true_beta)

    fig, axes = plt.subplots(1, n_coef, figsize=(5 * n_coef, 4))
    if n_coef == 1:
        axes = [axes]

    coef_names = ["intercept"] + [f"x{i}" for i in range(1, n_coef)]
    for j, ax in enumerate(axes):
        beta_alt_col = [row[j] for row in beta_alt]
        beta_ols_col = [row[j] for row in beta_ols]
        ax.hist(beta_alt_col, bins=bins, alpha=0.6, color="orange", edgecolor="black", label="Alt")
        ax.hist(beta_ols_col, bins=bins, alpha=0.6, color="blue", edgecolor="black", label="OLS")
        ax.axvline(true_beta[j], color="red", linestyle="--", linewidth=2, label="true β")
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


def _nested_allclose(a: list[list[float]], b: list[list[float]], tol: float = 1e-10) -> bool:
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return False
        for x, y in zip(row_a, row_b):
            if abs(x - y) > tol:
                return False
    return True


# ---------------------------------------------------------------------------
# Unit Tests - F10
# ---------------------------------------------------------------------------


def test_monte_carlo_returns_expected_keys() -> bool:
    from test_utils import assert_true

    res = monte_carlo_gauss_markov(n_sim=200, n_obs=30, random_state=RANDOM_STATE)
    return assert_true(
        "ols_bias" in res and "beta_ols_all" in res and "beta_alt_all" in res,
        label="monte_carlo_gauss_markov returns expected result keys",
    )


def test_monte_carlo_beta_shapes() -> bool:
    from test_utils import assert_shape

    res = monte_carlo_gauss_markov(n_sim=200, n_obs=80, random_state=RANDOM_STATE)
    checks = [
        assert_shape(res["beta_ols_all"], (200, 3), label="beta_ols_all shape is n_sim by p+1"),
        assert_shape(res["beta_alt_all"], (200, 3), label="beta_alt_all shape is n_sim by p+1"),
    ]
    return all(checks)


def test_monte_carlo_summary_lengths() -> bool:
    from test_utils import assert_equal

    res = monte_carlo_gauss_markov(n_sim=200, n_obs=80, random_state=RANDOM_STATE)
    checks = [
        assert_equal(len(res["ols_mean"]), 3, label="ols_mean length matches p+1"),
        assert_equal(len(res["alt_var"]), 3, label="alt_var length matches p+1"),
    ]
    return all(checks)


def test_monte_carlo_ols_nearly_unbiased() -> bool:
    from test_utils import assert_true

    true_beta = (2.0, -1.5, 0.8)
    res = monte_carlo_gauss_markov(
        n_sim=800,
        n_obs=100,
        true_beta=true_beta,
        true_sigma=1.0,
        random_state=123,
    )
    ok = all(abs(estimated - truth) < 0.12 for estimated, truth in zip(res["ols_mean"], true_beta))
    return assert_true(ok, label="OLS estimates are nearly unbiased in Monte Carlo")


def test_monte_carlo_ols_variance_not_larger_than_alt() -> bool:
    from test_utils import assert_true

    res = monte_carlo_gauss_markov(n_sim=600, n_obs=90, alt_scale=0.4, random_state=99)
    ok = all(ols_var <= alt_var + 1e-12 for ols_var, alt_var in zip(res["ols_var"], res["alt_var"]))
    return assert_true(ok, label="OLS variance is not larger than alternative estimator")


def test_monte_carlo_reproducible() -> bool:
    from test_utils import assert_true

    res1 = monte_carlo_gauss_markov(n_sim=5, n_obs=10, random_state=1)
    res2 = monte_carlo_gauss_markov(n_sim=5, n_obs=10, random_state=1)
    return assert_true(
        _nested_allclose(res1["beta_ols_all"], res2["beta_ols_all"]),
        label="simulation is reproducible given random_state",
    )


def run_tests() -> tuple[int, int]:
    from test_utils import TestLogger

    TestLogger.print_suite_header("F10 - Gauss Markov Demo")
    tests = [
        test_monte_carlo_returns_expected_keys,
        test_monte_carlo_beta_shapes,
        test_monte_carlo_summary_lengths,
        test_monte_carlo_ols_nearly_unbiased,
        test_monte_carlo_ols_variance_not_larger_than_alt,
        test_monte_carlo_reproducible,
    ]
    passed = sum(int(test()) for test in tests)
    return passed, len(tests)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from test_utils import TestLogger

    print("=" * 55)
    print("  UNIT TESTS - gauss_markov_demo.py")
    print("=" * 55)
    passed, total = run_tests()
    TestLogger.print_summary(passed, total)
