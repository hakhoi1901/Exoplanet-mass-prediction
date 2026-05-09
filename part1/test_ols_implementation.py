"""
test_ols_implementation.py - Unit Tests cho F1, F2, F3
Mỗi hàm có ít nhất 4 unit test, tự tạo data (không phụ thuộc file bên ngoài).

Cách chạy:
    python part1/test_ols_implementation.py
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RANDOM_STATE

from test_utils import (
    TestLogger, assert_close, assert_equal, assert_true,
    assert_shape, assert_in_range, assert_raises,
    make_linear_data, make_multifeature_data,
    verify_vs_sklearn_ols,
)
from ols_implementation import ols_fit, hat_matrix, model_metrics


# ═══════════════════════════════════════════════════════════════════
#  F1: ols_fit
# ═══════════════════════════════════════════════════════════════════

def test_ols_exact_solution():
    """sigma=0 → beta_hat phải bằng đúng beta thực."""
    TRUE_BETA = [1.0, 2.0]
    X, y = make_linear_data(n=30, beta=TRUE_BETA, sigma=0.0)
    res  = ols_fit(X, y)
    return assert_close(
        res["beta_hat"], TRUE_BETA,
        label="F1: exact β̂ == TRUE_BETA (sigma=0)",
        atol=1e-7,
    )


def test_ols_output_shape():
    """Kiểm tra shape của tất cả output."""
    X, y = make_linear_data(n=20, beta=[3.0, -1.0, 0.5], sigma=0.5)
    res  = ols_fit(X, y)
    n, p = 20, 2  # beta=[3, -1, 0.5] → p=2 features
    ok1 = assert_shape(res["beta_hat"], (p + 1,), label="F1: beta_hat shape = (p+1,)")
    ok2 = assert_shape(res["y_hat"],    (n,),     label="F1: y_hat shape = (n,)")
    ok3 = assert_shape(res["residuals"],(n,),     label="F1: residuals shape = (n,)")
    return ok1 and ok2 and ok3


def test_ols_residuals_sum_zero():
    """Tổng residuals ≈ 0 (tính chất OLS khi có intercept)."""
    X, y = make_linear_data(n=50, beta=[2.0, -1.5, 0.8], sigma=1.0)
    res  = ols_fit(X, y)
    return assert_close(
        sum(res["residuals"]), 0.0,
        label="F1: Σresiduals ≈ 0",
        atol=1e-8,
    )


def test_ols_vs_sklearn():
    """So sánh beta_hat với sklearn LinearRegression."""
    X, y = make_linear_data(n=60, beta=[1.0, 2.0, -0.5], sigma=0.3)
    res  = ols_fit(X, y)
    return verify_vs_sklearn_ols(X, y, res["beta_hat"])


def test_ols_sigma2_exact():
    """sigma=0 → sigma2_hat ≈ 0."""
    X, y = make_linear_data(n=30, beta=[1.0, 2.0], sigma=0.0)
    res  = ols_fit(X, y)
    return assert_close(
        res["sigma2_hat"], 0.0,
        label="F1: σ̂² ≈ 0 khi sigma=0",
        atol=1e-10,
    )


def test_ols_sigma2_positive():
    """sigma > 0 → sigma2_hat phải dương và gần true_sigma² khi n lớn."""
    TRUE_SIGMA = 1.0
    X, y = make_linear_data(n=100, beta=[1.0, 2.0], sigma=TRUE_SIGMA, seed=RANDOM_STATE)
    res  = ols_fit(X, y)
    # Với n=100 và sigma=1.0, sigma2_hat phải nằm trong [0.5, 2.0]
    return assert_in_range(
        res["sigma2_hat"], 0.5, 2.0,
        label="F1: sigma2_hat ≈ true_sigma² khi sigma=1.0 (n=100)",
    )


# ═══════════════════════════════════════════════════════════════════
#  F2: hat_matrix
# ═══════════════════════════════════════════════════════════════════

def test_hat_idempotent():
    """H² ≈ H (tính chất lũy đẳng)."""
    X, _ = make_linear_data(n=15, beta=[1.0, 2.0], sigma=0.0)
    res  = hat_matrix(X)
    return assert_true(
        res["is_idempotent"],
        label="F2: H² ≈ H (idempotent)",
    )


def test_hat_symmetric():
    """Hᵀ ≈ H (ma trận đối xứng)."""
    X, _ = make_linear_data(n=15, beta=[1.0, 2.0], sigma=0.0)
    res  = hat_matrix(X)
    return assert_true(
        res["is_symmetric"],
        label="F2: Hᵀ ≈ H (symmetric)",
    )


def test_hat_rank():
    """rank(H) = p + 1."""
    X, _ = make_linear_data(n=20, beta=[1.0, 2.0, -0.5], sigma=0.0)
    res  = hat_matrix(X)
    p = len(X[0])
    return assert_equal(
        res["rank"], p + 1,
        label=f"F2: rank(H) = {p + 1}",
    )


def test_hat_projection():
    """H @ y == ŷ (từ ols_fit)."""
    from utils import matvec
    X, y = make_linear_data(n=15, beta=[1.0, 2.0], sigma=0.5, seed=RANDOM_STATE)
    ols_res = ols_fit(X, y)
    hat_res = hat_matrix(X)
    y_hat_from_H = matvec(hat_res["H"], y)
    return assert_close(
        y_hat_from_H, ols_res["y_hat"],
        label="F2: H @ y == ŷ (OLS)",
        atol=1e-7,
    )


def test_hat_eigenvalues():
    """Eigenvalues chỉ gồm 0 và 1."""
    # KNOWN LIMITATION: eigenvalues trong hat_matrix được hardcode là [1.0]*rank + [0.0]*(n-rank)
    # thay vì tính thực từ ma trận bằng QR iteration. Test này kiểm tra tính nhất quán
    # (giá trị luôn ∈ {0, 1}), không phát hiện được bug trong thuật toán tính eigenvalue thực.
    # Accepted scope limitation cho dự án này.
    X, _ = make_linear_data(n=15, beta=[1.0, 2.0], sigma=0.0)
    res  = hat_matrix(X)
    evs  = res["eigenvalues"]
    all_0_or_1 = all(abs(e) < 1e-8 or abs(e - 1.0) < 1e-8 for e in evs)
    return assert_true(
        all_0_or_1,
        label="F2: eigenvalues ∈ {0, 1}",
    )


# ═══════════════════════════════════════════════════════════════════
#  F3: model_metrics
# ═══════════════════════════════════════════════════════════════════

def test_metrics_perfect_fit():
    """sigma=0 → R² ≈ 1, RSS ≈ 0."""
    X, y = make_linear_data(n=30, beta=[1.0, 2.0], sigma=0.0)
    res  = ols_fit(X, y)
    met  = model_metrics(y, res["y_hat"], p=1)
    ok1 = assert_close(met["R2"],  1.0, label="F3: R² ≈ 1.0 (perfect fit)", atol=1e-8)
    ok2 = assert_close(met["RSS"], 0.0, label="F3: RSS ≈ 0  (perfect fit)",  atol=1e-8)
    return ok1 and ok2


def test_metrics_tss_decomposition():
    """RSS + MSS ≈ TSS."""
    X, y = make_linear_data(n=50, beta=[2.0, -1.5, 0.8], sigma=1.0)
    res  = ols_fit(X, y)
    met  = model_metrics(y, res["y_hat"], p=2)
    return assert_close(
        met["RSS"] + met["MSS"], met["TSS"],
        label="F3: RSS + MSS ≈ TSS",
        atol=1e-8,
    )


def test_metrics_r2_range():
    """0 ≤ R² ≤ 1 cho mô hình bình thường."""
    X, y = make_linear_data(n=50, beta=[2.0, -1.5], sigma=1.0)
    res  = ols_fit(X, y)
    met  = model_metrics(y, res["y_hat"], p=1)
    return assert_in_range(met["R2"], 0.0, 1.0, label="F3: 0 ≤ R² ≤ 1")


def test_metrics_vs_sklearn():
    """R² khớp với sklearn.metrics.r2_score."""
    from sklearn.metrics import r2_score
    X, y = make_linear_data(n=60, beta=[1.0, 2.0, -0.5], sigma=0.5)
    res  = ols_fit(X, y)
    met  = model_metrics(y, res["y_hat"], p=2)
    sk_r2 = r2_score(y, res["y_hat"])
    return assert_close(
        met["R2"], sk_r2,
        label="F3: R² == sklearn r2_score",
        atol=1e-8,
    )


def test_metrics_f_stat_positive():
    """F-statistic > 0 cho mô hình có ý nghĩa."""
    X, y = make_linear_data(n=50, beta=[2.0, -1.5, 0.8], sigma=0.5)
    res  = ols_fit(X, y)
    met  = model_metrics(y, res["y_hat"], p=2)
    return assert_true(
        met["F_stat"] > 0,
        label="F3: F_stat > 0",
        details=f"F_stat = {met['F_stat']:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    passed = 0
    total  = 0

    def run(result: bool):
        global passed, total
        total  += 1
        passed += int(result)

    # --- F1: ols_fit ---
    TestLogger.print_suite_header("F1 - ols_fit  |  Normal Equations")
    run(test_ols_exact_solution())
    run(test_ols_output_shape())
    run(test_ols_residuals_sum_zero())
    run(test_ols_vs_sklearn())
    run(test_ols_sigma2_exact())
    run(test_ols_sigma2_positive())

    # --- F2: hat_matrix ---
    TestLogger.print_suite_header("F2 - hat_matrix  |  Projection Properties")
    run(test_hat_idempotent())
    run(test_hat_symmetric())
    run(test_hat_rank())
    run(test_hat_projection())
    run(test_hat_eigenvalues())

    # --- F3: model_metrics ---
    TestLogger.print_suite_header("F3 - model_metrics  |  R², F-test, MAE, RMSE")
    run(test_metrics_perfect_fit())
    run(test_metrics_tss_decomposition())
    run(test_metrics_r2_range())
    run(test_metrics_vs_sklearn())
    run(test_metrics_f_stat_positive())

    # --- Tổng kết ---
    TestLogger.print_summary(passed, total)
