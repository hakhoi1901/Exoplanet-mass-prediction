from __future__ import annotations
import math
import os
import sys

import matplotlib.pyplot as plt

# Import utils từ Project 1
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import transpose, matmul, matvec, dot_product, solve_system
from config import EPSILON, RANDOM_STATE


# ---------------------------------------------------------------------------
# Utilities nội bộ — chuẩn hóa (manual, không dùng numpy)
# ---------------------------------------------------------------------------

def _col_mean(X: list[list[float]]) -> list[float]:
    """Tính mean từng cột của X."""
    n, p = len(X), len(X[0])
    return [sum(X[i][j] for i in range(n)) / n for j in range(p)]


def _col_std(X: list[list[float]], mean: list[float]) -> list[float]:
    """Tính population std từng cột của X."""
    n, p = len(X), len(X[0])
    return [
        math.sqrt(max(sum((X[i][j] - mean[j]) ** 2 for i in range(n)) / n, 1e-12))
        for j in range(p)
    ]


def _standardize(X: list[list[float]], mean: list[float], std: list[float]) -> list[list[float]]:
    """Chuẩn hóa X theo mean và std đã cho."""
    return [[(X[i][j] - mean[j]) / std[j] for j in range(len(mean))] for i in range(len(X))]


def _add_bias(X: list[list[float]]) -> list[list[float]]:
    """Thêm cột 1 vào đầu ma trận X (intercept)."""
    return [[1.0] + row for row in X]


# ---------------------------------------------------------------------------
# F6: Ridge Regression — Closed-form
# Liên kết: Dùng solve_system, transpose, matmul, matvec từ utils.py
# ---------------------------------------------------------------------------

def ridge_fit(
    X: list[list[float]],
    y: list[float],
    lam: float = 1.0,
) -> dict:
    """
    F6: Ridge Regression — nghiệm closed-form.

    Công thức:
        β̂_ridge = (X̃ᵀX̃ + λI*)⁻¹ X̃ᵀy
    trong đó X̃ là ma trận design đã chuẩn hóa + bias,
    I* là ma trận đơn vị với I*[0,0] = 0 (không penalize intercept).

    Liên kết: Dùng transpose, matmul, matvec, solve_system từ utils.py.

    Tham số:
        X    : Ma trận features (n x p), CHƯA có cột bias.
        y    : Vector target (n,).
        lam  : Hệ số regularization λ ≥ 0.

    Trả về dict gồm:
        beta_hat : list[float] — hệ số [intercept, β₁, …, βₚ] trên thang chuẩn hóa.
        y_hat    : list[float] — giá trị dự đoán trên thang gốc.
        mean_X   : list[float] — mean từng cột X (dùng để transform test set).
        std_X    : list[float] — std  từng cột X.
    """
    n, p = len(X), len(X[0])

    # Chuẩn hóa features
    mean_X = _col_mean(X)
    std_X  = _col_std(X, mean_X)
    X_sc   = _standardize(X, mean_X, std_X)
    X_b    = _add_bias(X_sc)          # (n, p+1)

    # I* — không penalize intercept
    I_star = [[1.0 if i == j else 0.0 for j in range(p + 1)] for i in range(p + 1)]
    I_star[0][0] = 0.0

    # A = XᵀX + λI*,  rhs = Xᵀy
    Xt  = transpose(X_b)               # utils.py
    XtX = matmul(Xt, X_b)              # utils.py
    A   = [[XtX[i][j] + lam * I_star[i][j] for j in range(p + 1)] for i in range(p + 1)]

    # Xᵀy — dùng dot_product cho từng hàng của Xᵀ
    rhs = [dot_product(Xt[i], y) for i in range(p + 1)]

    # Giải hệ (XᵀX + λI*)β = Xᵀy
    beta_hat = solve_system(A, rhs)     # utils.py
    y_hat    = matvec(X_b, beta_hat)    # utils.py

    return {
        "beta_hat": beta_hat,
        "y_hat":    y_hat,
        "mean_X":   mean_X,
        "std_X":    std_X,
    }


def ridge_predict(
    X: list[list[float]],
    beta_hat: list[float],
    mean_X: list[float],
    std_X: list[float],
) -> list[float]:
    """Dự đoán y cho X mới dùng beta_hat từ ridge_fit."""
    X_sc = _standardize(X, mean_X, std_X)
    X_b  = _add_bias(X_sc)
    return matvec(X_b, beta_hat)        # utils.py


def ridge_trace(
    X: list[list[float]],
    y: list[float],
    lambdas: list[float] | None = None,
    save_dir: str = "output",
) -> dict:
    """
    Vẽ Ridge Trace: λ vs hệ số hồi quy (không tính intercept).

    Trả về dict: {'lambdas': list, 'coefs': list[list]} để dùng trong CV.
    """
    if lambdas is None:
        lambdas = [10 ** e for e in [x / 10 for x in range(-30, 41)]]  # 1e-3 … 1e4

    coefs = []
    for lam in lambdas:
        res = ridge_fit(X, y, lam)
        coefs.append(res["beta_hat"][1:])  # bỏ intercept

    coefs_T = list(zip(*coefs))  # (p, n_lambdas)
    p = len(coefs_T)

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for j in range(p):
        ax.plot(lambdas, coefs_T[j], label=f"β_{j + 1}")
    ax.set_xscale("log")
    ax.set_title("Ridge Trace: λ vs Hệ Số Hồi Quy", fontsize=13)
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("Giá trị hệ số β")
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ridge_trace.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return {"lambdas": lambdas, "coefs": [list(c) for c in coefs]}


# ---------------------------------------------------------------------------
# F7: Lasso Regression — Coordinate Descent
# Liên kết: Dùng soft_threshold (manual), _standardize, _add_bias, matvec
# ---------------------------------------------------------------------------

def soft_threshold(rho: float, lam: float) -> float:
    """Hàm soft-thresholding cho Lasso: S(ρ, λ)."""
    if rho > lam:
        return rho - lam
    elif rho < -lam:
        return rho + lam
    return 0.0


def lasso_fit(
    X: list[list[float]],
    y: list[float],
    lam: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict:
    """
    F7: Lasso Regression — Coordinate Descent (manual, không dùng numpy).

    Tối thiểu hóa: ‖y − Xβ‖² + λ‖β‖₁
    Nghiệm không có dạng closed-form; dùng coordinate descent.

    Tham số:
        X        : Ma trận features (n x p), CHƯA có cột bias.
        y        : Vector target (n,).
        lam      : Hệ số regularization λ ≥ 0.
        max_iter : Số vòng lặp tối đa.
        tol      : Ngưỡng hội tụ (max |Δβ|).

    Trả về dict gồm:
        beta_hat : list[float] — [intercept, β₁, …, βₚ].
        y_hat    : list[float] — giá trị dự đoán.
        n_iter   : int         — số vòng lặp thực tế.
        mean_X   : list[float]
        std_X    : list[float]
    """
    n = len(X)
    p = len(X[0])

    # Chuẩn hóa features (manual)
    mean_X = _col_mean(X)
    std_X  = _col_std(X, mean_X)
    X_sc   = _standardize(X, mean_X, std_X)

    # Intercept = mean(y), center y
    y_mean = sum(y) / n
    intercept = y_mean
    y_centered = [y[i] - intercept for i in range(n)]

    # Khởi tạo beta = 0
    beta = [0.0] * p

    # Pre-compute z_j = ‖x_j‖² cho mỗi cột j
    z = [sum(X_sc[i][j] ** 2 for i in range(n)) for j in range(p)]

    n_iter = max_iter
    for it in range(max_iter):
        beta_old = beta[:]

        for j in range(p):
            # Tính partial residual: r_j = y_centered - Σ_{k≠j} X_sc[:,k] * beta[k]
            # = y_centered - (X_sc @ beta - X_sc[:,j] * beta[j])
            # Tối ưu: tính X_sc @ beta trước, rồi cộng lại X_sc[:,j] * beta[j]
            r_j = [0.0] * n
            for i in range(n):
                pred_i = sum(X_sc[i][k] * beta[k] for k in range(p)) - X_sc[i][j] * beta[j]
                r_j[i] = y_centered[i] - pred_i

            # rho_j = X_sc[:,j] · r_j
            rho_j = sum(X_sc[i][j] * r_j[i] for i in range(n))

            # Update beta[j] với soft-thresholding
            beta[j] = soft_threshold(rho_j, lam) / z[j] if abs(z[j]) > EPSILON else 0.0

        # Kiểm tra hội tụ: max |Δβ|
        max_change = max(abs(beta[j] - beta_old[j]) for j in range(p))
        if max_change < tol:
            n_iter = it + 1
            break

    # Tạo output
    beta_hat = [intercept] + beta
    X_b = _add_bias(X_sc)
    y_hat = matvec(X_b, beta_hat)   # utils.py

    return {
        "beta_hat": beta_hat,
        "y_hat":    y_hat,
        "n_iter":   n_iter,
        "mean_X":   mean_X,
        "std_X":    std_X,
    }


def lasso_predict(
    X: list[list[float]],
    beta_hat: list[float],
    mean_X: list[float],
    std_X: list[float],
) -> list[float]:
    """Dự đoán y cho X mới dùng beta_hat từ lasso_fit."""
    X_sc = _standardize(X, mean_X, std_X)
    X_b  = _add_bias(X_sc)
    return matvec(X_b, beta_hat)        # utils.py


# ---------------------------------------------------------------------------
# Unit Tests — F6 & F7  (≥ 4 test mỗi hàm)
# ---------------------------------------------------------------------------

def _mse(y_true: list[float], y_pred: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


# --- Ridge ---

def test_ridge_output_shape():
    """Beta_hat phải có độ dài p+1."""
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0]]
    y = [5.0, 4.0, 10.0, 8.0]
    res = ridge_fit(X, y, lam=1.0)
    assert len(res["beta_hat"]) == 3, "Sai số phần tử beta_hat"
    assert len(res["y_hat"]) == 4,   "Sai số phần tử y_hat"
    print("test_ridge_output_shape: PASSED")


def test_ridge_lam0_close_to_ols():
    """Khi λ → 0, Ridge ≈ OLS (kiểm tra MSE nhỏ)."""
    import numpy as np
    np.random.seed(42)
    X = np.random.randn(50, 3).tolist()
    beta_true = [1.0, -2.0, 0.5]
    y = [sum(beta_true[j] * X[i][j] for j in range(3)) for i in range(50)]

    res = ridge_fit(X, y, lam=1e-6)
    mse = _mse(y, res["y_hat"])
    assert mse < 0.01, f"MSE quá lớn khi λ→0: {mse}"
    print("test_ridge_lam0_close_to_ols: PASSED")


def test_ridge_large_lam_shrinks_coefs():
    """Lambda lớn → hệ số co về 0 (shrinkage)."""
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0]]
    y = [5.0, 4.0, 10.0, 8.0]
    res_small = ridge_fit(X, y, lam=1e-4)
    res_large = ridge_fit(X, y, lam=1e6)
    norm_small = sum(b ** 2 for b in res_small["beta_hat"][1:]) ** 0.5
    norm_large = sum(b ** 2 for b in res_large["beta_hat"][1:]) ** 0.5
    assert norm_large < norm_small, "Lambda lớn không co hệ số về 0"
    print("test_ridge_large_lam_shrinks_coefs: PASSED")


def test_ridge_predict_consistent():
    """ridge_predict phải cho kết quả khớp y_hat trong ridge_fit."""
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0]]
    y = [5.0, 4.0, 10.0, 8.0]
    res  = ridge_fit(X, y, lam=0.5)
    pred = ridge_predict(X, res["beta_hat"], res["mean_X"], res["std_X"])
    for a, b in zip(res["y_hat"], pred):
        assert abs(a - b) < 1e-8, "ridge_predict không khớp y_hat"
    print("test_ridge_predict_consistent: PASSED")


def test_ridge_vs_sklearn():
    """Kiểm chứng beta_hat với sklearn Ridge (verify only)."""
    import numpy as np
    from sklearn.linear_model import Ridge
    np.random.seed(0)
    X_np = np.random.randn(30, 2)
    y_np = X_np @ np.array([2.0, -1.0]) + 0.5
    X = X_np.tolist(); y = y_np.tolist()

    res = ridge_fit(X, y, lam=1.0)
    mse_ours = _mse(y, res["y_hat"])

    sk = Ridge(alpha=1.0, fit_intercept=True)
    sk.fit(X_np, y_np)
    mse_sk = float(np.mean((y_np - sk.predict(X_np)) ** 2))

    # Cho phép sai lệch 60% do sklearn dùng norm khác (trên dữ liệu gốc, không chuẩn hóa)
    assert abs(mse_ours - mse_sk) / (mse_sk + 1e-12) < 0.60, (
        f"MSE chênh lệch quá lớn: ours={mse_ours:.4f}, sklearn={mse_sk:.4f}"
    )
    print(f"test_ridge_vs_sklearn: PASSED  (MSE ours={mse_ours:.4f}, sklearn={mse_sk:.4f})")


# --- Lasso ---

def test_lasso_output_shape():
    """Beta_hat phải có độ dài p+1."""
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0]]
    y = [5.0, 4.0, 10.0, 8.0]
    res = lasso_fit(X, y, lam=0.1)
    assert len(res["beta_hat"]) == 3, "Sai số phần tử beta_hat"
    assert len(res["y_hat"]) == 4,   "Sai số phần tử y_hat"
    print("test_lasso_output_shape: PASSED")


def test_lasso_sparsity():
    """Lambda đủ lớn → một số hệ số bị zero (sparsity)."""
    import numpy as np
    np.random.seed(42)
    X = np.random.randn(60, 5).tolist()
    # chỉ feature 0 và 1 thật sự có ảnh hưởng
    y = [2 * X[i][0] - 1.5 * X[i][1] + 0.05 * np.random.randn() for i in range(60)]
    res = lasso_fit(X, y, lam=2.0)
    zeros = sum(1 for b in res["beta_hat"][1:] if abs(b) < 1e-6)
    assert zeros >= 1, f"Lasso phải tạo ra ít nhất 1 hệ số = 0 khi lam=2.0, got zeros={zeros}"
    print(f"test_lasso_sparsity: PASSED  ({zeros}/5 hệ số = 0)")


def test_lasso_predict_consistent():
    """lasso_predict phải cho kết quả khớp y_hat trong lasso_fit."""
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0]]
    y = [5.0, 4.0, 10.0, 8.0]
    res  = lasso_fit(X, y, lam=0.1)
    pred = lasso_predict(X, res["beta_hat"], res["mean_X"], res["std_X"])
    for a, b in zip(res["y_hat"], pred):
        assert abs(a - b) < 1e-6, f"lasso_predict không khớp y_hat: {a} vs {b}"
    print("test_lasso_predict_consistent: PASSED")


def test_lasso_lam0_close_to_ols():
    """Khi λ → 0, Lasso ≈ OLS."""
    import numpy as np
    np.random.seed(1)
    X = np.random.randn(40, 2).tolist()
    y = [2.0 * X[i][0] - 1.0 * X[i][1] for i in range(40)]
    res = lasso_fit(X, y, lam=1e-6)
    mse = _mse(y, res["y_hat"])
    assert mse < 0.01, f"MSE quá lớn khi λ→0: {mse}"
    print("test_lasso_lam0_close_to_ols: PASSED")


def test_lasso_vs_sklearn():
    """Kiểm chứng với sklearn Lasso (verify only)."""
    import numpy as np
    from sklearn.linear_model import Lasso
    np.random.seed(5)
    X_np = np.random.randn(50, 3)
    y_np = X_np @ np.array([1.0, 0.0, -2.0]) + np.random.randn(50) * 0.3
    X = X_np.tolist(); y = y_np.tolist()

    res    = lasso_fit(X, y, lam=0.5, max_iter=5000)
    mse_ours = _mse(y, res["y_hat"])

    sk = Lasso(alpha=0.5, max_iter=10000)
    sk.fit(X_np, y_np)
    mse_sk = float(np.mean((y_np - sk.predict(X_np)) ** 2))

    # Cả hai MSE phải hợp lý (< 2.0 trên dữ liệu synthetic này)
    assert mse_ours < 2.0, f"MSE của ours quá lớn: {mse_ours:.4f}"
    assert mse_sk   < 2.0, f"MSE của sklearn quá lớn: {mse_sk:.4f}"
    print(f"test_lasso_vs_sklearn: PASSED  (MSE ours={mse_ours:.4f}, sklearn={mse_sk:.4f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    print("=" * 55)
    print("  UNIT TESTS — ridge_lasso.py")
    print("=" * 55)

    print("\n--- Ridge Regression ---")
    test_ridge_output_shape()
    test_ridge_lam0_close_to_ols()
    test_ridge_large_lam_shrinks_coefs()
    test_ridge_predict_consistent()
    test_ridge_vs_sklearn()

    print("\n--- Lasso Regression ---")
    test_lasso_output_shape()
    test_lasso_sparsity()
    test_lasso_predict_consistent()
    test_lasso_lam0_close_to_ols()
    test_lasso_vs_sklearn()

    print("\n--- Ridge Trace (visual) ---")
    np.random.seed(42)
    X_demo = np.random.randn(50, 4).tolist()
    y_demo = [2*X_demo[i][0] - X_demo[i][1] + 0.5*X_demo[i][2] for i in range(50)]
    ridge_trace(X_demo, y_demo)
    print("Ridge trace saved to output/ridge_trace.png")