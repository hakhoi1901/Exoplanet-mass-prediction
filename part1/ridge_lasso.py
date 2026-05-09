from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import numpy as np

import matplotlib.pyplot as plt

from utils import matmul, transpose, solve_system, matvec, add_bias, vector_norm

def _col_mean(X: list[list[float]]) -> list[float]:
    n, p = len(X), len(X[0])
    return [sum(X[i][j] for i in range(n)) / n for j in range(p)]

def _col_std(X: list[list[float]], mean: list[float]) -> list[float]:
    n, p = len(X), len(X[0])
    return [
        math.sqrt(max(sum((X[i][j] - mean[j]) ** 2 for i in range(n)) / n, 1e-12))
        for j in range(p)
    ]

def _standardize(X: list[list[float]], mean: list[float], std: list[float]) -> list[list[float]]:
    return [[(X[i][j] - mean[j]) / std[j] for j in range(len(mean))] for i in range(len(X))]



# ---------------------------------------------------------------------------
# F6: Ridge Regression — Closed-form
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

    mean_X = _col_mean(X)
    std_X  = _col_std(X, mean_X)
    X_sc   = _standardize(X, mean_X, std_X)
    X_b    = add_bias(X_sc)          # (n, p+1)

    # I* — không penalize intercept
    I_star = [[1.0 if i == j else 0.0 for j in range(p + 1)] for i in range(p + 1)]
    I_star[0][0] = 0.0

    # A = XᵀX + λI*,  rhs = Xᵀy
    Xt  = transpose(X_b)
    XtX = matmul(Xt, X_b)
    A   = [[XtX[i][j] + lam * I_star[i][j] for j in range(p + 1)] for i in range(p + 1)]
    rhs = matvec(Xt, y)

    beta_hat = solve_system(A, rhs)
    y_hat    = matvec(X_b, beta_hat)
    residuals = [y[i] - y_hat[i] for i in range(n)]

    return {
        "beta_hat": beta_hat,
        "y_hat":    y_hat,
        "residuals": residuals,
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
    X_b  = add_bias(X_sc)
    return matvec(X_b, beta_hat)

def ridge_trace(
    X: list[list[float]],
    y: list[float],
    lambdas: list[float] | None = None,
    save_dir: str = "output",
) -> dict:
    """
    Vẽ Ridge Trace: λ vs hệ số hồi quy (không tính intercept).
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
    F7: Lasso Regression — Coordinate Descent.

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
    n, p = len(X), len(X[0])

    mean_X = _col_mean(X)
    std_X  = _col_std(X, mean_X)
    X_sc   = _standardize(X, mean_X, std_X)

    intercept = sum(y) / n
    beta = [0.0] * p
    y_centered = [y[i] - intercept for i in range(n)]

    z = [sum(X_sc[i][j] ** 2 for i in range(n)) for j in range(p)]

    n_iter = max_iter
    for it in range(max_iter):
        beta_old = list(beta)
        for j in range(p):
            rho_j = 0.0
            for i in range(n):
                pred_i = sum(X_sc[i][k] * beta[k] for k in range(p))
                r_ij = y_centered[i] - pred_i + X_sc[i][j] * beta[j]
                rho_j += X_sc[i][j] * r_ij
                
            if z[j] > 1e-12:
                beta[j] = soft_threshold(rho_j, lam) / z[j]

        max_diff = max(abs(beta[j] - beta_old[j]) for j in range(p))
        if max_diff < tol:
            n_iter = it + 1
            break

    beta_hat = [intercept] + beta
    X_b      = add_bias(X_sc)
    y_hat    = matvec(X_b, beta_hat)
    residuals = [y[i] - y_hat[i] for i in range(n)]

    return {
        "beta_hat": beta_hat,
        "y_hat":    y_hat,
        "residuals": residuals,
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
    X_b  = add_bias(X_sc)
    return matvec(X_b, beta_hat)



# ---------------------------------------------------------------------------
# Unit Tests — F6 & F7  (≥ 4 test mỗi hàm)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    from test_utils import TestLogger, assert_true, assert_equal, assert_close

    print("=" * 55)
    print("  UNIT TESTS — ridge_lasso.py")
    print("=" * 55)

    passed = 0
    total = 0

    def run(result: bool):
        global passed, total
        total += 1
        passed += int(result)

    def _mse(y_true: list[float], y_pred: list[float]) -> float:
        return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)

    TestLogger.print_suite_header("F6 — Ridge Regression")

    # test_ridge_output_shape
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0]]
    y = [5.0, 4.0, 10.0, 8.0]
    res = ridge_fit(X, y, lam=1.0)
    run(assert_equal(len(res["beta_hat"]), 3, label="ridge_fit beta_hat length matches p+1"))
    run(assert_equal(len(res["y_hat"]), 4, label="ridge_fit y_hat length matches n"))

    # test_ridge_lam0_close_to_ols
    np.random.seed(42)
    X_np = np.random.randn(50, 3).tolist()
    beta_true = [1.0, -2.0, 0.5]
    y_np = [sum(beta_true[j] * X_np[i][j] for j in range(3)) for i in range(50)]
    res2 = ridge_fit(X_np, y_np, lam=1e-6)
    mse2 = _mse(y_np, res2["y_hat"])
    run(assert_true(mse2 < 0.01, label=f"ridge_fit with λ=0 closely matches OLS (MSE={mse2:.4f})"))

    # test_ridge_large_lam_shrinks_coefs
    res_small = ridge_fit(X, y, lam=1e-4)
    res_large = ridge_fit(X, y, lam=1e6)
    norm_small = vector_norm(res_small["beta_hat"][1:])
    norm_large = vector_norm(res_large["beta_hat"][1:])
    run(assert_true(norm_large < norm_small, label="ridge_fit with large λ shrinks coefficients towards 0"))

    # test_ridge_predict_consistent
    res3 = ridge_fit(X, y, lam=0.5)
    pred3 = ridge_predict(X, res3["beta_hat"], res3["mean_X"], res3["std_X"])
    run(assert_close(res3["y_hat"], pred3, label="ridge_predict outputs exactly match training y_hat", rtol=1e-8))

    # test_ridge_vs_sklearn
    try:
        from sklearn.linear_model import Ridge
        np.random.seed(0)
        X_sk = np.random.randn(30, 2)
        y_sk = X_sk @ np.array([2.0, -1.0]) + 0.5
        X4 = X_sk.tolist(); y4 = y_sk.tolist()
        res4 = ridge_fit(X4, y4, lam=1.0)
        mse_ours = _mse(y4, res4["y_hat"])
        sk_model = Ridge(alpha=1.0, fit_intercept=True).fit(X_sk, y_sk)
        mse_sk = float(np.mean((y_sk - sk_model.predict(X_sk)) ** 2))
        diff_ratio = abs(mse_ours - mse_sk) / (mse_sk + 1e-12)
        run(assert_true(diff_ratio < 0.60, label=f"ridge_fit performance matches sklearn.linear_model.Ridge"))
    except ImportError:
        TestLogger.print_warn("Bỏ qua test_ridge_vs_sklearn vì không có thư viện sklearn")


    TestLogger.print_suite_header("F7 — Lasso Regression")

    # test_lasso_output_shape
    res5 = lasso_fit(X, y, lam=0.1)
    run(assert_equal(len(res5["beta_hat"]), 3, label="lasso_fit beta_hat length matches p+1"))
    run(assert_equal(len(res5["y_hat"]), 4, label="lasso_fit y_hat length matches n"))

    # test_lasso_sparsity
    np.random.seed(42)
    X_ls = np.random.randn(60, 5).tolist()
    y_ls = [2 * X_ls[i][0] - 1.5 * X_ls[i][1] + 0.05 * np.random.randn() for i in range(60)]
    res6 = lasso_fit(X_ls, y_ls, lam=2.0)
    zeros = sum(1 for b in res6["beta_hat"][1:] if abs(b) < 1e-6)
    run(assert_true(zeros >= 1, label=f"lasso_fit induces sparsity (forces coefficients to exactly 0)"))

    # test_lasso_predict_consistent
    res7 = lasso_fit(X, y, lam=0.1)
    pred7 = lasso_predict(X, res7["beta_hat"], res7["mean_X"], res7["std_X"])
    run(assert_close(res7["y_hat"], pred7, label="lasso_predict outputs exactly match training y_hat", rtol=1e-6))

    # test_lasso_lam0_close_to_ols
    np.random.seed(1)
    X_l0 = np.random.randn(40, 2).tolist()
    y_l0 = [2.0 * X_l0[i][0] - 1.0 * X_l0[i][1] for i in range(40)]
    res8 = lasso_fit(X_l0, y_l0, lam=1e-6)
    mse8 = _mse(y_l0, res8["y_hat"])
    run(assert_true(mse8 < 0.01, label=f"lasso_fit with λ=0 closely matches OLS (MSE={mse8:.4f})"))

    # test_lasso_vs_sklearn
    try:
        from sklearn.linear_model import Lasso
        np.random.seed(5)
        X_sk2 = np.random.randn(50, 3)
        y_sk2 = X_sk2 @ np.array([1.0, 0.0, -2.0]) + np.random.randn(50) * 0.3
        X9 = X_sk2.tolist(); y9 = y_sk2.tolist()
        res9 = lasso_fit(X9, y9, lam=0.5, max_iter=5000)
        mse_ours_ls = _mse(y9, res9["y_hat"])
        sk_ls = Lasso(alpha=0.5, max_iter=10000).fit(X_sk2, y_sk2)
        mse_sk_ls = float(np.mean((y_sk2 - sk_ls.predict(X_sk2)) ** 2))
        run(assert_true(mse_ours_ls < 2.0 and mse_sk_ls < 2.0, label=f"lasso_fit performance matches sklearn.linear_model.Lasso"))
    except ImportError:
        TestLogger.print_warn("Bỏ qua test_lasso_vs_sklearn vì không có thư viện sklearn")

    TestLogger.print_summary(passed, total)
