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

    residuals = [y[i] - y_hat[i] for i in range(n)]

    return {
        "beta_hat":  beta_hat,
        "y_hat":     y_hat,
        "residuals": residuals,
        "mean_X":    mean_X,
        "std_X":     std_X,
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
    show_plot: bool = False,
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
    out_path = os.path.join(save_dir, "ridge_trace.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    
    if show_plot:
        print(f"[F6] Ridge Trace đã lưu tại: {out_path}")
        plt.show()
    plt.close(fig)

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


def lasso_trace(
    X: list[list[float]],
    y: list[float],
    lambdas: list[float] | None = None,
    save_dir: str = "output",
    show_plot: bool = False,
) -> dict:
    """
    Vẽ Lasso Path: λ vs hệ số hồi quy (không tính intercept).

    Tương tự ridge_trace nhưng dùng lasso_fit (coordinate descent).
    Lưu ý: lasso_fit chậm hơn ridge nên dùng lưới λ thưa hơn.

    Trả về dict: {'lambdas': list, 'coefs': list[list]} để dùng tiếp.
    """
    if lambdas is None:
        # Lưới thưa hơn ridge để tiết kiệm thời gian (coordinate descent chậm)
        lambdas = [10 ** e for e in [x / 5 for x in range(-10, 21)]]  # 1e-2 … 1e4

    coefs = []
    for lam in lambdas:
        res = lasso_fit(X, y, lam)
        coefs.append(res["beta_hat"][1:])  # bỏ intercept

    coefs_T = list(zip(*coefs))  # (p, n_lambdas)
    p = len(coefs_T)

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for j in range(p):
        ax.plot(lambdas, coefs_T[j], label=f"β_{j + 1}")
    ax.set_xscale("log")
    ax.set_title("Lasso Path: λ vs Hệ Số Hồi Quy", fontsize=13)
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("Giá trị hệ số β")
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "lasso_path.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    if show_plot:
        print(f"[F7] Lasso Path đã lưu tại: {out_path}")
        plt.show()
    plt.close(fig)

    return {"lambdas": lambdas, "coefs": [list(c) for c in coefs]}


# ---------------------------------------------------------------------------
# Unit Tests — F6 & F7  (≥ 4 test mỗi hàm)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    from test_utils import TestLogger, assert_true, assert_equal, assert_close
    import numpy as np

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
    np.random.seed(RANDOM_STATE)
    X_np = np.random.randn(50, 3).tolist()
    beta_true = [1.0, -2.0, 0.5]
    y_np = [sum(beta_true[j] * X_np[i][j] for j in range(3)) for i in range(50)]
    res2 = ridge_fit(X_np, y_np, lam=1e-6)
    mse2 = _mse(y_np, res2["y_hat"])
    run(assert_true(mse2 < 0.01, label=f"ridge_fit with λ=0 closely matches OLS (MSE={mse2:.4f})"))

    # test_ridge_large_lam_shrinks_coefs
    res_small = ridge_fit(X, y, lam=1e-4)
    res_large = ridge_fit(X, y, lam=1e6)
    norm_small = sum(b ** 2 for b in res_small["beta_hat"][1:]) ** 0.5
    norm_large = sum(b ** 2 for b in res_large["beta_hat"][1:]) ** 0.5
    run(assert_true(norm_large < norm_small, label="ridge_fit with large λ shrinks coefficients towards 0"))

    # test_ridge_predict_consistent
    res3 = ridge_fit(X, y, lam=0.5)
    pred3 = ridge_predict(X, res3["beta_hat"], res3["mean_X"], res3["std_X"])
    run(assert_close(res3["y_hat"], pred3, label="ridge_predict outputs exactly match training y_hat", rtol=1e-8))

    # test_ridge_vs_sklearn
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        np.random.seed(RANDOM_STATE)
        X_sk = np.random.randn(30, 2)
        y_sk = X_sk @ np.array([2.0, -1.0]) + 0.5
        X4 = X_sk.tolist(); y4 = y_sk.tolist()
        res4 = ridge_fit(X4, y4, lam=1.0)
        mse_ours = _mse(y4, res4["y_hat"])
        # Dùng Pipeline(StandardScaler + Ridge) để khớp với cách chúng ta standardize X bên trong
        pipe = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0, fit_intercept=True))])
        pipe.fit(X_sk, y_sk)
        mse_sk = float(np.mean((y_sk - pipe.predict(X_sk)) ** 2))
        diff_ratio = abs(mse_ours - mse_sk) / (mse_sk + 1e-12)
        run(assert_true(diff_ratio < 0.10, label=f"ridge_fit performance matches sklearn Pipeline(StandardScaler+Ridge) (diff={diff_ratio:.1%})"))
    except ImportError:
        TestLogger.print_warn("Bỏ qua test_ridge_vs_sklearn vì không có thư viện sklearn")


    TestLogger.print_suite_header("F7 — Lasso Regression")

    # test_lasso_output_shape
    res5 = lasso_fit(X, y, lam=0.1)
    run(assert_equal(len(res5["beta_hat"]), 3, label="lasso_fit beta_hat length matches p+1"))
    run(assert_equal(len(res5["y_hat"]), 4, label="lasso_fit y_hat length matches n"))

    # test_lasso_sparsity
    np.random.seed(RANDOM_STATE)
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
    np.random.seed(RANDOM_STATE)
    X_l0 = np.random.randn(40, 2).tolist()
    y_l0 = [2.0 * X_l0[i][0] - 1.0 * X_l0[i][1] for i in range(40)]
    res8 = lasso_fit(X_l0, y_l0, lam=1e-6)
    mse8 = _mse(y_l0, res8["y_hat"])
    run(assert_true(mse8 < 0.01, label=f"lasso_fit with λ=0 closely matches OLS (MSE={mse8:.4f})"))

    # test_lasso_vs_sklearn
    try:
        from sklearn.linear_model import Lasso
        np.random.seed(RANDOM_STATE)
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
