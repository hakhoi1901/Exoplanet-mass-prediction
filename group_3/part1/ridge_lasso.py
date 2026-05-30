from __future__ import annotations
import math
import os
import sys

import matplotlib.pyplot as plt
PART1_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))

# Import utils từ Project 1
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import transpose, matmul, matvec, dot_product, solve_system
from config import EPSILON, RANDOM_STATE

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


def _to_original_scale(beta_scaled: list[float], mean: list[float], std: list[float]) -> list[float]:
    coefs = [beta_scaled[j + 1] / std[j] for j in range(len(mean))]
    intercept = beta_scaled[0] - sum(coefs[j] * mean[j] for j in range(len(mean)))
    return [intercept] + coefs


def _predict_original(X: list[list[float]], beta_hat: list[float]) -> list[float]:
    p = len(beta_hat) - 1
    return [beta_hat[0] + sum(row[j] * beta_hat[j + 1] for j in range(p)) for row in X]


# ---------------------------------------------------------------------------
# F6: Ridge Regression - Closed-form
# ---------------------------------------------------------------------------

def ridge_fit(
    X: list[list[float]],
    y: list[float],
    lam: float = 1.0,
) -> dict:
    """
    F6: Ridge Regression - nghiệm closed-form.

    Công thức:
        beta_hat_ridge = (X_tilde^T X_tilde + lambda I*)^-1 X_tilde^T y
    trong đó X_tilde là ma trận design đã chuẩn hóa + bias,
    I* là ma trận đơn vị với I*[0,0] = 0 (không penalize intercept).

    Liên kết: Dùng transpose, matmul, matvec, solve_system từ utils.py.

    Tham số:
        X    : Ma trận features (n x p), CHƯA có cột bias.
        y    : Vector target (n,).
        lam  : Hệ số regularization lambda >= 0.

    Trả về dict gồm:
        beta_hat : list[float] - hệ số [intercept, beta_1, ..., beta_p] trên thang chuẩn hóa.
        y_hat    : list[float] - giá trị dự đoán trên thang gốc.
        mean_X   : list[float] - mean từng cột X (dùng để transform test set).
        std_X    : list[float] - std  từng cột X.
    """
    n, p = len(X), len(X[0])

    # Chuẩn hóa features
    mean_X = _col_mean(X)
    std_X  = _col_std(X, mean_X)
    X_sc   = _standardize(X, mean_X, std_X)
    X_b    = _add_bias(X_sc)          # (n, p+1)

    # I* - không penalize intercept
    I_star = [[1.0 if i == j else 0.0 for j in range(p + 1)] for i in range(p + 1)]
    I_star[0][0] = 0.0

    # A = X^T X + lambda I*,  rhs = X^T y
    Xt  = transpose(X_b)               # utils.py
    XtX = matmul(Xt, X_b)              # utils.py
    A   = [[XtX[i][j] + lam * I_star[i][j] for j in range(p + 1)] for i in range(p + 1)]

    # X^T y - dùng dot_product cho từng hàng của X^T
    rhs = [dot_product(Xt[i], y) for i in range(p + 1)]

    # Giải hệ (X^T X + lambda I*) beta = X^T y
    beta_scaled = solve_system(A, rhs)     # utils.py
    beta_hat = _to_original_scale(beta_scaled, mean_X, std_X)
    y_hat = _predict_original(X, beta_hat)

    residuals = [y[i] - y_hat[i] for i in range(n)]

    return {
        "beta_hat":              beta_hat,
        "beta_hat_standardized": beta_scaled,
        "y_hat":                 y_hat,
        "residuals":             residuals,
        "mean_X":                mean_X,
        "std_X":                 std_X,
    }


def ridge_predict(
    X: list[list[float]],
    beta_hat: list[float],
    mean_X: list[float] | None = None,
    std_X: list[float] | None = None,
    *,
    standardized: bool = False,
) -> list[float]:
    """Dự đoán y cho X mới dùng beta_hat từ ridge_fit."""
    if standardized:
        if mean_X is None or std_X is None:
            raise ValueError("mean_X and std_X are required for standardized coefficients")
        X_sc = _standardize(X, mean_X, std_X)
        X_b = _add_bias(X_sc)
        return matvec(X_b, beta_hat)
    return _predict_original(X, beta_hat)


def ridge_trace(
    X: list[list[float]],
    y: list[float],
    lambdas: list[float] | None = None,
    save_dir: str = PART1_OUTPUT_DIR,
    show_plot: bool = False,
) -> dict:
    """
    Vẽ Ridge Trace: lambda vs hệ số hồi quy (không tính intercept).

    Trả về dict: {'lambdas': list, 'coefs': list[list]} để dùng trong CV.
    """
    if lambdas is None:
        lambdas = [10 ** e for e in [x / 10 for x in range(-30, 41)]]  # 1e-3 ... 1e4

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
# F7: Lasso Regression - Coordinate Descent
# ---------------------------------------------------------------------------

def soft_threshold(rho: float, lam: float) -> float:
    """Hàm soft-thresholding cho Lasso: S(rho, lambda)."""
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
    F7: Lasso Regression - Coordinate Descent (manual, không dùng numpy).

    Tối thiểu hóa: ||y - X beta||^2 + lambda ||beta||_1
    Nghiệm không có dạng closed-form; dùng coordinate descent.

    Tham số:
        X        : Ma trận features (n x p), CHƯA có cột bias.
        y        : Vector target (n,).
        lam      : Hệ số regularization lambda >= 0.
        max_iter : Số vòng lặp tối đa.
        tol      : Ngưỡng hội tụ (max |delta beta|).

    Trả về dict gồm:
        beta_hat : list[float] - [intercept, beta_1, ..., beta_p].
        y_hat    : list[float] - giá trị dự đoán.
        n_iter   : int         - số vòng lặp thực tế.
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

    # Pre-compute z_j = ||x_j||^2 cho mỗi cột j
    z = [sum(X_sc[i][j] ** 2 for i in range(n)) for j in range(p)]

    n_iter = max_iter
    for it in range(max_iter):
        beta_old = beta[:]

        for j in range(p):
            # Tính partial residual: r_j = y_centered - sum_{k != j} X_sc[:,k] * beta[k]
            # = y_centered - (X_sc @ beta - X_sc[:,j] * beta[j])
            # Tối ưu: tính X_sc @ beta trước, rồi cộng lại X_sc[:,j] * beta[j]
            r_j = [0.0] * n
            for i in range(n):
                pred_i = sum(X_sc[i][k] * beta[k] for k in range(p)) - X_sc[i][j] * beta[j]
                r_j[i] = y_centered[i] - pred_i

            # rho_j = X_sc[:,j] dot r_j
            rho_j = sum(X_sc[i][j] * r_j[i] for i in range(n))

            # Update beta[j] với soft-thresholding
            beta[j] = soft_threshold(rho_j, lam) / z[j] if abs(z[j]) > EPSILON else 0.0

        # Kiểm tra hội tụ: max |delta beta|
        max_change = max(abs(beta[j] - beta_old[j]) for j in range(p))
        if max_change < tol:
            n_iter = it + 1
            break

    # Tạo output
    beta_scaled = [intercept] + beta
    beta_hat = _to_original_scale(beta_scaled, mean_X, std_X)
    y_hat = _predict_original(X, beta_hat)

    return {
        "beta_hat":              beta_hat,
        "beta_hat_standardized": beta_scaled,
        "y_hat":                 y_hat,
        "n_iter":                n_iter,
        "mean_X":                mean_X,
        "std_X":                 std_X,
    }


def lasso_predict(
    X: list[list[float]],
    beta_hat: list[float],
    mean_X: list[float] | None = None,
    std_X: list[float] | None = None,
    *,
    standardized: bool = False,
) -> list[float]:
    """Dự đoán y cho X mới dùng beta_hat từ lasso_fit."""
    if standardized:
        if mean_X is None or std_X is None:
            raise ValueError("mean_X and std_X are required for standardized coefficients")
        X_sc = _standardize(X, mean_X, std_X)
        X_b = _add_bias(X_sc)
        return matvec(X_b, beta_hat)
    return _predict_original(X, beta_hat)


def lasso_trace(
    X: list[list[float]],
    y: list[float],
    lambdas: list[float] | None = None,
    save_dir: str = PART1_OUTPUT_DIR,
    show_plot: bool = False,
) -> dict:
    """
    Vẽ Lasso Path: lambda vs hệ số hồi quy.

    Tương tự ridge_trace nhưng dùng lasso_fit (coordinate descent).
    Lưu ý: lasso_fit chậm hơn ridge nên dùng lưới lambda thưa hơn.

    Trả về dict: {'lambdas': list, 'coefs': list[list]} để dùng tiếp.
    """
    if lambdas is None:
        # Lưới thưa hơn ridge để tiết kiệm thời gian
        lambdas = [10 ** e for e in [x / 5 for x in range(-10, 21)]]  # 1e-2 ... 1e4

    coefs = []
    for lam in lambdas:
        res = lasso_fit(X, y, lam)
        coefs.append(res["beta_hat"][1:]) 

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
# Unit Tests - F6 & F7
# ---------------------------------------------------------------------------


def _test_mse(y_true: list[float], y_pred: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def _test_randn_matrix(n: int, p: int, seed: int) -> list[list[float]]:
    import random

    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(p)] for _ in range(n)]


def _small_regression_data() -> tuple[list[list[float]], list[float]]:
    return [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0]], [5.0, 4.0, 10.0, 8.0]


def test_ridge_output_shape() -> bool:
    from test_utils import assert_equal

    X, y = _small_regression_data()
    res = ridge_fit(X, y, lam=1.0)
    checks = [
        assert_equal(len(res["beta_hat"]), 3, label="ridge_fit beta_hat length matches p+1"),
        assert_equal(len(res["y_hat"]), 4, label="ridge_fit y_hat length matches n"),
    ]
    return all(checks)


def test_ridge_lam0_close_to_ols() -> bool:
    from test_utils import assert_true

    X = _test_randn_matrix(50, 3, RANDOM_STATE)
    beta_true = [1.0, -2.0, 0.5]
    y = [sum(beta_true[j] * X[i][j] for j in range(3)) for i in range(50)]
    res = ridge_fit(X, y, lam=1e-6)
    mse = _test_mse(y, res["y_hat"])
    return assert_true(mse < 0.01, label=f"ridge_fit with lambda near 0 matches OLS (MSE={mse:.4f})")


def test_ridge_large_lam_shrinks_coefs() -> bool:
    from test_utils import assert_true

    X, y = _small_regression_data()
    res_small = ridge_fit(X, y, lam=1e-4)
    res_large = ridge_fit(X, y, lam=1e6)
    norm_small = sum(b ** 2 for b in res_small["beta_hat"][1:]) ** 0.5
    norm_large = sum(b ** 2 for b in res_large["beta_hat"][1:]) ** 0.5
    return assert_true(norm_large < norm_small, label="large ridge lambda shrinks coefficients")


def test_ridge_predict_consistent() -> bool:
    from test_utils import assert_close

    X, y = _small_regression_data()
    res = ridge_fit(X, y, lam=0.5)
    pred = ridge_predict(X, res["beta_hat"], res["mean_X"], res["std_X"])
    return assert_close(res["y_hat"], pred, label="ridge_predict matches training fitted values", rtol=1e-8)


def test_ridge_vs_sklearn_optional() -> bool:
    from test_utils import TestLogger, assert_true

    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        TestLogger.print_warn("Skip ridge sklearn verification because sklearn is not installed")
        return True

    X = _test_randn_matrix(30, 2, RANDOM_STATE)
    y = [2.0 * row[0] - 1.0 * row[1] + 0.5 for row in X]
    ours = ridge_fit(X, y, lam=1.0)
    mse_ours = _test_mse(y, ours["y_hat"])
    pipe = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0, fit_intercept=True))])
    pipe.fit(X, y)
    mse_sk = _test_mse(y, list(pipe.predict(X)))
    diff_ratio = abs(mse_ours - mse_sk) / (mse_sk + 1e-12)
    return assert_true(diff_ratio < 0.10, label=f"ridge_fit matches sklearn pipeline (diff={diff_ratio:.1%})")


def test_lasso_output_shape() -> bool:
    from test_utils import assert_equal

    X, y = _small_regression_data()
    res = lasso_fit(X, y, lam=0.1)
    checks = [
        assert_equal(len(res["beta_hat"]), 3, label="lasso_fit beta_hat length matches p+1"),
        assert_equal(len(res["y_hat"]), 4, label="lasso_fit y_hat length matches n"),
    ]
    return all(checks)


def test_lasso_sparsity() -> bool:
    import random
    from test_utils import assert_true

    rng = random.Random(RANDOM_STATE + 1)
    X = _test_randn_matrix(60, 5, RANDOM_STATE)
    y = [2 * X[i][0] - 1.5 * X[i][1] + 0.05 * rng.gauss(0.0, 1.0) for i in range(60)]
    res = lasso_fit(X, y, lam=2.0)
    zeros = sum(1 for b in res["beta_hat"][1:] if abs(b) < 1e-6)
    return assert_true(zeros >= 1, label="lasso_fit induces coefficient sparsity")


def test_lasso_predict_consistent() -> bool:
    from test_utils import assert_close

    X, y = _small_regression_data()
    res = lasso_fit(X, y, lam=0.1)
    pred = lasso_predict(X, res["beta_hat"], res["mean_X"], res["std_X"])
    return assert_close(res["y_hat"], pred, label="lasso_predict matches training fitted values", rtol=1e-6)


def test_lasso_lam0_close_to_ols() -> bool:
    from test_utils import assert_true

    X = _test_randn_matrix(40, 2, RANDOM_STATE)
    y = [2.0 * X[i][0] - 1.0 * X[i][1] for i in range(40)]
    res = lasso_fit(X, y, lam=1e-6)
    mse = _test_mse(y, res["y_hat"])
    return assert_true(mse < 0.01, label=f"lasso_fit with lambda near 0 matches OLS (MSE={mse:.4f})")


def test_lasso_vs_sklearn_optional() -> bool:
    import random
    from test_utils import TestLogger, assert_true

    try:
        from sklearn.linear_model import Lasso
    except ImportError:
        TestLogger.print_warn("Skip lasso sklearn verification because sklearn is not installed")
        return True

    rng = random.Random(RANDOM_STATE + 2)
    X = _test_randn_matrix(50, 3, RANDOM_STATE)
    y = [row[0] - 2.0 * row[2] + 0.3 * rng.gauss(0.0, 1.0) for row in X]
    ours = lasso_fit(X, y, lam=0.5, max_iter=5000)
    mse_ours = _test_mse(y, ours["y_hat"])
    sk = Lasso(alpha=0.5, max_iter=10000).fit(X, y)
    mse_sk = _test_mse(y, list(sk.predict(X)))
    return assert_true(mse_ours < 2.0 and mse_sk < 2.0, label="lasso_fit performance matches sklearn Lasso")


def _run_test_group(tests: list, suite_name: str) -> tuple[int, int]:
    from test_utils import TestLogger

    TestLogger.print_suite_header(suite_name)
    passed = sum(int(test()) for test in tests)
    return passed, len(tests)


def run_tests() -> tuple[int, int]:
    groups = [
        (
            "F6 - Ridge Regression",
            [
                test_ridge_output_shape,
                test_ridge_lam0_close_to_ols,
                test_ridge_large_lam_shrinks_coefs,
                test_ridge_predict_consistent,
                test_ridge_vs_sklearn_optional,
            ],
        ),
        (
            "F7 - Lasso Regression",
            [
                test_lasso_output_shape,
                test_lasso_sparsity,
                test_lasso_predict_consistent,
                test_lasso_lam0_close_to_ols,
                test_lasso_vs_sklearn_optional,
            ],
        ),
    ]
    total_passed = 0
    total_tests = 0
    for suite_name, tests in groups:
        passed, total = _run_test_group(tests, suite_name)
        total_passed += passed
        total_tests += total
    return total_passed, total_tests


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from test_utils import TestLogger

    print("=" * 55)
    print("  UNIT TESTS - ridge_lasso.py")
    print("=" * 55)
    passed, total = run_tests()
    TestLogger.print_summary(passed, total)
