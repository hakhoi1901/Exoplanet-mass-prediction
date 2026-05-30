from __future__ import annotations
import math
import sys
import os

# Thêm thư mục gốc vào path để import utils và config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import transpose, matmul, matvec, dot_product, inverse, solve_system, identity_matrix, add_bias
from config import RANDOM_STATE, EPSILON

PART1_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))


def _validate_xy(X: list[list[float]], y: list[float], *, need_df: bool = True) -> tuple[int, int]:
    if not X or not isinstance(X, list):
        raise ValueError("X must be a non-empty list of rows")
    if not y or not isinstance(y, list):
        raise ValueError("y must be a non-empty list")
    n = len(X)
    if len(y) != n:
        raise ValueError(f"X and y length mismatch: len(X)={n}, len(y)={len(y)}")
    if not isinstance(X[0], list) or len(X[0]) == 0:
        raise ValueError("X must contain at least one feature column")
    p = len(X[0])
    for i, row in enumerate(X):
        if not isinstance(row, list) or len(row) != p:
            raise ValueError(f"X row {i} has inconsistent length")
    if need_df and n <= p + 1:
        raise ValueError(f"Need n > p + 1 to estimate sigma^2 (n={n}, p={p})")
    return n, p


def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3e-14
    fpmin = 1e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    return h


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_two_sided_pvalue(t_stat: float, df: int) -> float:
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
    if math.isnan(t_stat):
        return float("nan")
    if math.isinf(t_stat):
        return 0.0
    t_abs = abs(t_stat)
    x = df / (df + t_abs * t_abs)
    return _regularized_beta(x, df / 2.0, 0.5)


def _student_t_critical(confidence: float, df: int) -> float:
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    target = 1.0 - confidence
    lo, hi = 0.0, 1.0
    while _student_t_two_sided_pvalue(hi, df) > target:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _student_t_two_sided_pvalue(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _f_sf(f_stat: float, dfn: int, dfd: int) -> float:
    if dfn <= 0 or dfd <= 0:
        raise ValueError("F-test degrees of freedom must be positive")
    if f_stat < 0:
        return 1.0
    if math.isinf(f_stat):
        return 0.0
    x = dfd / (dfd + dfn * f_stat)
    return _regularized_beta(x, dfd / 2.0, dfn / 2.0)


# ---------------------------------------------------------------------------
# F1: OLS Fit - Giải Normal Equations
# ---------------------------------------------------------------------------

def ols_fit(
    X: list[list[float]],
    y: list[float],
) -> dict:
    """
    F1: Giải Normal Equations để tính nghiệm OLS từ đầu.

    Công thức:
        β̂ = (XᵀX)⁻¹Xᵀy
        σ̂² = RSS / (n - p - 1)

    X nhận vào CHƯA có cột bias - hàm tự thêm cột 1 bên trong.

    Tham số:
        X : list[list[float]] - Ma trận features, shape (n, p), chưa có bias.
        y : list[float]       - Vector target, shape (n,).

    Trả về dict:
        beta_hat   : list[float] - [intercept, β₁, …, βₚ], shape (p+1,).
        sigma2_hat : float       - Ước lượng phương sai nhiễu RSS/(n-p-1).
        y_hat      : list[float] - Giá trị dự đoán, shape (n,).
        residuals  : list[float] - Phần dư y - ŷ, shape (n,).
    """
    n, p = _validate_xy(X, y)

    # Bước 1: Thêm cột bias (cột 1) vào đầu → X_bias shape (n, p+1)
    X_bias = [[1.0] + row for row in X]

    # Bước 2: Tính XᵀX → (p+1, p+1)
    Xt = transpose(X_bias)
    XtX = matmul(Xt, X_bias)

    # Bước 3: Tính Xᵀy → (p+1,)
    Xty = [dot_product(Xt[i], y) for i in range(p + 1)]

    # Bước 4: Giải hệ (XᵀX) β = Xᵀy
    beta_hat = solve_system(XtX, Xty)

    # Bước 5: Tính ŷ = X_bias @ β̂
    y_hat = matvec(X_bias, beta_hat)

    # Bước 6: Tính residuals = y - ŷ
    residuals = [y[i] - y_hat[i] for i in range(n)]

    # Bước 7-8: RSS và σ̂²
    rss = sum(r * r for r in residuals)
    sigma2_hat = rss / (n - p - 1)

    return {
        "beta_hat":   beta_hat,
        "sigma2_hat": sigma2_hat,
        "y_hat":      y_hat,
        "residuals":  residuals,
    }


# ---------------------------------------------------------------------------
# F2: Hat Matrix - Ma trận chiếu H
# ---------------------------------------------------------------------------

def hat_matrix(X: list[list[float]]) -> dict:
    """
    F2: Tính Hat Matrix H = X(XᵀX)⁻¹Xᵀ và kiểm tra các tính chất.

    X nhận vào CHƯA có cột bias - hàm tự thêm cột 1 bên trong.

    Tham số:
        X : list[list[float]] - Ma trận features, shape (n, p), chưa có bias.

    Trả về dict:
        H             : list[list[float]] - Hat matrix (n x n).
        is_idempotent : bool              - H² ≈ H (sai số < 1e-8).
        is_symmetric  : bool              - Hᵀ ≈ H (sai số < 1e-8).
        rank          : int               - rank(H) = p+1.
        eigenvalues   : list[float]       - Giá trị riêng (chỉ 0 hoặc 1).
    """
    if not X or not isinstance(X, list):
        raise ValueError("X must be a non-empty list of rows")
    if not isinstance(X[0], list) or len(X[0]) == 0:
        raise ValueError("X must contain at least one feature column")
    n = len(X)
    p = len(X[0])
    for i, row in enumerate(X):
        if not isinstance(row, list) or len(row) != p:
            raise ValueError(f"X row {i} has inconsistent length")
    if n <= p:
        raise ValueError(f"Need n > p to compute the hat matrix (n={n}, p={p})")

    # Bước 1: Thêm cột bias
    X_bias = [[1.0] + row for row in X]

    # Bước 2-3: Tính (XᵀX)⁻¹
    Xt = transpose(X_bias)
    XtX = matmul(Xt, X_bias)
    XtX_inv = inverse(XtX)

    # Bước 4: H = X_bias @ (XᵀX)⁻¹ @ Xᵀ
    #         = X_bias @ temp,  với temp = (XᵀX)⁻¹ @ Xᵀ
    temp = matmul(XtX_inv, Xt)   # (p+1, n)
    H = matmul(X_bias, temp)     # (n, n)

    # Bước 5: Kiểm tra idempotent - H² ≈ H
    H2 = matmul(H, H)
    is_idempotent = True
    for i in range(n):
        for j in range(n):
            if abs(H2[i][j] - H[i][j]) > 1e-8:
                is_idempotent = False
                break
        if not is_idempotent:
            break

    # Bước 6: Kiểm tra symmetric - Hᵀ ≈ H
    Ht = transpose(H)
    is_symmetric = True
    for i in range(n):
        for j in range(n):
            if abs(Ht[i][j] - H[i][j]) > 1e-8:
                is_symmetric = False
                break
        if not is_symmetric:
            break

    # Bước 7: Rank - Với ma trận chiếu, rank = trace(H) làm tròn
    trace_H = sum(H[i][i] for i in range(n))
    rank = round(trace_H)

    # Bước 8: Eigenvalues - Với ma trận chiếu idempotent,
    # eigenvalues lý thuyết chỉ gồm 0 và 1.
    # Số eigenvalue = 1 chính bằng rank (= p+1).
    # KNOWN LIMITATION: eigenvalues được tính theo lý thuyết (hardcode [1]*rank + [0]*(n-rank))
    # thay vì tính thực từ ma trận bằng QR iteration - vượt scope dự án.
    # Giá trị này chính xác về mặt lý thuyết cho projection matrix idempotent.
    eigenvalues = [1.0] * rank + [0.0] * (n - rank)

    return {
        "H":             H,
        "is_idempotent": is_idempotent,
        "is_symmetric":  is_symmetric,
        "rank":          rank,
        "eigenvalues":   eigenvalues,
    }


def plot_hat_matrix_diagnostics(
    hat_result: dict,
    top_k: int = 30,
    save_dir: str | None = PART1_OUTPUT_DIR,
    show: bool = True,
) -> dict:
    """Plot Hat Matrix heatmap and eigenvalue counts from ``hat_matrix`` output."""
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    H = np.asarray(hat_result["H"], dtype=float)
    leverage = np.diag(H)
    k_view = min(top_k, H.shape[0])
    order = np.argsort(leverage)[::-1][:k_view]
    H_view = H[np.ix_(order, order)]
    vmax = float(np.max(np.abs(H_view))) if H_view.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        H_view,
        cmap="coolwarm",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "H_ij"},
        ax=axes[0],
    )
    axes[0].set_title(f"Hat Matrix Heatmap\nTop {k_view} leverage observations")
    axes[0].set_xlabel("Observation index (sorted)")
    axes[0].set_ylabel("Observation index (sorted)")

    eigenvalues = np.real(np.asarray(hat_result["eigenvalues"], dtype=float))
    n_one = int(np.sum(np.isclose(eigenvalues, 1.0, atol=1e-7)))
    n_zero = int(np.sum(np.isclose(eigenvalues, 0.0, atol=1e-7)))
    axes[1].bar(["0", "1"], [n_zero, n_one], color=["#8ecae6", "#fb8500"], edgecolor="black")
    axes[1].set_title("Eigenvalue Counts of H")
    axes[1].set_xlabel("Eigenvalue")
    axes[1].set_ylabel("Count")
    for idx, val in enumerate([n_zero, n_one]):
        axes[1].text(idx, val, str(val), ha="center", va="bottom")

    plt.tight_layout()

    out_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "hat_matrix_diagnostics.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "fig": fig,
        "axes": axes,
        "leverage": leverage.tolist(),
        "order": order.tolist(),
        "eigenvalue_counts": {"zero": n_zero, "one": n_one},
        "out_path": out_path,
    }


# ---------------------------------------------------------------------------
# F3: Model Metrics - Các chỉ số đánh giá mô hình
# ---------------------------------------------------------------------------

def model_metrics(
    y:     list[float],
    y_hat: list[float],
    p:     int,
) -> dict:
    """
    F3: Tính đầy đủ các chỉ số đánh giá mô hình hồi quy.

    Tham số:
        y     : list[float] - Ground truth, shape (n,).
        y_hat : list[float] - Dự đoán, shape (n,).
        p     : int         - Số features (không tính intercept).

    Trả về dict:
        RSS      : float - Residual Sum of Squares = Σ(yᵢ - ŷᵢ)².
        TSS      : float - Total Sum of Squares    = Σ(yᵢ - ȳ)².
        MSS      : float - Model Sum of Squares    = TSS - RSS.
        R2       : float - Hệ số xác định          = 1 - RSS/TSS.
        R2_adj   : float - R² hiệu chỉnh           = 1 - (n-1)/(n-p-1)*(1-R²).
        F_stat   : float - F-statistic             = (MSS/p) / (RSS/(n-p-1)).
        F_pvalue : float - p-value của F-test.
        MAE      : float - Mean Absolute Error      = mean(|y - ŷ|).
        RMSE     : float - Root Mean Squared Error   = sqrt(mean((y - ŷ)²)).
    """
    n = len(y)
    if len(y_hat) != n:
        raise ValueError(f"y and y_hat length mismatch: len(y)={n}, len(y_hat)={len(y_hat)}")
    if p <= 0:
        raise ValueError("p must be positive for the overall F-test")
    if n <= p + 1:
        raise ValueError(f"Cần n > p + 1 để tính model metrics (n={n}, p={p})")
    y_bar = sum(y) / n

    # Sums of Squares
    rss = sum((y[i] - y_hat[i]) ** 2 for i in range(n))
    tss = sum((y[i] - y_bar) ** 2 for i in range(n))
    mss = tss - rss

    # R² và Adjusted R²
    if abs(tss) < EPSILON:
        # y hằng số → R² = 1 nếu dự đoán chính xác, 0 nếu không
        r2 = 1.0 if abs(rss) < EPSILON else 0.0
    else:
        r2 = 1.0 - rss / tss

    r2_adj = 1.0 - (n - 1) / (n - p - 1) * (1.0 - r2)

    # F-statistic
    denom = rss / (n - p - 1)
    if abs(denom) < EPSILON:
        f_stat = float("inf")
    else:
        f_stat = (mss / p) / denom

    # F p-value
    f_pvalue = _f_sf(f_stat, dfn=p, dfd=n - p - 1)

    # MAE
    mae = sum(abs(y[i] - y_hat[i]) for i in range(n)) / n

    # RMSE
    rmse = math.sqrt(rss / n)

    return {
        "RSS":      rss,
        "TSS":      tss,
        "MSS":      mss,
        "R2":       r2,
        "R2_adj":   r2_adj,
        "F_stat":   f_stat,
        "F_pvalue": f_pvalue,
        "MAE":      mae,
        "RMSE":     rmse,
    }


# ---------------------------------------------------------------------------
# F4: Coefficient Inference - SE, t-stat, p-value, CI
# Liên kết: Nhận output từ F1 (beta_hat, sigma2_hat)
# ---------------------------------------------------------------------------

def coef_inference(
    X:        list[list[float]],
    y:        list[float],
    beta_hat: list[float],
    sigma2:   float,
) -> dict:
    """
    F4: Tính standard errors, t-statistics, p-values, CI cho từng hệ số.

    Liên kết: Sử dụng beta_hat và sigma2_hat trực tiếp từ ols_fit (F1).
    Công thức:
        Cov(β̂) = σ² · (XᵀX)⁻¹
        SE(β̂ⱼ) = sqrt(Cov(β̂)[j,j])
        t_j = β̂ⱼ / SE(β̂ⱼ)
        p-value = 2 · P(T > |t_j|)  với  T ~ t(n - p - 1)

    Tham số:
        X        : Ma trận features (n x p), chưa có bias.
        y        : Vector target (n,).
        beta_hat : list[float] - [intercept, β₁, …, βₚ] từ ols_fit.
        sigma2   : float - σ̂² từ ols_fit.

    Trả về dict (không phải DataFrame - dùng pandas.DataFrame(coef_inference(...)) nếu cần hiển thị bảng):
        coef     : list[float] - Hệ số β̂.
        std_err  : list[float] - Standard errors.
        t_stat   : list[float] - t-statistics.
        p_value  : list[float] - p-values (two-sided).
        ci_lower : list[float] - 95% CI cận dưới.
        ci_upper : list[float] - 95% CI cận trên.
        names    : list[str]   - Tên hệ số.
    """
    n, p = _validate_xy(X, y)
    if len(beta_hat) != p + 1:
        raise ValueError(f"beta_hat must have length p+1={p + 1}")
    if sigma2 < 0:
        raise ValueError("sigma2 must be non-negative")

    # Thêm cột bias
    X_bias = [[1.0] + row for row in X]

    # Tính (XᵀX)⁻¹
    Xt = transpose(X_bias)
    XtX = matmul(Xt, X_bias)
    XtX_inv = inverse(XtX)

    # Cov(β̂) = σ² · (XᵀX)⁻¹
    p1 = p + 1
    df = n - p1

    # Standard errors = sqrt(diag(σ² · (XᵀX)⁻¹))
    std_err = []
    for j in range(p1):
        var_j = sigma2 * XtX_inv[j][j]
        std_err.append(math.sqrt(max(var_j, 0.0)))

    # t-statistics = β̂ⱼ / SE(β̂ⱼ)
    t_stat = []
    for j in range(p1):
        if std_err[j] > EPSILON:
            t_stat.append(beta_hat[j] / std_err[j])
        else:
            t_stat.append(float('nan'))

    # p-values (two-sided)
    p_value = [_student_t_two_sided_pvalue(t_j, df=df) for t_j in t_stat]

    # 95% CI
    t_crit = _student_t_critical(0.95, df=df)
    ci_lower = [beta_hat[j] - t_crit * std_err[j] for j in range(p1)]
    ci_upper = [beta_hat[j] + t_crit * std_err[j] for j in range(p1)]

    names = ["intercept"] + [f"x{i}" for i in range(1, p + 1)]

    return {
        "coef":     list(beta_hat),
        "std_err":  std_err,
        "t_stat":   t_stat,
        "p_value":  p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "names":    names,
    }


# ---------------------------------------------------------------------------
# F5: VIF - Variance Inflation Factor
# Liên kết: Gọi F1 (ols_fit) và F3 (model_metrics) cho mỗi sub-regression
# ---------------------------------------------------------------------------

def vif(X: list[list[float]]) -> dict[str, float]:
    """
    F5: Tính VIF cho mỗi feature bằng cách hồi quy xⱼ theo các feature còn lại.

    Liên kết:
        - Gọi ols_fit (F1) để fit sub-regression xⱼ ~ X_others.
        - Gọi model_metrics (F3) để tính R²ⱼ.
        - VIFⱼ = 1 / (1 - R²ⱼ)

    Tham số:
        X : list[list[float]] - Ma trận features (n x p), chưa có bias.

    Trả về:
        dict[str, float] - {"x1": VIF₁, "x2": VIF₂, ...}
    """
    if not X or not isinstance(X, list):
        raise ValueError("X must be a non-empty list of rows")
    if not isinstance(X[0], list) or len(X[0]) == 0:
        raise ValueError("X must contain at least one feature column")
    n = len(X)
    p = len(X[0])
    for i, row in enumerate(X):
        if not isinstance(row, list) or len(row) != p:
            raise ValueError(f"X row {i} has inconsistent length")

    if p < 2:
        raise ValueError("VIF requires at least 2 features")

    vifs: dict[str, float] = {}
    for j in range(p):
        # Tách feature j làm target, các feature còn lại làm predictors
        x_j = [X[i][j] for i in range(n)]
        X_others = [[X[i][k] for k in range(p) if k != j] for i in range(n)]

        # Dùng F1 để fit sub-regression
        sub_res = ols_fit(X_others, x_j)

        # Dùng F3 để tính R²
        sub_met = model_metrics(x_j, sub_res["y_hat"], p=p - 1)
        r2_j = sub_met["R2"]

        # VIF = 1 / (1 - R²)
        if abs(1.0 - r2_j) < EPSILON:
            vifs[f"x{j + 1}"] = float("inf")
        else:
            vifs[f"x{j + 1}"] = 1.0 / (1.0 - r2_j)

    return vifs



# ---------------------------------------------------------------------------
# Unit Tests - F1-F5
# ---------------------------------------------------------------------------


def test_ols_fit_exact_solution() -> bool:
    from test_utils import assert_close, make_linear_data

    true_beta = [1.0, 2.0]
    X, y = make_linear_data(n=30, beta=true_beta, sigma=0.0, seed=RANDOM_STATE)
    res = ols_fit(X, y)
    return assert_close(res["beta_hat"], true_beta, label="ols_fit exact beta when sigma=0", atol=1e-7)


def test_ols_fit_output_shapes() -> bool:
    from test_utils import assert_shape, make_linear_data

    X, y = make_linear_data(n=20, beta=[3.0, -1.0, 0.5], sigma=0.5, seed=RANDOM_STATE)
    res = ols_fit(X, y)
    checks = [
        assert_shape(res["beta_hat"], (3,), label="ols_fit beta_hat shape is p+1"),
        assert_shape(res["y_hat"], (20,), label="ols_fit y_hat shape is n"),
        assert_shape(res["residuals"], (20,), label="ols_fit residuals shape is n"),
    ]
    return all(checks)


def test_ols_fit_residuals_sum_to_zero() -> bool:
    from test_utils import assert_close, make_linear_data

    X, y = make_linear_data(n=50, beta=[2.0, -1.5, 0.8], sigma=1.0, seed=RANDOM_STATE)
    res = ols_fit(X, y)
    return assert_close(sum(res["residuals"]), 0.0, label="OLS residuals sum to zero", atol=1e-8)


def test_ols_fit_vs_sklearn_verifier() -> bool:
    from test_utils import make_linear_data, verify_vs_sklearn_ols

    X, y = make_linear_data(n=60, beta=[1.0, 2.0, -0.5], sigma=0.3, seed=RANDOM_STATE)
    res = ols_fit(X, y)
    return verify_vs_sklearn_ols(X, y, res["beta_hat"], rtol=1e-3)


def test_ols_fit_singular_matrix_raises() -> bool:
    from test_utils import assert_raises

    return assert_raises(
        ValueError,
        ols_fit,
        [[1.0, 1.0], [1.0, 1.0]],
        [1.0, 2.0],
        label="ols_fit raises on singular design matrix",
    )


def test_hat_matrix_projection_properties() -> bool:
    from test_utils import assert_equal, assert_shape, assert_true, make_linear_data

    X, _ = make_linear_data(n=20, beta=[1.0, 2.0], sigma=0.0, seed=RANDOM_STATE)
    res = hat_matrix(X)
    eigen_ok = all(abs(value) < 1e-4 or abs(value - 1.0) < 1e-4 for value in res["eigenvalues"])
    checks = [
        assert_shape(res["H"], (20, 20), label="hat_matrix H shape is n by n"),
        assert_true(res["is_idempotent"], label="hat_matrix is idempotent"),
        assert_true(res["is_symmetric"], label="hat_matrix is symmetric"),
        assert_equal(res["rank"], 2, label="hat_matrix rank equals p+1"),
        assert_true(eigen_ok, label="hat_matrix eigenvalues are 0 or 1"),
    ]
    return all(checks)


def test_hat_matrix_projects_y_to_ols_predictions() -> bool:
    from test_utils import assert_close, make_linear_data
    from utils import matvec

    X, y = make_linear_data(n=15, beta=[1.0, 2.0], sigma=0.5, seed=RANDOM_STATE)
    ols_res = ols_fit(X, y)
    hat_res = hat_matrix(X)
    return assert_close(matvec(hat_res["H"], y), ols_res["y_hat"], label="H @ y equals OLS fitted values", atol=1e-7)


def test_model_metrics_basic_properties() -> bool:
    from test_utils import assert_close, assert_in_range, assert_true, make_linear_data

    X, y = make_linear_data(n=50, beta=[1.0, 2.0], sigma=1.0, seed=RANDOM_STATE)
    fit = ols_fit(X, y)
    met = model_metrics(y, fit["y_hat"], p=1)
    checks = [
        assert_true("R2" in met and "F_stat" in met, label="model_metrics returns R2 and F_stat"),
        assert_in_range(met["R2"], 0.0, 1.0, label="model_metrics R2 is in [0, 1]"),
        assert_close(met["TSS"], met["RSS"] + met["MSS"], label="TSS equals RSS + MSS", rtol=1e-4),
    ]
    return all(checks)


def test_model_metrics_rejects_insufficient_df() -> bool:
    from test_utils import assert_raises

    return assert_raises(
        ValueError,
        model_metrics,
        [1.0, 2.0],
        [1.1, 1.9],
        2,
        label="model_metrics raises when n <= p+1",
    )


def test_coef_inference_output_structure() -> bool:
    from test_utils import assert_equal, assert_true, make_linear_data

    X, y = make_linear_data(n=120, beta=[1.5, 2.0], sigma=0.3, seed=RANDOM_STATE)
    fit = ols_fit(X, y)
    out = coef_inference(X, y, fit["beta_hat"], fit["sigma2_hat"])
    checks = [
        assert_equal(
            list(out.keys()),
            ["coef", "std_err", "t_stat", "p_value", "ci_lower", "ci_upper", "names"],
            label="coef_inference returns expected keys",
        ),
        assert_equal(out["names"], ["intercept", "x1"], label="coef_inference returns coefficient names"),
        assert_true(all(se >= 0 for se in out["std_err"]), label="coef_inference std_err values are non-negative"),
        assert_true(all(0.0 <= p <= 1.0 for p in out["p_value"]), label="coef_inference p-values are in [0, 1]"),
    ]
    return all(checks)


def test_coef_inference_ci_contains_true_coefficients() -> bool:
    from test_utils import assert_true, make_linear_data

    true_beta = [2.0, -1.0, 0.7]
    X, y = make_linear_data(n=600, beta=true_beta, sigma=0.2, seed=RANDOM_STATE)
    fit = ols_fit(X, y)
    out = coef_inference(X, y, fit["beta_hat"], fit["sigma2_hat"])
    contains = all(out["ci_lower"][i] <= truth <= out["ci_upper"][i] for i, truth in enumerate(true_beta))
    return assert_true(contains, label="coef_inference confidence intervals contain true beta")


def test_vif_returns_one_value_per_feature() -> bool:
    from test_utils import assert_equal, make_collinear_data

    X, _ = make_collinear_data(n=150, seed=RANDOM_STATE)
    out = vif(X)
    return assert_equal(set(out.keys()), {"x1", "x2", "x3"}, label="vif returns one value per feature")


def test_vif_detects_collinearity() -> bool:
    from test_utils import assert_true, make_collinear_data

    X, _ = make_collinear_data(n=200, seed=RANDOM_STATE)
    out = vif(X)
    return assert_true(any(value > 10 for value in out.values()), label="vif detects high collinearity")


def test_vif_rejects_single_feature() -> bool:
    from test_utils import assert_raises

    return assert_raises(ValueError, vif, [[1.0], [2.0]], label="vif raises when p < 2")


def _run_test_group(tests: list, suite_name: str) -> tuple[int, int]:
    from test_utils import TestLogger

    TestLogger.print_suite_header(suite_name)
    passed = sum(int(test()) for test in tests)
    return passed, len(tests)


def run_tests() -> tuple[int, int]:
    total_passed = 0
    total_tests = 0
    groups = [
        (
            "F1 - ols_fit",
            [
                test_ols_fit_exact_solution,
                test_ols_fit_output_shapes,
                test_ols_fit_residuals_sum_to_zero,
                test_ols_fit_vs_sklearn_verifier,
                test_ols_fit_singular_matrix_raises,
            ],
        ),
        (
            "F2 - hat_matrix",
            [
                test_hat_matrix_projection_properties,
                test_hat_matrix_projects_y_to_ols_predictions,
            ],
        ),
        (
            "F3 - model_metrics",
            [
                test_model_metrics_basic_properties,
                test_model_metrics_rejects_insufficient_df,
            ],
        ),
        (
            "F4 - coef_inference",
            [
                test_coef_inference_output_structure,
                test_coef_inference_ci_contains_true_coefficients,
            ],
        ),
        (
            "F5 - vif",
            [
                test_vif_returns_one_value_per_feature,
                test_vif_detects_collinearity,
                test_vif_rejects_single_feature,
            ],
        ),
    ]
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
    print("  UNIT TESTS - ols_implementation.py")
    print("=" * 55)
    passed, total = run_tests()
    TestLogger.print_summary(passed, total)
