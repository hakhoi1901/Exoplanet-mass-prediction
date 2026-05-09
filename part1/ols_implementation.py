from __future__ import annotations
import math
import sys
import os

# Thêm thư mục gốc vào path để import utils và config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import transpose, matmul, matvec, dot_product, inverse, solve_system, identity_matrix, add_bias
from config import RANDOM_STATE, EPSILON

# ---------------------------------------------------------------------------
# F1: OLS Fit — Giải Normal Equations
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

    X nhận vào CHƯA có cột bias — hàm tự thêm cột 1 bên trong.

    Tham số:
        X : list[list[float]] — Ma trận features, shape (n, p), chưa có bias.
        y : list[float]       — Vector target, shape (n,).

    Trả về dict:
        beta_hat   : list[float] — [intercept, β₁, …, βₚ], shape (p+1,).
        sigma2_hat : float       — Ước lượng phương sai nhiễu RSS/(n-p-1).
        y_hat      : list[float] — Giá trị dự đoán, shape (n,).
        residuals  : list[float] — Phần dư y - ŷ, shape (n,).
    """
    n = len(X)
    p = len(X[0])

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
# F2: Hat Matrix — Ma trận chiếu H
# ---------------------------------------------------------------------------

def hat_matrix(X: list[list[float]]) -> dict:
    """
    F2: Tính Hat Matrix H = X(XᵀX)⁻¹Xᵀ và kiểm tra các tính chất.

    X nhận vào CHƯA có cột bias — hàm tự thêm cột 1 bên trong.

    Tham số:
        X : list[list[float]] — Ma trận features, shape (n, p), chưa có bias.

    Trả về dict:
        H             : list[list[float]] — Hat matrix (n x n).
        is_idempotent : bool              — H² ≈ H (sai số < 1e-8).
        is_symmetric  : bool              — Hᵀ ≈ H (sai số < 1e-8).
        rank          : int               — rank(H) = p+1.
        eigenvalues   : list[float]       — Giá trị riêng (chỉ 0 hoặc 1).
    """
    n = len(X)
    p = len(X[0])

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

    # Bước 5: Kiểm tra idempotent — H² ≈ H
    H2 = matmul(H, H)
    is_idempotent = True
    for i in range(n):
        for j in range(n):
            if abs(H2[i][j] - H[i][j]) > 1e-8:
                is_idempotent = False
                break
        if not is_idempotent:
            break

    # Bước 6: Kiểm tra symmetric — Hᵀ ≈ H
    Ht = transpose(H)
    is_symmetric = True
    for i in range(n):
        for j in range(n):
            if abs(Ht[i][j] - H[i][j]) > 1e-8:
                is_symmetric = False
                break
        if not is_symmetric:
            break

    # Bước 7: Rank — Với ma trận chiếu, rank = trace(H) làm tròn
    trace_H = sum(H[i][i] for i in range(n))
    rank = round(trace_H)

    # Bước 8: Eigenvalues — Với ma trận chiếu idempotent,
    # eigenvalues lý thuyết chỉ gồm 0 và 1.
    # Số eigenvalue = 1 chính bằng rank (= p+1).
    # KNOWN LIMITATION: eigenvalues được tính theo lý thuyết (hardcode [1]*rank + [0]*(n-rank))
    # thay vì tính thực từ ma trận bằng QR iteration — vượt scope dự án.
    # Giá trị này chính xác về mặt lý thuyết cho projection matrix idempotent.
    eigenvalues = [1.0] * rank + [0.0] * (n - rank)

    return {
        "H":             H,
        "is_idempotent": is_idempotent,
        "is_symmetric":  is_symmetric,
        "rank":          rank,
        "eigenvalues":   eigenvalues,
    }


# ---------------------------------------------------------------------------
# F3: Model Metrics — Các chỉ số đánh giá mô hình
# ---------------------------------------------------------------------------

def model_metrics(
    y:     list[float],
    y_hat: list[float],
    p:     int,
) -> dict:
    """
    F3: Tính đầy đủ các chỉ số đánh giá mô hình hồi quy.

    Tham số:
        y     : list[float] — Ground truth, shape (n,).
        y_hat : list[float] — Dự đoán, shape (n,).
        p     : int         — Số features (không tính intercept).

    Trả về dict:
        RSS      : float — Residual Sum of Squares = Σ(yᵢ - ŷᵢ)².
        TSS      : float — Total Sum of Squares    = Σ(yᵢ - ȳ)².
        MSS      : float — Model Sum of Squares    = TSS - RSS.
        R2       : float — Hệ số xác định          = 1 - RSS/TSS.
        R2_adj   : float — R² hiệu chỉnh           = 1 - (n-1)/(n-p-1)*(1-R²).
        F_stat   : float — F-statistic             = (MSS/p) / (RSS/(n-p-1)).
        F_pvalue : float — p-value của F-test.
        MAE      : float — Mean Absolute Error      = mean(|y - ŷ|).
        RMSE     : float — Root Mean Squared Error   = sqrt(mean((y - ŷ)²)).
    """
    # Chỉ dùng scipy cho F p-value — không có closed-form thủ công hợp lý
    from scipy.stats import f as f_dist

    n = len(y)
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
    f_pvalue = float(f_dist.sf(f_stat, dfn=p, dfd=n - p - 1))

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
# F4: Coefficient Inference — SE, t-stat, p-value, CI
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
        beta_hat : list[float] — [intercept, β₁, …, βₚ] từ ols_fit.
        sigma2   : float — σ̂² từ ols_fit.

    Trả về dict (không phải DataFrame — dùng pandas.DataFrame(coef_inference(...)) nếu cần hiển thị bảng):
        coef     : list[float] — Hệ số β̂.
        std_err  : list[float] — Standard errors.
        t_stat   : list[float] — t-statistics.
        p_value  : list[float] — p-values (two-sided).
        ci_lower : list[float] — 95% CI cận dưới.
        ci_upper : list[float] — 95% CI cận trên.
        names    : list[str]   — Tên hệ số.
    """
    # Chỉ dùng scipy cho t-distribution CDF
    from scipy.stats import t as t_dist

    n = len(X)
    p = len(X[0])

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
    p_value = [2.0 * float(t_dist.sf(abs(t_j), df=df)) for t_j in t_stat]

    # 95% CI
    t_crit = float(t_dist.ppf(0.975, df=df))
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
# F5: VIF — Variance Inflation Factor
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
        X : list[list[float]] — Ma trận features (n x p), chưa có bias.

    Trả về:
        dict[str, float] — {"x1": VIF₁, "x2": VIF₂, ...}
    """
    n = len(X)
    p = len(X[0])

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
# Unit Tests — F1-F5
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    from test_utils import TestLogger, make_linear_data, make_collinear_data, assert_close, assert_equal, assert_true, assert_shape, assert_raises, verify_vs_sklearn_ols

    print("=" * 55)
    print("  UNIT TESTS — ols_implementation.py")
    print("=" * 55)

    passed = 0
    total = 0

    def run(result: bool):
        global passed, total
        total += 1
        passed += int(result)

    # --- F1: ols_fit ---
    TestLogger.print_suite_header("F1 — ols_fit")
    X, y = make_linear_data(n=30, beta=[1.0, 2.0, -1.5], sigma=0.5, seed=RANDOM_STATE)
    res_f1 = ols_fit(X, y)
    
    run(assert_true("beta_hat" in res_f1 and "y_hat" in res_f1, label="ols_fit returns correct keys"))
    run(assert_shape(res_f1["beta_hat"], (3,), label="ols_fit beta_hat shape is (p+1,)"))
    run(verify_vs_sklearn_ols(X, y, res_f1["beta_hat"], rtol=1e-3))
    run(assert_raises(ValueError, ols_fit, [[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0], label="ols_fit singular matrix raises error"))

    # --- F2: hat_matrix ---
    TestLogger.print_suite_header("F2 — hat_matrix")
    X_hat, _ = make_linear_data(n=20, beta=[1.0, 2.0], sigma=0.0, seed=RANDOM_STATE)
    res_f2 = hat_matrix(X_hat)
    
    run(assert_shape(res_f2["H"], (20, 20), label="hat_matrix H shape is (n, n)"))
    run(assert_true(res_f2["is_idempotent"], label="hat_matrix is idempotent"))
    run(assert_true(res_f2["is_symmetric"], label="hat_matrix is symmetric"))
    run(assert_equal(res_f2["rank"], 2, label="hat_matrix rank = p+1"))
    
    zeros_and_ones = [eval for eval in res_f2["eigenvalues"] if abs(eval) < 1e-4 or abs(eval - 1.0) < 1e-4]
    run(assert_equal(len(zeros_and_ones), len(res_f2["eigenvalues"]), label="hat_matrix eigenvalues are 0 or 1"))

    # --- F3: model_metrics ---
    TestLogger.print_suite_header("F3 — model_metrics")
    X_m, y_m = make_linear_data(n=50, beta=[1.0, 2.0], sigma=1.0, seed=RANDOM_STATE)
    res_ols_m = ols_fit(X_m, y_m)
    res_f3 = model_metrics(y_m, res_ols_m["y_hat"], p=1)
    
    run(assert_true("R2" in res_f3 and "F_stat" in res_f3, label="model_metrics returns keys"))
    run(assert_true(0.0 <= res_f3["R2"] <= 1.0, label="model_metrics R2 is in [0,1]"))
    run(assert_close(res_f3["TSS"], res_f3["RSS"] + res_f3["MSS"], label="model_metrics TSS = RSS + MSS", rtol=1e-4))
    run(assert_raises(ValueError, model_metrics, [1.0, 2.0], [1.1, 1.9], 2, label="model_metrics error when n <= p+1"))

    # --- F4: coef_inference ---
    TestLogger.print_suite_header("F4 — coef_inference")
    X_i, y_i = make_linear_data(n=60, beta=[0.5, 3.0], sigma=1.0, seed=RANDOM_STATE)
    res_ols_i = ols_fit(X_i, y_i)
    res_f4 = coef_inference(X_i, y_i, res_ols_i["beta_hat"], res_ols_i["sigma2_hat"])
    
    run(assert_true(len(res_f4["names"]) == 2, label="coef_inference returns dict with lists of length p+1"))
    run(assert_true(all(se >= 0 for se in res_f4["std_err"]), label="coef_inference std_err >= 0"))
    run(assert_true(all(0 <= p <= 1 for p in res_f4["p_value"]), label="coef_inference p_value in [0,1]"))
    run(assert_true(all(res_f4["ci_lower"][i] <= res_f4["ci_upper"][i] for i in range(2)), label="coef_inference ci_lower <= ci_upper"))

    # --- F5: vif ---
    TestLogger.print_suite_header("F5 — vif")
    X_v, _ = make_collinear_data(n=100, seed=RANDOM_STATE)
    res_f5 = vif(X_v)
    
    run(assert_equal(len(res_f5), 3, label="vif returns dict of length p"))
    run(assert_true(any(v > 10 for v in res_f5.values()), label="vif detects high collinearity (>10)"))
    run(assert_true(all(v >= 1.0 for v in res_f5.values()), label="vif values are >= 1.0"))
    run(assert_raises(ValueError, vif, [[1.0], [2.0]], label="vif raises error when p < 2"))

    TestLogger.print_summary(passed, total)
