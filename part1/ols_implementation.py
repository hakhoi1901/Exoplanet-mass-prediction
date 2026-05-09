from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np
import pandas as pd
from scipy.stats import t, f

from utils import matmul, transpose, inverse, add_bias, matvec, solve_system

def ols_fit(X: list[list[float]], y: list[float]) -> dict:
    X_bias = add_bias(X)
    n = len(X_bias)
    p1 = len(X_bias[0])
    
    Xt = transpose(X_bias)
    XtX = matmul(Xt, X_bias)
    
    Xty_vec = matvec(Xt, y)
    try:
        beta_hat = solve_system(XtX, Xty_vec)
    except ValueError:
        raise ValueError("Matrix is singular")
    
    y_hat = matvec(X_bias, beta_hat)
        
    residuals = [y[i] - y_hat[i] for i in range(n)]
    rss = sum(r**2 for r in residuals)
    sigma2_hat = rss / (n - p1)
    
    return {
        "beta_hat": beta_hat,
        "y_hat": y_hat,
        "residuals": residuals,
        "sigma2_hat": sigma2_hat,
    }

def _r2_score_list(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    y_bar = sum(y_true) / n
    rss = sum((t_val - p_val)**2 for t_val, p_val in zip(y_true, y_pred))
    tss = sum((t_val - y_bar)**2 for t_val in y_true)
    if abs(tss) < 1e-12:
        return 1.0 if abs(rss) < 1e-12 else 0.0
    return 1.0 - rss / tss

def coef_inference(
    X: list[list[float]],
    y: list[float],
    beta_hat: list[float],
    sigma2: float,
) -> pd.DataFrame:
    """
    Compute coefficient inference statistics for an OLS model.

    Returns a DataFrame with columns:
    coef, std_err, t_stat, p_value, ci_lower, ci_upper
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if sigma2 < 0:
        raise ValueError("sigma2 must be non-negative")
    
    X_bias = add_bias(X)
    n = len(X_bias)
    p1 = len(X_bias[0])
    df = n - p1
    if df <= 0:
        raise ValueError("Need n > p + 1 for coefficient inference")
    
    Xt = transpose(X_bias)
    XtX = matmul(Xt, X_bias)
    try:
        XtX_inv = inverse(XtX)
    except ValueError:
        raise ValueError("Matrix is singular")
        
    std_err = []
    for j in range(p1):
        var_j = sigma2 * XtX_inv[j][j]
        std_err.append(math.sqrt(max(0.0, var_j)))
        
    t_stat = []
    p_value = []
    ci_lower = []
    ci_upper = []
    
    t_crit = t.ppf(0.975, df=df)
    
    for j in range(p1):
        coef = beta_hat[j]
        se = std_err[j]
        if se > 1e-12:
            ts = coef / se
            t_stat.append(ts)
            p_value.append(2.0 * t.sf(abs(ts), df=df))
        else:
            t_stat.append(float('nan'))
            p_value.append(float('nan'))
            
        ci_lower.append(coef - t_crit * se)
        ci_upper.append(coef + t_crit * se)
        
    feature_names = ["intercept"] + [f"x{i}" for i in range(1, p1)]
    return pd.DataFrame(
        {
            "coef": beta_hat,
            "std_err": std_err,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
        index=feature_names,
    )

def vif(X: list[list[float]]) -> dict[str, float]:
    """
    Compute Variance Inflation Factor (VIF) for each feature in X.
    """
    n = len(X)
    p = len(X[0]) if n > 0 else 0
    if p < 2:
        raise ValueError("VIF requires at least 2 features")
    if n <= p:
        raise ValueError("Need n > p for stable VIF estimation")
    
    vifs = {}
    for j in range(p):
        x_j = [row[j] for row in X]
        X_others = [[row[i] for i in range(p) if i != j] for row in X]
        
        result_j = ols_fit(X_others, x_j)
        r2_j = _r2_score_list(x_j, result_j["y_hat"])
        
        if 1.0 - r2_j < 1e-12:
            vif_j = float("inf")
        else:
            vif_j = 1.0 / (1.0 - r2_j)
            
        vifs[f"x{j + 1}"] = vif_j
        
    return vifs

def hat_matrix(X: list[list[float]]) -> dict:
    """
    F2: Tính Hat Matrix H = X(XᵀX)⁻¹Xᵀ và kiểm tra các tính chất.
    """
    X_bias = add_bias(X)
    n = len(X_bias)
    p1 = len(X_bias[0]) if n > 0 else 0
    
    Xt = transpose(X_bias)
    XtX = matmul(Xt, X_bias)
    
    try:
        A_inv = inverse(XtX)
    except ValueError:
        raise ValueError("Matrix is singular")
        
    H = matmul(matmul(X_bias, A_inv), Xt)
    
    # Check idempotent: H^2 = H
    H_sq = matmul(H, H)
    is_idempotent = True
    for i in range(n):
        for j in range(n):
            if abs(H_sq[i][j] - H[i][j]) > 1e-8:
                is_idempotent = False
                break
        if not is_idempotent:
            break
            
    # Check symmetric: H^T = H
    H_T = transpose(H)
    is_symmetric = True
    for i in range(n):
        for j in range(n):
            if abs(H_T[i][j] - H[i][j]) > 1e-8:
                is_symmetric = False
                break
        if not is_symmetric:
            break
            
    # rank = trace(H) for idempotent matrix
    rank = int(round(sum(H[i][i] for i in range(n))))
    
    # eigenvalues
    eigenvalues = np.linalg.eigvals(np.array(H)).real.tolist()
    
    return {
        "H": H,
        "is_idempotent": is_idempotent,
        "is_symmetric": is_symmetric,
        "rank": rank,
        "eigenvalues": eigenvalues
    }

def model_metrics(y: list[float], y_hat: list[float], p: int) -> dict:
    """
    F3: Tính đầy đủ các chỉ số đánh giá mô hình.
    """
    n = len(y)
    if n <= p + 1:
        raise ValueError("Need n > p + 1 for model metrics")
        
    y_bar = sum(y) / n
    rss = sum((y[i] - y_hat[i])**2 for i in range(n))
    tss = sum((y[i] - y_bar)**2 for i in range(n))
    mss = tss - rss
    
    r2 = 1.0 - rss / tss if tss > 1e-12 else 0.0
    r2_adj = 1.0 - (n - 1) / (n - p - 1) * (1.0 - r2)
    
    mse_model = mss / p if p > 0 else 0.0
    mse_res = rss / (n - p - 1)
    
    f_stat = mse_model / mse_res if mse_res > 1e-12 else float('inf')
    if f_stat != float('inf') and p > 0:
        f_pvalue = f.sf(f_stat, p, n - p - 1)
    else:
        f_pvalue = float('nan')
        
    mae = sum(abs(y[i] - y_hat[i]) for i in range(n)) / n
    rmse = math.sqrt(rss / n)
    
    return {
        "RSS": rss,
        "TSS": tss,
        "MSS": mss,
        "R2": r2,
        "R2_adj": r2_adj,
        "F_stat": f_stat,
        "F_pvalue": f_pvalue,
        "MAE": mae,
        "RMSE": rmse
    }

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
    X, y = make_linear_data(n=30, beta=[1.0, 2.0, -1.5], sigma=0.5, seed=42)
    res_f1 = ols_fit(X, y)
    
    run(assert_true("beta_hat" in res_f1 and "y_hat" in res_f1, label="ols_fit returns correct keys"))
    run(assert_shape(res_f1["beta_hat"], (3,), label="ols_fit beta_hat shape is (p+1,)"))
    run(verify_vs_sklearn_ols(X, y, res_f1["beta_hat"], rtol=1e-3))
    run(assert_raises(ValueError, ols_fit, [[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0], label="ols_fit singular matrix raises error"))

    # --- F2: hat_matrix ---
    TestLogger.print_suite_header("F2 — hat_matrix")
    X_hat, _ = make_linear_data(n=20, beta=[1.0, 2.0], sigma=0.0, seed=42)
    res_f2 = hat_matrix(X_hat)
    
    run(assert_shape(res_f2["H"], (20, 20), label="hat_matrix H shape is (n, n)"))
    run(assert_true(res_f2["is_idempotent"], label="hat_matrix is idempotent"))
    run(assert_true(res_f2["is_symmetric"], label="hat_matrix is symmetric"))
    run(assert_equal(res_f2["rank"], 2, label="hat_matrix rank = p+1"))
    
    zeros_and_ones = [eval for eval in res_f2["eigenvalues"] if abs(eval) < 1e-4 or abs(eval - 1.0) < 1e-4]
    run(assert_equal(len(zeros_and_ones), len(res_f2["eigenvalues"]), label="hat_matrix eigenvalues are 0 or 1"))

    # --- F3: model_metrics ---
    TestLogger.print_suite_header("F3 — model_metrics")
    X_m, y_m = make_linear_data(n=50, beta=[1.0, 2.0], sigma=1.0, seed=1)
    res_ols_m = ols_fit(X_m, y_m)
    res_f3 = model_metrics(y_m, res_ols_m["y_hat"], p=1)
    
    run(assert_true("R2" in res_f3 and "F_stat" in res_f3, label="model_metrics returns keys"))
    run(assert_true(0.0 <= res_f3["R2"] <= 1.0, label="model_metrics R2 is in [0,1]"))
    run(assert_close(res_f3["TSS"], res_f3["RSS"] + res_f3["MSS"], label="model_metrics TSS = RSS + MSS", rtol=1e-4))
    run(assert_raises(ValueError, model_metrics, [1.0, 2.0], [1.1, 1.9], 2, label="model_metrics error when n <= p+1"))

    # --- F4: coef_inference ---
    TestLogger.print_suite_header("F4 — coef_inference")
    X_i, y_i = make_linear_data(n=60, beta=[0.5, 3.0], sigma=1.0, seed=2)
    res_ols_i = ols_fit(X_i, y_i)
    res_f4 = coef_inference(X_i, y_i, res_ols_i["beta_hat"], res_ols_i["sigma2_hat"])
    
    run(assert_shape(res_f4, (2, 6), label="coef_inference returns DataFrame of shape (p+1, 6)"))
    run(assert_true(all(res_f4["std_err"] >= 0), label="coef_inference std_err >= 0"))
    run(assert_true(all(res_f4["p_value"] >= 0) and all(res_f4["p_value"] <= 1), label="coef_inference p_value in [0,1]"))
    run(assert_true(all(res_f4["ci_lower"] <= res_f4["ci_upper"]), label="coef_inference ci_lower <= ci_upper"))

    # --- F5: vif ---
    TestLogger.print_suite_header("F5 — vif")
    X_v, _ = make_collinear_data(n=100, seed=42)
    res_f5 = vif(X_v)
    
    run(assert_equal(len(res_f5), 3, label="vif returns dict of length p"))
    run(assert_true(any(v > 10 for v in res_f5.values()), label="vif detects high collinearity (>10)"))
    run(assert_true(all(v >= 1.0 for v in res_f5.values()), label="vif values are >= 1.0"))
    run(assert_raises(ValueError, vif, [[1.0], [2.0]], label="vif raises error when p < 2"))

    TestLogger.print_summary(passed, total)
