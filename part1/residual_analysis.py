from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math

import matplotlib.pyplot as plt
import scipy.stats as stats

from utils import matmul, transpose, inverse


# ---------------------------------------------------------------------------
# F8: Phân Tích Phần Dư — 4 biểu đồ chẩn đoán
# ---------------------------------------------------------------------------

def residual_plots(
    y: list[float],
    y_hat: list[float],
    X: list[list[float]] | None = None,
    save_dir: str = "output",
) -> tuple[plt.Figure, dict]:
    """
    F8: Vẽ 4 biểu đồ chẩn đoán phần dư chuẩn.

    Biểu đồ:
        1. Residuals vs Fitted  — kiểm tra tính tuyến tính & đồng phương sai.
        2. Normal Q-Q           — kiểm tra tính chuẩn của phần dư (GM5).
        3. Scale-Location       — kiểm tra homoscedasticity.
        4. Cook's Distance      — phát hiện influential points.

    Tham số:
        y       : Giá trị thực (n,).
        y_hat   : Giá trị dự đoán (n,).
        X       : Ma trận design ĐÃ có cột bias (n x (p+1)). Bắt buộc để tính
                  Cook's Distance chính xác. Nếu None, dùng |residuals| thay thế.
        save_dir: Thư mục lưu ảnh.

    Trả về dict:
        residuals   : list[float]
        std_residuals: list[float]  — standardized residuals
        cooks_d     : list[float]   — Cook's Distance (hoặc |e| nếu X=None)
    """
    os.makedirs(save_dir, exist_ok=True)

    n = len(y)
    e = [y[i] - y_hat[i] for i in range(n)]

    p1 = 1
    if X is not None and len(X) > 0:
        p1 = len(X[0])
        
    rss = sum(ei ** 2 for ei in e)
    sigma2 = rss / max(n - p1, 1)
    sigma_hat = math.sqrt(max(sigma2, 1e-12))
    e_std = [ei / sigma_hat for ei in e]

    if X is not None:
        try:
            Xt = transpose(X)
            XtX = matmul(Xt, X)
            XtX_inv = inverse(XtX)
            H = matmul(X, matmul(XtX_inv, Xt))
            h = [H[i][i] for i in range(n)]
            
            cooks_d = []
            for i in range(n):
                denom = p1 * sigma2 * ((1 - h[i]) ** 2)
                if abs(denom) < 1e-12:
                    cooks_d.append(abs(e[i]))
                else:
                    cooks_d.append((e[i]**2 * h[i]) / denom)
        except ValueError:
            cooks_d = [abs(ei) for ei in e]
    else:
        cooks_d = [abs(ei) for ei in e]

    sqrt_abs_std = [math.sqrt(abs(es)) for es in e_std]
    indices = list(range(n))

    # --- Vẽ ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Phân Tích Phần Dư (Residual Diagnostic Plots)", fontsize=14, y=1.01)

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(y_hat, e, alpha=0.55, edgecolors="steelblue", facecolors="none", linewidths=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2, label="e = 0")
    _smooth_line(ax, y_hat, e, color="orange", label="LOWESS approx.")
    ax.set_title("Residuals vs Fitted", fontsize=12)
    ax.set_xlabel("Fitted values (ŷ)")
    ax.set_ylabel("Residuals (e = y − ŷ)")
    ax.legend()

    # 2. Normal Q-Q
    ax = axes[0, 1]
    (osm, osr), (slope, intercept_q, _) = stats.probplot(e, dist="norm")
    ax.scatter(osm, osr, alpha=0.55, edgecolors="steelblue", facecolors="none", linewidths=0.8,
               label="Quantile")
    qqx = [min(osm), max(osm)]
    ax.plot(qqx, [slope * x + intercept_q for x in qqx], color="red", linestyle="--", linewidth=1.2,
            label="Normal line")
    ax.set_title("Normal Q-Q", fontsize=12)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles (phần dư chuẩn hoá)")
    ax.legend()

    # 3. Scale-Location
    ax = axes[1, 0]
    ax.scatter(y_hat, sqrt_abs_std, alpha=0.55, edgecolors="steelblue", facecolors="none",
               linewidths=0.8)
    _smooth_line(ax, y_hat, sqrt_abs_std, color="orange", label="LOWESS approx.")
    ax.set_title("Scale-Location", fontsize=12)
    ax.set_xlabel("Fitted values (ŷ)")
    ax.set_ylabel("√|Standardized Residuals|")
    ax.legend()

    # 4. Cook's Distance
    ax = axes[1, 1]
    markerline, stemlines, baseline = ax.stem(
        indices, cooks_d, linefmt="steelblue", markerfmt=" ", basefmt="black"
    )
    stemlines.set_linewidths(0.8)
    # Ngưỡng phổ biến: 4/n
    threshold = 4.0 / n
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"Ngưỡng 4/n = {threshold:.3f}")
    
    influential = [i for i, d in enumerate(cooks_d) if d > threshold]
    if influential:
        inf_d = [cooks_d[i] for i in influential]
        ax.scatter(influential, inf_d, color="red", zorder=5,
                   label=f"Influential ({len(influential)} pts)")
    ax.set_title("Cook's Distance", fontsize=12)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's Distance (Dᵢ)")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, "residual_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    plt.close()

    return fig, {
        "residuals": e,
        "std_residuals": e_std,
        "cooks_d": cooks_d,
    }


def _smooth_line(ax, x: list[float], y: list[float], n_bins: int = 20, **kwargs):
    """Vẽ đường LOWESS đơn giản bằng moving average theo bin."""
    order = sorted(range(len(x)), key=lambda k: x[k])
    xs = [x[i] for i in order]
    ys = [y[i] for i in order]
    
    bin_size = max(1, len(xs) // n_bins)
    bx = []
    by = []
    for i in range(0, len(xs), bin_size):
        chunk_x = xs[i:i+bin_size]
        chunk_y = ys[i:i+bin_size]
        if chunk_x:
            bx.append(sum(chunk_x) / len(chunk_x))
            by.append(sum(chunk_y) / len(chunk_y))
            
    ax.plot(bx, by, linewidth=1.5, **kwargs)


# ---------------------------------------------------------------------------
# Unit Tests — F8  (≥ 4 tests)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    from test_utils import TestLogger, assert_true, assert_equal, assert_close

    print("=" * 55)
    print("  UNIT TESTS — residual_analysis.py")
    print("=" * 55)

    passed = 0
    total = 0

    def run(result: bool):
        global passed, total
        total += 1
        passed += int(result)

    TestLogger.print_suite_header("F8 — Residual Analysis")

    # test_residual_plots_returns_correct_keys
    y     = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_hat = [1.1, 1.9, 3.2, 3.8, 5.1]
    fig, res = residual_plots(y, y_hat, save_dir="output/test")
    keys_ok = all(key in res for key in ("residuals", "std_residuals", "cooks_d"))
    run(assert_true(keys_ok, label="residual_plots returns correct dictionary keys"))

    # test_residuals_correct_values
    y2     = [3.0, 5.0, 7.0]
    y_hat2 = [2.5, 5.5, 6.0]
    fig2, res2 = residual_plots(y2, y_hat2, save_dir="output/test")
    expected = [0.5, -0.5, 1.0]
    run(assert_close(res2["residuals"], expected, label="residuals exactly match formula (y - y_hat)"))

    # test_cooks_distance_with_X
    import random
    random.seed(0)
    n, p = 30, 2
    X_np  = [[random.gauss(0, 1) for _ in range(p)] for _ in range(n)]
    X_b   = [[1.0] + row for row in X_np]
    beta  = [1.0, 2.0, -1.0]
    y3    = [sum(X_b[i][j] * beta[j] for j in range(3)) for i in range(n)]
    y_hat3 = [y3[i] + 0.1 * random.gauss(0, 1) for i in range(n)]
    fig3, res3 = residual_plots(y3, y_hat3, X=X_b, save_dir="output/test")
    run(assert_equal(len(res3["cooks_d"]), n, label="Cook's D list length matches number of observations (n)"))
    run(assert_true(all(d >= 0 for d in res3["cooks_d"]), label="all calculated Cook's distances are non-negative"))

    # test_influential_point_detected
    random.seed(1)
    X_np2 = [[random.gauss(0, 1)] for _ in range(n)]
    X_b2  = [[1.0] + row for row in X_np2]
    beta2 = [0.0, 1.0]
    y4    = [sum(X_b2[i][j] * beta2[j] for j in range(2)) for i in range(n)]
    y_hat4 = list(y4)
    y4[-1]     = 100.0
    y_hat4[-1] = 0.0
    fig4, res4 = residual_plots(y4, y_hat4, X=X_b2, save_dir="output/test")
    threshold = 4.0 / n
    run(assert_true(res4["cooks_d"][-1] > threshold, label="Cook's D detects extreme outlier above 4/n threshold"))

    # test_residual_plots_no_X
    y5     = [1.0, 2.0, 3.0, 4.0]
    y_hat5 = [1.0, 2.0, 3.0, 4.5]
    fig5, res5 = residual_plots(y5, y_hat5, X=None, save_dir="output/test")
    run(assert_equal(len(res5["cooks_d"]), 4, label="residual_plots correctly falls back to |e| when X is None"))

    TestLogger.print_summary(passed, total)
