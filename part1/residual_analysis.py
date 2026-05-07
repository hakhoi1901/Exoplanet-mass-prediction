from __future__ import annotations
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


# ---------------------------------------------------------------------------
# F8: Phân Tích Phần Dư — 4 biểu đồ chẩn đoán
# ---------------------------------------------------------------------------

def residual_plots(
    y: list[float],
    y_hat: list[float],
    X: list[list[float]] | None = None,
    save_dir: str = "output",
) -> dict:
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

    y_arr     = np.array(y, dtype=float)
    y_hat_arr = np.array(y_hat, dtype=float)
    e         = y_arr - y_hat_arr          # phần dư thô
    n         = len(y_arr)

    # --- Standardized residuals ---
    sigma_hat = math.sqrt(max(float(np.sum(e ** 2)) / max(n - 2, 1), 1e-12))
    e_std     = e / sigma_hat

    # --- Hat matrix leverage & Cook's Distance ---
    if X is not None:
        X_np = np.array(X, dtype=float)           # đã có bias
        # leverage h_ii = diag(X(XᵀX)⁻¹Xᵀ)
        try:
            XtX_inv = np.linalg.inv(X_np.T @ X_np)
            H       = X_np @ XtX_inv @ X_np.T
            h       = np.diag(H)                  # leverage values
            p1      = X_np.shape[1]               # p + 1
            # Cook's D: D_i = e_i² / (p1 * s²) * h_ii / (1 - h_ii)²
            denom   = p1 * (sigma_hat ** 2) * ((1 - h) ** 2)
            denom   = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            cooks_d = (e ** 2 * h) / denom
        except np.linalg.LinAlgError:
            cooks_d = np.abs(e)   # fallback
    else:
        cooks_d = np.abs(e)       # placeholder khi không có X

    sqrt_abs_std = np.sqrt(np.abs(e_std))
    indices      = np.arange(n)

    # --- Vẽ ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Phân Tích Phần Dư (Residual Diagnostic Plots)", fontsize=14, y=1.01)

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(y_hat_arr, e, alpha=0.55, edgecolors="steelblue", facecolors="none", linewidths=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2, label="e = 0")
    # LOWESS smoother (dùng numpy đơn giản — trung bình cục bộ)
    _smooth_line(ax, y_hat_arr, e, color="orange", label="LOWESS approx.")
    ax.set_title("Residuals vs Fitted", fontsize=12)
    ax.set_xlabel("Fitted values (ŷ)")
    ax.set_ylabel("Residuals (e = y − ŷ)")
    ax.legend()

    # 2. Normal Q-Q
    ax = axes[0, 1]
    (osm, osr), (slope, intercept_q, _) = stats.probplot(e, dist="norm")
    ax.scatter(osm, osr, alpha=0.55, edgecolors="steelblue", facecolors="none", linewidths=0.8,
               label="Quantile")
    qqx = np.array([min(osm), max(osm)])
    ax.plot(qqx, slope * qqx + intercept_q, color="red", linestyle="--", linewidth=1.2,
            label="Normal line")
    ax.set_title("Normal Q-Q", fontsize=12)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles (phần dư chuẩn hoá)")
    ax.legend()

    # 3. Scale-Location (√|e_std| vs Fitted)
    ax = axes[1, 0]
    ax.scatter(y_hat_arr, sqrt_abs_std, alpha=0.55, edgecolors="steelblue", facecolors="none",
               linewidths=0.8)
    _smooth_line(ax, y_hat_arr, sqrt_abs_std, color="orange", label="LOWESS approx.")
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
    # Đánh dấu các điểm vượt ngưỡng
    influential = np.where(cooks_d > threshold)[0]
    if len(influential) > 0:
        ax.scatter(influential, cooks_d[influential], color="red", zorder=5,
                   label=f"Influential ({len(influential)} pts)")
    ax.set_title("Cook's Distance", fontsize=12)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's Distance (Dᵢ)")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, "residual_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[F8] Biểu đồ phần dư đã lưu tại: {out_path}")

    return {
        "residuals":    e.tolist(),
        "std_residuals": e_std.tolist(),
        "cooks_d":      cooks_d.tolist(),
    }


def _smooth_line(ax, x: np.ndarray, y: np.ndarray, n_bins: int = 20, **kwargs):
    """Vẽ đường LOWESS đơn giản bằng moving average theo bin."""
    order  = np.argsort(x)
    xs, ys = x[order], y[order]
    bins   = np.array_split(np.arange(len(xs)), n_bins)
    bx     = [xs[b].mean() for b in bins if len(b)]
    by     = [ys[b].mean() for b in bins if len(b)]
    ax.plot(bx, by, linewidth=1.5, **kwargs)


# ---------------------------------------------------------------------------
# Unit Tests — F8  (≥ 4 tests)
# ---------------------------------------------------------------------------

def test_residual_plots_returns_correct_keys():
    """Hàm phải trả về dict đủ 3 key."""
    y     = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_hat = [1.1, 1.9, 3.2, 3.8, 5.1]
    res = residual_plots(y, y_hat, save_dir="output/test")
    for key in ("residuals", "std_residuals", "cooks_d"):
        assert key in res, f"Thiếu key '{key}'"
    print("test_residual_plots_returns_correct_keys: PASSED")


def test_residuals_correct_values():
    """residuals = y − y_hat."""
    y     = [3.0, 5.0, 7.0]
    y_hat = [2.5, 5.5, 6.0]
    res   = residual_plots(y, y_hat, save_dir="output/test")
    expected = [0.5, -0.5, 1.0]
    for a, b in zip(res["residuals"], expected):
        assert abs(a - b) < 1e-9, f"residual sai: {a} vs {b}"
    print("test_residuals_correct_values: PASSED")


def test_cooks_distance_with_X():
    """Cook's Distance phải là list độ dài n khi X được cung cấp."""
    np.random.seed(0)
    n, p = 30, 2
    X_np  = np.random.randn(n, p)
    X_b   = np.column_stack([np.ones(n), X_np])   # design matrix với bias
    beta  = np.array([1.0, 2.0, -1.0])
    y     = (X_b @ beta).tolist()
    y_hat = (X_b @ beta + 0.1 * np.random.randn(n)).tolist()
    res   = residual_plots(y, y_hat, X=X_b.tolist(), save_dir="output/test")
    assert len(res["cooks_d"]) == n
    assert all(d >= 0 for d in res["cooks_d"]), "Cook's D phải >= 0"
    print("test_cooks_distance_with_X: PASSED")


def test_influential_point_detected():
    """Điểm outlier rõ ràng phải có Cook's D lớn hơn 4/n."""
    n = 30
    np.random.seed(1)
    X_np = np.random.randn(n, 1)
    X_b  = np.column_stack([np.ones(n), X_np])
    y    = (X_b @ np.array([0.0, 1.0])).tolist()
    y_hat = y[:]
    # Tạo một điểm outlier cực đoan ở cuối
    y[-1]     = 100.0
    y_hat[-1] = 0.0
    res = residual_plots(y, y_hat, X=X_b.tolist(), save_dir="output/test")
    threshold = 4.0 / n
    assert res["cooks_d"][-1] > threshold, "Điểm outlier phải vượt ngưỡng Cook's D"
    print("test_influential_point_detected: PASSED")


def test_residual_plots_no_X():
    """Chạy được khi X=None (Cook's D fallback = |e|)."""
    y     = [1.0, 2.0, 3.0, 4.0]
    y_hat = [1.0, 2.0, 3.0, 4.5]
    res   = residual_plots(y, y_hat, X=None, save_dir="output/test")
    assert len(res["cooks_d"]) == 4
    print("test_residual_plots_no_X: PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  UNIT TESTS — residual_analysis.py")
    print("=" * 55)
    test_residual_plots_returns_correct_keys()
    test_residuals_correct_values()
    test_cooks_distance_with_X()
    test_influential_point_detected()
    test_residual_plots_no_X()
    print("\nAll tests PASSED.")

    # Demo
    print("\n--- Demo với synthetic data ---")
    np.random.seed(42)
    n, p = 80, 3
    X_np   = np.random.randn(n, p)
    X_bias = np.column_stack([np.ones(n), X_np])
    beta   = np.array([2.0, 1.5, -1.0, 0.5])
    y_demo = (X_bias @ beta + np.random.randn(n) * 0.8).tolist()
    y_hat_demo = (X_bias @ beta).tolist()
    residual_plots(y_demo, y_hat_demo, X=X_bias.tolist())