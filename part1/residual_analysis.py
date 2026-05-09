from __future__ import annotations
import math
import os
import sys

import matplotlib.pyplot as plt

# Import utils và F2 hat_matrix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))



# ---------------------------------------------------------------------------
# F8: Phân Tích Phần Dư — 4 biểu đồ chẩn đoán
# Liên kết: Dùng F2 hat_matrix để tính leverage và Cook's Distance
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

    Liên kết:
        - Gọi hat_matrix (F2) để tính leverage h_ii cho Cook's Distance.

    Tham số:
        y       : Giá trị thực (n,).
        y_hat   : Giá trị dự đoán (n,).
        X       : Ma trận features (n x p), CHƯA có bias.
                  Bắt buộc để tính Cook's Distance chính xác qua F2.
                  Nếu None, dùng |residuals| thay thế.
        save_dir: Thư mục lưu ảnh.

    Trả về dict:
        residuals    : list[float]
        std_residuals: list[float]  — standardized residuals
        cooks_d      : list[float]  — Cook's Distance (hoặc |e| nếu X=None)
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(y)

    # --- Residuals ---
    residuals = [y[i] - y_hat[i] for i in range(n)]

    # --- Standardized residuals ---
    rss = sum(r * r for r in residuals)
    sigma_hat = math.sqrt(max(rss / max(n - 2, 1), 1e-12))
    std_residuals = [r / sigma_hat for r in residuals]

    # --- Cook's Distance (liên kết F2) ---
    if X is not None:
        # Dùng F2 hat_matrix để tính leverage
        from part1.ols_implementation import hat_matrix as compute_hat_matrix
        hat_res = compute_hat_matrix(X)
        H = hat_res["H"]

        # h_ii = diagonal(H) = leverage values
        h = [H[i][i] for i in range(n)]
        p1 = len(X[0]) + 1   # p + 1 (bao gồm intercept)

        # Cook's D_i = (e_i² / (p1 * σ̂²)) * (h_ii / (1 - h_ii)²)
        cooks_d = []
        for i in range(n):
            denom = p1 * (sigma_hat ** 2) * ((1.0 - h[i]) ** 2)
            if abs(denom) < 1e-12:
                denom = 1e-12
            d_i = (residuals[i] ** 2 * h[i]) / denom
            cooks_d.append(d_i)
    else:
        # Fallback khi không có X
        cooks_d = [abs(r) for r in residuals]

    # --- sqrt(|e_std|) ---
    sqrt_abs_std = [math.sqrt(abs(s)) for s in std_residuals]

    # --- Vẽ (dùng matplotlib cho visualization — cho phép) ---
    import scipy.stats as stats
    import numpy as np

    y_hat_np = np.array(y_hat)
    e_np = np.array(residuals)
    e_std_np = np.array(std_residuals)
    sqrt_abs_np = np.array(sqrt_abs_std)
    cooks_np = np.array(cooks_d)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Phân Tích Phần Dư (Residual Diagnostic Plots)", fontsize=14, y=1.01)

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(y_hat_np, e_np, alpha=0.55, edgecolors="steelblue", facecolors="none", linewidths=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2, label="e = 0")
    _smooth_line(ax, y_hat_np, e_np, color="orange", label="LOWESS approx.")
    ax.set_title("Residuals vs Fitted", fontsize=12)
    ax.set_xlabel("Fitted values (ŷ)")
    ax.set_ylabel("Residuals (e = y − ŷ)")
    ax.legend()

    # 2. Normal Q-Q
    ax = axes[0, 1]
    (osm, osr), (slope, intercept_q, _) = stats.probplot(e_np, dist="norm")
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
    ax.scatter(y_hat_np, sqrt_abs_np, alpha=0.55, edgecolors="steelblue", facecolors="none",
               linewidths=0.8)
    _smooth_line(ax, y_hat_np, sqrt_abs_np, color="orange", label="LOWESS approx.")
    ax.set_title("Scale-Location", fontsize=12)
    ax.set_xlabel("Fitted values (ŷ)")
    ax.set_ylabel("√|Standardized Residuals|")
    ax.legend()

    # 4. Cook's Distance
    ax = axes[1, 1]
    indices = np.arange(n)
    markerline, stemlines, baseline = ax.stem(
        indices, cooks_np, linefmt="steelblue", markerfmt=" ", basefmt="black"
    )
    stemlines.set_linewidths(0.8)
    threshold = 4.0 / n
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"Ngưỡng 4/n = {threshold:.3f}")
    influential = np.where(cooks_np > threshold)[0]
    if len(influential) > 0:
        ax.scatter(influential, cooks_np[influential], color="red", zorder=5,
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
        "residuals":    residuals,
        "std_residuals": std_residuals,
        "cooks_d":      cooks_d,
    }


def _smooth_line(ax, x, y, n_bins: int = 20, **kwargs):
    """Vẽ đường LOWESS đơn giản bằng moving average theo bin."""
    import numpy as np
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
    import numpy as np
    np.random.seed(0)
    n, p = 30, 2
    X_np  = np.random.randn(n, p)
    beta  = np.array([1.0, 2.0, -1.0])
    X_b   = np.column_stack([np.ones(n), X_np])
    y     = (X_b @ beta).tolist()
    y_hat = (X_b @ beta + 0.1 * np.random.randn(n)).tolist()
    # Truyền X chưa có bias (F8 sẽ gọi F2 hat_matrix, tự thêm bias)
    res   = residual_plots(y, y_hat, X=X_np.tolist(), save_dir="output/test")
    assert len(res["cooks_d"]) == n
    assert all(d >= 0 for d in res["cooks_d"]), "Cook's D phải >= 0"
    print("test_cooks_distance_with_X: PASSED")


def test_influential_point_detected():
    """Điểm outlier rõ ràng phải có Cook's D lớn hơn 4/n."""
    import numpy as np
    n = 30
    np.random.seed(1)
    X_np = np.random.randn(n, 1)
    beta = np.array([0.0, 1.0])
    X_b  = np.column_stack([np.ones(n), X_np])
    y    = (X_b @ beta).tolist()
    y_hat = y[:]
    # Tạo một điểm outlier cực đoan ở cuối
    y[-1]     = 100.0
    y_hat[-1] = 0.0
    res = residual_plots(y, y_hat, X=X_np.tolist(), save_dir="output/test")
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
    import numpy as np
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
    beta   = np.array([2.0, 1.5, -1.0, 0.5])
    X_bias_np = np.column_stack([np.ones(n), X_np])
    y_demo = (X_bias_np @ beta + np.random.randn(n) * 0.8).tolist()
    y_hat_demo = (X_bias_np @ beta).tolist()
    # Truyền X chưa có bias cho F8 (F8 sẽ gọi F2, F2 tự thêm bias)
    residual_plots(y_demo, y_hat_demo, X=X_np.tolist())