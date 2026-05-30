from __future__ import annotations

import math
import os
import sys

import matplotlib.pyplot as plt
PART1_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from config import RANDOM_STATE
from utils import matvec


# ---------------------------------------------------------------------------
# F8: Residual Analysis - 4 biểu đồ chẩn đoán phần dư
# ---------------------------------------------------------------------------

def residual_plots(
    X_or_y,
    y_or_y_hat,
    beta_hat=None,
    *,
    X: list[list[float]] | None = None,
    save_dir: str = PART1_OUTPUT_DIR,
    show_plot: bool = False,
) -> dict:
    """
    F8: Vẽ bốn biểu đồ chẩn đoán phần dư cho mô hình hồi quy.
        residual_plots(X, y, beta_hat)
        residual_plots(y, y_hat, X=X)
        residual_plots(y, y_hat, X)

    Tham số:
        X_or_y     : Ma trận feature X hoặc vector y, tùy cách gọi.
        y_or_y_hat : Vector target y hoặc vector dự đoán y_hat.
        beta_hat   : Hệ số hồi quy [intercept, beta_1, ..., beta_p] nếu truyền X và y.
        X          : Ma trận feature tùy chọn khi đã có sẵn y_hat.
        save_dir   : Thư mục lưu ảnh residual_plots.png.
        show_plot  : Có hiển thị biểu đồ sau khi lưu hay không.

    Trả về dict:
        fig           : Figure Matplotlib.
        out_path      : Đường dẫn ảnh đã lưu.
        residuals     : Phần dư y - y_hat.
        std_residuals : Phần dư chuẩn hóa.
        cooks_d       : Cook's Distance hoặc fallback theo độ lớn phần dư.
        y_hat         : Giá trị dự đoán dùng trong biểu đồ.
    """
    X_features, y, y_hat = _resolve_inputs(X_or_y, y_or_y_hat, beta_hat, X)
    os.makedirs(save_dir, exist_ok=True)
    n = len(y)

    # Tính phần dư và phần dư chuẩn hóa.
    residuals = [y[i] - y_hat[i] for i in range(n)]
    rss = sum(r * r for r in residuals)
    p_feat = len(X_features[0]) if X_features is not None else 1
    df = max(n - p_feat - 1, 1)
    sigma_hat = math.sqrt(max(rss / df, 1e-12))
    std_residuals = [r / sigma_hat for r in residuals]

    if X_features is not None:
        # Nếu có X, dùng leverage từ Hat Matrix để tính Cook's Distance đúng công thức.
        from part1.ols_implementation import hat_matrix as compute_hat_matrix

        hat_res = compute_hat_matrix(X_features)
        H = hat_res["H"]
        h = [H[i][i] for i in range(n)]
        p1 = p_feat + 1
        cooks_d = []
        for i in range(n):
            denom = p1 * (sigma_hat ** 2) * ((1.0 - h[i]) ** 2)
            if abs(denom) < 1e-12:
                denom = 1e-12
            cooks_d.append((residuals[i] ** 2 * h[i]) / denom)
    else:
        # Fallback cho cách gọi chỉ truyền y và y_hat: dùng độ lớn phần dư để vẫn vẽ được panel thứ tư.
        cooks_d = [abs(r) for r in residuals]

    sqrt_abs_std = [math.sqrt(abs(s)) for s in std_residuals]
    qq_x, qq_y, qq_slope, qq_intercept = _qq_line(residuals)

    # Bốn panel chẩn đoán: Residuals vs Fitted, Q-Q, Scale-Location, Cook's Distance.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Residual Diagnostic Plots", fontsize=14, y=1.01)

    ax = axes[0, 0]
    ax.scatter(y_hat, residuals, s=15, alpha=0.55, edgecolors="steelblue", facecolors="none", linewidths=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2, label="e = 0")
    _smooth_line(ax, y_hat, residuals, color="orange", label="Binned mean")
    ax.set_title("Residuals vs Fitted", fontsize=12)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(qq_x, qq_y, s=15, alpha=0.55, edgecolors="steelblue", facecolors="none", linewidths=0.8)
    if qq_x:
        line_x = [min(qq_x), max(qq_x)]
        line_y = [qq_slope * x + qq_intercept for x in line_x]
        ax.plot(line_x, line_y, color="red", linestyle="--", linewidth=1.2, label="Normal line")
    ax.set_title("Normal Q-Q", fontsize=12)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(y_hat, sqrt_abs_std, s=15, alpha=0.55, edgecolors="steelblue", facecolors="none", linewidths=0.8)
    _smooth_line(ax, y_hat, sqrt_abs_std, color="orange", label="Binned mean")
    ax.set_title("Scale-Location", fontsize=12)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("sqrt(|Standardized Residuals|)")
    ax.legend()

    ax = axes[1, 1]
    indices = list(range(n))
    markerline, stemlines, baseline = ax.stem(indices, cooks_d, linefmt="steelblue", markerfmt=" ", basefmt="black")
    stemlines.set_linewidths(0.8)
    threshold = 4.0 / n
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"4/n = {threshold:.3f}")
    influential = [i for i, d in enumerate(cooks_d) if d > threshold]
    if influential:
        ax.scatter(influential, [cooks_d[i] for i in influential], s=15, color="red", zorder=5,
                   label=f"Influential ({len(influential)} pts)")
    ax.set_title("Cook's Distance", fontsize=12)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's Distance")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, "residual_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    if show_plot:
        print(f"[F8] Saved residual plots to: {out_path}")
        plt.show()
    plt.close(fig)

    return {
        "fig": fig,
        "out_path": out_path,
        "residuals": residuals,
        "std_residuals": std_residuals,
        "cooks_d": cooks_d,
        "y_hat": y_hat,
    }


def _resolve_inputs(X_or_y, y_or_y_hat, beta_hat, legacy_X):
    """Chuẩn hóa các cách gọi khác nhau về bộ ba X_features, y và y_hat."""
    if beta_hat is not None and _is_matrix(beta_hat) and legacy_X is None and not _is_matrix(X_or_y):
        legacy_X = beta_hat
        beta_hat = None

    if beta_hat is not None:
        X_features = _to_matrix(X_or_y, name="X")
        y = _to_vector(y_or_y_hat, name="y")
        if len(X_features) != len(y):
            raise ValueError(f"X and y length mismatch: len(X)={len(X_features)}, len(y)={len(y)}")
        if len(beta_hat) != len(X_features[0]) + 1:
            raise ValueError("beta_hat must have length p + 1")
        X_bias = [[1.0] + row for row in X_features]
        y_hat = matvec(X_bias, [float(b) for b in beta_hat])
        return X_features, y, y_hat

    y = _to_vector(X_or_y, name="y")
    y_hat = _to_vector(y_or_y_hat, name="y_hat")
    if len(y) != len(y_hat):
        raise ValueError(f"y and y_hat length mismatch: len(y)={len(y)}, len(y_hat)={len(y_hat)}")
    X_features = _to_matrix(legacy_X, name="X") if legacy_X is not None else None
    if X_features is not None and len(X_features) != len(y):
        raise ValueError(f"X and y length mismatch: len(X)={len(X_features)}, len(y)={len(y)}")
    return X_features, y, y_hat


def _is_matrix(value) -> bool:
    """Kiểm tra nhanh một object có dạng ma trận 2 chiều hay không."""
    try:
        if value is None or len(value) == 0:
            return False
        first = value[0]
        return hasattr(first, "__iter__") and not isinstance(first, (str, bytes))
    except (TypeError, KeyError):
        return False


def _to_vector(values, *, name: str) -> list[float]:
    """Ép một sequence 1 chiều sang list float và kiểm tra rỗng."""
    if values is None:
        raise ValueError(f"{name} must not be None")
    try:
        out = [float(v) for v in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric sequence") from exc
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def _to_matrix(values, *, name: str) -> list[list[float]]:
    """Ép một sequence 2 chiều sang ma trận float và kiểm tra số cột nhất quán."""
    if not _is_matrix(values):
        raise ValueError(f"{name} must be a non-empty matrix")
    matrix = [[float(v) for v in row] for row in values]
    p = len(matrix[0])
    if p == 0:
        raise ValueError(f"{name} must contain at least one feature")
    for i, row in enumerate(matrix):
        if len(row) != p:
            raise ValueError(f"{name} row {i} has inconsistent length")
    return matrix


def _qq_line(residuals: list[float]) -> tuple[list[float], list[float], float, float]:
    """Tạo dữ liệu cho biểu đồ Normal Q-Q và đường tham chiếu tuyến tính."""
    n = len(residuals)
    y_sorted = sorted(residuals)
    x_theoretical = [_normal_ppf((i + 0.5) / n) for i in range(n)]
    x_mean = sum(x_theoretical) / n
    y_mean = sum(y_sorted) / n
    ss_x = sum((x - x_mean) ** 2 for x in x_theoretical)
    if ss_x <= 1e-12:
        return x_theoretical, y_sorted, 1.0, 0.0
    slope = sum((x_theoretical[i] - x_mean) * (y_sorted[i] - y_mean) for i in range(n)) / ss_x
    intercept = y_mean - slope * x_mean
    return x_theoretical, y_sorted, slope, intercept


def _normal_ppf(p: float) -> float:
    """Xấp xỉ inverse CDF của phân phối chuẩn chuẩn hóa."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")

    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
        )

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r) + 1.0
    )


def _smooth_line(ax, x, y, n_bins: int = 20, **kwargs):
    """Vẽ đường xu hướng bằng trung bình theo các bin của trục x."""
    pairs = sorted((float(xi), float(yi)) for xi, yi in zip(x, y))
    if not pairs:
        return
    n = len(pairs)
    n_bins = max(1, min(n_bins, n))
    bx, by = [], []
    for b in range(n_bins):
        start = b * n // n_bins
        end = (b + 1) * n // n_bins
        chunk = pairs[start:end]
        if not chunk:
            continue
        bx.append(sum(v[0] for v in chunk) / len(chunk))
        by.append(sum(v[1] for v in chunk) / len(chunk))
    ax.plot(bx, by, linewidth=1.5, **kwargs)


# ---------------------------------------------------------------------------
# Unit Tests - F8
# ---------------------------------------------------------------------------


def test_residual_plots_returns_required_keys() -> bool:
    from test_utils import assert_true

    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_hat = [1.1, 1.9, 3.2, 3.8, 5.1]
    res = residual_plots(y, y_hat, save_dir=os.path.join(PART1_OUTPUT_DIR, "test"))
    keys_ok = all(key in res for key in ("residuals", "std_residuals", "cooks_d", "y_hat"))
    return assert_true(keys_ok, label="residual_plots returns required keys")


def test_residual_plots_residual_formula() -> bool:
    from test_utils import assert_close

    y = [3.0, 5.0, 7.0]
    y_hat = [2.5, 5.5, 6.0]
    res = residual_plots(y, y_hat, save_dir=os.path.join(PART1_OUTPUT_DIR, "test"))
    return assert_close(res["residuals"], [0.5, -0.5, 1.0], label="residuals equal y - y_hat")


def test_residual_plots_cooks_distance_shape() -> bool:
    import random
    from test_utils import assert_equal, assert_true

    random.seed(RANDOM_STATE)
    n, p = 30, 2
    X = [[random.gauss(0, 1) for _ in range(p)] for _ in range(n)]
    beta = [1.0, 2.0, -1.0]
    X_b = [[1.0] + row for row in X]
    y = [sum(X_b[i][j] * beta[j] for j in range(3)) for i in range(n)]
    y_hat = [y[i] + 0.1 * random.gauss(0, 1) for i in range(n)]
    res = residual_plots(y, y_hat, X=X, save_dir=os.path.join(PART1_OUTPUT_DIR, "test"))
    checks = [
        assert_equal(len(res["cooks_d"]), n, label="Cook's D length matches n"),
        assert_true(all(d >= 0 for d in res["cooks_d"]), label="Cook's D values are non-negative"),
    ]
    return all(checks)


def test_residual_plots_detects_extreme_influence() -> bool:
    import random
    from test_utils import assert_true

    random.seed(RANDOM_STATE)
    n = 30
    X = [[random.gauss(0, 1)] for _ in range(n)]
    beta = [0.0, 1.0]
    X_b = [[1.0] + row for row in X]
    y = [sum(X_b[i][j] * beta[j] for j in range(2)) for i in range(n)]
    y[-1] = 100.0
    res = residual_plots(X, y, beta, save_dir=os.path.join(PART1_OUTPUT_DIR, "test"))
    return assert_true(res["cooks_d"][-1] > 4.0 / n, label="Cook's D flags extreme outlier")


def test_residual_plots_fallback_without_x() -> bool:
    from test_utils import assert_equal

    y = [1.0, 2.0, 3.0, 4.0]
    y_hat = [1.0, 2.0, 3.0, 4.5]
    res = residual_plots(y, y_hat, X=None, save_dir=os.path.join(PART1_OUTPUT_DIR, "test"))
    return assert_equal(len(res["cooks_d"]), 4, label="fallback Cook's D list length matches n")


def _generate_demo_plot() -> None:
    """Sinh ảnh residual_plots.png demo để dùng trong báo cáo."""
    from test_utils import TestLogger, make_linear_data

    try:
        from part1.ols_implementation import ols_fit
    except ImportError:
        TestLogger.print_warn("Could not import ols_fit to generate demo residual plot")
        return

    TestLogger.print_info("Generate demo residual_plots.png for report")
    X_demo, y_demo = make_linear_data(n=200, beta=[2.0, 3.0, -1.5], sigma=1.0)
    fit = ols_fit(X_demo, y_demo)
    residual_plots(X_demo, y_demo, fit["beta_hat"], save_dir=PART1_OUTPUT_DIR)


def run_tests() -> tuple[int, int]:
    """Chạy toàn bộ unit tests của F8 và trả về số test pass/tổng số test."""
    from test_utils import TestLogger

    TestLogger.print_suite_header("F8 - Residual Analysis")
    tests = [
        test_residual_plots_returns_required_keys,
        test_residual_plots_residual_formula,
        test_residual_plots_cooks_distance_shape,
        test_residual_plots_detects_extreme_influence,
        test_residual_plots_fallback_without_x,
    ]
    passed = sum(int(test()) for test in tests)
    _generate_demo_plot()
    return passed, len(tests)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from test_utils import TestLogger

    print("=" * 55)
    print("  UNIT TESTS - residual_analysis.py")
    print("=" * 55)
    passed, total = run_tests()
    TestLogger.print_summary(passed, total)
