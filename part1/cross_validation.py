from __future__ import annotations
import math
import os
import sys
from typing import Callable

# Import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RANDOM_STATE, EPSILON


# ---------------------------------------------------------------------------
# F9: k-Fold Cross-Validation
# Liên kết: Nhận model_fn (F1 ols_fit / F6 ridge_fit / F7 lasso_fit)
#           và predict_fn để fit + đánh giá trên mỗi fold
# ---------------------------------------------------------------------------

def kfold_cv(
    X: list[list[float]],
    y: list[float],
    k: int,
    model_fn: Callable,
    predict_fn: Callable,
    **model_kwargs,
) -> dict:
    """
    F9: k-Fold Cross-Validation từ đầu (manual, không dùng numpy cho logic).

    Liên kết:
        - model_fn: Nhận hàm fit từ F1/F6/F7 (ols_fit, ridge_fit, lasso_fit).
        - predict_fn: Nhận hàm predict tương ứng.

    Tham số:
        X          : Ma trận features (n x p), CHƯA có cột bias.
        y          : Vector target (n,).
        k          : Số fold (khuyến nghị 5 hoặc 10).
        model_fn   : Hàm fit, signature: model_fn(X_train, y_train, **kwargs) → model_dict.
                     model_dict PHẢI chứa 'beta_hat', 'mean_X', 'std_X' (hoặc 'y_hat').
        predict_fn : Hàm predict, signature: predict_fn(X_val, model_dict) → list[float].
        **model_kwargs: Tham số bổ sung truyền vào model_fn (vd: lam=0.5).

    Trả về dict:
        mean_cv_mse  : float      — trung bình MSE qua k fold.
        std_cv_mse   : float      — độ lệch chuẩn MSE.
        cv_mse_list  : list[float]— MSE từng fold.
        mean_cv_r2   : float      — trung bình R² qua k fold.
        cv_r2_list   : list[float]— R² từng fold.
    """
    n = len(y)

    # Shuffle indices với seed cố định (RANDOM_STATE = 42)
    # Dùng Fisher-Yates shuffle manual thay vì numpy
    indices = list(range(n))
    _manual_shuffle(indices, seed=RANDOM_STATE)

    # Chia thành k fold
    folds = _split_into_k(indices, k)

    cv_mse = []
    cv_r2  = []

    for i in range(k):
        # Tách validation và training indices
        val_idx = folds[i]
        train_idx = []
        for j in range(k):
            if j != i:
                train_idx.extend(folds[j])

        # Tạo X_train, y_train, X_val, y_val
        X_train = [X[idx] for idx in train_idx]
        y_train = [y[idx] for idx in train_idx]
        X_val   = [X[idx] for idx in val_idx]
        y_val   = [y[idx] for idx in val_idx]

        # Fit model (gọi F1/F6/F7)
        model = model_fn(X_train, y_train, **model_kwargs)

        # Predict
        y_pred = predict_fn(X_val, model)

        # MSE = mean((y_val - y_pred)²)
        n_val = len(y_val)
        mse = sum((y_val[q] - y_pred[q]) ** 2 for q in range(n_val)) / n_val
        cv_mse.append(mse)

        # R² = 1 - SS_res / SS_tot
        y_val_mean = sum(y_val) / n_val
        ss_res = sum((y_val[q] - y_pred[q]) ** 2 for q in range(n_val))
        ss_tot = sum((y_val[q] - y_val_mean) ** 2 for q in range(n_val))
        r2 = 1.0 - ss_res / ss_tot if abs(ss_tot) > EPSILON else 0.0
        cv_r2.append(r2)

    # Tính mean và std (manual)
    mean_mse = sum(cv_mse) / k
    mean_r2  = sum(cv_r2) / k
    std_mse  = math.sqrt(sum((m - mean_mse) ** 2 for m in cv_mse) / k)

    return {
        "mean_cv_mse": mean_mse,
        "std_cv_mse":  std_mse,
        "cv_mse_list": cv_mse,
        "mean_cv_r2":  mean_r2,
        "cv_r2_list":  cv_r2,
    }


def _manual_shuffle(lst: list, seed: int) -> None:
    """Fisher-Yates shuffle in-place với LCG pseudo-random generator."""
    # Linear Congruential Generator (đơn giản, reproducible)
    state = seed
    n = len(lst)
    for i in range(n - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        lst[i], lst[j] = lst[j], lst[i]


def _split_into_k(indices: list[int], k: int) -> list[list[int]]:
    """Chia list indices thành k phần (gần bằng nhau)."""
    n = len(indices)
    fold_size = n // k
    remainder = n % k
    folds = []
    start = 0
    for i in range(k):
        # Các fold đầu nhận thêm 1 phần tử nếu có dư
        end = start + fold_size + (1 if i < remainder else 0)
        folds.append(indices[start:end])
        start = end
    return folds


def cv_lambda_search(
    X: list[list[float]],
    y: list[float],
    model_fn: Callable,
    predict_fn: Callable,
    lambdas: list[float] | None = None,
    k: int = 5,
    save_dir: str = "output",
) -> dict:
    """
    Tìm λ tối ưu qua k-fold CV và vẽ biểu đồ λ vs CV-MSE (log scale).

    Trả về:
        best_lam      : float — λ cho MSE thấp nhất.
        lambdas       : list[float]
        mean_cv_mse   : list[float] — MSE trung bình mỗi λ.
        std_cv_mse    : list[float]
    """
    if lambdas is None:
        lambdas = [10 ** e for e in [x / 4 for x in range(-12, 17)]]  # 1e-3 … 1e4

    mean_mse_list = []
    std_mse_list  = []

    for lam in lambdas:
        res = kfold_cv(X, y, k=k, model_fn=model_fn, predict_fn=predict_fn, lam=lam)
        mean_mse_list.append(res["mean_cv_mse"])
        std_mse_list.append(res["std_cv_mse"])

    # Tìm best λ (manual argmin)
    best_idx = 0
    for i in range(1, len(mean_mse_list)):
        if mean_mse_list[i] < mean_mse_list[best_idx]:
            best_idx = i
    best_lam = lambdas[best_idx]

    # --- Vẽ λ vs CV-MSE ---
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        lambdas, mean_mse_list, yerr=std_mse_list,
        fmt="o-", color="steelblue", ecolor="lightblue",
        elinewidth=1.5, capsize=3, markersize=4, linewidth=1.2,
        label="CV MSE (mean ± std)"
    )
    ax.axvline(best_lam, color="red", linestyle="--", linewidth=1.5,
               label=f"λ* = {best_lam:.4g}  (MSE={mean_mse_list[best_idx]:.4f})")
    ax.set_xscale("log")
    ax.set_title(f"λ vs CV MSE ({k}-Fold Cross-Validation)", fontsize=13)
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("Mean CV MSE")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(save_dir, "lambda_cv_score.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[F9] λ vs CV score đã lưu tại: {out_path}")
    print(f"[F9] λ tối ưu = {best_lam:.4g}  (CV MSE = {mean_mse_list[best_idx]:.4f})")

    return {
        "best_lam":    best_lam,
        "lambdas":     lambdas,
        "mean_cv_mse": mean_mse_list,
        "std_cv_mse":  std_mse_list,
    }


# ---------------------------------------------------------------------------
# Unit Tests — F9  (≥ 4 tests)
# ---------------------------------------------------------------------------

def _make_linear_data(n=60, seed=42):
    """Tạo data test (dùng numpy CHỈ trong tests)."""
    import numpy as np
    rng  = np.random.default_rng(seed)
    X    = rng.standard_normal((n, 3))
    beta = np.array([1.5, -1.0, 0.5])
    y    = X @ beta + 0.3 * rng.standard_normal(n)
    return X.tolist(), y.tolist()


# Adapter cho ridge (dùng trong tests)
def _ridge_predict(X_val: list[list[float]], model: dict) -> list[float]:
    from part1.ridge_lasso import ridge_predict
    return ridge_predict(X_val, model["beta_hat"], model["mean_X"], model["std_X"])


def test_kfold_cv_returns_correct_keys():
    """Kết quả phải có đủ 5 key."""
    from part1.ridge_lasso import ridge_fit
    X, y = _make_linear_data()
    res  = kfold_cv(X, y, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=0.1)
    for key in ("mean_cv_mse", "std_cv_mse", "cv_mse_list", "mean_cv_r2", "cv_r2_list"):
        assert key in res, f"Thiếu key '{key}'"
    print("test_kfold_cv_returns_correct_keys: PASSED")


def test_kfold_cv_number_of_folds():
    """cv_mse_list phải có đúng k phần tử."""
    from part1.ridge_lasso import ridge_fit
    X, y = _make_linear_data()
    for k in (3, 5, 10):
        res = kfold_cv(X, y, k=k, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=0.1)
        assert len(res["cv_mse_list"]) == k, f"k={k}: có {len(res['cv_mse_list'])} fold"
    print("test_kfold_cv_number_of_folds: PASSED")


def test_kfold_cv_mse_positive():
    """MSE từng fold phải >= 0."""
    from part1.ridge_lasso import ridge_fit
    X, y = _make_linear_data()
    res  = kfold_cv(X, y, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=1.0)
    for i, mse in enumerate(res["cv_mse_list"]):
        assert mse >= 0, f"Fold {i}: MSE âm ({mse})"
    print("test_kfold_cv_mse_positive: PASSED")


def test_kfold_cv_mean_matches_list():
    """mean_cv_mse phải bằng trung bình cv_mse_list."""
    from part1.ridge_lasso import ridge_fit
    X, y = _make_linear_data()
    res  = kfold_cv(X, y, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=0.5)
    expected_mean = sum(res["cv_mse_list"]) / len(res["cv_mse_list"])
    assert abs(res["mean_cv_mse"] - expected_mean) < 1e-9, "mean_cv_mse không khớp"
    print("test_kfold_cv_mean_matches_list: PASSED")


def test_kfold_cv_r2_range():
    """R² của mô hình tốt trên dữ liệu tuyến tính phải > 0.5."""
    from part1.ridge_lasso import ridge_fit
    X, y = _make_linear_data(n=100, seed=7)
    res  = kfold_cv(X, y, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=1e-4)
    assert res["mean_cv_r2"] > 0.5, f"R² quá thấp: {res['mean_cv_r2']:.3f}"
    print(f"test_kfold_cv_r2_range: PASSED  (mean R² = {res['mean_cv_r2']:.3f})")


def test_kfold_cv_reproducible():
    """Kết quả phải giống nhau khi gọi hai lần (seed cố định)."""
    from part1.ridge_lasso import ridge_fit
    X, y = _make_linear_data()
    res1 = kfold_cv(X, y, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=1.0)
    res2 = kfold_cv(X, y, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=1.0)
    assert res1["mean_cv_mse"] == res2["mean_cv_mse"], "Kết quả không reproducible"
    print("test_kfold_cv_reproducible: PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  UNIT TESTS — cross_validation.py")
    print("=" * 55)
    test_kfold_cv_returns_correct_keys()
    test_kfold_cv_number_of_folds()
    test_kfold_cv_mse_positive()
    test_kfold_cv_mean_matches_list()
    test_kfold_cv_r2_range()
    test_kfold_cv_reproducible()
    print("\nAll tests PASSED.")

    # Demo: tìm λ tối ưu
    print("\n--- Demo: λ Search với Ridge ---")
    from part1.ridge_lasso import ridge_fit, ridge_predict
    X_d, y_d = _make_linear_data(n=120, seed=0)
    cv_lambda_search(
        X_d, y_d,
        model_fn=ridge_fit,
        predict_fn=lambda X_v, m: ridge_predict(X_v, m["beta_hat"], m["mean_X"], m["std_X"]),
        lambdas=[10 ** (e / 4) for e in range(-8, 17)],
        k=5,
    )