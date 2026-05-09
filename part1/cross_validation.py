from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import random
from typing import Callable, Any

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# F9: k-Fold Cross-Validation
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
    F9: k-Fold Cross-Validation từ đầu.

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
    indices = list(range(n))
    rng = random.Random(42)
    rng.shuffle(indices)

    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    folds = []
    current = 0
    for size in fold_sizes:
        folds.append(indices[current:current + size])
        current += size

    cv_scores = []
    cv_r2 = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = []
        for j in range(k):
            if j != i:
                train_idx.extend(folds[j])

        X_train = [X[idx] for idx in train_idx]
        y_train = [y[idx] for idx in train_idx]
        X_val = [X[idx] for idx in val_idx]
        y_val = [y[idx] for idx in val_idx]

        model = model_fn(X_train, y_train, **model_kwargs)
        y_pred = predict_fn(X_val, model)

        mse = sum((y_val[idx] - y_pred[idx])**2 for idx in range(len(y_val))) / len(y_val)
        cv_scores.append(mse)

        y_bar = sum(y_val) / len(y_val)
        ss_res = sum((y_val[idx] - y_pred[idx])**2 for idx in range(len(y_val)))
        ss_tot = sum((y_val[idx] - y_bar)**2 for idx in range(len(y_val)))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        cv_r2.append(r2)

    mean_cv_score = sum(cv_scores) / k
    std_cv_score = math.sqrt(sum((x - mean_cv_score)**2 for x in cv_scores) / k)

    mean_cv_r2 = sum(cv_r2) / k

    return {
        "mean_cv_score": mean_cv_score,
        "std_cv_score":  std_cv_score,
        "cv_scores":     cv_scores,
        "mean_cv_r2":    mean_cv_r2,
        "cv_r2_list":    cv_r2,
    }


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
        lambdas = [10 ** e for e in [x / 4 for x in range(-12, 17)]]

    mean_mse_list = []
    std_mse_list = []

    for lam in lambdas:
        res = kfold_cv(X, y, k=k, model_fn=model_fn, predict_fn=predict_fn, lam=lam)
        mean_mse_list.append(res["mean_cv_score"])
        std_mse_list.append(res["std_cv_score"])

    best_idx = 0
    for i in range(1, len(mean_mse_list)):
        if mean_mse_list[i] < mean_mse_list[best_idx]:
            best_idx = i
            
    best_lam = lambdas[best_idx]

    # --- Vẽ λ vs CV-MSE ---
    os.makedirs(save_dir, exist_ok=True)
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
    plt.show(block=False)
    plt.close()
    
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
if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    from test_utils import TestLogger, assert_true, assert_equal, assert_close
    from part1.ridge_lasso import ridge_fit, ridge_predict
    
    print("=" * 55)
    print("  UNIT TESTS — cross_validation.py")
    print("=" * 55)

    passed = 0
    total = 0

    def run(result: bool):
        global passed, total
        total += 1
        passed += int(result)

    def _make_linear_data(n=60, seed=42):
        import random
        rng = random.Random(seed)
        X = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(n)]
        beta = [1.5, -1.0, 0.5]
        y = [sum(X[i][j] * beta[j] for j in range(3)) + 0.3 * rng.gauss(0, 1) for i in range(n)]
        return X, y

    def _ridge_predict(X_val: list[list[float]], model: dict) -> list[float]:
        return ridge_predict(X_val, model["beta_hat"], model["mean_X"], model["std_X"])

    TestLogger.print_suite_header("F9 — k-Fold Cross-Validation")

    # test_kfold_cv_returns_correct_keys
    X, y = _make_linear_data()
    res  = kfold_cv(X, y, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=0.1)
    keys_ok = all(key in res for key in ("mean_cv_score", "std_cv_score", "cv_scores", "mean_cv_r2", "cv_r2_list"))
    run(assert_true(keys_ok, label="kfold_cv returns correct dictionary keys"))

    # test_kfold_cv_number_of_folds
    ok_folds = True
    for k in (3, 5, 10):
        res2 = kfold_cv(X, y, k=k, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=0.1)
        if len(res2["cv_scores"]) != k:
            ok_folds = False
    run(assert_true(ok_folds, label="cv_scores list length matches number of folds (k)"))

    # test_kfold_cv_mse_positive
    res3 = kfold_cv(X, y, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=1.0)
    run(assert_true(all(mse >= 0 for mse in res3["cv_scores"]), label="all cv_scores (MSE) are non-negative"))

    # test_kfold_cv_mean_matches_list
    res4 = kfold_cv(X, y, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=0.5)
    expected_mean = sum(res4["cv_scores"]) / len(res4["cv_scores"])
    run(assert_close(res4["mean_cv_score"], expected_mean, label="mean_cv_score matches exact average of cv_scores list", rtol=1e-9))

    # test_kfold_cv_r2_range
    X2, y2 = _make_linear_data(n=100, seed=7)
    res5 = kfold_cv(X2, y2, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=1e-4)
    run(assert_true(res5["mean_cv_r2"] > 0.5, label="CV R2 score is within expected valid range"))

    # test_kfold_cv_reproducible
    res6_1 = kfold_cv(X, y, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=1.0)
    res6_2 = kfold_cv(X, y, k=5, model_fn=ridge_fit, predict_fn=_ridge_predict, lam=1.0)
    run(assert_equal(res6_1["mean_cv_score"], res6_2["mean_cv_score"], label="kfold_cv results are completely reproducible given same seed"))

    TestLogger.print_summary(passed, total)


