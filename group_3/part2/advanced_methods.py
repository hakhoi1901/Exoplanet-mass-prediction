from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def _as_float_matrix(values) -> list[list[float]]:
    """Chuyển dữ liệu 2 chiều thành ma trận float thuần Python."""
    return [[float(value) for value in row] for row in values]


def _as_float_list(values) -> list[float]:
    """Chuyển dữ liệu 1 chiều thành danh sách float thuần Python."""
    return [float(value) for value in values]


def _squared_distance(row1: list[float], row2: list[float]) -> float:
    """Tính bình phương khoảng cách Euclid giữa hai dòng dữ liệu."""
    return sum((a - b) ** 2 for a, b in zip(row1, row2))


def rbf_kernel(X1: list[list[float]], X2: list[list[float]], gamma: float) -> list[list[float]]:
    """Tạo ma trận Gram RBF giữa hai ma trận đặc trưng."""
    return [
        [math.exp(-gamma * max(_squared_distance(row1, row2), 0.0)) for row2 in X2]
        for row1 in X1
    ]


def _solve_linear_system(A: list[list[float]], b: list[float]) -> list[float]:
    """Giải hệ Ax=b bằng khử Gauss-Jordan có chọn pivot."""
    n = len(A)
    if n == 0:
        return []
    augmented = [row[:] + [float(rhs)] for row, rhs in zip(A, b)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        pivot = augmented[pivot_row][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular or near-singular")
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        augmented[col] = [value / pivot for value in augmented[col]]

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) <= 1e-15:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[col])
            ]

    return [augmented[row][-1] for row in range(n)]


def kernel_ridge_fit(
    X: list[list[float]],
    y: list[float],
    alpha: float = 1.0,
    gamma: float = 0.1,
) -> dict:
    """Huấn luyện Kernel Ridge Regression dạng đối ngẫu với RBF kernel."""
    K = rbf_kernel(X, X, gamma=gamma)
    regularized = [
        [value + (alpha if i == j else 0.0) for j, value in enumerate(row)]
        for i, row in enumerate(K)
    ]
    dual_coef = _solve_linear_system(regularized, y)
    return {"X_train": X, "dual_coef": dual_coef, "alpha": alpha, "gamma": gamma}


def kernel_ridge_predict(model: dict, X: list[list[float]]) -> list[float]:
    """Dự đoán target cho dữ liệu mới bằng mô hình Kernel Ridge đã fit."""
    K = rbf_kernel(X, model["X_train"], gamma=model["gamma"])
    return [sum(k_value * coef for k_value, coef in zip(row, model["dual_coef"])) for row in K]


def regression_scores(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    """Tính MAE, RMSE và R2 cho kết quả hồi quy."""
    residual = [actual - predicted for actual, predicted in zip(y_true, y_pred)]
    mean_y = sum(y_true) / len(y_true) if y_true else 0.0
    rss = sum(value**2 for value in residual)
    tss = sum((value - mean_y) ** 2 for value in y_true)
    mse = rss / len(residual) if residual else 0.0
    return {
        "MAE": sum(abs(value) for value in residual) / len(residual) if residual else 0.0,
        "RMSE": math.sqrt(mse),
        "R2": 1.0 - rss / tss if tss > 1e-12 else 0.0,
    }


def run_kernel_ridge_bonus(
    preprocessed_path: str | Path = ROOT_DIR / "part2" / "output" / "preprocessed.pkl",
    output_dir: str | Path = ROOT_DIR / "part2" / "output",
    max_train: int = 800,
) -> dict:
    """Chạy thí nghiệm Kernel Ridge bonus và lưu các file kết quả."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(preprocessed_path).open("rb") as f:
        data = pickle.load(f)

    X_train = _as_float_matrix(data["X_train"].values.tolist())
    y_train = _as_float_list(data["y_train"].tolist())
    X_test = _as_float_matrix(data["X_test"].values.tolist())
    y_test = _as_float_list(data["y_test"].tolist())

    if len(X_train) > max_train:
        rng = random.Random(42)
        idx = rng.sample(range(len(X_train)), max_train)
        X_fit = [X_train[i] for i in idx]
        y_fit = [y_train[i] for i in idx]
    else:
        X_fit = X_train
        y_fit = y_train

    if len(X_fit) < 2:
        raise ValueError("Kernel ridge needs at least 2 training rows")

    val_size = max(1, int(len(X_fit) * 0.2))
    if val_size >= len(X_fit):
        val_size = len(X_fit) - 1
    X_sub_train, y_sub_train = X_fit[:-val_size], y_fit[:-val_size]
    X_val, y_val = X_fit[-val_size:], y_fit[-val_size:]

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    gammas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

    rows = []
    best_val_rmse = float("inf")
    best_params = {}
    for alpha in alphas:
        for gamma in gammas:
            model = kernel_ridge_fit(X_sub_train, y_sub_train, alpha=alpha, gamma=gamma)
            pred_val = kernel_ridge_predict(model, X_val)
            val_scores = regression_scores(y_val, pred_val)

            row = {"alpha": alpha, "gamma": gamma, "Val_RMSE": val_scores["RMSE"]}
            rows.append(row)

            if val_scores["RMSE"] < best_val_rmse:
                best_val_rmse = val_scores["RMSE"]
                best_params = {"alpha": alpha, "gamma": gamma}

    print(f"Best validation params: {best_params}")
    final_model = kernel_ridge_fit(X_fit, y_fit, alpha=best_params["alpha"], gamma=best_params["gamma"])

    final_pred = kernel_ridge_predict(final_model, X_test)
    final_scores = regression_scores(y_test, final_pred)

    best_result = {
        "alpha": best_params["alpha"],
        "gamma": best_params["gamma"],
        **final_scores,
    }

    pd.DataFrame(rows).sort_values("Val_RMSE").to_csv(output_dir / "kernel_ridge_search.csv", index=False)

    result = {
        "method": "Kernel Ridge Regression with RBF kernel",
        "note": "Tuning hyperparams on Validation set, evaluated once on Test set to prevent leakage.",
        "max_train": max_train,
        "best_on_test": best_result,
    }

    (output_dir / "advanced_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """Đọc tham số dòng lệnh và chạy thí nghiệm nâng cao."""
    parser = argparse.ArgumentParser(description="Part 2 bonus advanced methods")
    parser.add_argument("--preprocessed", default=str(ROOT_DIR / "part2" / "output" / "preprocessed.pkl"))
    parser.add_argument("--outdir", default=str(ROOT_DIR / "part2" / "output"))
    parser.add_argument("--max-train", type=int, default=800)
    args = parser.parse_args()
    result = run_kernel_ridge_bonus(args.preprocessed, args.outdir, max_train=args.max_train)
    print("Advanced method completed")
    print(f"  Best Kernel Ridge: {result['best_on_test']}")
    print(f"  Saved: {Path(args.outdir) / 'advanced_results.json'}")


if __name__ == "__main__":
    main()
