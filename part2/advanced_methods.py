from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    X1_sq = np.sum(X1 * X1, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2 * X2, axis=1).reshape(1, -1)
    dist2 = np.maximum(X1_sq + X2_sq - 2.0 * X1 @ X2.T, 0.0)
    return np.exp(-gamma * dist2)


def kernel_ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, gamma: float = 0.1) -> dict:
    K = rbf_kernel(X, X, gamma=gamma)
    dual_coef = np.linalg.solve(K + alpha * np.eye(K.shape[0]), y)
    return {"X_train": X, "dual_coef": dual_coef, "alpha": alpha, "gamma": gamma}


def kernel_ridge_predict(model: dict, X: np.ndarray) -> np.ndarray:
    K = rbf_kernel(X, model["X_train"], gamma=model["gamma"])
    return K @ model["dual_coef"]


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    rss = float(np.sum(residual**2))
    tss = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "R2": float(1.0 - rss / tss) if tss > 1e-12 else 0.0,
    }


def run_kernel_ridge_bonus(
    preprocessed_path: str | Path = ROOT_DIR / "part2" / "output" / "preprocessed.pkl",
    output_dir: str | Path = ROOT_DIR / "part2" / "output",
    max_train: int = 800,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(preprocessed_path).open("rb") as f:
        data = pickle.load(f)

    X_train = data["X_train"].to_numpy(dtype=float)
    y_train = data["y_train"].to_numpy(dtype=float)
    X_test = data["X_test"].to_numpy(dtype=float)
    y_test = data["y_test"].to_numpy(dtype=float)

    if len(X_train) > max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=max_train, replace=False)
        X_fit = X_train[idx]
        y_fit = y_train[idx]
    else:
        X_fit = X_train
        y_fit = y_train

    val_size = int(len(X_fit) * 0.2) 
    X_sub_train, y_sub_train = X_fit[:-val_size], y_fit[:-val_size]
    X_val, y_val = X_fit[-val_size:], y_fit[-val_size:]

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    gammas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

    rows = []
    best_val_rmse = float('inf')
    best_params = {}
    for alpha in alphas:
        for gamma in gammas:
            # Huấn luyện trên tập con Sub-train
            model = kernel_ridge_fit(X_sub_train, y_sub_train, alpha=alpha, gamma=gamma)
            
            # Đánh giá trên tập Validation
            pred_val = kernel_ridge_predict(model, X_val)
            val_scores = regression_scores(y_val, pred_val)
            
            row = {"alpha": alpha, "gamma": gamma, "Val_RMSE": val_scores["RMSE"]}
            rows.append(row)
            
            # Lưu lại cấu hình tốt nhất
            if val_scores["RMSE"] < best_val_rmse:
                best_val_rmse = val_scores["RMSE"]
                best_params = {"alpha": alpha, "gamma": gamma}

    print(f"Tham số tốt nhất tìm được: {best_params}")
    final_model = kernel_ridge_fit(X_fit, y_fit, alpha=best_params["alpha"], gamma=best_params["gamma"])
    
    final_pred = kernel_ridge_predict(final_model, X_test)
    final_scores = regression_scores(y_test, final_pred)
    
    best_result = {
        "alpha": best_params["alpha"], 
        "gamma": best_params["gamma"], 
        **final_scores
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
    parser = argparse.ArgumentParser(description="Part 2 bonus advanced methods")
    parser.add_argument("--preprocessed", default=str(ROOT_DIR / "part2" / "output" / "preprocessed.pkl"))
    parser.add_argument("--outdir", default=str(ROOT_DIR / "part2" / "output"))
    parser.add_argument("--max-train", type=int, default=800)
    args = parser.parse_args()
    result = run_kernel_ridge_bonus(args.preprocessed, args.outdir, max_train=args.max_train)
    print("Advanced method completed")
    print(f"  Best Kernel Ridge: {result['best']}")
    print(f"  Saved: {Path(args.outdir) / 'advanced_results.json'}")


if __name__ == "__main__":
    main()
