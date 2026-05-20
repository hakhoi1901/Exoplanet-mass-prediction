from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PART1_DIR = ROOT_DIR / "part1"
for path in (ROOT_DIR, PART1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ols_implementation import coef_inference, model_metrics, ols_fit
from ridge_lasso import lasso_fit, lasso_predict, ridge_fit, ridge_predict
from cross_validation import kfold_cv
from residual_analysis import residual_plots


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[ModelComparison] {message}", flush=True)


def _log_table(title: str, table: pd.DataFrame, verbose: bool = True, max_rows: int | None = None) -> None:
    if not verbose or table.empty:
        return
    shown = table if max_rows is None else table.head(max_rows)
    print(f"\n[ModelComparison] {title}", flush=True)
    print(shown.to_string(index=False), flush=True)


def _as_list_frame(df: pd.DataFrame) -> list[list[float]]:
    return df.astype(float).values.tolist()


def _as_list_series(series: pd.Series) -> list[float]:
    return series.astype(float).tolist()


def ols_predict(X: list[list[float]], model: dict) -> list[float]:
    beta = model["beta_hat"]
    return [beta[0] + sum(row[j] * beta[j + 1] for j in range(len(row))) for row in X]


def _ridge_predict_adapter(X_val: list[list[float]], model: dict) -> list[float]:
    return ridge_predict(X_val, model["beta_hat"], model["mean_X"], model["std_X"])


def _lasso_predict_adapter(X_val: list[list[float]], model: dict) -> list[float]:
    return lasso_predict(X_val, model["beta_hat"], model["mean_X"], model["std_X"])


def regression_scores(y_true: list[float], y_pred: list[float], p: int) -> dict[str, float]:
    metrics = model_metrics(y_true, y_pred, p=min(p, max(1, len(y_true) - 2)))
    return {
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "R2_adj": metrics["R2_adj"],
        "RSS": metrics["RSS"],
    }


def evaluate_model(
    name: str,
    fit_fn: Callable[..., dict],
    predict_fn: Callable[[list[list[float]], dict], list[float]],
    X_train: list[list[float]],
    y_train: list[float],
    X_test: list[list[float]],
    y_test: list[float],
    feature_names: list[str],
    cv_k: int = 5,
    verbose: bool = False,
    **fit_kwargs,
) -> dict:
    start = time.perf_counter()
    _log(f"Fitting {name}: features={len(feature_names)}, cv_k={cv_k}, params={fit_kwargs or '{}'}", verbose)
    model = fit_fn(X_train, y_train, **fit_kwargs)
    y_train_pred = model.get("y_hat", predict_fn(X_train, model))
    y_test_pred = predict_fn(X_test, model)

    cv = kfold_cv(
        X_train,
        y_train,
        k=cv_k,
        model_fn=fit_fn,
        predict_fn=predict_fn,
        **fit_kwargs,
    )

    train_metrics = regression_scores(y_train, y_train_pred, p=len(feature_names))
    test_metrics = regression_scores(y_test, y_test_pred, p=len(feature_names))
    _log(
        f"{name} metrics: train_R2={train_metrics['R2']:.4f}, "
        f"test_R2={test_metrics['R2']:.4f}, test_RMSE={test_metrics['RMSE']:.4f}, "
        f"test_MAE={test_metrics['MAE']:.4f}, cv_MSE={cv['mean_cv_score']:.6f}, "
        f"cv_R2={cv['mean_cv_r2']:.4f}, elapsed={time.perf_counter() - start:.2f}s",
        verbose,
    )

    return {
        "name": name,
        "model": model,
        "feature_names": feature_names,
        "train_pred": y_train_pred,
        "test_pred": y_test_pred,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv": cv,
        "fit_kwargs": fit_kwargs,
    }


def lambda_search(
    X_train: list[list[float]],
    y_train: list[float],
    model_fn: Callable[..., dict],
    predict_fn: Callable[[list[list[float]], dict], list[float]],
    lambdas: list[float],
    k: int = 5,
    extra_kwargs: dict | None = None,
    verbose: bool = False,
    label: str = "model",
) -> dict:
    extra_kwargs = extra_kwargs or {}
    rows = []
    _log(f"Lambda search for {label}: candidates={lambdas}, k={k}", verbose)
    for lam in lambdas:
        cv = kfold_cv(
            X_train,
            y_train,
            k=k,
            model_fn=model_fn,
            predict_fn=predict_fn,
            lam=lam,
            **extra_kwargs,
        )
        rows.append(
            {
                "lambda": lam,
                "mean_cv_mse": cv["mean_cv_score"],
                "std_cv_mse": cv["std_cv_score"],
                "mean_cv_r2": cv["mean_cv_r2"],
            }
        )
        _log(
            f"{label} lambda={lam:.6g}: mean_cv_mse={cv['mean_cv_score']:.6f}, "
            f"std={cv['std_cv_score']:.6f}, mean_cv_r2={cv['mean_cv_r2']:.4f}",
            verbose,
        )
    best = min(rows, key=lambda row: row["mean_cv_mse"])
    return {"best_lam": float(best["lambda"]), "rows": rows}


def select_features_by_pvalue(
    X_train: list[list[float]],
    y_train: list[float],
    feature_names: list[str],
    alpha: float = 0.05,
) -> tuple[list[int], pd.DataFrame]:
    full = ols_fit(X_train, y_train)
    inference = coef_inference(X_train, y_train, full["beta_hat"], full["sigma2_hat"])
    table = pd.DataFrame(
        {
            "feature": ["intercept"] + feature_names,
            "coef": inference["coef"],
            "std_err": inference["std_err"],
            "t_stat": inference["t_stat"],
            "p_value": inference["p_value"],
            "ci_lower": inference["ci_lower"],
            "ci_upper": inference["ci_upper"],
        }
    )
    selected = [
        i
        for i, p_value in enumerate(inference["p_value"][1:])
        if math.isfinite(float(p_value)) and float(p_value) < alpha
    ]
    if not selected:
        selected = (
            table.iloc[1:]
            .assign(abs_t=lambda df: df["t_stat"].abs())
            .sort_values("abs_t", ascending=False)
            .head(min(3, len(feature_names)))
            .index.to_series()
            .sub(1)
            .astype(int)
            .tolist()
        )
    return selected, table


def _subset_columns(X: list[list[float]], indices: list[int]) -> list[list[float]]:
    return [[row[i] for i in indices] for row in X]


def _serializable_result(result: dict) -> dict:
    model = result["model"]
    beta = model.get("beta_hat")
    return {
        "name": result["name"],
        "feature_names": result["feature_names"],
        "train_metrics": result["train_metrics"],
        "test_metrics": result["test_metrics"],
        "cv_mean_mse": result["cv"]["mean_cv_score"],
        "cv_std_mse": result["cv"]["std_cv_score"],
        "cv_mean_r2": result["cv"]["mean_cv_r2"],
        "fit_kwargs": result["fit_kwargs"],
        "beta_hat": beta,
    }


def plot_model_comparison(results: dict[str, dict], output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    r2 = [results[name]["test_metrics"]["R2"] for name in names]
    rmse = [results[name]["test_metrics"]["RMSE"] for name in names]
    mae = [results[name]["test_metrics"]["MAE"] for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [(r2, "Test R2", True), (rmse, "Test RMSE", False), (mae, "Test MAE", False)]
    for ax, (values, title, higher_is_better) in zip(axes, specs):
        colors = ["#0f766e" if (v == max(values) if higher_is_better else v == min(values)) else "#3b82f6" for v in values]
        bars = ax.barh(names, values, color=colors)
        ax.set_title(title)
        ax.invert_yaxis()
        for bar, value in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    path = output_dir / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_feature_importance(result: dict, output_dir: Path) -> str | None:
    beta = result["model"].get("beta_hat")
    if beta is None or len(beta) <= 1:
        return None
    coef = pd.DataFrame({"feature": result["feature_names"], "coef": beta[1:]})
    coef = coef.sort_values("coef", key=lambda s: s.abs(), ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(coef))))
    colors = ["#dc2626" if value < 0 else "#0f766e" for value in coef["coef"]]
    ax.barh(coef["feature"], coef["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Feature importance - {result['name']}")
    ax.set_xlabel("Coefficient on standardized features")
    fig.tight_layout()
    path = output_dir / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_actual_vs_predicted(y_true: list[float], y_pred: list[float], name: str, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=16, alpha=0.45, color="#2563eb")
    low = min(min(y_true), min(y_pred))
    high = max(max(y_true), max(y_pred))
    ax.plot([low, high], [low, high], color="#dc2626", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs predicted - {name}")
    fig.tight_layout()
    path = output_dir / "actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _target_summary(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    rows = []
    for split, y in [("train", y_train), ("test", y_test)]:
        rows.append(
            {
                "split": split,
                "n": int(len(y)),
                "mean": float(y.mean()),
                "std": float(y.std(ddof=0)),
                "min": float(y.min()),
                "q25": float(y.quantile(0.25)),
                "median": float(y.median()),
                "q75": float(y.quantile(0.75)),
                "max": float(y.max()),
            }
        )
    return pd.DataFrame(rows).round(6)


def _feature_matrix_summary(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "train",
                "rows": X_train.shape[0],
                "features": X_train.shape[1],
                "missing_cells": int(X_train.isna().sum().sum()),
                "max_abs_mean": float(X_train.mean().abs().max()),
                "min_std": float(X_train.std(ddof=0).min()),
                "max_std": float(X_train.std(ddof=0).max()),
            },
            {
                "split": "test",
                "rows": X_test.shape[0],
                "features": X_test.shape[1],
                "missing_cells": int(X_test.isna().sum().sum()),
                "max_abs_mean": float(X_test.mean().abs().max()),
                "min_std": float(X_test.std(ddof=0).min()),
                "max_std": float(X_test.std(ddof=0).max()),
            },
        ]
    ).round(6)


def _coef_table(result: dict) -> pd.DataFrame:
    beta = result["model"].get("beta_hat")
    if beta is None or len(beta) <= 1:
        return pd.DataFrame()
    table = pd.DataFrame({"feature": result["feature_names"], "coef": beta[1:]})
    table["abs_coef"] = table["coef"].abs()
    return table.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"]).round(6)


def _prediction_summary(y_true: list[float], y_pred: list[float]) -> pd.DataFrame:
    residual = [float(actual) - float(predicted) for actual, predicted in zip(y_true, y_pred)]
    pred = [float(value) for value in y_pred]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def std(values: list[float]) -> float:
        avg = mean(values)
        return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values)) if values else 0.0

    return pd.DataFrame(
        [
            {
                "pred_mean": mean(pred),
                "pred_std": std(pred),
                "pred_min": min(pred) if pred else 0.0,
                "pred_max": max(pred) if pred else 0.0,
                "residual_mean": mean(residual),
                "residual_std": std(residual),
                "residual_min": min(residual) if residual else 0.0,
                "residual_max": max(residual) if residual else 0.0,
            }
        ]
    ).round(6)


def run_model_comparison(
    preprocessed_path: str | Path = ROOT_DIR / "part2" / "output" / "preprocessed.pkl",
    output_dir: str | Path = ROOT_DIR / "part2" / "output",
    include_lasso: bool = True,
    verbose: bool = True,
) -> dict:
    start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(preprocessed_path).open("rb") as f:
        data = pickle.load(f)

    X_train_df: pd.DataFrame = data["X_train"]
    X_test_df: pd.DataFrame = data["X_test"]
    y_train_s: pd.Series = data["y_train"]
    y_test_s: pd.Series = data["y_test"]
    feature_names = list(data["feature_names"])

    X_train = _as_list_frame(X_train_df)
    X_test = _as_list_frame(X_test_df)
    y_train = _as_list_series(y_train_s)
    y_test = _as_list_series(y_test_s)

    _log(f"Loaded preprocessed data from {preprocessed_path}", verbose)
    _log_table("Feature matrix summary", _feature_matrix_summary(X_train_df, X_test_df), verbose)
    _log_table("Target distribution", _target_summary(y_train_s, y_test_s), verbose)
    _log(f"Features used ({len(feature_names)}): {feature_names}", verbose)

    results: dict[str, dict] = {}

    results["OLS full"] = evaluate_model(
        "OLS full",
        ols_fit,
        ols_predict,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        verbose=verbose,
    )
    _log_table("OLS full largest coefficients", _coef_table(results["OLS full"]), verbose, max_rows=8)

    selected_idx, inference_table = select_features_by_pvalue(X_train, y_train, feature_names)
    selected_features = [feature_names[i] for i in selected_idx]
    inference_for_log = inference_table.copy()
    inference_for_log["abs_t"] = inference_for_log["t_stat"].abs()
    _log_table(
        "OLS coefficient inference sorted by p-value",
        inference_for_log.sort_values("p_value")[["feature", "coef", "std_err", "t_stat", "p_value", "ci_lower", "ci_upper"]].round(6),
        verbose,
        max_rows=10,
    )
    _log(f"OLS selected features at alpha=0.05 ({len(selected_features)}): {selected_features}", verbose)
    X_train_sel = _subset_columns(X_train, selected_idx)
    X_test_sel = _subset_columns(X_test, selected_idx)
    results["OLS selected"] = evaluate_model(
        "OLS selected",
        ols_fit,
        ols_predict,
        X_train_sel,
        y_train,
        X_test_sel,
        y_test,
        selected_features,
        verbose=verbose,
    )

    ridge_grid = [10 ** (-3 + 0.5 * i) for i in range(13)]
    ridge_search = lambda_search(X_train, y_train, ridge_fit, _ridge_predict_adapter, ridge_grid, k=5, verbose=verbose, label="Ridge")
    ridge_search_table = pd.DataFrame(ridge_search["rows"]).sort_values("mean_cv_mse").round(6)
    _log_table("Ridge lambda search sorted by CV MSE", ridge_search_table, verbose, max_rows=8)
    _log(f"Ridge selected lambda={ridge_search['best_lam']:.6g}", verbose)
    results["Ridge"] = evaluate_model(
        "Ridge",
        ridge_fit,
        _ridge_predict_adapter,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        lam=ridge_search["best_lam"],
        verbose=verbose,
    )
    _log_table("Ridge largest coefficients", _coef_table(results["Ridge"]), verbose, max_rows=8)

    lasso_search = None
    if include_lasso:
        lasso_grid = [10 ** (-4 + 0.5 * i) for i in range(13)]
        lasso_kwargs = {"max_iter": 300, "tol": 1e-5}
        lasso_search = lambda_search(
            X_train,
            y_train,
            lasso_fit,
            _lasso_predict_adapter,
            lasso_grid,
            k=3,
            extra_kwargs=lasso_kwargs,
            verbose=verbose,
            label="Lasso",
        )
        lasso_search_table = pd.DataFrame(lasso_search["rows"]).sort_values("mean_cv_mse").round(6)
        _log_table("Lasso lambda search sorted by CV MSE", lasso_search_table, verbose, max_rows=8)
        _log(f"Lasso selected lambda={lasso_search['best_lam']:.6g}", verbose)
        results["Lasso"] = evaluate_model(
            "Lasso",
            lasso_fit,
            _lasso_predict_adapter,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            cv_k=3,
            lam=lasso_search["best_lam"],
            verbose=verbose,
            **lasso_kwargs,
        )
        lasso_coef = _coef_table(results["Lasso"])
        if not lasso_coef.empty:
            zeros = int((lasso_coef["coef"].abs() < 1e-10).sum())
            _log(f"Lasso zeroed coefficients: {zeros}/{len(lasso_coef)}", verbose)
            _log_table("Lasso largest coefficients", lasso_coef, verbose, max_rows=8)

    summary = pd.DataFrame(
        [
            {
                "model": name,
                "test_R2": result["test_metrics"]["R2"],
                "test_RMSE": result["test_metrics"]["RMSE"],
                "test_MAE": result["test_metrics"]["MAE"],
                "cv_MSE": result["cv"]["mean_cv_score"],
                "cv_R2": result["cv"]["mean_cv_r2"],
            }
            for name, result in results.items()
        ]
    ).sort_values("test_R2", ascending=False)
    _log_table("Model comparison summary sorted by test R2", summary.round(6), verbose)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    inference_table.to_csv(output_dir / "ols_inference.csv", index=False)
    pd.DataFrame(ridge_search["rows"]).to_csv(output_dir / "ridge_lambda_search.csv", index=False)
    if lasso_search is not None:
        pd.DataFrame(lasso_search["rows"]).to_csv(output_dir / "lasso_lambda_search.csv", index=False)

    best_name = str(summary.iloc[0]["model"])
    best = results[best_name]
    _log(f"Best model by test R2: {best_name}", verbose)
    _log_table("Best model prediction/residual summary on test set", _prediction_summary(y_test, best["test_pred"]), verbose)
    plot_paths = {
        "model_comparison": plot_model_comparison(results, output_dir),
        "actual_vs_predicted": plot_actual_vs_predicted(y_test, best["test_pred"], best_name, output_dir),
    }
    importance_path = plot_feature_importance(results["Ridge"], output_dir)
    if importance_path:
        plot_paths["feature_importance"] = importance_path
    residual_info = residual_plots(
        y_test,
        best["test_pred"],
        X=_subset_columns(X_test, [feature_names.index(col) for col in best["feature_names"]]) if best["feature_names"] else None,
        save_dir=str(output_dir),
        show_plot=False,
    )
    plot_paths["residual_plots"] = str(output_dir / "residual_plots.png")

    serializable = {
        "summary": summary.to_dict(orient="records"),
        "best_model": best_name,
        "selected_features": selected_features,
        "ridge_search": ridge_search,
        "lasso_search": lasso_search,
        "plots": plot_paths,
        "results": {name: _serializable_result(result) for name, result in results.items()},
        "residual_cooks_distance_max": max(residual_info["cooks_d"]) if residual_info["cooks_d"] else None,
    }
    (output_dir / "model_results.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    _log(f"Saved model artifacts to {output_dir}", verbose)

    with (output_dir / "results.pkl").open("wb") as f:
        pickle.dump(
            {
                "results": results,
                "summary": summary,
                "feature_names": feature_names,
                "selected_features": selected_features,
                "X_train": X_train_df,
                "X_test": X_test_df,
                "y_train": y_train_s,
                "y_test": y_test_s,
            },
            f,
        )

    _log(f"Model comparison finished in {time.perf_counter() - start:.2f}s", verbose)
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2 model comparison using part1 implementations")
    parser.add_argument("--preprocessed", default=str(ROOT_DIR / "part2" / "output" / "preprocessed.pkl"))
    parser.add_argument("--outdir", default=str(ROOT_DIR / "part2" / "output"))
    parser.add_argument("--no-lasso", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_model_comparison(
        preprocessed_path=args.preprocessed,
        output_dir=args.outdir,
        include_lasso=not args.no_lasso,
        verbose=not args.quiet,
    )
    print("Model comparison completed")
    for row in result["summary"]:
        print(
            f"  {row['model']:<14} R2={row['test_R2']:.4f} "
            f"RMSE={row['test_RMSE']:.4f} MAE={row['test_MAE']:.4f}"
        )
    print(f"  Best model: {result['best_model']}")
    print(f"  Saved: {Path(args.outdir) / 'model_results.json'}")


if __name__ == "__main__":
    main()
