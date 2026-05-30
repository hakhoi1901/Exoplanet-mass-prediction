from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
PART1_DIR = ROOT_DIR / "part1"
for path in (ROOT_DIR, PART1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import RANDOM_STATE
from ols_implementation import vif as part1_vif
from ridge_lasso import ridge_fit, ridge_predict


DEFAULT_TARGET = "pl_rade"
DEFAULT_DATA_PATH = ROOT_DIR / "part2" / "data" / "data.csv"
DEFAULT_MODEL_COLUMNS = (
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen",
    "pl_trandur",
    "pl_trandep",
    "pl_imppar",
    "pl_eqt",
    "pl_insol",
    "pl_bmasse",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_met",
    "st_logg",
    "sy_dist",
    "pl_rade",
)
DEFAULT_LOG_COLUMNS = (
    "pl_orbper",
    "pl_orbsmax",
    "pl_trandep",
    "pl_insol",
    "pl_bmasse",
    "pl_eqt",
    "sy_dist",
    "pl_rade",
)
DEFAULT_WINSOR_COLUMNS = ("pl_orbeccen", "pl_trandur", "pl_imppar", "st_rad")
DEFAULT_DROP_COLUMNS = ("st_mass", "st_logg")


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    sep = "\t" if first_line.count("\t") > first_line.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8", comment="#", low_memory=False)
    df.columns = [str(col).strip().lstrip("@") for col in df.columns]
    return df


def select_model_columns(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    model_columns: Iterable[str] = DEFAULT_MODEL_COLUMNS,
) -> pd.DataFrame:
    requested = list(dict.fromkeys(model_columns))
    if target not in requested:
        requested.append(target)
    missing = [col for col in requested if col not in df.columns]
    if target in missing:
        raise KeyError(f"Target column '{target}' was not found")

    selected = [col for col in requested if col in df.columns]
    out = df[selected].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def train_test_split_frame(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    rng = np.random.default_rng(random_state)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    n_test = max(1, int(round(len(df) * test_size)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _safe_log1p(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric
    min_value = numeric.min(skipna=True)
    if min_value <= -1:
        shift = abs(min_value) + 1.0
        numeric = numeric + shift
    return numeric.map(lambda value: math.log1p(value) if pd.notna(value) else value)


def _to_float_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.astype(float)


def _replace_non_finite(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([math.inf, -math.inf], math.nan)


def _flatten_axes(axes) -> list:
    if isinstance(axes, (list, tuple)):
        flattened = []
        for item in axes:
            flattened.extend(_flatten_axes(item))
        return flattened
    if hasattr(axes, "flat"):
        return list(axes.flat)
    return [axes]


def _axes_grid(axes, rows: int, cols: int) -> list[list]:
    flattened = _flatten_axes(axes)
    return [flattened[row * cols : (row + 1) * cols] for row in range(rows)]


@dataclass
class DataPipeline:
    target: str = DEFAULT_TARGET
    log_columns: Iterable[str] = DEFAULT_LOG_COLUMNS
    winsor_columns: Iterable[str] = DEFAULT_WINSOR_COLUMNS
    initial_drop_columns: Iterable[str] = DEFAULT_DROP_COLUMNS
    vif_threshold: float = 10.0
    max_vif_drops: int = 10
    standardize: bool = True
    one_hot_encode: bool = True
    imputation_method: str = "mice"
    mice_imputations: int = 5
    mice_iterations: int = 5
    mice_regularization: float = 1e-3
    mice_noise_scale: float = 0.15
    random_state: int = RANDOM_STATE
    verbose: bool = True
    log_top_n: int = 8

    target_transformed_: str | None = None
    feature_names_: list[str] = field(default_factory=list)
    categorical_levels_: dict[str, list[str]] = field(default_factory=dict)
    winsor_bounds_: dict[str, tuple[float, float]] = field(default_factory=dict)
    log_transform_report_: list[dict] = field(default_factory=list)
    winsor_report_: list[dict] = field(default_factory=list)
    imputation_report_: list[dict] = field(default_factory=list)
    impute_values_: dict[str, float] = field(default_factory=dict)
    mice_missing_columns_: list[str] = field(default_factory=list)
    mice_chains_: list[list[dict]] = field(default_factory=list)
    scale_mean_: dict[str, float] = field(default_factory=dict)
    scale_std_: dict[str, float] = field(default_factory=dict)
    vif_drop_columns_: list[str] = field(default_factory=list)
    vif_history_: list[dict[str, float | str]] = field(default_factory=list)
    final_vif_: dict[str, float] = field(default_factory=dict)
    missing_report_: pd.DataFrame | None = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[DataPipeline] {message}", flush=True)

    def _log_table(self, title: str, table: pd.DataFrame, max_rows: int | None = None) -> None:
        if not self.verbose or table.empty:
            return
        shown = table if max_rows is None else table.head(max_rows)
        print(f"\n[DataPipeline] {title}", flush=True)
        print(shown.to_string(index=False), flush=True)

    @staticmethod
    def _safe_skew(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.skew()) if len(values) >= 3 else float("nan")

    def _feature_missing_table(self, X: pd.DataFrame) -> pd.DataFrame:
        total = len(X)
        rows = []
        for col in X.columns:
            missing = int(X[col].isna().sum())
            if missing > 0:
                observed = X[col].dropna()
                rows.append(
                    {
                        "column": col,
                        "missing": missing,
                        "missing_%": round(missing / total * 100, 3) if total else 0.0,
                        "observed": int(observed.shape[0]),
                        "observed_mean": round(float(observed.mean()), 6) if len(observed) else float("nan"),
                        "observed_std": round(float(observed.std(ddof=0)), 6) if len(observed) else float("nan"),
                    }
                )
        return pd.DataFrame(rows).sort_values("missing_%", ascending=False) if rows else pd.DataFrame()

    def fit(self, df: pd.DataFrame) -> "DataPipeline":
        start = time.perf_counter()
        target_missing = int(df[self.target].isna().sum()) if self.target in df.columns else 0
        self._log(
            f"Raw data: rows={df.shape[0]}, columns={df.shape[1]}, "
            f"target='{self.target}', missing target rows={target_missing}"
        )
        prepared = self._prepare_frame(df, fitting=True)
        X = prepared.drop(columns=[self.target_transformed_])
        self.missing_report_ = self._missing_report(prepared)
        self._log(f"After target cleanup and log-transform: rows={len(prepared)}, features={X.shape[1]}, target='{self.target_transformed_}'")
        self._log_table(
            "Log-transform effect on skewness and scale",
            pd.DataFrame(self.log_transform_report_),
            max_rows=self.log_top_n,
        )
        self._log_table("Feature missing values before imputation", self._feature_missing_table(X), max_rows=self.log_top_n)

        X = self._fit_categorical_encoder(X)
        if self.categorical_levels_:
            encoded_counts = pd.DataFrame(
                {"column": list(self.categorical_levels_), "levels": [len(v) for v in self.categorical_levels_.values()]}
            )
            self._log_table("Categorical encoding levels", encoded_counts, max_rows=self.log_top_n)
        X = self._drop_initial_columns(X)
        actual_initial_drops = [c for c in self.initial_drop_columns if c in X.columns or c in prepared.columns]
        if actual_initial_drops:
            self._log(f"Initial feature drops before modeling: {actual_initial_drops}")
        X = _to_float_frame(X)
        X = _replace_non_finite(X)

        self._fit_winsor_bounds(X)
        X = self._apply_winsor_bounds(X)
        self._log_table("Winsorization summary fitted on train data", pd.DataFrame(self.winsor_report_), max_rows=self.log_top_n)

        X_before_impute = X.copy()
        self._fit_imputer(X)
        X = self._apply_imputer(X)
        self._build_imputation_report(X_before_impute, X)
        self._log_table(
            f"Missing-value imputation summary ({self.imputation_method.upper()})",
            pd.DataFrame(self.imputation_report_),
            max_rows=self.log_top_n,
        )
        self._log(f"Remaining missing cells after imputation: {int(X.isna().sum().sum())}")

        self._fit_vif_filter(X)
        X = X.drop(columns=self.vif_drop_columns_, errors="ignore")
        self._log(f"VIF dropped columns: {self.vif_drop_columns_ if self.vif_drop_columns_ else 'none'}")

        self._fit_scaler(X)
        self.feature_names_ = X.columns.tolist()
        scale_table = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "mean_train": [round(self.scale_mean_[c], 6) for c in self.feature_names_],
                "std_train": [round(self.scale_std_[c], 6) for c in self.feature_names_],
            }
        )
        self._log_table("Standardization parameters fitted on train data", scale_table, max_rows=self.log_top_n)
        elapsed = time.perf_counter() - start
        self._log(f"Final training matrix: rows={X.shape[0]}, features={X.shape[1]}, elapsed={elapsed:.2f}s")
        return self

    def transform(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        if self.target_transformed_ is None:
            raise RuntimeError("DataPipeline must be fitted before transform")

        start = time.perf_counter()
        prepared = self._prepare_frame(df, fitting=False, include_target=include_target)
        y = None
        if include_target:
            y = prepared[self.target_transformed_].astype(float).reset_index(drop=True)
            X = prepared.drop(columns=[self.target_transformed_])
        else:
            X = prepared

        X = self._apply_categorical_encoder(X)
        X = self._drop_initial_columns(X)
        X = _to_float_frame(X)
        X = _replace_non_finite(X)
        X = self._apply_winsor_bounds(X)
        X = self._apply_imputer(X)
        X = X.drop(columns=self.vif_drop_columns_, errors="ignore")
        X = X.reindex(columns=self.feature_names_, fill_value=0.0)
        X = self._apply_scaler(X)
        elapsed = time.perf_counter() - start
        self._log(
            f"Transformed data: input rows={df.shape[0]}, output shape={X.shape}, "
            f"remaining missing cells={int(X.isna().sum().sum())}"
            f", elapsed={elapsed:.2f}s"
        )
        return X.reset_index(drop=True), y

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        self.fit(df)
        X, y = self.transform(df, include_target=True)
        if y is None:
            raise RuntimeError("Target was not returned from transform")
        return X, y

    def _prepare_frame(
        self,
        df: pd.DataFrame,
        fitting: bool,
        include_target: bool = True,
    ) -> pd.DataFrame:
        out = df.copy()
        if include_target and self.target not in out.columns:
            raise KeyError(f"Target column '{self.target}' was not found")

        log_columns = list(self.log_columns)
        log_set = set(log_columns)
        if fitting:
            self.log_transform_report_ = []
        for col in log_columns:
            if col in out.columns:
                before = pd.to_numeric(out[col], errors="coerce")
                transformed = _safe_log1p(out[col])
                if fitting:
                    self.log_transform_report_.append(
                        {
                            "source": col,
                            "created": f"log_{col}",
                            "missing": int(before.isna().sum()),
                            "min_before": round(float(before.min(skipna=True)), 6) if before.notna().any() else float("nan"),
                            "max_before": round(float(before.max(skipna=True)), 6) if before.notna().any() else float("nan"),
                            "skew_before": round(self._safe_skew(before), 6),
                            "skew_after": round(self._safe_skew(transformed), 6),
                        }
                    )
                out[f"log_{col}"] = transformed
                out = out.drop(columns=[col])

        transformed_target = f"log_{self.target}" if self.target in log_set else self.target
        if fitting:
            self.target_transformed_ = transformed_target

        if include_target:
            out = out.dropna(subset=[self.target_transformed_])
        elif self.target_transformed_ in out.columns:
            out = out.drop(columns=[self.target_transformed_])

        return out.reset_index(drop=True)

    def _fit_categorical_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.one_hot_encode:
            self.categorical_levels_ = {}
            self._log("One-hot encoding disabled; keeping numeric columns only")
            return X.select_dtypes(include="number").copy()

        self.categorical_levels_ = {}
        for col in X.select_dtypes(exclude="number").columns:
            values = X[col].fillna("__missing__").astype(str)
            self.categorical_levels_[col] = sorted(values.unique().tolist())
        if self.categorical_levels_:
            self._log(f"One-hot columns fitted: {list(self.categorical_levels_.keys())}")
        return self._apply_categorical_encoder(X)

    def _apply_categorical_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = X.select_dtypes(include="number").copy()
        if not self.one_hot_encode:
            return numeric

        encoded_parts = [numeric]
        for col, levels in self.categorical_levels_.items():
            values = X[col].fillna("__missing__").astype(str) if col in X.columns else pd.Series("__missing__", index=X.index)
            for level in levels:
                encoded_parts.append(pd.DataFrame({f"{col}__{level}": (values == level).astype(float)}))
        return pd.concat(encoded_parts, axis=1)

    def _drop_initial_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.initial_drop_columns if c in X.columns], errors="ignore")

    def _fit_winsor_bounds(self, X: pd.DataFrame) -> None:
        self.winsor_bounds_ = {}
        self.winsor_report_ = []
        for col in self.winsor_columns:
            if col in X.columns:
                low = float(X[col].quantile(0.01))
                high = float(X[col].quantile(0.99))
                self.winsor_bounds_[col] = (low, high)
                non_missing = int(X[col].notna().sum())
                clipped_low = int((X[col] < low).sum())
                clipped_high = int((X[col] > high).sum())
                clipped_total = clipped_low + clipped_high
                self.winsor_report_.append(
                    {
                        "column": col,
                        "p01": round(low, 6),
                        "p99": round(high, 6),
                        "below_p01": clipped_low,
                        "above_p99": clipped_high,
                        "clipped_total": clipped_total,
                        "clipped_%": round(clipped_total / non_missing * 100, 3) if non_missing else 0.0,
                    }
                )

    def _apply_winsor_bounds(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, (low, high) in self.winsor_bounds_.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=low, upper=high)
        return out

    def _fit_imputer(self, X: pd.DataFrame) -> None:
        self.impute_values_ = {}
        for col in X.columns:
            median = X[col].median(skipna=True)
            self.impute_values_[col] = 0.0 if pd.isna(median) else float(median)
        if self.imputation_method.lower() != "mice":
            self._log("Using initial median fallback only because MICE is disabled")
            self.mice_missing_columns_ = []
            self.mice_chains_ = []
            return

        self._fit_mice_imputer(X)

    def _apply_imputer(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.imputation_method.lower() == "mice" and self.mice_chains_:
            return self._apply_mice_imputer(X)

        out = X.copy()
        for col, value in self.impute_values_.items():
            if col in out.columns:
                out[col] = out[col].fillna(value)
        return out.fillna(0.0)

    def _build_imputation_report(self, before: pd.DataFrame, after: pd.DataFrame) -> None:
        self.imputation_report_ = []
        total = len(before)
        for col in before.columns:
            missing_mask = before[col].isna()
            missing_count = int(missing_mask.sum())
            if missing_count == 0:
                continue

            observed = before.loc[~missing_mask, col].dropna()
            imputed = after.loc[missing_mask, col].dropna()
            self.imputation_report_.append(
                {
                    "column": col,
                    "missing": missing_count,
                    "missing_%": round(missing_count / total * 100, 3) if total else 0.0,
                    "initial_median": round(float(self.impute_values_.get(col, 0.0)), 6),
                    "observed_mean": round(float(observed.mean()), 6) if len(observed) else float("nan"),
                    "imputed_mean": round(float(imputed.mean()), 6) if len(imputed) else float("nan"),
                    "imputed_std": round(float(imputed.std(ddof=0)), 6) if len(imputed) else float("nan"),
                    "imputed_min": round(float(imputed.min()), 6) if len(imputed) else float("nan"),
                    "imputed_max": round(float(imputed.max()), 6) if len(imputed) else float("nan"),
                }
            )

        self.imputation_report_.sort(key=lambda row: row["missing_%"], reverse=True)

    def _initial_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in out.columns:
            out[col] = out[col].fillna(self.impute_values_.get(col, 0.0))
        return out.fillna(0.0).astype(float)

    def _fit_mice_imputer(self, X: pd.DataFrame) -> None:
        self.mice_missing_columns_ = [col for col in X.columns if X[col].isna().any()]
        self.mice_chains_ = []
        if not self.mice_missing_columns_:
            return

        self._log(
            f"MICE settings: m={self.mice_imputations}, chained iterations={self.mice_iterations}, "
            f"ridge lambda={self.mice_regularization}, stochastic noise scale={self.mice_noise_scale}"
        )
        columns = X.columns.tolist()
        base = self._initial_impute(X)

        for imputation_idx in range(self.mice_imputations):
            rng = np.random.default_rng(self.random_state + imputation_idx)
            current = base.copy()
            chain: list[dict] = []

            # Tao cac dataset khoi tao khac nhau cho "multiple" imputations.
            for col in self.mice_missing_columns_:
                missing_mask = X[col].isna()
                observed = X.loc[~missing_mask, col].dropna()
                observed_std = float(observed.std(ddof=0)) if len(observed) > 1 else 0.0
                if observed_std > 1e-12 and missing_mask.any():
                    current.loc[missing_mask, col] = (
                        self.impute_values_[col]
                        + rng.normal(0.0, observed_std * self.mice_noise_scale, size=int(missing_mask.sum()))
                    )

            for iteration in range(self.mice_iterations):
                for target_col in self.mice_missing_columns_:
                    observed_mask = X[target_col].notna()
                    missing_mask = X[target_col].isna()
                    predictors = [col for col in columns if col != target_col]

                    step = self._fit_mice_step(current, X, target_col, predictors, observed_mask)
                    step["iteration"] = iteration
                    step["imputation"] = imputation_idx
                    chain.append(step)

                    if missing_mask.any():
                        predicted = self._predict_mice_step(current.loc[missing_mask, predictors], step)
                        residual_std = float(step.get("residual_std", 0.0))
                        if residual_std > 1e-12:
                            predicted = [
                                float(value) + float(rng.normal(0.0, residual_std * self.mice_noise_scale))
                                for value in predicted
                            ]
                        current.loc[missing_mask, target_col] = predicted

            self.mice_chains_.append(chain)

    def _fit_mice_step(
        self,
        current: pd.DataFrame,
        original: pd.DataFrame,
        target_col: str,
        predictors: list[str],
        observed_mask: pd.Series,
    ) -> dict:
        y_observed = original.loc[observed_mask, target_col].astype(float)
        if len(y_observed) < max(5, len(predictors) + 2):
            return {
                "target": target_col,
                "predictors": predictors,
                "kind": "constant",
                "value": float(y_observed.mean()) if len(y_observed) else self.impute_values_.get(target_col, 0.0),
                "residual_std": float(y_observed.std(ddof=0)) if len(y_observed) > 1 else 0.0,
            }

        X_observed = current.loc[observed_mask, predictors].astype(float)
        try:
            model = ridge_fit(
                X_observed.values.tolist(),
                y_observed.tolist(),
                lam=self.mice_regularization,
            )
            fitted = ridge_predict(
                X_observed.values.tolist(),
                model["beta_hat"],
                model["mean_X"],
                model["std_X"],
            )
            residuals = [actual - pred for actual, pred in zip(y_observed.tolist(), fitted)]
            residual_mean = sum(residuals) / len(residuals) if residuals else 0.0
            residual_std = math.sqrt(
                sum((value - residual_mean) ** 2 for value in residuals) / len(residuals)
            ) if residuals else 0.0
            return {
                "target": target_col,
                "predictors": predictors,
                "kind": "ridge",
                "model": model,
                "residual_std": residual_std,
            }
        except Exception:
            return {
                "target": target_col,
                "predictors": predictors,
                "kind": "constant",
                "value": float(y_observed.mean()),
                "residual_std": float(y_observed.std(ddof=0)) if len(y_observed) > 1 else 0.0,
            }

    @staticmethod
    def _predict_mice_step(X_predictors: pd.DataFrame, step: dict) -> list[float]:
        if len(X_predictors) == 0:
            return []
        if step["kind"] == "constant":
            return [float(step["value"]) for _ in range(len(X_predictors))]

        model = step["model"]
        return ridge_predict(
            X_predictors.astype(float).values.tolist(),
            model["beta_hat"],
            model["mean_X"],
            model["std_X"],
        )

    def _apply_mice_imputer(self, X: pd.DataFrame) -> pd.DataFrame:
        base = self._initial_impute(X)
        imputed_versions: list[pd.DataFrame] = []

        for chain in self.mice_chains_:
            current = base.copy()
            for step in chain:
                target_col = step["target"]
                if target_col not in X.columns:
                    continue
                missing_mask = X[target_col].isna()
                if not missing_mask.any():
                    continue

                predictors = [col for col in step["predictors"] if col in current.columns]
                if len(predictors) != len(step["predictors"]):
                    continue
                predicted = self._predict_mice_step(current.loc[missing_mask, predictors], step)
                current.loc[missing_mask, target_col] = predicted
            imputed_versions.append(current)

        if not imputed_versions:
            return base

        averaged = imputed_versions[0].copy()
        for extra in imputed_versions[1:]:
            averaged = averaged + extra
        averaged = averaged / len(imputed_versions)
        return averaged.reindex(columns=X.columns).fillna(0.0).astype(float)

    def _fit_vif_filter(self, X: pd.DataFrame) -> None:
        self.vif_drop_columns_ = []
        self.vif_history_ = []
        self.final_vif_ = {}
        working = X.copy()
        self._log(
            f"VIF rule: threshold={self.vif_threshold}, max iterative drops={self.max_vif_drops}"
        )

        for round_idx in range(self.max_vif_drops):
            if working.shape[1] < 2:
                break
            try:
                vif_values = part1_vif(working.values.tolist())
            except Exception as exc:
                self.vif_history_.append({"error": str(exc)})
                self._log(f"VIF filtering stopped because part1_vif raised: {exc}")
                break

            mapped = {
                working.columns[int(key[1:]) - 1]: float(value)
                for key, value in vif_values.items()
            }
            vif_table = (
                pd.DataFrame({"feature": list(mapped.keys()), "VIF": list(mapped.values())})
                .sort_values("VIF", ascending=False)
                .reset_index(drop=True)
            )
            self._log_table(f"VIF round {round_idx + 1} - highest values", vif_table, max_rows=self.log_top_n)
            worst_col, worst_vif = max(mapped.items(), key=lambda item: item[1])
            self.vif_history_.append({"feature": worst_col, "vif": worst_vif})
            if not math.isfinite(worst_vif) or worst_vif > self.vif_threshold:
                self.vif_drop_columns_.append(worst_col)
                working = working.drop(columns=[worst_col])
                self._log(f"Dropped '{worst_col}' due to VIF>{self.vif_threshold}")
            else:
                self.final_vif_ = mapped
                break

        if not self.final_vif_ and working.shape[1] >= 2:
            try:
                vif_values = part1_vif(working.values.tolist())
                self.final_vif_ = {
                    working.columns[int(key[1:]) - 1]: float(value)
                    for key, value in vif_values.items()
                }
            except Exception:
                self.final_vif_ = {}

    def _fit_scaler(self, X: pd.DataFrame) -> None:
        self.scale_mean_ = {}
        self.scale_std_ = {}
        for col in X.columns:
            mean = float(X[col].mean())
            std = float(X[col].std(ddof=0))
            self.scale_mean_[col] = mean
            self.scale_std_[col] = std if std > 1e-12 else 1.0

    def _apply_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.standardize:
            return X.astype(float)
        out = X.copy().astype(float)
        for col in out.columns:
            out[col] = (out[col] - self.scale_mean_.get(col, 0.0)) / self.scale_std_.get(col, 1.0)
        return out

    @staticmethod
    def _missing_report(df: pd.DataFrame) -> pd.DataFrame:
        total = len(df)
        return pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": df.isna().sum().values,
                "missing_percent": (df.isna().mean() * 100).round(3).values,
                "dtype": [str(dtype) for dtype in df.dtypes],
            }
        ).sort_values("missing_percent", ascending=False)

    def metadata(self) -> dict:
        return {
            "target": self.target,
            "target_transformed": self.target_transformed_,
            "feature_names": self.feature_names_,
            "log_columns": list(self.log_columns),
            "winsor_bounds": self.winsor_bounds_,
            "log_transform_report": self.log_transform_report_,
            "winsor_report": self.winsor_report_,
            "initial_drop_columns": list(self.initial_drop_columns),
            "vif_threshold": self.vif_threshold,
            "vif_drop_columns": self.vif_drop_columns_,
            "vif_history": self.vif_history_,
            "final_vif": self.final_vif_,
            "imputation_method": self.imputation_method,
            "initial_impute_values": self.impute_values_,
            "imputation_report": self.imputation_report_,
            "mice_imputations": self.mice_imputations,
            "mice_iterations": self.mice_iterations,
            "mice_regularization": self.mice_regularization,
            "mice_noise_scale": self.mice_noise_scale,
            "random_state": self.random_state,
            "mice_missing_columns": self.mice_missing_columns_,
            "standardize": self.standardize,
        }


def write_eda_outputs(df: pd.DataFrame, target: str, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes(include="number")

    paths: dict[str, str] = {}
    describe_path = output_dir / "describe_numeric.csv"
    numeric.describe().T.to_csv(describe_path)
    paths["describe_numeric"] = str(describe_path)

    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (df.isna().mean() * 100).round(3).values,
            "dtype": [str(dtype) for dtype in df.dtypes],
        }
    ).sort_values("missing_percent", ascending=False)
    missing_path = output_dir / "missing_values.csv"
    missing.to_csv(missing_path, index=False)
    paths["missing_values"] = str(missing_path)

    duplicate_path = output_dir / "duplicates.json"
    duplicate_path.write_text(
        json.dumps({"duplicate_rows": int(df.duplicated().sum()), "rows": int(len(df))}, indent=2),
        encoding="utf-8",
    )
    paths["duplicates"] = str(duplicate_path)

    if numeric.shape[1] > 0:
        sns.set_theme(style="whitegrid", font_scale=0.8)

        fig, axes = plt.subplots(math.ceil(numeric.shape[1] / 4), 4, figsize=(16, 3.0 * math.ceil(numeric.shape[1] / 4)))
        axes = _flatten_axes(axes)
        for i, col in enumerate(numeric.columns):
            axes[i].hist(numeric[col].dropna(), bins=40, color="#3b82f6", alpha=0.8)
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()
        hist_path = output_dir / "histograms.png"
        fig.savefig(hist_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths["histograms"] = str(hist_path)

        corr = numeric.corr(method="pearson")
        corr_path = output_dir / "correlation_matrix.csv"
        corr.to_csv(corr_path)
        paths["correlation_matrix"] = str(corr_path)

        corr_mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
        fig_size = max(9, 0.72 * len(corr.columns))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.82))
        sns.heatmap(
            corr,
            mask=corr_mask,
            cmap="RdBu",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7, "color": "black"},
            linewidths=0.45,
            linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.82, "ticks": [-1.0 + 0.2 * i for i in range(11)]},
            ax=ax,
        )
        ax.set_title("Correlation heatmap (Pearson)", pad=24)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", labelrotation=90, labelsize=8, length=0)
        ax.tick_params(axis="y", labelrotation=0, labelsize=8, length=0)
        ax.grid(False)
        fig.tight_layout()
        heatmap_path = output_dir / "correlation_heatmap.png"
        fig.savefig(heatmap_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths["correlation_heatmap"] = str(heatmap_path)

        if target in numeric.columns:
            corr_target = corr[target].drop(index=target).abs().sort_values(ascending=False).head(5)
            fig, axes = plt.subplots(1, len(corr_target), figsize=(4 * len(corr_target), 4))
            if len(corr_target) == 1:
                axes = [axes]
            for ax, col in zip(axes, corr_target.index):
                sub = numeric[[col, target]].dropna()
                ax.scatter(sub[col], sub[target], s=10, alpha=0.35, color="#0f766e")
                ax.set_xlabel(col)
                ax.set_ylabel(target)
                ax.set_title(f"r={corr.loc[col, target]:.3f}")
            fig.tight_layout()
            scatter_path = output_dir / "scatter_top5_target.png"
            fig.savefig(scatter_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            paths["scatter_top5_target"] = str(scatter_path)

    return paths


def _vif_mapping_for_frame(X: pd.DataFrame) -> dict[str, float]:
    if X.shape[1] < 2:
        return {}
    vif_values = part1_vif(X.values.tolist())
    return {
        col: float(vif_values.get(f"x{i + 1}", vif_values.get(i, vif_values.get(str(i), float("nan")))))
        for i, col in enumerate(X.columns)
    }


def write_preprocessing_diagnostic_plots(
    train_raw: pd.DataFrame,
    pipeline: DataPipeline,
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    sns.set_theme(style="whitegrid", font_scale=0.85)

    log_cols = [col for col in ("pl_orbper", "pl_insol", "pl_bmasse", "pl_trandep") if col in train_raw.columns]
    if log_cols:
        fig, axes = plt.subplots(len(log_cols), 2, figsize=(11, 2.7 * len(log_cols)))
        axes = _axes_grid(axes, len(log_cols), 2)
        for row_idx, col in enumerate(log_cols):
            before = pd.to_numeric(train_raw[col], errors="coerce").dropna()
            after = _safe_log1p(train_raw[col]).dropna()
            axes[row_idx][0].hist(before, bins=35, color="#2563eb", alpha=0.78)
            axes[row_idx][0].set_title(f"{col} - before")
            axes[row_idx][1].hist(after, bins=35, color="#0f766e", alpha=0.78)
            axes[row_idx][1].set_title(f"log_{col} - after")
        fig.suptitle("Before/after log-transform", y=1.01)
        fig.tight_layout()
        path = output_dir / "log_transform_before_after.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["log_transform_before_after"] = str(path)

    prepared = pipeline._prepare_frame(train_raw, fitting=False, include_target=True)
    X_stage = prepared.drop(columns=[pipeline.target_transformed_], errors="ignore")
    X_stage = pipeline._apply_categorical_encoder(X_stage)
    X_stage = pipeline._drop_initial_columns(X_stage)

    before_winsor = X_stage.copy()
    after_winsor = pipeline._apply_winsor_bounds(before_winsor)
    winsor_cols = [col for col in pipeline.winsor_columns if col in before_winsor.columns]
    if winsor_cols:
        fig, axes = plt.subplots(1, len(winsor_cols), figsize=(4 * len(winsor_cols), 4))
        axes = _flatten_axes(axes)
        for ax, col in zip(axes, winsor_cols):
            before = before_winsor[col].dropna()
            after = after_winsor[col].dropna()
            ax.boxplot([before, after], tick_labels=["Before", "After"], patch_artist=True)
            ax.set_title(col)
            ax.set_ylabel("Value")
        fig.suptitle("Before/after winsorization", y=1.02)
        fig.tight_layout()
        path = output_dir / "winsorization_before_after.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["winsorization_before_after"] = str(path)

    before_mice = after_winsor.copy()
    after_mice = pipeline._apply_imputer(before_mice.copy())
    mice_report = pd.DataFrame(pipeline.imputation_report_)
    if not mice_report.empty:
        mice_cols = [col for col in mice_report.sort_values("missing", ascending=False)["column"].head(4) if col in before_mice.columns]
        if mice_cols:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            axes = axes.reshape(-1)
            for ax, col in zip(axes, mice_cols):
                missing_mask = before_mice[col].isna()
                observed = before_mice.loc[~missing_mask, col].dropna()
                imputed = after_mice.loc[missing_mask, col].dropna()
                ax.hist(observed, bins=30, alpha=0.55, color="#2563eb", density=True, label="Observed")
                if not imputed.empty:
                    ax.hist(imputed, bins=18, alpha=0.65, color="#dc2626", density=True, label="MICE imputed")
                ax.set_title(col)
                ax.legend()
            for ax in axes[len(mice_cols):]:
                ax.set_visible(False)
            fig.suptitle("Observed vs MICE-imputed distributions", y=1.02)
            fig.tight_layout()
            path = output_dir / "mice_observed_vs_imputed.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths["mice_observed_vs_imputed"] = str(path)

    before_vif = _vif_mapping_for_frame(after_mice)
    after_vif = _vif_mapping_for_frame(after_mice.drop(columns=pipeline.vif_drop_columns_, errors="ignore"))
    if before_vif and after_vif:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
        before_top = pd.Series(before_vif).sort_values(ascending=False).head(8).sort_values()
        after_top = pd.Series(after_vif).sort_values(ascending=False).head(8).sort_values()
        axes[0].barh(before_top.index, before_top.values, color="#dc2626", alpha=0.82)
        axes[0].axvline(pipeline.vif_threshold, color="black", linestyle="--", linewidth=1)
        axes[0].set_title("Before VIF filtering")
        axes[0].set_xlabel("VIF")
        axes[1].barh(after_top.index, after_top.values, color="#0f766e", alpha=0.82)
        axes[1].axvline(pipeline.vif_threshold, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("After VIF filtering")
        axes[1].set_xlabel("VIF")
        fig.suptitle("VIF before/after filtering", y=1.02)
        fig.tight_layout()
        path = output_dir / "vif_before_after.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["vif_before_after"] = str(path)

    return paths


def run_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = ROOT_DIR / "part2" / "output",
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    make_plots: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_dataset(data_path)
    raw = select_model_columns(source, target=target)
    dropped_columns = source.shape[1] - raw.shape[1]
    model_input_shape = raw.shape
    schema_filter_columns = list(raw.columns)
    print(
        f"[Schema Filter] Source shape {source.shape} -> model columns {raw.shape}; "
        f"dropped {dropped_columns} non-model columns",
        flush=True,
    )

    raw = raw[(raw["pl_rade"] <= 1.6) & (raw["pl_bmasse"] <= 10)]
    print(f"[Domain Restriction] Rocky/Super-Earth samples kept: {len(raw)}", flush=True)
    
    raw = raw.dropna(subset=[target]).reset_index(drop=True)
    train_raw, test_raw = train_test_split_frame(raw, test_size=test_size, random_state=random_state)

    pipeline = DataPipeline(target=target, random_state=random_state)
    X_train, y_train = pipeline.fit_transform(train_raw)
    X_test, y_test = pipeline.transform(test_raw, include_target=True)
    if y_test is None:
        raise RuntimeError("Could not transform test target")

    eda_paths = {}
    if make_plots:
        eda_paths = write_eda_outputs(raw, target=target, output_dir=output_dir / "eda")
        eda_paths.update(write_preprocessing_diagnostic_plots(train_raw, pipeline, output_dir / "diagnostics"))

    payload = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "feature_names": pipeline.feature_names_,
        "target": target,
        "target_transformed": pipeline.target_transformed_,
        "metadata": pipeline.metadata(),
        "eda_paths": eda_paths,
    }
    payload["metadata"].update(
        {
            "source_data_path": str(Path(data_path)),
            "source_data_shape": list(source.shape),
            "model_input_shape_before_domain_filter": list(model_input_shape),
            "schema_filter_columns": schema_filter_columns,
            "schema_filter_dropped_columns": int(dropped_columns),
        }
    )

    with (output_dir / "preprocessed.pkl").open("wb") as f:
        pickle.dump(payload, f)

    (output_dir / "preprocessing_metadata.json").write_text(
        json.dumps(payload["metadata"], indent=2),
        encoding="utf-8",
    )
    if pipeline.missing_report_ is not None:
        pipeline.missing_report_.to_csv(output_dir / "missing_report_after_transforms.csv", index=False)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2 data preprocessing pipeline")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--outdir", default=str(ROOT_DIR / "part2" / "output"))
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(
        data_path=args.data,
        output_dir=args.outdir,
        target=args.target,
        test_size=args.test_size,
        make_plots=not args.skip_plots,
        random_state=args.random_state,
    )

    print("Preprocessing completed")
    print(f"  Train: {result['X_train'].shape}, Test: {result['X_test'].shape}")
    print(f"  Target: {result['target']} -> {result['target_transformed']}")
    print(f"  Random state: {result['metadata']['random_state']}")
    print(f"  Features: {result['feature_names']}")
    print(f"  VIF drops: {result['metadata']['vif_drop_columns']}")
    print(f"  Saved: {Path(args.outdir) / 'preprocessed.pkl'}")


if __name__ == "__main__":
    main()
