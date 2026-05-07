import unittest

import numpy as np

from part1.gauss_markov_demo import monte_carlo_gauss_markov
from part1.ols_implementation import coef_inference, vif
from test_utils import make_collinear_data, make_linear_data


class TestF4CoefInference(unittest.TestCase):
    def test_coef_inference_columns_and_index(self) -> None:
        X, y = make_linear_data(n=120, beta=[1.5, 2.0], sigma=0.3, seed=1)
        X_np = np.asarray(X, dtype=float)
        y_np = np.asarray(y, dtype=float)
        X_bias = np.column_stack([np.ones(len(X_np)), X_np])
        beta_hat = np.linalg.lstsq(X_bias, y_np, rcond=None)[0]
        y_hat = X_bias @ beta_hat
        sigma2 = float(np.sum((y_np - y_hat) ** 2) / (len(y_np) - X_bias.shape[1]))

        out = coef_inference(X, y, beta_hat.tolist(), sigma2)

        self.assertEqual(
            list(out.columns),
            ["coef", "std_err", "t_stat", "p_value", "ci_lower", "ci_upper"],
        )
        self.assertEqual(list(out.index), ["intercept", "x1"])

    def test_coef_inference_ci_contains_true_coefficients(self) -> None:
        true_beta = [2.0, -1.0, 0.7]
        X, y = make_linear_data(n=600, beta=true_beta, sigma=0.2, seed=11)
        X_np = np.asarray(X, dtype=float)
        y_np = np.asarray(y, dtype=float)
        X_bias = np.column_stack([np.ones(len(X_np)), X_np])
        beta_hat = np.linalg.lstsq(X_bias, y_np, rcond=None)[0]
        y_hat = X_bias @ beta_hat
        sigma2 = float(np.sum((y_np - y_hat) ** 2) / (len(y_np) - X_bias.shape[1]))

        out = coef_inference(X, y, beta_hat.tolist(), sigma2)

        for idx, truth in zip(["intercept", "x1", "x2"], true_beta):
            self.assertLessEqual(out.loc[idx, "ci_lower"], truth)
            self.assertGreaterEqual(out.loc[idx, "ci_upper"], truth)


class TestF5Vif(unittest.TestCase):
    def test_vif_returns_one_value_per_feature(self) -> None:
        X, _ = make_collinear_data(n=150, seed=7)
        out = vif(X)
        self.assertEqual(set(out.keys()), {"x1", "x2", "x3"})

    def test_vif_detects_collinearity(self) -> None:
        X, _ = make_collinear_data(n=200, seed=13)
        out = vif(X)
        self.assertTrue(any(v > 10 for v in out.values()))


class TestF10MonteCarlo(unittest.TestCase):
    def test_monte_carlo_returns_expected_shapes(self) -> None:
        result = monte_carlo_gauss_markov(n_sim=200, n_obs=80, random_state=42)
        self.assertEqual(result["beta_ols_arr"].shape, (200, 3))
        self.assertEqual(result["beta_alt_arr"].shape, (200, 3))
        self.assertEqual(list(result["summary"].index), ["intercept", "x1", "x2"])

    def test_monte_carlo_ols_is_nearly_unbiased(self) -> None:
        true_beta = (2.0, -1.5, 0.8)
        result = monte_carlo_gauss_markov(
            n_sim=800,
            n_obs=100,
            true_beta=true_beta,
            true_sigma=1.0,
            random_state=123,
        )
        mean_beta = result["beta_ols_arr"].mean(axis=0)
        self.assertTrue(np.all(np.abs(mean_beta - np.asarray(true_beta)) < 0.12))

    def test_monte_carlo_ols_has_lower_variance_than_alt(self) -> None:
        result = monte_carlo_gauss_markov(
            n_sim=600,
            n_obs=90,
            alt_scale=0.4,
            random_state=99,
        )
        summary = result["summary"]
        self.assertTrue(np.all(summary["ols_var"].values <= summary["alt_var"].values + 1e-12))


if __name__ == "__main__":
    unittest.main()
