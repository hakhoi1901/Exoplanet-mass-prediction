import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_utils import make_linear_data
from gauss_markov_demo import monte_carlo_gauss_markov, plot_beta_histograms
from residual_analysis import residual_plots
from ols_implementation import ols_fit
from ridge_lasso import ridge_trace, lasso_trace, ridge_fit, ridge_predict
from cross_validation import cv_lambda_search


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    print("=== START GENERATING PART 1 FIGURES ===")

    print("1. Generating Gauss-Markov figure (gauss_markov_histograms.png)...")
    res_gm = monte_carlo_gauss_markov(n_sim=1000, n_obs=100)
    plot_beta_histograms(
        res_gm["beta_ols_all"],
        res_gm["beta_alt_all"],
        res_gm["true_beta"],
        save_dir=out_dir,
    )

    print("2. Generating residual diagnostic figure (residual_plots.png)...")
    X_res, y_res = make_linear_data(n=200, beta=[2.0, 3.0, -1.5], sigma=1.0)
    res_ols = ols_fit(X_res, y_res)
    residual_plots(X_res, y_res, res_ols["beta_hat"], save_dir=out_dir)

    print("3. Generating Ridge Trace and Lasso Path figures...")
    X_rl, y_rl = make_linear_data(n=100, beta=[1.0, 0.5, -0.5, 2.0], sigma=1.0)
    ridge_trace(X_rl, y_rl, save_dir=out_dir)
    lasso_trace(X_rl, y_rl, save_dir=out_dir)

    print("4. Generating cross-validation figure (lambda_cv_score.png)...")

    def _predict(X_val, model):
        return ridge_predict(X_val, model["beta_hat"], model["mean_X"], model["std_X"])

    cv_lambda_search(X_rl, y_rl, model_fn=ridge_fit, predict_fn=_predict, k=5, save_dir=out_dir)

    print(f"=== DONE. Figures saved to: {out_dir} ===")


if __name__ == "__main__":
    main()
