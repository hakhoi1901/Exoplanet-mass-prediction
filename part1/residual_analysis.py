import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def residual_plots(y: list[float], y_hat: list[float], X: list[list[float]] = None):
    """
    F8: Vẽ 4 biểu đồ chẩn đoán phần dư.
    """
    y = np.array(y); y_hat = np.array(y_hat)
    residuals = y - y_hat
    n = len(y)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_hat, residuals, alpha=0.5)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_title("Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")

    # 2. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q")

    # 3. Scale-Location
    residuals_std = (residuals - np.mean(residuals)) / np.std(residuals)
    axes[1, 0].scatter(y_hat, np.sqrt(np.abs(residuals_std)), alpha=0.5)
    axes[1, 0].set_title("Scale-Location")
    
    # 4. Cook's Distance (Sơ khai)
    axes[1, 1].stem(np.arange(n), np.abs(residuals), markerfmt=" ")
    axes[1, 1].set_title("Residuals Leverage (Cook's D Placeholder)")

    plt.tight_layout()
    plt.show()