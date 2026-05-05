import numpy as np
from typing import Any

def soft_threshold(rho: float, lam: float) -> float:
    """Hàm soft-thresholding cho Lasso."""
    if rho > lam:
        return rho - lam
    elif rho < -lam:
        return rho + lam
    else:
        return 0.0

def ridge_fit(X: list[list[float]], y: list[float], lam: float = 1.0) -> dict[str, Any]:
    """
    F6: Ridge Regression - Closed-form solution.
    Công thức: beta = (X.T @ X + lam * I)^-1 @ X.T @ y
    """
    X_np = np.array(X)
    y_np = np.array(y)
    n, p = X_np.shape

    # 1. Chuẩn hóa X (trừ bias)
    mean_X = np.mean(X_np, axis=0)
    std_X = np.std(X_np, axis=0)
    X_scaled = (X_np - mean_X) / std_X

    # 2. Thêm cột bias
    X_bias = np.column_stack([np.ones(n), X_scaled])
    
    # 3. Tạo ma trận I (p+1, p+1) và đặt I[0,0] = 0 để không hiệu chỉnh intercept
    I = np.eye(p + 1)
    I[0, 0] = 0
    
    # 4. Giải hệ phương trình (X.T @ X + lam * I) @ beta = X.T @ y
    A = X_bias.T @ X_bias + lam * I
    b = X_bias.T @ y_np
    beta_hat = np.linalg.solve(A, b) # Sau này thay bằng solve_system từ Utils.py
    
    return {
        'beta_hat': beta_hat.tolist(),
        'y_hat': (X_bias @ beta_hat).tolist(),
        'mean_X': mean_X.tolist(),
        'std_X': std_X.tolist()
    }

def lasso_fit(X: list[list[float]], y: list[float], lam: float = 1.0, max_iter: int = 1000, tol: float = 1e-6) -> dict[str, Any]:
    """
    F7: Lasso Regression - Coordinate Descent.
    """
    X_np = np.array(X)
    y_np = np.array(y)
    n, p = X_np.shape

    # Chuẩn hóa
    mean_X = np.mean(X_np, axis=0); std_X = np.std(X_np, axis=0)
    X_scaled = (X_np - mean_X) / std_X
    
    # Khởi tạo
    intercept = np.mean(y_np)
    beta = np.zeros(p)
    y_centered = y_np - intercept
    
    for i in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            # Tính partial residual
            r_j = y_centered - (X_scaled @ beta - X_scaled[:, j] * beta[j])
            rho_j = X_scaled[:, j] @ r_j
            z_j = np.sum(X_scaled[:, j]**2)
            
            beta[j] = soft_threshold(rho_j, lam) / z_j
            
        if np.max(np.abs(beta - beta_old)) < tol:
            break
            
    beta_hat = [intercept] + beta.tolist()
    return {'beta_hat': beta_hat, 'n_iter': i + 1}

# --- Unit Tests ---
def test_ridge_fit():
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0]]
    y = [5.0, 4.0, 10.0]
    res = ridge_fit(X, y, lam=0.1)
    assert len(res['beta_hat']) == 3
    print("test_ridge_fit: PASSED")