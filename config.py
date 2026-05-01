import math

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────

RANDOM_STATE: int   = 42
EPSILON:      float = 1e-12

# ─────────────────────────────────────────────────────────────────
#  NUMERIC HELPERS  (tái sử dụng từ Utils.py)
# ─────────────────────────────────────────────────────────────────

def is_zero(x: float) -> bool:
    """Kiểm tra một số có xấp xỉ bằng 0 hay không."""
    return abs(x) < EPSILON

def zero_rectify(value: float) -> float:
    """Khử giá trị về 0 nếu nó đủ nhỏ."""
    return 0.0 if is_zero(value) else value

def calculate_relative_error(A: list, x_hat: list, b: list) -> float:
    """
    Tính sai số tương đối ||A·x_hat - b||₂ / ||b||₂.

    Args:
        A:     Ma trận hệ số  list[list[float]]  (n × m)
        x_hat: Vector nghiệm  list[float]         (m,)
        b:     Vector vế phải list[float]         (n,)

    Returns:
        float — sai số tương đối; 0.0 nếu b = 0 và r = 0.
    """
    n        = len(b)
    residual = [sum(A[i][j] * x_hat[j] for j in range(len(x_hat))) - b[i]
                for i in range(n)]
    norm_r   = math.sqrt(sum(r * r for r in residual))
    norm_b   = math.sqrt(sum(bi * bi for bi in b))
    if is_zero(norm_b):
        return 0.0 if is_zero(norm_r) else float('inf')
    return norm_r / norm_b
