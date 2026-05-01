from __future__ import annotations
import math
from config import is_zero, zero_rectify

def identity_matrix(n: int) -> list[list[float]]:
    """
    Tạo ma trận đơn vị kích thước n x n.

    Tham số:
        n: Kích thước của ma trận đơn vị cần tạo.

    Trả về:
        list[list[float]]: Ma trận đơn vị I kích thước n x n.
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def transpose(A: list[list[float]]) -> list[list[float]]:
    """
    Tính ma trận chuyển vị của ma trận A.

    Tham số:
        A: Ma trận đầu vào kích thước m x n.

    Trả về:
        list[list[float]]: Ma trận chuyển vị A^T kích thước n x m.
    """
    return [list(col) for col in zip(*A)]

def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Nhân hai ma trận A và B.

    Thuật toán:
    1. Kiểm tra tính tương thích của kích thước (số cột A = số dòng B).
    2. Khởi tạo ma trận kết quả C toàn số 0.
    3. Duyệt qua từng dòng i của A, từng cột k của A, và từng dòng j của B để cộng dồn tích aik * bkj.
    4. Tối ưu hóa: Bỏ qua các phần tử aik bằng 0.

    Tham số:
        A: Ma trận thứ nhất kích thước m x p.
        B: Ma trận thứ hai kích thước p x n.

    Trả về:
        list[list[float]]: Ma trận tích C = A x B kích thước m x n.

    Ngoại lệ:
        ValueError: Nếu số cột của A không bằng số dòng của B.
    """
    m = len(A)
    p = len(A[0])
    if len(B) != p:
        raise ValueError(f"Không thể nhân ma trận: kích thước không tương thích ({m}x{p} và {len(B)}x{len(B[0]) if B else 0})")

    n = len(B[0])
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for k in range(p):
            aik = A[i][k]
            if is_zero(aik):
                continue
            for j in range(n):
                C[i][j] += aik * B[k][j]
    return C

def matvec(A: list[list[float]], v: list[float]) -> list[float]:
    """
    Nhân ma trận A với vector v.

    Tham số:
        A: Ma trận kích thước m x n.
        v: Vector kích thước n.

    Trả về:
        list[float]: Vector kết quả b = A x v kích thước m.

    Ngoại lệ:
        ValueError: Nếu số cột của A không bằng kích thước của v.
    """
    m = len(A)
    n = len(A[0])
    if len(v) != n:
        raise ValueError("Kích thước ma trận và vector không tương thích")
    
    res = [0.0] * m
    for i in range(m):
        for j in range(n):
            res[i] += A[i][j] * v[j]
    return res

def dot_product(u: list[float], v: list[float]) -> float:
    """
    Tính tích vô hướng của hai vector u và v.

    Tham số:
        u: Vector thứ nhất.
        v: Vector thứ hai.

    Trả về:
        float: Giá trị tích vô hướng (scalar).
    """
    return sum(ui * vi for ui, vi in zip(u, v))

def vector_norm(v: list[float]) -> float:
    """
    Tính chuẩn Euclidean (L2 norm) của vector v.

    Tham số:
        v: Vector cần tính chuẩn.

    Trả về:
        float: Độ dài (chuẩn) của vector v.
    """
    return math.sqrt(max(0.0, dot_product(v, v)))

def normalize(v: list[float]) -> list[float]:
    """
    Chuẩn hóa vector v về đơn vị (unit vector).

    Tham số:
        v: Vector cần chuẩn hóa.

    Trả về:
        list[float]: Vector đã được chuẩn hóa có độ dài bằng 1. 
                    Trả về vector 0 nếu chuẩn ban đầu xấp xỉ 0.
    """
    nrm = vector_norm(v)
    if nrm < 1e-15:
        return [0.0] * len(v)
    return [x / nrm for x in v]

def is_upper_triangular(U: list[list[float]], tol: float = 1e-9) -> bool:
    """
    Kiểm tra xem ma trận U có phải là ma trận tam giác trên hay không.

    Tham số:
        U: Ma trận cần kiểm tra.
        tol: Độ dung sai cho phép các phần tử dưới đường chéo khác 0.

    Trả về:
        bool: True nếu các phần tử dưới đường chéo đều <= tol, ngược lại False.
    """
    m = len(U)
    n = len(U[0])
    for i in range(1, min(m, n)):
        for j in range(i):
            if abs(U[i][j]) > tol:
                return False
    return True

def check_identity(M: list[list[float]], tol: float = 1e-5) -> bool:
    """
    Kiểm tra xem ma trận M có xấp xỉ ma trận đơn vị hay không.

    Tham số:
        M: Ma trận cần kiểm tra.
        tol: Độ dung sai cho phép sai lệch so với ma trận đơn vị.

    Trả về:
        bool: True nếu M xấp xỉ ma trận đơn vị, ngược lại False.
    """
    n = len(M)
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(M[i][j] - expected) > tol:
                return False
    return True

def is_zero_tol(x: float, tol: float) -> bool:
    """
    Kiểm tra xem một số có gần bằng 0 hay không.

    Tham số:
        x: Số cần kiểm tra
        tol: Độ dung sai

    Trả về:
        True nếu x gần bằng 0, False ngược lại
    """
    return is_zero(x) or abs(x) <= tol

def rectify_matrix(M: list[list[float]]) -> list[list[float]]:
    """
    Chuẩn hóa ma trận M bằng cách loại bỏ các giá trị gần bằng 0.

    Tham số:
        M: Ma trận cần chuẩn hóa

    Trả về:
        Ma trận đã được chuẩn hóa
    """
    rectified: list[list[float]] = []
    for row in M:
        rectified_row: list[float] = []
        for value in row:
            v = zero_rectify(float(value))
            if is_zero(v):
                v = 0.0
            rectified_row.append(v)
        rectified.append(rectified_row)
    return rectified

def rectify_vector(v: list[float]) -> list[float]:
    """
    Chuẩn hóa vector v bằng cách loại bỏ các giá trị gần bằng 0.

    Tham số:
        v: Vector cần chuẩn hóa

    Trả về:
        Vector đã được chuẩn hóa
    """
    rectified: list[float] = []
    for value in v:
        x = zero_rectify(float(value))
        if is_zero(x):
            x = 0.0
        rectified.append(x)
    return rectified

def max_abs_diff(A: list[list[float]], B: list[list[float]]) -> float:
    """
    Tính sai số lớn nhất giữa hai ma trận.

    Tham số:
        A: Ma trận thứ nhất
        B: Ma trận thứ hai

    Trả về:
        Sai số lớn nhất

    Xử lý ngoại lệ:
        ValueError: Nếu hai ma trận không có cùng kích thước
    """
    if len(A) != len(B):
        raise ValueError("Hai ma trận phải có cùng số dòng")
    if A and len(A[0]) != len(B[0]):
        raise ValueError("Hai ma trận phải có cùng số cột")

    max_err = 0.0
    for i in range(len(A)):
        for j in range(len(A[0])):
            diff = abs(A[i][j] - B[i][j])
            if diff > max_err:
                max_err = diff
    return max_err

def orthogonalize(v: list[float], basis: list[list[float]]) -> list[float]:
    """
    Trực giao hóa vector v với cơ sở đã cho.

    Tham số:
        v: Vector cần trực giao hóa
        basis: Cơ sở đã cho

    Trả về:
        Vector đã được trực giao hóa
    """
    w = [float(x) for x in v]
    for b in basis:
        coeff = dot_product(w, b)
        if is_zero(coeff):
            continue
        for i in range(len(w)):
            w[i] = zero_rectify(w[i] - coeff * b[i])
    return w

def find_new_unit_vector(basis: list[list[float]], dim: int) -> list[float]:
    """
    Tìm vector đơn vị mới trực giao với cơ sở đã cho.

    Tham số:
        basis: Cơ sở đã cho
        dim: Kích thước của vector

    Trả về:
        Vector đơn vị mới

    Xử lý ngoại lệ:
        ValueError: Nếu không thể xây dựng cơ sở trực giao
    """
    for idx in range(dim):
        candidate = [0.0] * dim
        candidate[idx] = 1.0
        w = orthogonalize(candidate, basis)
        nw = vector_norm(w)
        if not is_zero(nw):
            return [zero_rectify(x / nw) for x in w]

    candidate = [float(i + 1) for i in range(dim)]
    w = orthogonalize(candidate, basis)
    nw = vector_norm(w)
    if is_zero(nw):
        raise ValueError("Không thể xây dựng cơ sở trực giao")
    return [zero_rectify(x / nw) for x in w]
