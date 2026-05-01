"""
test_utils.py — MTH00051 Đồ Án 2: Data Fitting & OLS
Shared testing utilities cho tất cả thành viên.

Cách dùng:
    from test_utils import (
        TestLogger, assert_close, assert_equal, assert_true,
        calculate_relative_error, make_linear_data,
        RANDOM_STATE, EPSILON, is_zero
    )
"""

import os
import math
import numpy as np
from config import *

os.system('')

# ─────────────────────────────────────────────────────────────────
#  DATA FACTORIES — tạo data test
# ─────────────────────────────────────────────────────────────────

def make_linear_data(
    n:     int   = 50,
    beta:  list  = None,
    sigma: float = 0.0,
    seed:  int   = RANDOM_STATE,
) -> tuple:
    """
    Sinh dataset tuyến tính  y = X·beta + noise.

    Args:
        n:     Số quan sát.
        beta:  [intercept, b1, b2, ...]  — mặc định [1.0, 2.0].
        sigma: Độ lệch chuẩn nhiễu. 0 = không noise (nghiệm exact).
        seed:  Random seed.

    Returns:
        (X, y)  —  list[list[float]], list[float]
        X chưa có cột bias, shape (n, p).
    """
    rng  = np.random.default_rng(seed)
    beta = np.asarray(beta if beta is not None else [1.0, 2.0], dtype=float)
    p    = len(beta) - 1
    X_np = rng.normal(size=(n, p))
    Xb   = np.column_stack([np.ones(n), X_np])
    y_np = Xb @ beta + (rng.normal(0, sigma, n) if sigma > 0 else 0.0)
    return X_np.tolist(), y_np.tolist()


def make_multifeature_data(
    n:     int   = 100,
    p:     int   = 4,
    sigma: float = 1.0,
    seed:  int   = RANDOM_STATE,
) -> tuple:
    """
    Sinh dataset hồi quy nhiều biến ngẫu nhiên.
    
    Args:
        n:     Số quan sát.
        p:     Số biến độc lập (không bao gồm bias).
        sigma: Độ lệch chuẩn nhiễu.
        seed:  Random seed.
        
    Returns:
        (X, y)  —  list[list[float]], list[float]
        X chưa có cột bias, shape (n, p).
    """
    rng  = np.random.default_rng(seed)
    X_np = rng.normal(size=(n, p))
    beta = rng.normal(size=p + 1)
    Xb   = np.column_stack([np.ones(n), X_np])
    y_np = Xb @ beta + rng.normal(0, sigma, n)
    return X_np.tolist(), y_np.tolist()


def make_collinear_data(
    n:    int = 100,
    seed: int = RANDOM_STATE,
) -> tuple:
    """
    Sinh dataset có đa cộng tuyến: x3 ≈ x1 + x2.
    
    Args:
        n:     Số quan sát.
        seed:  Random seed.
        
    Returns:
        (X, y)  —  list[list[float]], list[float]
        X chưa có cột bias, shape (n, 3).
    """
    rng = np.random.default_rng(seed)
    x1  = rng.normal(size=n)
    x2  = rng.normal(size=n)
    x3  = x1 + x2 + rng.normal(0, 0.01, n)
    X   = np.column_stack([x1, x2, x3])
    y   = 2*x1 - x2 + rng.normal(0, 0.5, n)
    return X.tolist(), y.tolist()


# ─────────────────────────────────────────────────────────────────
#  ASSERT HELPERS — trả về bool, không raise
# ─────────────────────────────────────────────────────────────────

def assert_close(
    actual,
    expected,
    label:   str   = "",
    rtol:    float = 1e-5,
    atol:    float = 1e-8,
) -> bool:
    """
    So sánh hai giá trị/array với sai số cho phép.
    In kết quả PASSED / FAILED.  Trả về True nếu pass.
    
    Args:
        actual:   Giá trị hoặc mảng thực tế.
        expected: Giá trị hoặc mảng kỳ vọng.
        label:    Nhãn mô tả test (hiển thị trong log).
        rtol:     Relative tolerance (dung sai tương đối).
        atol:     Absolute tolerance (dung sai tuyệt đối).
        
    Returns:
        True nếu hai giá trị/array bằng nhau trong dung sai cho phép,
        False nếu khác nhau.
    """
    try:
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            rtol=rtol, atol=atol,
        )
        TestLogger.print_result(label or "assert_close", passed=True,
                                details=f"rtol={rtol}, atol={atol}")
        return True
    except AssertionError as e:
        TestLogger.print_result(label or "assert_close", passed=False,
                                details=str(e).splitlines()[0][:80])
        return False


def assert_equal(actual, expected, label: str = "") -> bool:
    """
    So sánh bằng nhau (==). In PASSED / FAILED.
    
    Args:
        actual:   Giá trị hoặc mảng thực tế.
        expected: Giá trị hoặc mảng kỳ vọng.
        label:    Nhãn mô tả test (hiển thị trong log).
        
    Returns:
        True nếu hai giá trị/array bằng nhau, False nếu khác.
    """
    passed = (actual == expected)
    details = f"got {actual!r}" + ("" if passed else f", expected {expected!r}")
    TestLogger.print_result(label or "assert_equal", passed=passed, details=details)
    return passed


def assert_true(condition: bool, label: str = "", details: str = "") -> bool:
    """
    Kiểm tra điều kiện Boolean.
    
    Args:
        condition: Giá trị Boolean cần kiểm tra.
        label:     Nhãn mô tả test (hiển thị trong log).
        details:   Thông tin chi tiết (hiển thị khi fail).
        
    Returns:
        True nếu điều kiện đúng, False nếu sai.
    """
    TestLogger.print_result(label or "assert_true", passed=bool(condition),
                            details=details)
    return bool(condition)


def assert_shape(arr, expected_shape: tuple, label: str = "") -> bool:
    """
    Kiểm tra shape của array/list.
    
    Args:
        arr:            Array hoặc list đầu vào.
        expected_shape: Tuple shape kỳ vọng.
        label:          Nhãn mô tả test.
        
    Returns:
        True nếu shape khớp, False nếu khác.
    """
    actual = np.asarray(arr).shape
    passed = (actual == expected_shape)
    details = f"shape={actual}" + ("" if passed else f", expected={expected_shape}")
    TestLogger.print_result(label or "assert_shape", passed=passed, details=details)
    return passed


def assert_in_range(val: float, lo: float, hi: float, label: str = "") -> bool:
    """
    Kiểm tra lo ≤ val ≤ hi.
    
    Args:
        val:    Giá trị cần kiểm tra.
        lo:     Cận dưới của khoảng.
        hi:     Cận trên của khoảng.
        label:  Nhãn mô tả test.
        
    Returns:
        True nếu val nằm trong khoảng [lo, hi], False nếu không.
    """
    passed  = (lo <= val <= hi)
    details = f"{lo} ≤ {val:.6g} ≤ {hi}"
    TestLogger.print_result(label or "assert_in_range", passed=passed, details=details)
    return passed


def assert_raises(exc_type: type, fn, *args, label: str = "", **kwargs) -> bool:
    """
    Kiểm tra fn(*args) ném đúng loại exception.
    
    Args:
        exc_type: Loại exception kỳ vọng.
        fn:       Hàm cần gọi.
        *args:    Tham số Positional của fn.
        **kwargs: Tham số Keyword của fn.
        label:    Nhãn mô tả test.
        
    Returns:
        True nếu fn ném đúng exc_type, False nếu không.
    """
    try:
        fn(*args, **kwargs)
        TestLogger.print_result(
            label or f"assert_raises({exc_type.__name__})",
            passed=False, details="Không có exception nào được ném ra",
        )
        return False
    except exc_type:
        TestLogger.print_result(
            label or f"assert_raises({exc_type.__name__})",
            passed=True, details=f"Đúng: ném ra {exc_type.__name__}",
        )
        return True
    except Exception as e:
        TestLogger.print_result(
            label or f"assert_raises({exc_type.__name__})",
            passed=False, details=f"Sai loại: {type(e).__name__}: {e}",
        )
        return False


# ─────────────────────────────────────────────────────────────────
#  TestLogger
# ─────────────────────────────────────────────────────────────────

class TestLogger:
    """
    Giao diện in kết quả test ra terminal.
    Tất cả thành viên đều dùng class này — đảm bảo output đồng bộ.

    Endpoints:
        TestLogger.print_suite_header(suite_name)
        TestLogger.print_group(group_name)
        TestLogger.print_result(test_name, passed, details="")
        TestLogger.print_value(label, actual, expected=None)
        TestLogger.print_warn(message)
        TestLogger.print_info(message)
        TestLogger.print_summary(passed_count, total_count)
    """

    # ── ANSI codes ─────────────────────────────────
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    GRAY    = '\033[90m'
    WHITE   = '\033[97m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'
    RESET   = '\033[0m'

    # ── Ký hiệu ────────────────────────────────────
    _OK   = '[OK]'
    _FAIL = '[FAIL]'
    _WARN = '[WARN]'
    _INFO = '[INFO]'
    _SEP  = '─'
    _W    = 72   # độ rộng cố định

    # ─────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────

    @classmethod
    def print_suite_header(cls, suite_name: str) -> None:
        """
        In tiêu đề lớn — gọi một lần đầu mỗi hàm/nhóm test.

        Ví dụ:
            TestLogger.print_suite_header("F1 — ols_fit")
        """
        bar = cls._SEP * cls._W
        print(f"\n{cls.CYAN}{cls.BOLD}{bar}{cls.RESET}")
        print(f"{cls.CYAN}{cls.BOLD}  {suite_name}{cls.RESET}")
        print(f"{cls.CYAN}{bar}{cls.RESET}\n")

    @classmethod
    def print_group(cls, group_name: str) -> None:
        """
        In tiêu đề nhóm nhỏ, gọi trước một nhóm test liên quan.

        Ví dụ:
            TestLogger.print_group("Kiểm tra beta_hat")
        """
        print(f"\n  {cls.BOLD}{cls.WHITE}{group_name}{cls.RESET}")
        print(f"  {cls.DIM}{'·' * (cls._W - 4)}{cls.RESET}")

    @classmethod
    def print_result(cls, test_name: str, passed: bool, details: str = "") -> None:
        """
        In một dòng kết quả PASSED / FAILED.

        Args:
            test_name: Tên test hiển thị (tối đa ~45 ký tự).
            passed:    True = PASSED, False = FAILED.
            details:   Thông tin thêm (sai số, giá trị...).

        Ví dụ:
            TestLogger.print_result("beta_hat shape", True,  "shape=(3,)")
            TestLogger.print_result("beta_hat value", False, "got [1.1], expected [1.0]")
        """
        # Cắt tên nếu quá dài
        name = test_name if len(test_name) <= 45 else test_name[:42] + '...'
        name_col = f"{name:<45}"

        if passed:
            icon    = f"{cls.GREEN}{cls.BOLD}{cls._OK}{cls.RESET}"
            status  = f"{cls.GREEN}{cls.BOLD}PASSED{cls.RESET}"
            detail  = f"  {cls.DIM}{cls.GREEN}{details}{cls.RESET}" if details else ""
        else:
            icon    = f"{cls.RED}{cls.BOLD}{cls._FAIL}{cls.RESET}"
            status  = f"{cls.RED}{cls.BOLD}FAILED{cls.RESET}"
            detail  = f"  {cls.RED}{details}{cls.RESET}" if details else ""

        print(f"  {icon}  {name_col}  {status}{detail}")

    @classmethod
    def print_value(cls, label: str, actual, expected=None) -> None:
        """
        In giá trị actual (và expected nếu có) — tiện debug.

        Ví dụ:
            TestLogger.print_value("beta[0]",  1.0001, expected=1.0)
            TestLogger.print_value("sigma2",   0.9823)
            TestLogger.print_value("n_iter",   47,     expected="< 100")
        """
        line = f"  {cls.DIM}{cls.GRAY}  │  {label:<18}{cls.RESET}"
        line += f"  got = {cls.WHITE}{actual}{cls.RESET}"
        if expected is not None:
            line += f"  │  expected = {cls.CYAN}{expected}{cls.RESET}"
        print(line)

    @classmethod
    def print_warn(cls, message: str, detail: str = "") -> None:
        """
        In cảnh báo — không ảnh hưởng pass/fail, chỉ để lưu ý.

        Ví dụ:
            TestLogger.print_warn("VIF = 12.3 > 10", "feature: x3")
        """
        line = f"  {cls.YELLOW}{cls._WARN}  {message}{cls.RESET}"
        if detail:
            line += f"  {cls.DIM}{detail}{cls.RESET}"
        print(line)

    @classmethod
    def print_info(cls, message: str) -> None:
        """
        In thông tin phụ — màu xám nhạt, không phải pass/fail.

        Ví dụ:
            TestLogger.print_info("Dùng sklearn để kiểm chứng")
        """
        print(f"  {cls.DIM}{cls.GRAY}{cls._INFO}  {message}{cls.RESET}")

    @classmethod
    def print_summary(cls, passed_count: int, total_count: int) -> None:
        """
        In tổng kết cuối suite — gọi sau khi chạy hết test.

        Ví dụ:
            TestLogger.print_summary(passed, total)
        """
        bar    = cls._SEP * cls._W
        failed = total_count - passed_count
        all_ok = (failed == 0)

        print(f"\n{cls.CYAN}{bar}{cls.RESET}")

        # Progress bar đơn giản
        if total_count > 0:
            filled    = int((cls._W - 4) * passed_count / total_count)
            remaining = (cls._W - 4) - filled
            bar_str   = (f"{cls.GREEN}{'█' * filled}{cls.RESET}"
                         f"{cls.DIM}{'░' * remaining}{cls.RESET}")
            print(f"  {bar_str}")

        # Dòng số liệu
        print(
            f"  {cls.GREEN}{cls.BOLD}{cls._OK} Passed: {passed_count:<4}{cls.RESET}  "
            f"{cls.RED}{cls._FAIL} Failed: {failed:<4}{cls.RESET}  "
            f"{cls.GRAY}Total: {total_count}{cls.RESET}"
        )

        # Verdict
        print(f"{cls.CYAN}{bar}{cls.RESET}")
        if all_ok:
            msg = f"  {cls._OK}  TẤT CẢ {total_count} TEST PASSED"
            print(f"{cls.GREEN}{cls.BOLD}{msg}{cls.RESET}")
        else:
            msg = f"  {cls._FAIL}  {failed}/{total_count} TEST FAILED — kiểm tra lại!"
            print(f"{cls.RED}{cls.BOLD}{msg}{cls.RESET}")
        print()


# ─────────────────────────────────────────────────────────────────
#  SKLEARN VERIFIERS — kiểm chứng nhanh với thư viện chuẩn
# ─────────────────────────────────────────────────────────────────

def verify_vs_sklearn_ols(
    X:        list,
    y:        list,
    beta_hat: list,
    rtol:     float = 1e-4,
) -> bool:
    """
    So sánh beta_hat với sklearn.LinearRegression.
    Gọi sau ols_fit để xác nhận kết quả đúng.

    Returns:
        True nếu cả intercept lẫn coef đều khớp.
    """
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        TestLogger.print_warn("sklearn không có — bỏ qua kiểm chứng OLS")
        return True

    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y, dtype=float)
    sk   = LinearRegression().fit(X_np, y_np)
    bh   = np.asarray(beta_hat, dtype=float)

    ok1  = assert_close(bh[0],  sk.intercept_, label="vs sklearn — intercept", rtol=rtol)
    ok2  = assert_close(bh[1:], sk.coef_,      label="vs sklearn — coef",      rtol=rtol)
    return ok1 and ok2


def verify_vs_sklearn_ridge(
    X:        list,
    y:        list,
    beta_hat: list,
    lam:      float,
    rtol:     float = 1e-3,
) -> bool:
    """
    So sánh beta_hat với sklearn.Ridge(alpha=lam).

    Returns:
        True nếu cả intercept lẫn coef đều khớp.
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        TestLogger.print_warn("sklearn không có — bỏ qua kiểm chứng Ridge")
        return True

    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y, dtype=float)
    sk   = Ridge(alpha=lam, fit_intercept=True).fit(X_np, y_np)
    bh   = np.asarray(beta_hat, dtype=float)

    ok1  = assert_close(bh[0],  sk.intercept_, label=f"vs sklearn Ridge(λ={lam}) — intercept", rtol=rtol)
    ok2  = assert_close(bh[1:], sk.coef_,      label=f"vs sklearn Ridge(λ={lam}) — coef",      rtol=rtol)
    return ok1 and ok2


# ─────────────────────────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    passed = 0
    total  = 0

    def run(result: bool):
        global passed, total
        total  += 1
        passed += int(result)

    # ── Suite F1: ols_fit ────────────────────────────
    TestLogger.print_suite_header("F1 — ols_fit  |  Normal Equations & residuals")

    TestLogger.print_group("Output format")
    run(assert_true(True,  label="output có key 'beta_hat'"))
    run(assert_true(False,  label="output có key 'sigma2_hat'"))
    run(assert_true(True,  label="output có key 'y_hat'"))
    run(assert_true(True,  label="output có key 'residuals'"))

    TestLogger.print_group("Nghiệm exact  (sigma = 0)")
    X, y = make_linear_data(n=20, beta=[1.0, 2.0], sigma=0.0)
    # Giả lập kết quả đúng để demo
    Xb   = np.column_stack([np.ones(len(X)), X])
    beta_fake = np.linalg.lstsq(Xb, y, rcond=None)[0].tolist()
    run(assert_close(beta_fake, [1.0, 2.0], label="beta = [1.0, 2.0]", atol=1e-7))
    run(assert_shape(beta_fake, (2,),        label="beta shape = (p+1,)"))

    TestLogger.print_group("Kiểm chứng với sklearn")
    TestLogger.print_info("Dùng sklearn.LinearRegression để verify")
    run(verify_vs_sklearn_ols(X, y, beta_fake))

    TestLogger.print_group("Các giá trị debug")
    TestLogger.print_value("beta[0]",   beta_fake[0],  expected=1.0)
    TestLogger.print_value("beta[1]",   beta_fake[1],  expected=2.0)
    TestLogger.print_value("sigma2",    0.0,           expected="≈ 0")
    TestLogger.print_warn("Dataset nhỏ — chỉ dùng để test giao diện", "n=20")

    # ── Suite F3: model_metrics ──────────────────────
    TestLogger.print_suite_header("F3 — model_metrics  |  R², RSS, TSS, F-stat")

    TestLogger.print_group("R² và các chỉ số cơ bản")
    run(assert_in_range(0.87,  0.0, 1.0,  label="R² ∈ [0, 1]"))
    run(assert_in_range(0.85,  0.0, 1.0,  label="R²_adj ∈ [0, 1]"))
    run(assert_true(True,                  label="RSS + MSS = TSS"))

    TestLogger.print_group("Test FAILED (demo)")
    run(assert_close([1.0, 2.0], [1.0, 2.1], label="beta không khớp (demo FAILED)", atol=1e-8))
    run(assert_equal(5, 4,                    label="shape không khớp  (demo FAILED)"))

    # ── Tổng kết ────────────────────────────────────
    TestLogger.print_summary(passed, total)