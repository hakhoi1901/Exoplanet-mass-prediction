from __future__ import annotations


def _assert_embedded_suite(module_name: str, run_tests) -> None:
    passed, total = run_tests()
    assert passed == total, f"{module_name}: {passed}/{total} embedded tests passed"


def test_ols_implementation_embedded_suite() -> None:
    from ols_implementation import run_tests

    _assert_embedded_suite("ols_implementation.py", run_tests)


def test_ridge_lasso_embedded_suite() -> None:
    from ridge_lasso import run_tests

    _assert_embedded_suite("ridge_lasso.py", run_tests)


def test_cross_validation_embedded_suite() -> None:
    from cross_validation import run_tests

    _assert_embedded_suite("cross_validation.py", run_tests)


def test_residual_analysis_embedded_suite() -> None:
    from residual_analysis import run_tests

    _assert_embedded_suite("residual_analysis.py", run_tests)


def test_gauss_markov_embedded_suite() -> None:
    from gauss_markov_demo import run_tests

    _assert_embedded_suite("gauss_markov_demo.py", run_tests)
