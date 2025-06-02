import numpy as np
import sympy as sp

from fem_solver import FEMSolver


def test_fem_solver_accuracy():
    """Test the FEM solver with known analytical solution."""
    # Parameters
    alpha = sp.Rational("1.0")
    h = sp.Rational("0.2")

    # Define symbolic variable
    x = sp.Symbol("x", real=True, positive=True)

    # Define the test problem
    f_exact = (x ** (3 - alpha) - 2 * (3 - alpha) * x - 1) / (3 - alpha)
    u_exact = (x ** (3 - alpha) - 1) / (3 - alpha)

    # Create solver
    solver = FEMSolver(h=h, alpha=alpha, x=x)

    # Basic solver setup assertions
    assert solver.n > 0, "Grid size should be positive"
    assert solver.h == h, "Step size should match input"
    assert solver.alpha == alpha, "Alpha parameter should match input"

    # Solve the system
    solution_coeffs = solver.solve(f_exact)

    # Assert solution exists and has correct dimensions
    assert solution_coeffs is not None, "Solution should exist"
    assert len(solution_coeffs) == solver.n + 1, "Solution should have n+1 coefficients"

    # Get solution function
    u_numerical = solver.get_solution_function(solution_coeffs)
    assert callable(u_numerical), "Solution function should be callable"

    # Create exact solution function
    u_exact_func = sp.lambdify(x, u_exact, "numpy")

    # Test points (avoid endpoints)
    x_test = np.linspace(0.01, 0.99, 100)

    # Evaluate solutions
    u_num_vals = np.array([u_numerical(xi) for xi in x_test])
    u_exact_vals = u_exact_func(x_test)

    # Basic shape assertions
    assert (
        u_num_vals.shape == u_exact_vals.shape
    ), "Solution arrays should have same shape"
    assert len(u_num_vals) == len(
        x_test
    ), "Solution should be evaluated at all test points"

    # Compute errors
    abs_errors = np.abs(u_num_vals - u_exact_vals)
    max_error = np.max(abs_errors)
    mean_error = np.mean(abs_errors)

    # Error assertions
    assert not np.isnan(max_error), "Maximum error should not be NaN"
    assert not np.isnan(mean_error), "Mean error should not be NaN"
    assert max_error >= 0, "Maximum error should be non-negative"
    assert mean_error >= 0, "Mean error should be non-negative"

    # Accuracy assertion (main test criterion)
    assert max_error <= 0.01, f"Maximum absolute error {max_error:.6f} should be ≤ 0.01"

    # Additional reasonable bounds
    assert mean_error <= max_error, "Mean error should not exceed maximum error"
    assert max_error < 1.0, "Maximum error should be reasonable (< 1.0)"


def test_fem_solver_boundary_conditions():
    """Test that boundary conditions are satisfied."""
    alpha = sp.Rational("1.0")
    h = sp.Rational("0.1")
    x = sp.Symbol("x", real=True, positive=True)

    # Simple test function
    f_test = x**2

    solver = FEMSolver(h=h, alpha=alpha, x=x)
    solution_coeffs = solver.solve(f_test)
    u_numerical = solver.get_solution_function(solution_coeffs)

    # Test boundary conditions u(1) = 0
    # Note: We test very close to boundaries since exact boundaries might not be defined
    assert (
        abs(u_numerical(0.999)) < 0.01
    ), "Solution should be close to 0 at right boundary"
