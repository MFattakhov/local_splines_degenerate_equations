import numpy as np
import sympy as sp
from tqdm import tqdm


class FEMSolver:
    """
    Finite Element Method solver for equations of the form:
    -D[x^alpha * D[u](x)] + u(x) = f(x) on interval (0,1)
    Uses piecewise linear basis functions with special handling for the singularity at x=0.
    """

    def __init__(
        self, h: float, alpha: float, x: sp.Symbol = None, bc_right: float = 0.0
    ):
        """
        Initialize the FEM solver.
        Args:
            h: Grid spacing (should divide 1 evenly)
            alpha: Exponent parameter in the differential operator
            x: Symbolic variable to use (if None, creates a new one)
            bc_right: Right boundary condition u(1) = bc_right
        """
        self.h = sp.Rational(h).limit_denominator(10**12)
        self.alpha = sp.Rational(alpha).limit_denominator(10**16)
        self.n = int(round(1 / h - 1))  # Number of interior nodes
        self.bc_right = bc_right

        # Use provided x or create new one
        if x is None:
            self.x = sp.Symbol("x", real=True)
        else:
            self.x = x

        # Define symbolic variables
        self._setup_symbolic_variables()
        self._setup_basis_functions()
        self._compute_matrix_elements()

        # Precompute the system matrix
        print("Assembling system matrix...")
        self.M = self._assemble_matrix()

    def _setup_symbolic_variables(self):
        """Setup symbolic variables for computation."""
        self.j = sp.symbols("j", integer=True)
        self.y = sp.symbols("y", real=True, positive=True)

    def _setup_basis_functions(self):
        """Define piecewise linear basis functions."""
        # Left piece: y (linear from 0 to 1)
        wlh = self.y
        # Right piece: 2-y (linear from 1 to 0)
        wrh = sp.S(2) - self.y

        # Transform to actual coordinates using specific h value
        self.w0lh_j = wlh.subs({self.y: (self.x / self.h - self.j)})
        self.w0rh_j = wrh.subs({self.y: (self.x / self.h - self.j)})
        self.w0lh_j_next = wlh.subs({self.y: (self.x / self.h - self.j - 1)})

        # Derivatives
        self.w0lh_j_p = sp.diff(self.w0lh_j, self.x)
        self.w0rh_j_p = sp.diff(self.w0rh_j, self.x)
        self.w0lh_j_p_next = sp.diff(self.w0lh_j_next, self.x)

    def _compute_matrix_elements(self):
        """Compute symbolic expressions for matrix elements."""
        print("Computing matrix elements...")

        # Bilinear form: ∫(x^α * u' * v' + u * v) dx

        # Diagonal elements (inner nodes)
        res1 = sp.integrate(
            self.x**self.alpha * self.w0lh_j_p**2 + self.w0lh_j**2,
            (self.x, self.h * self.j, self.h * (self.j + 1)),
        )
        res2 = sp.integrate(
            self.x**self.alpha * self.w0rh_j_p**2 + self.w0rh_j**2,
            (self.x, self.h * (self.j + 1), self.h * (self.j + 2)),
        )
        self.M_diag_inner = (res1 + res2).simplify()

        # First diagonal element (special handling for x=0)
        self.M_diag_first = (
            sp.integrate(
                self.x**self.alpha * self.w0rh_j_p**2 + self.w0rh_j**2,
                (self.x, sp.S(1) / 2**16, self.h),  # Avoid singularity at x=0
            )
            .subs({self.j: -1})
            .simplify()
        )

        # Last diagonal element (special handling for boundary at x=1)
        self.M_diag_last = sp.integrate(
            self.x**self.alpha * self.w0lh_j_p**2 + self.w0lh_j**2,
            (self.x, self.h * self.j, sp.S(1)),  # Integrate up to x=1
        ).simplify()

        # Off-diagonal elements
        self.M_offdiag = sp.integrate(
            self.x**self.alpha * self.w0rh_j_p * self.w0lh_j_p_next
            + self.w0rh_j * self.w0lh_j_next,
            (self.x, self.h * (self.j + 1), self.h * (self.j + 2)),
        ).simplify()

        # First off-diagonal element
        self.M_offdiag_first = (
            sp.integrate(
                self.x**self.alpha * self.w0rh_j_p * self.w0lh_j_p_next
                + self.w0rh_j * self.w0lh_j_next,
                (self.x, sp.S(1) / 2**16, self.h),
            )
            .subs({self.j: -1})
            .simplify()
        )

        # Last off-diagonal element (from second-to-last to last node)
        self.M_offdiag_last = sp.integrate(
            self.x**self.alpha * self.w0rh_j_p * self.w0lh_j_p_next
            + self.w0rh_j * self.w0lh_j_next,
            (self.x, self.h * (self.j + 1), sp.S(1)),  # Integrate up to x=1
        ).simplify()

    def _assemble_matrix(self):
        """Assemble the system matrix."""
        # If we have a non-zero boundary condition, we need to modify the system
        # The last node is fixed, so we solve for n nodes instead of n+1
        if self.bc_right != 0:
            matrix_size = self.n
        else:
            matrix_size = self.n + 1

        M = sp.zeros(matrix_size, matrix_size)

        if self.bc_right != 0:
            # Modified assembly for non-zero boundary condition
            assembly_iter = range(-1, self.n - 1)
        else:
            # Original assembly
            assembly_iter = range(-1, self.n)

        assembly_iter = tqdm(assembly_iter, desc="Assembling matrix")

        for j_val in assembly_iter:
            row = j_val + 1

            if row == 0:  # First row
                M[row, row] = self.M_diag_first
            elif (
                self.bc_right != 0 and row == self.n - 1
            ):  # Last row with boundary condition
                M[row, row] = self.M_diag_last.subs({self.j: j_val})
            else:  # Inner rows
                M[row, row] = self.M_diag_inner.subs({self.j: j_val})

            # Off-diagonal elements
            if row < matrix_size - 1:
                if row == 0:  # First off-diagonal
                    M[row, row + 1] = self.M_offdiag_first
                    M[row + 1, row] = self.M_offdiag_first
                elif (
                    self.bc_right != 0 and row == self.n - 2
                ):  # Last off-diagonal with boundary condition
                    offdiag_val = self.M_offdiag_last.subs({self.j: j_val})
                    M[row, row + 1] = offdiag_val
                    M[row + 1, row] = offdiag_val
                else:  # Other off-diagonals
                    offdiag_val = self.M_offdiag.subs({self.j: j_val})
                    M[row, row + 1] = offdiag_val
                    M[row + 1, row] = offdiag_val

        return M

    def _solve_tridiagonal(self, f0):
        """
        Solve tridiagonal system M * x = f0 using Thomas algorithm with high precision.
        Args:
            f0: Right-hand side vector (sympy expressions)
        Returns:
            Solution vector as numpy array of longdouble
        """
        n = len(f0)

        # Extract the diagonals from M
        a = [np.longdouble(0)]  # subdiagonal (below main diagonal)
        b = []  # main diagonal
        c = []  # superdiagonal (above main diagonal)

        # Extract the diagonals
        extraction_iter = range(n)
        extraction_iter = tqdm(extraction_iter, desc="Extracting diagonals")

        def transform(v):
            """Convert sympy expression to longdouble."""
            return np.longdouble(v.p) / np.longdouble(v.q)

        for i in extraction_iter:
            b.append(transform(self.M[i, i]))
            if i < n - 1:
                c.append(transform(self.M[i, i + 1]))
            if i > 0:
                a.append(transform(self.M[i, i - 1]))

        d = [transform(f0[i]) for i in range(n)]

        # Forward elimination
        forward_iter = range(1, n)
        forward_iter = tqdm(forward_iter, desc="Forward elimination")
        for i in forward_iter:
            w = a[i] / b[i - 1]
            b[i] = b[i] - w * c[i - 1]
            d[i] = d[i] - w * d[i - 1]

        # Back substitution
        x = [np.longdouble(0)] * n
        x[n - 1] = d[n - 1] / b[n - 1]
        back_iter = range(n - 2, -1, -1)
        back_iter = tqdm(back_iter, desc="Back substitution")
        for i in back_iter:
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

        return np.array(x, dtype=np.longdouble)

    def solve(self, f: sp.Expr):
        """
        Solve the differential equation for given right-hand side f.
        Args:
            f: Right-hand side function (sympy expression)
        Returns:
            Solution coefficients at grid points as numpy array of longdouble
        """
        # Assemble right-hand side vector
        f_vec = self._assemble_rhs(f)

        # Solve using high-precision tridiagonal solver
        return self._solve_tridiagonal(f_vec)

    def _assemble_rhs(self, f):
        """Assemble the right-hand side vector."""
        print("Assembling right-hand side vector...")

        if self.bc_right != 0:
            matrix_size = self.n
            rhs_iter = range(self.n)
        else:
            matrix_size = self.n + 1
            rhs_iter = range(self.n + 1)

        f_vec = sp.zeros(matrix_size, 1)
        rhs_iter = tqdm(rhs_iter, desc="Computing RHS integrals")

        for row in rhs_iter:
            if row == 0:
                # First element (special handling)
                f_vec[0] = sp.integrate(
                    f * self.w0rh_j.subs({self.j: -1}),
                    (self.x, sp.S(1) / 2**16, self.h),
                )
            elif self.bc_right != 0 and row == self.n - 1:
                # Last element with boundary condition
                j_val = row - 1

                # Left piece integral
                integral1 = sp.integrate(
                    f * self.w0lh_j.subs({self.j: j_val}),
                    (self.x, self.h * j_val, self.h * (j_val + 1)),
                )

                # Right piece integral (up to x=1)
                integral2 = sp.integrate(
                    f * self.w0rh_j.subs({self.j: j_val}),
                    (self.x, self.h * (j_val + 1), sp.S(1)),
                )

                # Subtract boundary condition contribution
                # The boundary condition contributes through the bilinear form
                bc_contribution = self.bc_right * self.M_offdiag_last.subs(
                    {self.j: j_val}
                )

                f_vec[row] = (integral1 + integral2 - bc_contribution).simplify()
            else:
                # Other elements
                j_val = row - 1

                # Left piece integral
                integral1 = sp.integrate(
                    f * self.w0lh_j.subs({self.j: j_val}),
                    (self.x, self.h * j_val, self.h * (j_val + 1)),
                )

                # Right piece integral
                integral2 = sp.integrate(
                    f * self.w0rh_j.subs({self.j: j_val}),
                    (self.x, self.h * (j_val + 1), self.h * (j_val + 2)),
                )

                f_vec[row] = (integral1 + integral2).simplify()

        print(f"f_vec: {f_vec}")
        return f_vec

    def get_solution_function(self, coefficients):
        """
        Get a vectorized callable function representing the solution.
        Args:
            coefficients: Solution coefficients from solve()
        Returns:
            Vectorized callable function that evaluates the solution at any point(s)
        """
        # Convert coefficients to numpy array and add boundary condition
        A0 = np.asarray(coefficients, dtype=np.longdouble)
        A0 = np.append(A0, [self.bc_right])  # Add boundary condition value

        h_float = float(self.h)

        def w0(y):
            """Base w0 function - fully vectorized"""
            y = np.asarray(y, dtype=np.longdouble)
            result = np.zeros_like(y)
            # Region 0 <= y <= 1: linear ramp up
            mask1 = (y >= 0) & (y <= 1)
            result[mask1] = y[mask1]
            # Region 1 < y <= 2: linear ramp down
            mask2 = (y > 1) & (y <= 2)
            result[mask2] = 2 - y[mask2]
            return result

        def make_phi0(j):
            """Create a shifted w0 function for specific j value"""

            def phi(x):
                y = np.asarray(x, dtype=np.longdouble) / h_float - j
                return w0(y)

            return phi

        # Create basis functions for j from -1 to n
        phi0_compiled = [make_phi0(j) for j in range(-1, self.n + 1)]

        def u_approx(x):
            """Approximate solution at point(s) x"""
            x = np.asarray(x, dtype=np.longdouble)
            scalar_input = x.ndim == 0
            if scalar_input:
                x = x.reshape(1)

            result = np.zeros_like(x, dtype=np.longdouble)

            for x_idx, x_val in enumerate(x):
                if x_val < 0 or x_val > 1:
                    result[x_idx] = 0
                    continue

                # Find nearby basis functions (only non-zero ones contribute)
                k = int(round(x_val / h_float))
                from_idx = max(0, k - 2)
                to_idx = min(len(A0), k + 2)

                # Sum contributions from nearby basis functions
                for idx in range(from_idx, to_idx):
                    result[x_idx] += A0[idx] * phi0_compiled[idx](x_val)

            return result[0] if scalar_input else result

        # Return vectorized version
        return np.vectorize(u_approx, otypes=[np.longdouble])


# Example usage
if __name__ == "__main__":
    # Create solver with different boundary conditions
    x = sp.Symbol("x")

    # Example 1: u(1) = 0 (original case)
    print("=== Case 1: u(1) = 0 ===")
    solver1 = FEMSolver(h=0.25, alpha=1.0, x=x, bc_right=0.0)
    f = x**2  # Example: f(x) = x^2
    solution_coeffs1 = solver1.solve(f)
    print("Solution coefficients:", solution_coeffs1)
    u_func1 = solver1.get_solution_function(solution_coeffs1)

    test_points = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for x_val in test_points:
        print(f"u({x_val}) = {u_func1(x_val)}")

    print("\n=== Case 2: u(1) = 2 ===")
    solver2 = FEMSolver(h=0.25, alpha=1.0, x=x, bc_right=2.0)
    solution_coeffs2 = solver2.solve(f)
    print("Solution coefficients:", solution_coeffs2)
    u_func2 = solver2.get_solution_function(solution_coeffs2)

    for x_val in test_points:
        print(f"u({x_val}) = {u_func2(x_val)}")

    print("\n=== Case 3: u(1) = -1.5 ===")
    solver3 = FEMSolver(h=0.25, alpha=1.0, x=x, bc_right=-1.5)
    solution_coeffs3 = solver3.solve(f)
    print("Solution coefficients:", solution_coeffs3)
    u_func3 = solver3.get_solution_function(solution_coeffs3)

    for x_val in test_points:
        print(f"u({x_val}) = {u_func3(x_val)}")
