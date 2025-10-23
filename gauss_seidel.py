#
# Gauss-Seidel Method Report
#

# Aim:
# To solve a system of linear equations using the Gauss-Seidel iterative method.

# Problem Statement:
# Solve the following system of linear equations:
#   10x +  y +  z = 12
#    x + 10y +  z = 12
#    x +  y + 10z = 12
# Start with an initial guess of x=0, y=0, z=0.

# Software Required:
# Python 3.x, NumPy library

# Algorithm:
# 1. Arrange the system of equations into a diagonally dominant form (if possible).
# 2. Express each variable from one equation (e.g., x from eq1, y from eq2, z from eq3).
#    x = (12 - y - z) / 10
#    y = (12 - x - z) / 10
#    z = (12 - x - y) / 10
# 3. Start with an initial guess [x, y, z] = [0, 0, 0].
# 4. In each iteration, calculate the new value for x, y, and z using the *most recent*
#    values of the other variables.
#    x_new = (12 - y_old - z_old) / 10
#    y_new = (12 - x_new - z_old) / 10  <-- Uses x_new
#    z_new = (12 - x_new - y_new) / 10  <-- Uses x_new and y_new
# 5. Repeat until the solution converges (i.e., the change between iterations
#    is below a specified tolerance).

# --- Program (Python) ---

import numpy as np

def gauss_seidel(A, b, initial_guess, tolerance=0.0001, max_iter=100):
    """
    Performs the Gauss-Seidel iteration.
    :param A: Coefficient matrix (NumPy array)
    :param b: Constant vector (NumPy array)
    :param initial_guess: Starting vector
    :param tolerance: Convergence criteria
    :param max_iter: Maximum iterations
    :return: Solution vector
    """
    n = len(b)
    x = initial_guess.copy()
    
    print("--- Gauss-Seidel Iteration ---")
    print("Iter\t x\t\t y\t\t z")
    print("-------------------------------------------------")
    print(f"0\t {x[0]:.6f}\t {x[1]:.6f}\t {x[2]:.6f}")

    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]
            
        print(f"{k+1}\t {x[0]:.6f}\t {x[1]:.6f}\t {x[2]:.6f}")
        
        # Check for convergence
        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            print("-------------------------------------------------")
            print(f"Converged after {k+1} iterations.")
            return x
            
    print("Failed to converge within maximum iterations.")
    return x

# --- Main execution ---
if __name__ == "__main__":
    # Coefficient matrix (A)
    A = np.array([
        [10.0, 1.0, 1.0],
        [1.0, 10.0, 1.0],
        [1.0, 1.0, 10.0]
    ])
    
    # Constant vector (b)
    b = np.array([12.0, 12.0, 12.0])
    
    # Initial guess
    initial_guess = np.zeros(len(b))
    
    solution = gauss_seidel(A, b, initial_guess)
    
    print("\n--- Output ---")
    if solution is not None:
        print(f"The solution is:")
        print(f"x = {solution[0]:.6f}")
        print(f"y = {solution[1]:.6f}")
        print(f"z = {solution[2]:.6f}")
    
    print("\n--- Result ---")
    if solution is not None:
        print(f"The solution to the system is approximately x=1.0, y=1.0, z=1.0.")
        print(f"The exact solution is x=1, y=1, z=1.")
    else:
        print("The method did not converge.")
