#
# Newton-Raphson Method Report
#

# Aim:
# To find a real root of a non-linear equation using the Newton-Raphson iterative method.

# Problem Statement:
# Find a root of the equation f(x) = x^3 - x - 1 = 0, which is accurate to four decimal places.
# We will start with an initial guess of x0 = 1.0.

# Software Required:
# Python 3.x, math library

# Algorithm:
# 1. Define the function f(x) = x^3 - x - 1
# 2. Define the derivative of the function g(x) = f'(x) = 3x^2 - 1
# 3. Choose an initial guess (x0) and a tolerable error (e).
# 4. Start the iteration:
# 5. Calculate x1 = x0 - f(x0) / g(x0)
# 6. Check if abs(x1 - x0) is less than the tolerable error e.
# 7. If yes, x1 is the root. Stop.
# 8. If no, set x0 = x1 and repeat from step 5.

# --- Program (Python) ---

import math

def f(x):
    """The function for which we are finding the root."""
    return x**3 - x - 1

def g(x):
    """The derivative of the function f(x)."""
    return 3 * x**2 - 1

def newton_raphson(x0, e, max_iter=100):
    """
    Performs the Newton-Raphson iteration.
    :param x0: Initial guess
    :param e: Tolerable error (precision)
    :param max_iter: Maximum number of iterations
    :return: The root, or None if not found
    """
    print("--- Newton-Raphson Iteration ---")
    print("Iter\t x0\t\t f(x0)\t\t x1\t\t |x1 - x0|")
    print("-------------------------------------------------------------------")
    
    step = 1
    while step <= max_iter:
        if g(x0) == 0.0:
            print("Error: Derivative is zero. Cannot proceed.")
            return None
        
        x1 = x0 - f(x0) / g(x0)
        error = abs(x1 - x0)
        
        print(f"{step}\t {x0:.6f}\t {f(x0):.6f}\t {x1:.6f}\t {error:.6f}")
        
        if error < e:
            print("-------------------------------------------------------------------")
            return x1
            
        x0 = x1
        step += 1
        
    print("Iteration failed to converge within max iterations.")
    return None

# --- Main execution ---
if __name__ == "__main__":
    # Initial parameters
    initial_guess = 1.0
    tolerance = 0.0001
    
    root = newton_raphson(initial_guess, tolerance)
    
    print("\n--- Output ---")
    if root is not None:
        print(f"The root is: {root:.6f}")
    else:
        print("Root not found.")
        
    print("\n--- Result ---")
    if root is not None:
        print(f"The root of the equation x^3 - x - 1 = 0 is approximately {root:.4f}.")
    else:
        print("The method did not converge to a root.")
