#
# Euler's Method Report
#

# Aim:
# To find the numerical solution of a first-order ordinary differential
# equation (ODE) with a given initial value using Euler's method.

# Problem Statement:
# Given the ODE: dy/dx = x + y
# With the initial condition: y(0) = 1
# Find the approximate value of y(1) using Euler's method with a step size h = 0.1.

# Software Required:
# Python 3.x, NumPy, Matplotlib

# Algorithm:
# 1. Define the function f(x, y) = x + y.
# 2. Set initial values x0 = 0, y0 = 1.
# 3. Set step size h = 0.1 and target x_target = 1.
# 4. Calculate the number of steps: n = (x_target - x0) / h.
# 5. Start iteration from i = 0 to n-1:
#    a. Calculate y_next = y_current + h * f(x_current, y_current)
#    b. Calculate x_next = x_current + h
#    c. Set y_current = y_next, x_current = x_next
# 6. The final y_next is the approximation for y(1).

# --- Program (Python) ---

import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    """The differential equation dy/dx = f(x, y)"""
    return x + y

def euler_method(x0, y0, h, x_target):
    """
    Solves an ODE using Euler's method.
    :param x0: Initial x value
    :param y0: Initial y value (y(x0))
    :param h: Step size
    :param x_target: The x-value at which to find y
    :return: Arrays of x and y values
    """
    n_steps = int((x_target - x0) / h)
    
    x = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    
    x[0] = x0
    y[0] = y0
    
    print("--- Euler's Method Iteration ---")
    print("Step\t x_n\t\t y_n (approx)")
    print("---------------------------------------")
    print(f"0\t {x[0]:.2f}\t\t {y[0]:.6f}")

    for i in range(n_steps):
        y[i+1] = y[i] + h * f(x[i], y[i])
        x[i+1] = x[i] + h
        print(f"{i+1}\t {x[i+1]:.2f}\t\t {y[i+1]:.6f}")
        
    print("---------------------------------------")
    return x, y

def exact_solution(x):
    """Exact solution y(x) = 2e^x - x - 1 for dy/dx = x+y, y(0)=1"""
    return 2 * np.exp(x) - x - 1

# --- Main execution ---
if __name__ == "__main__":
    # Initial conditions
    x0 = 0.0
    y0 = 1.0
    
    # Parameters
    h = 0.1
    x_target = 1.0
    
    x_euler, y_euler = euler_method(x0, y0, h, x_target)
    
    # Get the final value
    y_final = y_euler[-1]
    
    print("\n--- Output ---")
    print(f"The approximate value of y({x_target}) is: {y_final:.6f}")
    
    # --- Result ---
    y_exact = exact_solution(x_target)
    error = abs(y_exact - y_final)
    print("\n--- Result ---")
    print(f"Using Euler's method with h={h}, the approximate value of y(1) is {y_final:.4f}.")
    print(f"The exact value is {y_exact:.4f}.")
    print(f"The absolute error is {error:.4f}.")
    
    # --- Visualization ---
    x_exact = np.linspace(x0, x_target, 50)
    y_exact_vals = exact_solution(x_exact)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_euler, y_euler, 'bo--', label=f"Euler's Method (h={h})", markersize=4)
    plt.plot(x_exact, y_exact_vals, 'r-', label="Exact Solution (y = 2e^x - x - 1)")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Euler's Method vs. Exact Solution for dy/dx = x + y")
    plt.legend()
    plt.grid(True)
    plt.savefig("euler_method_plot.png")
    print("\nPlot saved as 'euler_method_plot.png'")
