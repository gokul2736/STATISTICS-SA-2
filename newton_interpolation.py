#
# Newton's Forward and Backward Interpolation Report
#

# Aim:
# To estimate the value of a function at a specific point using
# Newton's forward and backward difference interpolation formulas.

# Problem Statement:
# Given the following data table of (x, y) points:
# x: 0, 1, 2, 3
# y: 1, 2, 4, 7
# 1. Use Newton's Forward Interpolation to find y(0.5) (near the start).
# 2. Use Newton's Backward Interpolation to find y(2.5) (near the end).

# Software Required:
# Python 3.x, NumPy library

# Algorithm:
# 1. Create a set of n data points (x, y).
# 2. Generate the difference table.
# 3. For Forward Interpolation (estimating near the start of the table):
#    a. Calculate 'u' = (value - x[0]) / (x[1] - x[0])
#    b. Apply the formula:
#       y_val = y[0] + u*delta_y[0] + (u*(u-1)/2!)*delta_2_y[0] + ...
# 4. For Backward Interpolation (estimating near the end of the table):
#    a. Calculate 'u' = (value - x[n-1]) / (x[1] - x[0])
#    b. Apply the formula:
#       y_val = y[n-1] + u*nabla_y[n-1] + (u*(u+1)/2!)*nabla_2_y[n-1] + ...

# --- Program (Python) ---

import numpy as np

def create_difference_table(y):
    """Creates a forward difference table."""
    n = len(y)
    # Initialize a 2D array (n x n) with zeros
    table = np.zeros((n, n))
    # First column is the y values
    table[:, 0] = y
    
    for j in range(1, n):  # Column
        for i in range(n - j):  # Row
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]
            
    return table

def factorial(n):
    """Helper function to calculate factorial."""
    if n == 0:
        return 1
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res

def newton_forward(x_data, diff_table, value):
    """Calculates interpolated value using forward formula."""
    n = len(x_data)
    h = x_data[1] - x_data[0] # Assumes equally spaced x data
    u = (value - x_data[0]) / h
    
    y_val = diff_table[0, 0]  # This is y[0]
    
    print("\n--- Calculating Forward Interpolation ---")
    print(f"y({value}) = {y_val:.4f}")
    
    for i in range(1, n):
        u_term = u
        for j in range(1, i):
            u_term = u_term * (u - j)
        
        term_val = (u_term / factorial(i)) * diff_table[0, i]
        y_val += term_val
        print(f" + ({u_term:.4f} / {factorial(i)}) * {diff_table[0, i]:.0f}  =  {y_val:.4f}")
        
    return y_val

def newton_backward(x_data, diff_table, value):
    """Calculates interpolated value using backward formula."""
    n = len(x_data)
    h = x_data[1] - x_data[0] # Assumes equally spaced x data
    u = (value - x_data[n - 1]) / h
    
    y_val = diff_table[n - 1, 0]  # This is y[n-1]
    
    print("\n--- Calculating Backward Interpolation ---")
    print(f"y({value}) = {y_val:.4f}")
    
    for i in range(1, n):
        u_term = u
        for j in range(1, i):
            u_term = u_term * (u + j)
            
        term_val = (u_term / factorial(i)) * diff_table[n - 1 - i, i]
        y_val += term_val
        print(f" + ({u_term:.4f} / {factorial(i)}) * {diff_table[n - 1 - i, i]:.0f}  =  {y_val:.4f}")
        
    return y_val

# --- Main execution ---
if __name__ == "__main__":
    x_data = np.array([0, 1, 2, 3], dtype=float)
    y_data = np.array([1, 2, 4, 7], dtype=float)
    
    # 1. Generate the difference table
    diff_table = create_difference_table(y_data)
    
    print("--- Difference Table (Forward) ---")
    print("y\t Δy\t Δ²y\t Δ³y")
    print("---------------------------------")
    # This print logic works for the forward table display
    for i in range(len(y_data)):
        for j in range(len(y_data) - i):
            print(f"{diff_table[i, j]:.0f}", end="\t ")
        print("")
    print("---------------------------------")

    
    # 2. Problem 1: Newton's Forward Interpolation for y(0.5)
    val_forward = 0.5
    result_forward = newton_forward(x_data, diff_table, val_forward)
    
    # 3. Problem 2: Newton's Backward Interpolation for y(2.5)
    val_backward = 2.5
    result_backward = newton_backward(x_data, diff_table, val_backward)

    print("\n--- Output ---")
    print(f"Final Forward interpolation for y({val_forward}): {result_forward:.6f}")
    print(f"Final Backward interpolation for y({val_backward}): {result_backward:.6f}")
    
    print("\n--- Result ---")
    print(f"1. Using Newton's forward interpolation, the estimated value at x=0.5 is {result_forward:.4f}.")
    print(f"2. Using Newton's backward interpolation, the estimated value at x=2.5 is {result_backward:.4f}.")

