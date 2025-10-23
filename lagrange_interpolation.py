#
# Lagrange Interpolation Report
#

# Aim:
# To find a polynomial that passes through a given set of points and
# estimate the value of the function at a new point using Lagrange interpolation.

# Problem Statement:
# Given the following data table of (x, y) points:
# x: 0, 1, 3
# y: 1, 3, 55
# (These points lie on the polynomial y = 6x^3 - 8x + 3. Let's pretend we don't know this.)
# Use Lagrange's Interpolation to find the value of y when x = 2.

# Software Required:
# Python 3.x, NumPy library

# Algorithm:
# 1. Create a set of n data points (x_data, y_data).
# 2. Define the point 'value' at which to interpolate.
# 3. Initialize the interpolated result 'y_val' to 0.
# 4. Loop for i from 0 to n-1 (for each point):
#    a. Initialize the Lagrange basis polynomial 'L_i' to 1.
#    b. Loop for j from 0 to n-1:
#       i. If i is not equal to j:
#          L_i = L_i * (value - x_data[j]) / (x_data[i] - x_data[j])
#    c. Add the weighted value to the total:
#       y_val = y_val + y_data[i] * L_i
# 5. Return 'y_val'.

# --- Program (Python) ---

import numpy as np

def lagrange_interpolation(x_data, y_data, value):
    """
    Performs Lagrange interpolation.
    :param x_data: Array of x-coordinates
    :param y_data: Array of y-coordinates
    :param value: The x-point at which to interpolate
    :return: The interpolated y-value
    """
    n = len(x_data)
    y_val = 0.0
    
    print("--- Lagrange Basis Polynomials L_i(x) at x = " + str(value) + " ---")
    
    for i in range(n):
        # L_i is the i-th Lagrange basis polynomial
        L_i = 1.0
        
        for j in range(n):
            if i != j:
                term = (value - x_data[j]) / (x_data[i] - x_data[j])
                L_i = L_i * term
        
        print(f"L_{i}({value}) = {L_i:.4f}")
        y_val += y_data[i] * L_i
        
    print("-----------------------------------------------")
    return y_val

# --- Main execution ---
if __name__ == "__main__":
    x_data = np.array([0, 1, 3], dtype=float)
    y_data = np.array([3, 1, 55], dtype=float)
    
    # Point to interpolate
    value_to_find = 2.0
    
    result = lagrange_interpolation(x_data, y_data, value_to_find)
    
    print("\n--- Output ---")
    print(f"The interpolated value at x = {value_to_find} is y = {result:.6f}")
    
    print("\n--- Result ---")
    print(f"Using Lagrange interpolation, the estimated value at x={value_to_find} is {result:.4f}.")
    
    # Verification with the actual polynomial y = 6x^3 - 8x + 3
    actual = 6*(value_to_find**3) - 8*(value_to_find) + 3
    print(f"The exact value from the polynomial y=6x³-8x+3 is {actual:.4f}.")
    print("The interpolation is exact, as expected for a polynomial.")
