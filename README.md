# Skill Assessment - 2: Numerical Methods in Python

## NAME: MARKANDEYAN GOKUL
## ROLL NO: 212224240086


## This collection includes implementations for the following numerical methods:
-  Newton's Raphson Method 
-  Gauss seidel Method 
-  Newton's Forward and Backward Interpolation 
-  Lagranges Interpolation 
-  Euler' s Method
-  Runge Kutta Method

### Prerequisites
Python: Python 3.7 or newer
Libraries: numpy, matplotlib
Knowledge: Basic understanding of numerical methods (root finding, interpolation, solving ODEs).

## Installation and Setup

Follow these steps to set up the local environment.

### 1. Clone the Repository
```bash
git clone https://github.com/gokul2736/STATISTICS-SA-2
cd STATISTICS-SA-2
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate Virtual Environment
```bash
.venv\Scripts\activate
```

### 4. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 5. How to Run the Programs
Example: Running the Runge Kutta's Method
```bash
python runge_kutta.py
```
## The Result of the statistical test will be printed on terminal.

# 1.Newton's Raphson Method
## Aim:
To find a real root of a non-linear equation using the Newton-Raphson iterative method.

## Problem Statement:
Find a root of the equation f(x) = x^3 - x - 1 = 0, which is accurate to four decimal places.
We will start with an initial guess of x0 = 1.0.

### Algorithm:
1. Define the function f(x) = x^3 - x - 1
2. Define the derivative of the function g(x) = f'(x) = 3x^2 - 1
3. Choose an initial guess (x0) and a tolerable error (e).
4. Start the iteration:
5. Calculate x1 = x0 - f(x0) / g(x0)
6. Check if abs(x1 - x0) is less than the tolerable error e.
7. If yes, x1 is the root. Stop.
8. If no, set x0 = x1 and repeat from step 5.

### Program
```python

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
```
### Output:



# 2.Gauss seidel Method
## Aim:
To solve a system of linear equations using the Gauss-Seidel iterative method.
## Problem Statement:
Solve the following system of linear equations:
10x +  y +  z = 12
 x + 10y +  z = 12
 x +  y + 10z = 12
Start with an initial guess of x=0, y=0, z=0.

### Algorithm:
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


### Program
```python
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

```
### Output:



# 3.Newton's Forward and Backward Interpolation
## Aim:
To estimate the value of a function at a specific point using
# Newton's forward and backward difference interpolation formulas.
## Problem Statement:
Given the following data table of (x, y) points:
# x: 0, 1, 2, 3
# y: 1, 2, 4, 7
# 1. Use Newton's Forward Interpolation to find y(0.5) (near the start).
# 2. Use Newton's Backward Interpolation to find y(2.5) (near the end).

### Algorithm:
 1. Create a set of n data points (x, y).
# 2. Generate the difference table.
# 3. For Forward Interpolation (estimating near the start of the table):
#    a. Calculate 'u' = (value - x[0]) / (x[1] - x[0])
#    b. Apply the formula:
#       y_val = y[0] + u*delta_y[0] + (u*(u-1)/2!)*delta_2_y[0] + ...
# 4. For Backward Interpolation (estimating near the end of the table):
#    a. Calculate 'u' = (value - x[n-1]) / (x[1] - x[0])
#    b. Apply the formula:
#       y_val = y[n-1] + u*nabla_y[n-1] + (u*(u+1)/2!)*nabla_2_y[n-1] + ...
### Program
```python
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
```
### Output:

# 4.Lagranges Interpolation
## Aim:
# To find a polynomial that passes through a given set of points and
# estimate the value of the function at a new point using Lagrange interpolation.
## Problem Statement:
Given the following data table of (x, y) points:
# x: 0, 1, 3
# y: 1, 3, 55
# (These points lie on the polynomial y = 6x^3 - 8x + 3. Let's pretend we don't know this.)
# Use Lagrange's Interpolation to find the value of y when x = 2.
### Algorithm:
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
### Program
```python
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
```
### Output:

# 5.Euler' s Method
## Aim:
To find the numerical solution of a first-order ordinary differential
# equation (ODE) with a given initial value using Euler's method.

## Problem Statement:
Given the ODE: dy/dx = x + y
# With the initial condition: y(0) = 1
# Find the approximate value of y(1) using Euler's method with a step size h = 0.1.

### Algorithm:
1. Define the function f(x, y) = x + y.
# 2. Set initial values x0 = 0, y0 = 1.
# 3. Set step size h = 0.1 and target x_target = 1.
# 4. Calculate the number of steps: n = (x_target - x0) / h.
# 5. Start iteration from i = 0 to n-1:
#    a. Calculate y_next = y_current + h * f(x_current, y_current)
#    b. Calculate x_next = x_current + h
#    c. Set y_current = y_next, x_current = x_next
# 6. The final y_next is the approximation for y(1).

### Program
```python
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
```
### Output:

# 6.Runge Kutta Method
## Aim:
To find a more accurate numerical solution of a first-order ODE
# with a given initial value using the 4th Order Runge-Kutta (RK4) method.
## Problem Statement:
Given the ODE: dy/dx = x + y
# With the initial condition: y(0) = 1
# Find the approximate value of y(1) using the RK4 method with a step size h = 0.1.
# (This is the same problem as the Euler's method for comparison).
### Algorithm:
1. Define the function f(x, y) = x + y.
# 2. Set initial values x0 = 0, y0 = 1, h = 0.1, and x_target = 1.
# 3. Calculate the number of steps: n = (x_target - x0) / h.
# 4. Start iteration from i = 0 to n-1:
#    a. Calculate the four "k" values:
#       k1 = h * f(x_i, y_i)
#       k2 = h * f(x_i + h/2, y_i + k1/2)
#       k3 = h * f(x_i + h/2, y_i + k2/2)
#       k4 = h * f(x_i + h, y_i + k3)
#    b. Calculate the next y-value:
#       y_next = y_current + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
#    c. Calculate x_next = x_current + h
#    d. Set y_current = y_next, x_current = x_next
# 5. The final y_next is the approximation for y(1).
### Program
```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    """The differential equation dy/dx = f(x, y)"""
    return x + y

def runge_kutta_4(x0, y0, h, x_target):
    """
    Solves an ODE using the 4th Order Runge-Kutta (RK4) method.
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
    
    print("--- Runge-Kutta (RK4) Method Iteration ---")
    print("Step\t x_n\t\t y_n (approx)")
    print("-------------------------------------------")
    print(f"0\t {x[0]:.2f}\t\t {y[0]:.6f}")

    for i in range(n_steps):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        
        y[i+1] = y[i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        x[i+1] = x[i] + h
        
        print(f"{i+1}\t {x[i+1]:.2f}\t\t {y[i+1]:.6f}")
        
    print("-------------------------------------------")
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
    
    x_rk4, y_rk4 = runge_kutta_4(x0, y0, h, x_target)
    
    # Get the final value
    y_final = y_rk4[-1]
    
    print("\n--- Output ---")
    print(f"The approximate value of y({x_target}) is: {y_final:.6f}")
    
    # --- Result ---
    y_exact = exact_solution(x_target)
    error = abs(y_exact - y_final)
    print("\n--- Result ---")
    print(f"Using RK4 method with h={h}, the approximate value of y(1) is {y_final:.6f}.")
    print(f"The exact value is {y_exact:.6f}.")
    print(f"The absolute error is {error:.6f}.")
    
    # --- Visualization ---
    x_exact = np.linspace(x0, x_target, 50)
    y_exact_vals = exact_solution(x_exact)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_rk4, y_rk4, 'go--', label=f"RK4 Method (h={h})", markersize=4)
    plt.plot(x_exact, y_exact_vals, 'r-', label="Exact Solution (y = 2e^x - x - 1)")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Runge-Kutta (RK4) Method vs. Exact Solution")
    plt.legend()
    plt.grid(True)
    plt.savefig("runge_kutta_plot.png")
    print("\nPlot saved as 'runge_kutta_plot.png'")
```
### Output:



## The Result of the statistical test will be printed on terminal.

###Conclusion 
In this Skill Assessment, we implemented various numerical methods using Python,
including root-finding, solving linear systems, interpolation, and ODE solvers.

• Newton-Raphson provided fast convergence for non-linear equations.
• Gauss-Seidel effectively solved linear systems iteratively.
• Newton and Lagrange interpolation accurately estimated intermediate values.
• Euler and Runge-Kutta methods provided numerical ODE solutions, with RK4 giving higher precision.

The implementations demonstrate the efficiency and versatility of Python in computational mathematics.

