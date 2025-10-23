# Skill Assessment - 2 Numerical Methords in Python  
## NAME: MARKANDEYAN GOKUL  
## ROLL NO: 212224240086  


Python Scripts for Numerical Methods (Skill Assessment 2)

This repository contains a collection of Python scripts that implement common numerical methods as part of a skill assessment. Each script is designed to solve a specific mathematical problem and provides a clear output and interpretation.

Aim of the Assessment

The main goal of this assessment is to provide clear, runnable Python examples for foundational numerical methods used in science, engineering, and finance.

Implemented Numerical Methods

This collection includes implementations for the following:

Newton-Raphson Method: (newton_raphson.py) - An iterative root-finding algorithm for a real-valued function.

Gauss-Seidel Method: (gauss_seidel.py) - An iterative method for solving a system of linear equations.

Newton's Interpolation: (newton_interpolation.py) - Implements both forward and backward difference formulas for polynomial interpolation.

Lagrange Interpolation: (lagrange_interpolation.py) - An alternative method for polynomial interpolation.

Euler's Method: (euler_method.py) - A simple numerical method for solving ordinary differential equations (ODEs) with a given initial value.

Runge-Kutta Method: (runge_kutta.py) - A more accurate method (RK4) for solving ODEs.

Getting Started

Follow these instructions to get the scripts running on your local machine.

Prerequisites

Python (version 3.7 or newer)

Git

Installation and Setup

1. Clone the Repository

git clone <YOUR_REPOSITORY_URL>
cd skill_assessment_numerical_methods



2. Create and Activate a Virtual Environment

# Create the environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate

# Activate on macOS/Linux
source .venv/bin/activate



3. Install Required Libraries

This assessment requires numpy for calculations and matplotlib to visualize the results of the ODE solvers.

pip install -r requirements.txt



How to Run the Programs

All scripts can be run directly from the main directory. Make sure your virtual environment is activated.

Example: Running the Newton-Raphson Method

python newton_raphson.py



Example: Running the Euler's Method (will generate a plot)

python euler_method.py



The output of each test will be printed directly to your terminal.

## Installation and Setup

Follow these steps to set up the local environment.

### 1. Clone the Repository
```bash
git clone https://github.com/gokul2736/STATISTICS-SA-2
cd STATISTICS-SA-2
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
```
### 3. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 4. How to Run the Programs
Example: Running the T-Test Program
```bash
python t-test.py
```
## The Result of the statistical test will be printed on terminal.
