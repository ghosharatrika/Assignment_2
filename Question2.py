""" This program solves the first order ODE using Euler method and 
    calculates the absolute and the relative error in each step. 
    Finally it plots the numerical and exact solution as well as the errors."""

import numpy as np
import matplotlib.pyplot as plt


# Euler method
def euler_method(f, x0, y0, h, N):
    t_values = [x0]
    y_values = [y0]
    error = [0.0]
    rel_err = [0.0]

    for i in range(N):
        y_new = y_values[-1] + h * f(t_values[-1], y_values[-1])
        t_new = t_values[-1] + h

        # Calculating the errors in each step
        error_val = abs(y_new - f_sol(t_new))
        rel_error_val = abs(error_val / y_new)

        error.append(error_val)
        rel_err.append(rel_error_val)

        print(f"Step {i + 1}: Error = {error_val}, Relative Error = {rel_error_val}")

        t_values.append(t_new)
        y_values.append(y_new)

    total_error = np.sum(error)
    print(f"Total error in {N} steps: {total_error}")
    return t_values, y_values, error, rel_err


# Defining the differential equation dy/dt = f(y, t)
def f(t, y):
    return y / t - (y / t) ** 2


# Defining the exact solution
def f_sol(t):
    return t / (1 + np.log(t)) 


# Defining initial conditions
x0 = 1
y0 = 1  # Initial value of y
h = 0.1  # Step size
N = int((2.0 - x0) / h)
t_sol, y_sol, error, rel_err = euler_method(f, x0, y0, h, N)

# Plotting the solution
t = np.linspace(1, 2, 100)
plt.plot(t, f_sol(t), label="True solution", linestyle='--')
plt.plot(t_sol, y_sol, '-o', color='blue', label="Euler's Method solution")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.title("Euler's Method Solution to ODE")
plt.show()

# Plotting the errors
steps = np.arange(N+1)
plt.plot(steps, error, '-o', color='red', label="Absolute Error")
plt.plot(steps, rel_err, '-o', color='green', label="Relative Error")
plt.xlabel("Step number")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.title("Absolute and Relative Error of Euler's Method")
plt.show()
