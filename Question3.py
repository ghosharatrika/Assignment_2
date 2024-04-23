# This program Second order ODE with initial values using RK4 method and plots the solution

import numpy as np
import matplotlib.pyplot as plt


# Defining the differential equations
def F(y, x):
    return np.array([y[1], 2 * y[1] - y[0] + x * np.exp(x) - x])


# Exact solution obtained using wolfram alpha
def exact(x):
    return 1 / 6 * np.exp(x) * (x ** 3 - 6 * x + 12) - x - 2


# RK4 method
def rk_solver(f, x0, y0, h, N):
    x = np.zeros(N + 1)
    y = np.zeros((N + 1, 2))

    x[0], y[0, :] = x0, y0

    for i in range(N):
        k1 = h * f(y[i], x[i])
        k2 = h * f(y[i] + k1 / 2, x[i] + h / 2)
        k3 = h * f(y[i] + k2 / 2, x[i] + h / 2)
        k4 = h * f(y[i] + k3, x[i] + h)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[i + 1] = x[i] + h

    return x, y


# Defining initial conditions
x0 = 0
y0 = 0  # Initial value of y
y_prime0 = 0
h = 0.1  # Step size
N = int((1.0 - x0) / h)

x_sol, y_sol = rk_solver(F, x0, np.array([y0,y_prime0]), h, N)
# Plotting the solution
plt.plot(x_sol, y_sol[:,0], '-o', color='blue', label="RK solution")
plt.plot(x_sol, exact(x_sol), color='black', label="Exact solution")
plt.xlabel("x")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.title("RK method Solution to ODE")
plt.show()
