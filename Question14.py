# This code solves second order ODE with initial values using euler method

import numpy as np
import matplotlib.pyplot as plt


# Defining the right-hand side functions of y''(t)
def f(t, y):
    return np.array([y[1], 2 * y[1] / t - 2 * y[0] / t**2 + t * np.log(t)])


# Initial conditions
t0 = 1
tf = 2  # Final time
h = 0.001  # Step size
N = int((tf - t0) / h)
t = np.linspace(t0, tf, N+1)  # Defining the mesh points
y = np.zeros((N+1, 2))
y[0, :] = np.array([1, 0])  # y(1) = 1 and y'(1) = 0


for i in range(N):
    y[i+1] = y[i] + h * f(t[i], y[i])  # Euler Method


# Exact solution
def true_sol(t):
    return 7 * t / 4 + ((t ** 3) / 2) * np.log(t) - (3 / 4) * t ** 3


# Plotting the results
plt.plot(t, y[:, 0], label='Euler\'s method solution', color='blue')
plt.plot(t, true_sol(t), label='Exact solution', color='red', linestyle='dashed')
plt.xlabel('t')
plt.ylabel('y')
plt.title("Solution of the Second-Order ODE using Euler's Method")
plt.legend()
plt.grid(True)
plt.show()
