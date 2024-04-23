# This program calculates the solution of First order ODE using backward euler method 
# and plots the solution

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# Defining the backward euler method
def backward_euler_method(f, x0, y0, h, N):
    x_values = [x0]
    y_values = [y0]

    for _ in range(N):
        x_new = x_values[-1] + h

        def F(y_new):
            return y_values[-1] + h * f(x_new, y_new) - y_new

        y_new = fsolve(F, y_values[-1])
        x_values.append(x_new)
        y_values.append(y_new[0])

    return x_values, y_values


# Defining the differential equation dy/dt = f(y, t)
def f1(x, y):
    return -9.0 * y  # dy/dx = -9y


def f2(x, y):
    return -20 * (y - x) ** 2 + 2 * x  # dy/dt = -20(y-x)^2 + 2x


x0 = 0
h = 0.1
N = int((1.0 - x0) / h)

# Defining the initial condition for first ODE and plotting the solution
y0 = np.exp(1)
x_values, y_values = backward_euler_method(f1, x0, y0, h, N)
plt.plot(x_values, y_values, '-o', color='black', label="Exponential Decay")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Backward Euler\'s Method for Solving dy/dx = -9y')
plt.legend()
plt.grid(True)
plt.show()

# Defining the initial condition for second ODE and plotting the solution
y0 = 1/3.0
x_values, y_values = backward_euler_method(f2, x0, y0, h, N)
plt.plot(x_values, y_values, '-o', color='red', label="Euler method")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Backward Euler\'s Method for Solving $dy/dx = -20(y-x)^2 + 2x$')
plt.legend()
plt.grid(True)
plt.show()
