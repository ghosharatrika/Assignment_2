""" This code uses scipy.integrate.solve_ivp to solve initial value problems
    and plots the solution as well compares it with analytical solution. """

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Defining the differential equations
def fun1(t, y):
    return t * np.exp(3*t) - 2*y


def fun2(t, y):
    return 1 - (t - y)**2


def fun3(t, y):
    return 1 + y/t


def fun4(t, y):
    return np.cos(2*t) + np.sin(3*t)


# Defining true solutions
def true_sol1(t):
    return 1/25 * np.exp(-2 * t) * (np.exp(5 * t) * (5 * t - 1) + 1)


def true_sol2(t):
    return (t**2 - 3 * t + 1) / (t - 3)


def true_sol3(t):
    return t * (np.log(t) + 2)


def true_sol4(t):
    return 1/6 * (3 * np.sin(2 * t) - 2 * np.cos(3 * t) + 8)


# Solving the initial value problems
sol1 = solve_ivp(fun1, [0, 1], [0], t_eval=np.linspace(0, 1, 100))
sol2 = solve_ivp(fun2, [2, 3], [1], t_eval=np.linspace(2, 3, 100))
sol3 = solve_ivp(fun3, [1, 2], [2], t_eval=np.linspace(1, 2, 100))
sol4 = solve_ivp(fun4, [0, 1], [1], t_eval=np.linspace(0, 1, 100))

# Plot for y′(t) = te^3t − 2y
plt.plot(sol1.t, true_sol1(sol1.t), label='Exact solution', color='black')
plt.plot(sol1.t, sol1.y[0], '--', label='Numerical Solution', color='red')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('y′(t) = te^3t − 2y with y(0) = 0')
plt.grid(True)
plt.legend()
plt.show()

# Plot for y′(t) = 1 − (t − y)^2
plt.plot(sol2.t, true_sol2(sol2.t), label='Exact solution', color='black')
plt.plot(sol2.t, sol2.y[0], '--', label='Numerical Solution', color='red')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('y′(t) = 1 − (t − y)^2 with y(2) = 1')
plt.grid(True)
plt.legend()
plt.show()

# Plot for y′(t) = 1 + y/t
plt.plot(sol3.t, true_sol3(sol3.t), label='Exact solution', color='black')
plt.plot(sol3.t, sol3.y[0], '--', label='Numerical Solution', color='red')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('y′(t) = 1 + y/t with y(1) = 2')
plt.grid(True)
plt.legend()
plt.show()

# Plot for y′(t) = cos(2t) + sin(3t)
plt.plot(sol4.t, true_sol4(sol4.t), label='Exact solution', color='black')
plt.plot(sol4.t, sol4.y[0], '--', label='Numerical Solution', color='red')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('y′(t) = cos(2t) + sin(3t) with y(0) = 1')
plt.grid(True)
plt.legend()
plt.show()
