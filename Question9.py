# This code uses scipy.integrate.solve_bvp to solve boundary value problems

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp


# Defining the boundary value problems
def fun1(x, y):
    return np.vstack((y[1], -np.exp(-2*y[0])))


def bc1(ya, yb):
    return np.array([ya[0] - 0, yb[0] - np.log(2)])


def fun2(x, y):
    return np.vstack((y[1], y[1]*np.cos(x) - y[0]*np.log(y[0])))


def bc2(ya, yb):
    return np.array([ya[0] - 1, yb[0] - np.exp(1)])


def fun3(x, y):
    return np.vstack((y[1], -(2*y[1]**3 + y[0]**2*y[1])/np.cos(x)))


def bc3(ya, yb):
    return np.array([ya[0] - 2**(-1/4), yb[0] - (12**(1/4))/2])


def fun4(x, y):
    return np.vstack((y[1], 0.5 - 0.5*y[1]**2 - y[0]*np.sin(x)/2))


def bc4(ya, yb):
    return np.array([ya[0] - 2, yb[0] - 2])


# Discretising the x domain
x1 = np.linspace(1, 2, 100)
x2 = np.linspace(0, np.pi/2, 100)
x3 = np.linspace(np.pi/4, np.pi/3, 100)
x4 = np.linspace(0, np.pi, 100)

# Guessed Initial Conditions
y_guess1 = np.zeros((2, x1.size))
y_guess2 = np.ones((2, x2.size))
y_guess3 = np.zeros((2, x3.size))
y_guess4 = np.zeros((2, x4.size))

# Solving the BVPs
sol1 = solve_bvp(fun1, bc1, x1, y_guess1)
sol2 = solve_bvp(fun2, bc2, x2, y_guess2)
sol3 = solve_bvp(fun3, bc3, x3, y_guess3)
sol4 = solve_bvp(fun4, bc4, x4, y_guess4)

# Plot for y′′ = −e^−2y
plt.plot(sol1.x, sol1.y[0], label='Numerical Solution', color='red')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('y′′ = −e^−2y with y(1) = 0 and y(2) = ln 2')
plt.grid(True)
plt.legend()
plt.show()

# Plot for y′′ = y′cos x − y ln y
plt.plot(sol2.x, sol2.y[0], label='Numerical Solution', color='red')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('y′′ = y′cos x − y ln y with y(0) = 1 and y(π/2) = e')
plt.grid(True)
plt.legend()
plt.show()

# Plot for y′′ = −(2(y′)^3 + y^2y′) sec x
plt.plot(sol3.x, sol3.y[0], label='Numerical Solution', color='red')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('y′′ = −(2(y′)^3 + y^2y′) sec x with y(π/4) = 2^−1/4 and y(π/3) =12^(1/4)/2')
plt.grid(True)
plt.legend()
plt.show()

# Plot for y′′ = 1/2 − (y′)^2/2 − y sin x/2
plt.plot(sol4.x, sol4.y[0], label='Numerical Solution', color='red')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('y′′ = 1/2 − (y′)^2/2 − y sin x/2 with y(0) = 2 and y(π) = 2')
plt.grid(True)
plt.legend()
plt.show()
