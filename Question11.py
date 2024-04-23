""" This code calculates the solution of ODE with inifinite domains
    by changing the variable and then plots the original ODE solution. """

import numpy as np
import matplotlib.pyplot as plt


# Defining the differential equations
def F_tilde(x, u):
    return 1 / ((x*(1-u)) ** 2 + u ** 2)


# RK4 method
def rk_solver(f, u, x_tilde, h, N):

    for i in range(N):
        k1 = h * f(x_tilde[i], u[i])
        k2 = h * f(x_tilde[i] + k1 / 2, u[i] + h / 2)
        k3 = h * f(x_tilde[i] + k2 / 2, u[i] + h / 2)
        k4 = h * f(x_tilde[i] + k3, u[i] + h)
        x_tilde[i+1] = x_tilde[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_tilde


# Define initial conditions
u0 = 0.0
x0_tilde = 1  # Initial value of x(t=0) or x_tilde(u=0)
uf = 1
h = 0.01  # Step size
N = int((uf - u0) / h)
u = np.linspace(u0, uf, N + 1)
x_tilde = np.zeros(N + 1)
u[0], x_tilde[0] = u0, x0_tilde

# Solving the ODE for u between [0,1]
x_tilde = rk_solver(F_tilde, u, x_tilde, h, N)

# Find the value of x at t = 3.5 * 10^6
u_dash = 3.5e6 / (3.5e6 + 1)
x_dash = rk_solver(F_tilde, np.linspace(u0, u_dash, N+1), np.ones(N+1), h, N)[-1]

t = u / (1 - u)

print(f"x(t= 3.5 * $10^6$) = {x_dash}")

# Plotting the solution
plt.plot(t[:-1], x_tilde[:-1], '--', color='green', label="RK solution")
plt.scatter(u_dash/(1 - u_dash), x_dash, color='red', alpha=0.0, label=f'x(t= 3.5 * $10^6$) = {x_dash:.3f}')
plt.xlim(-2, 100)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)
plt.title("RK method Solution to ODE")
plt.show()
