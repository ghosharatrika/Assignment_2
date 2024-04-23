""" This code solves the initial value problem for coupled
    First order ODE using RK4 and then plots the solution. """

import numpy as np
import matplotlib.pyplot as plt


# Defining the system of ODEs
def coupled_f(t, u):
    u1, u2, u3 = u
    du1dt = u1 + 2 * u2 - 2 * u3 + np.exp(-t)
    du2dt = u2 + u3 - 2 * np.exp(-t)
    du3dt = u1 + 2 * u2 + np.exp(-t)
    return np.array([du1dt, du2dt, du3dt])


# Single step of RK4
def RK4step(f, t, u, h):
    k1 = h*f(t, u)
    k2 = h*f(t + h / 2, u + 0.5 * k1)
    k3 = h*f(t + h / 2, u + 0.5 * k2)
    k4 = h*f(t + h, u + k3)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Fourth-order Runge-Kutta method
def RK4Solver(f, t0, t_end, u0, h, N):
    t_values = np.linspace(t0, t_end, N + 1)
    u_values = np.zeros((N + 1, 3))
    u_values[0, :] = u0

    for i in range(1, N+1):
        u_values[i] = RK4step(f, t_values[i-1], u_values[i-1], h)  # updates ith row of u
    return t_values, u_values


# Initial conditions
u0 = np.array([3, -1, 1])
t0, t_end = 0, 1
h = 0.1
N = int((t_end - t0) / h)

# Solving the ODEs
t, u = RK4Solver(coupled_f, t0, t_end, u0, h, N)

# Plotting results
plt.plot(t, u[:, 0], label='u1(t)')
plt.plot(t, u[:, 1], label='u2(t)')
plt.plot(t, u[:, 2], label='u3(t)')
plt.title('Solution of the System of ODEs using RK4')
plt.xlabel('Time t')
plt.ylabel('u(t)')
plt.legend()
plt.grid(True)
plt.show()
