""" This code computes the solution of ODE with initial value
    using Adavptive Step Size method and then plots the 
    solution and the mesh points. """

import numpy as np
import matplotlib.pyplot as plt


def func(t, y):
    return (y ** 2 + y) / t  # The function defining the ODE y' = (y^2 + y) / t


def true_sol(t):
    return (2 * t)/(1 - 2 * t)  # Exact solution obtained using wolfram alpha


def runge_kutta_step(t, y, h):
    # Performing one step of the fourth-order Runge-Kutta method
    k1 = h * func(t, y)
    k2 = h * func(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * func(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * func(t + h, y + k3)

    y_new = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return y_new


def adaptive_runge_kutta(t0, y0, t_end, tol):
    # Adaptive Runge-Kutta method with a given tolerance
    t = [t0]
    y = [y0]

    h = 0.1  # Initial step size
    while t[-1] < t_end:
        # Performing one step with the current dt
        y1 = runge_kutta_step(t[-1], y[-1], 2 * h)

        # Performing two half steps with half the dt
        y2_half = runge_kutta_step(t[-1], y[-1], h)
        y2 = runge_kutta_step(t[-1] + h, y2_half, h)

        # Estimating the error
        error = np.abs(y2 - y1) / 30
        if error < tol:
            t.append(t[-1] + h)
            y.append(y2_half)

        h *= (tol / error) ** 0.25  # Updating the value of the step size

    return np.array(t), np.array(y)


# Initial conditions
t0 = 1.0
y0 = -2.0
t_end = 3.0
tol = 1e-4

# Solving the ODE using adaptive Runge-Kutta
t, y = adaptive_runge_kutta(t0, y0, t_end, tol)
t_span = np.linspace(1, 3, 101)
# Plotting the solution
print("The mesh points are:\n", t)
plt.plot(t, y, 'o-', color='red', label='Numerical Solution')
plt.plot(t_span, true_sol(t_span), color='black', label='Analytical Solution')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Solution of y\' = (y^2 + y)/t using Adaptive Runge-Kutta')
plt.grid(True)
plt.legend()
plt.show()
