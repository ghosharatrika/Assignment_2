""" This code solves a second order ODE with boundary values using 
    the relaxation method and prints the no. of iterations taken to converge 
    to the solution and plots the solution. """

import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 10
t1 = 10
num_points = 100  # Number of discrete points

# Discretization of the time domain
t = np.linspace(0, t1, num_points + 1)
h = t[1] - t[0]

# Initial guess
y = np.zeros(num_points + 1)
colors = ('brown', 'blue', 'green', 'orange', 'purple')
j = 0

# Relaxation method
tolerance = 1e-6
for iteration in range(10**6):
    y_old = y.copy()
    # Updating y(t) using relaxation formula where y[2:] = y(t+h) and y[:2] = y(t-h)
    y[1:-1] = 0.5 * (y[2:] + y[:-2] + g * h ** 2)

    # Checking for convergence
    if np.max(np.abs(y - y_old)) < tolerance:
        print(f"Converged after {iteration} iterations")
        break

    # Plotting candidate solutions for iterations divisible by 1200
    if iteration % 1200 == 0 and j <= 4 and iteration != 0:
        plt.plot(t, y, label=f'Candidate Solution {j + 1}', color=colors[j])
        j += 1


# Exact solution
exact_solution = -0.5 * g * t ** 2 + 50 * t

# Plotting solution
plt.plot(t, exact_solution, label='Exact Solution', color='black')
plt.plot(t, y, '--', label='Numerical Solution', color='red')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Relaxation Method for Freely falling object')
plt.grid(True)
plt.legend()
plt.show()
