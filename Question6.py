""" This code finds the solution of a Second order ODE with BOUNDARY VALUES 
    using the Shooting method and also plots the solution as well as candidate
    solutions. It also tries to find the solution using numpy.argmin()"""

# ---------------------------------------------------------------------------------------------------------------

""" This code finds the solution of a Second order ODE with BOUNDARY VALUES 
    using the Shooting method and also plots the solution as well as candidate
    solutions. It also tries to find the solution using numpy.argmin()"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Constants
g = 10  # acceleration due to gravity
t1 = 10  # end time
t_eval = np.linspace(0, t1, 100)


# Define the differential equation dy^2/dt^2 = -10
def F(t, y):
    return [y[1], -g]


# Define a function to find the correct dy/dt(0) using the shooting method
def find_v0(v0_guess):
    sol = solve_ivp(F, [0, t1], [0.0, v0_guess[0]], t_eval=t_eval)
    return sol.y[0][-1] - 0.0  # Return the difference between final value of y and the desired value, 0.0


# Use fsolve to find the correct dy/dt(0)
v0_guess = np.array([0.0])  # Initial guess for dy/dt(0)
v0 = fsolve(find_v0, v0_guess)[0]  # Extract the first element as the solution
print("v0 for shooting method:", v0)

# Solve the differential equation with the correct initial condition
sol = solve_ivp(F, [0, 10], [0.0, v0], t_eval=t_eval, method='RK45')
candidate_v0 = [v0-15.0, v0-10.0, v0-5.0, v0+10, v0+20]
# Plot the solutions
plt.plot(sol.t, sol.y[0], color='red', label=f'y(t) for shooting method with v(0) = {v0:.2f}')
for v in candidate_v0:
    sol1 = solve_ivp(F, [0, 10], [0.0, v], t_eval=t_eval, method='RK45')
    plt.plot(sol1.t, sol1.y[0], color='black', label=f'Candidate solution with v(0) = {v:.2f}', linestyle = '--')
plt.xlabel('t')
plt.ylabel('$y(t)$')
plt.legend()
plt.grid()
plt.show()

# -----------------------------------------------------------------------------------------------------------------------

# This part of the code uses numpy.argmin() to find the v0 and plots the candidate as well as exact solution

# Use numpy.argmin to find the solution
def find_solution(v_guesses):
    # It calculates the boundary value of x(t) for guess velocity
    x_ends = np.array([find_v0(np.array([v])) for v in v_guesses])

    # Find the index of the minimum difference
    min_index = np.argmin(np.abs(x_ends))
    return v_guesses[min_index]


# Find the solution
v_start = 0
v_end = 10
bv =100 # value of x(t=10)
while abs(bv) > 10e-3:
    v_guesses = np.linspace(v_start, v_end, 100)
    v_solution = find_solution(v_guesses)
    bv = find_v0(np.array([v_solution]))
    t_span = [0, t1]
    y0 = [0, v_solution]
    sol = solve_ivp(F, t_span, y0, t_eval=np.linspace(0, t1, 100))
    plt.plot(sol.t, sol.y[0], '--', label=f'Candidate solution with v0 = {v_solution:.3f}')
    v_start = v_end
    v_end += 10.0

# Correct solution
t_span = [0, t1]
y0 = [0, v_solution]
sol = solve_ivp(F, t_span, y0, t_eval=np.linspace(0, t1, 100))

# Plot the solution
plt.plot(sol.t, sol.y[0], color='black', label=f'Position x(t) with v0 = {v_solution:.3f}')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Shooting Method using numpy.argmin()')
plt.legend()
plt.grid(True)
plt.show()
