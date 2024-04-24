/* This code solves the initial value problem using Euler
    method and calculates the error in each step by comparing
    the solution obtained numerically and analytically and also
    prints the error bounds for each step. */

#include <stdio.h>
#include <math.h>

// This function defines the exact solution of  ODE
double exact_solution(double t) {
    return pow(t + 1, 2) - 0.5 * exp(t);
}

// This function defines the ODE
double derivative(double t, double y) {
    return y - t * t  + 1;
}

// This function calculates the error bound at each step
double error_bound(double h, double t, double t0) {
    double M = 2;  // Maximum absolute value of the second derivative
    double L = 1;  // Lipschitz constant
    
    return (h * M / (2 * L)) * (exp(M * (t - t0) / L) - 1);

}

// Euler method
void euler_method(double t0, double y0, double h, int N) {
    double y = y0;
    double t = t0;
    double y_exact, y_new, error;
    int i = 0;
    // Tabulating the errors in each step
    printf("Step\t\tt\t\tEuler's y\tExact y\t\tError\t\tError Bound\n");
    printf("-------------------------------------------------------------------------------------------\n");

    
    y_exact = exact_solution(t);
    printf("%d\t\t%.1f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", i, t0, y0, y_exact, fabs(y0-y_exact), error_bound(h, t0, t0));
    
    for(int i = 1; i <= N; i++) {
        t = t + h;
        y_exact = exact_solution(t);
        y_new = y + h * derivative(t - h, y);
        
        error = fabs(y_exact - y_new);
        printf("%d\t\t%.1f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", i, t, y_new, y_exact, error, error_bound(h, t, t0));
        
        y = y_new;
    }
}

int main() {
    // Initial condition
    double t0 = 0.0;
    double y0 = 0.5;
    double h = 0.2;
    double tf = 2.0;
    int N = ceil((tf - t0) / h);
  
    euler_method(t0, y0, h, N);

    return 0;
}
