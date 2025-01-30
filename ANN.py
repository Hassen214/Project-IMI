import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from scipy.special import comb

class Vasicek:
    def __init__(self, kappa, theta, sigma):
        """
        Initialize the parameters for the Vasicek model.
        
        :param kappa: Speed of mean reversion
        :param theta: Long-term mean level
        :param sigma: Volatility of the process
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def generate(self, X0, T, N, n):
        """
        Generate N trajectories of the Vasicek process on the interval [0, T] with a time step of T/n.
        
        :param X0: Initial value of the process
        :param T: The total time period
        :param N: Number of trajectories to generate
        :param n: Number of discretization steps (so time step is T/n)
        
        :return: Array of N trajectories discretized over the interval [0, T]
        """
        dt = T / n  # Adjusted time step for the interval [0, T]
        trajectories = np.zeros((N, n))  # Matrix to store the trajectories
        trajectories[:, 0] = X0  # Initialize all trajectories at X0

        for i in range(1, n):
            # Generate standard normal noise
            Z = np.random.randn(N)  # Standard normal random variables for each trajectory
            # Update the process value according to the Vasicek model
            trajectories[:, i] = (
                trajectories[:, i - 1] * np.exp(-self.kappa * dt)
                + self.theta * (1 - np.exp(-self.kappa * dt))
                +  np.sqrt((self.sigma**2) *(1 - np.exp(-2 * self.kappa * dt)) / (2 * self.kappa)) * Z
            )

        return trajectories

def expected_value_vasicek(power, theta, sigma, kappa):
    """
    Compute the expected value of the Vasicek process at X_n using the closed-form formula.
    
    :param power: Number of steps in the discretization
    :param theta: Long-term mean level
    :param sigma: Volatility
    :param kappa: Speed of mean reversion
    :return: The theoretical expected value
    """
    expected_sum = 0
    for k in range(power + 1):
        # Binomial coefficient
        binom_coeff = comb(power, k)
        
        # Factor for theta and sigma
        theta_factor = theta**(power - k)
        sigma_factor = sigma**k / (2 * kappa)**(k//2)
        
        # Expectation of Z^k
        if k % 2 == 0:  # Even powers of Z
            m = k // 2
            # For even k, the expectation of Z^{2m} is (2m-1)!! = (2m-1)*(2m-3)*...*1
            z_expectation = math.factorial(2 * m) // (2**m * math.factorial(m))  # Double factorial calculation
        else:  # Odd powers of Z
            z_expectation = 0
        
        # Add to the summation
        expected_sum += binom_coeff * theta_factor * sigma_factor * z_expectation
    
    return expected_sum

def empirical_covariance(X, n, T, theta_emp, h):
    k0 = int(n * h / T)  # Convert lag h to index
    cov = 0
    for k in range(n - k0):
        cov += (X[k] * X[k + k0]) / (n - k0)
    return cov - theta_emp**2

def emp_cov(X, h, n, T, theta_emp):
    return empirical_covariance(X, n, T, theta_emp, h)

# Parameters for Vasicek

n = 1000
num_lags_list = np.arange(20, 100, 20)  # List of lags to evaluate

# Values for kappa, theta, sigma
values = np.linspace(0.01, 3, 30)  # Range for the parameters
label = []
num_lags = 90
total = 0 
results = []
for n in range(10000, 11000, 1000):
        for kappa in values:
            for sigma in values:
                for theta in values:
                    total+=1
                    vasicek_process = Vasicek(kappa, theta, sigma)
                    X0 = np.random.uniform(0, 1)  # Initial value of the process
                    T = np.random.randint(100, 5000)  # Random total time period
                    N = 1  # Single trajectory
                    
                    # Generate the lag array
                    lags = np.array([i * T / n for i in range(num_lags)])
                    # Generate the trajectory
                    trajectory = vasicek_process.generate(X0, T, N, n)[0]
                    print(f"Kappa: {kappa}, Theta: {theta}, Sigma: {sigma}, N: {N}, n: {n}")
                    result_temp = []
                    # Calculate expected values for different powers
                    for power in range(10):
                        expected = expected_value_vasicek(power, theta, sigma, kappa)
                        result_temp.append(expected)
                    
                    # Calculate empirical covariance
                    theta_emp = np.mean(trajectory)                
                    covariances = np.array([emp_cov(trajectory, h, n, T, theta_emp) for h in lags])
                    for x in covariances:
                        result_temp.append(x)
                    
                    results.append(result_temp)
                    print(len(result_temp))
                    label.append((kappa, theta, sigma)) 

print(total)
# Create DataFrame from results
column_names = [f"Expected_{i}" for i in range(10)] + [f"Covariance_{i}" for i in range(num_lags)]
df_results = pd.DataFrame(results, columns=column_names)

# Add kappa, theta, sigma columns
df_results['Kappa'] = [x[0] for x in label]
df_results['Theta'] = [x[1] for x in label]
df_results['Sigma'] = [x[2] for x in label]

# Save to CSV
df_results.to_csv("vasicek_results.csv", mode='a', header=False, index=False)

print(f"Total calculations: {total}")
print(df_results)