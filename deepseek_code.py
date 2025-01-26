import numpy as np
from scipy.optimize import minimize

# Simulate Vasicek model data
np.random.seed(42)  # Set seed for reproducibility
n = 20000  # Increased number of observations
kappa_true = 0.5  # True speed of mean reversion
theta_true = 0.05  # True long-term mean
sigma_true = 0.02  # True volatility
dt = 1  # Time step

# Simulate interest rate process using the Vasicek model
# The Vasicek model is defined by the SDE: dr_t = κ(θ - r_t)dt + σ dW_t
# Where:
# - r_t is the interest rate at time t
# - κ is the speed of mean reversion
# - θ is the long-term mean
# - σ is the volatility
# - dW_t is a Wiener process (Brownian motion)

r = np.zeros(n)  # Initialize array for interest rates
r[0] = theta_true  # Start at the long-term mean
for t in range(1, n):
    # Euler-Maruyama discretization of the Vasicek SDE:
    # r_t = r_{t-1} + κ(θ - r_{t-1})Δt + σ√Δt * ε_t
    # Where ε_t ~ N(0, 1) is a standard normal random variable
    r[t] = r[t-1] + kappa_true * (theta_true - r[t-1]) * dt + sigma_true * np.sqrt(dt) * np.random.normal()

# Define moment conditions for GMM estimation
# GMM (Generalized Method of Moments) estimates parameters by matching sample moments to theoretical moments
def moment_conditions(params, r):
    kappa, theta, sigma = params  # Unpack parameters
    residuals = r - theta  # Residuals: r_t - θ
    
    # Theoretical variance of the Vasicek process:
    # Var(r_t) = σ² / (2κ)
    variance = sigma**2 / (2 * kappa)
    
    # Define moments:
    # 1. E[r_t - θ] = 0 (Mean of residuals)
    # 2. E[(r_t - θ)^2] = σ²/(2κ) (Variance of residuals)
    # 3. E[(r_t - θ)(r_{t-1} - θ)] = σ²/(2κ) * exp(-κΔt) (Autocovariance at lag 1)
    # 4. E[(r_t - θ)(r_{t-2} - θ)] = σ²/(2κ) * exp(-2κΔt) (Autocovariance at lag 2)
    # 5. E[(r_t - θ)(r_{t-3} - θ)] = σ²/(2κ) * exp(-3κΔt) (Autocovariance at lag 3)
    # 6. E[(r_t - θ)(r_{t-4} - θ)] = σ²/(2κ) * exp(-4κΔt) (Autocovariance at lag 4)
    # 7. E[(r_t - θ)(r_{t-5} - θ)] = σ²/(2κ) * exp(-5κΔt) (Autocovariance at lag 5)
    # 8. E[(r_t - θ)^3] = 0 (Skewness of residuals, assuming normality)
    # 9. E[(r_t - θ)^4] = 3(σ²/(2κ))^2 (Kurtosis of residuals, assuming normality)
    # 10. E[(r_t - θ)^5] = 0 (Fifth moment, assuming normality)
    # 11. E[(r_t - θ)^6] = 15(σ²/(2κ))^3 (Sixth moment, assuming normality)
    
    moments = np.array([
        np.mean(residuals),  # First moment: E[r_t - θ] = 0
        np.mean(residuals**2) - variance,  # Second moment: E[(r_t - θ)^2] = σ²/(2κ)
        np.mean(residuals[1:] * residuals[:-1]) - variance * np.exp(-kappa * dt),  # Autocovariance (lag 1)
        np.mean(residuals[2:] * residuals[:-2]) - variance * np.exp(-kappa * 2 * dt),  # Autocovariance (lag 2)
        np.mean(residuals[3:] * residuals[:-3]) - variance * np.exp(-kappa * 3 * dt),  # Autocovariance (lag 3)
        np.mean(residuals[4:] * residuals[:-4]) - variance * np.exp(-kappa * 4 * dt),  # Autocovariance (lag 4)
        np.mean(residuals[5:] * residuals[:-5]) - variance * np.exp(-kappa * 5 * dt),  # Autocovariance (lag 5)
        np.mean(residuals**3),  # Third moment: E[(r_t - θ)^3] = 0
        np.mean(residuals**4) - 3 * variance**2,  # Fourth moment: E[(r_t - θ)^4] = 3(σ²/(2κ))^2
        np.mean(residuals**5),  # Fifth moment: E[(r_t - θ)^5] = 0
        np.mean(residuals**6) - 15 * variance**3,  # Sixth moment: E[(r_t - θ)^6] = 15(σ²/(2κ))^3
    ])
    return moments

# Define GMM objective function
# The objective function minimizes the weighted sum of squared moment conditions:
# Q(θ) = g(θ)' W g(θ)
# Where:
# - g(θ) is the vector of moment conditions
# - W is the weighting matrix
def gmm_objective(params, r, W):
    moments = moment_conditions(params, r)
    return moments.T @ W @ moments  # Q(θ) = g' W g

# Initial parameter guesses
params_init = np.array([0.1, 0.03, 0.01])  # Initial guesses for κ, θ, σ

# Weighting matrix (start with identity matrix)
W = np.eye(11)  # 11 moments now

# Step 1: First-stage estimation with identity matrix
# Minimize the GMM objective function using the BFGS algorithm
result1 = minimize(gmm_objective, params_init, args=(r, W), method='BFGS')
params_est1 = result1.x  # First-stage parameter estimates

# Step 2: Compute the optimal weighting matrix
# The optimal weighting matrix is the inverse of the covariance matrix of the moment conditions
moments1 = moment_conditions(params_est1, r)
S = np.outer(moments1, moments1)  # Covariance matrix of moment conditions

# Regularize the covariance matrix to avoid singularity
regularization = 1e-6
S_reg = S + regularization * np.eye(S.shape[0])

# Compute the optimal weighting matrix using pseudo-inverse
W_optimal = np.linalg.pinv(S_reg)

# Step 3: Second-stage estimation with optimal weighting matrix
# Re-estimate parameters using the optimal weighting matrix
result2 = minimize(gmm_objective, params_init, args=(r, W_optimal), method='BFGS')
params_est2 = result2.x  # Second-stage parameter estimates

# Estimated parameters
kappa_est, theta_est, sigma_est = params_est2
print("Estimated Parameters (with increased moments):")
print(f"kappa: {kappa_est}, theta: {theta_est}, sigma: {abs(sigma_est)}")

# Compare with true values
print("\nTrue Parameters:")
print(f"kappa: {kappa_true}, theta: {theta_true}, sigma: {abs(sigma_true)}")