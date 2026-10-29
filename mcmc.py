import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data
# We know the 'truth', but the algorithm must guess it.
np.random.seed(42)
true_mu = 5.5
true_sigma = 2.0
data = np.random.normal(true_mu, true_sigma, size=1000)

# 2. Define the Bayesian Math (in Log Space)
def log_prior(mu, sigma):
    # Uniform uninformative prior: mu must be between [-10, 10], sigma > 0
    if -10.0 < mu < 10.0 and 0.1 < sigma < 10.0:
        return 0.0 # log(1)
    return -np.inf # log(0)

def log_likelihood(mu, sigma, data):
    # Log-probability density function of the Normal distribution
    return np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - ((data - mu)**2) / (2 * sigma**2))

def log_posterior(mu, sigma, data):
    lp = log_prior(mu, sigma)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(mu, sigma, data)
# 3. Metropolis-Hastings MCMC Sampler
def run_mcmc(data, iterations=10000):
    # Start with a terrible initial guess
    current_mu = 0.0
    current_sigma = 1.0
    
    # Store the trace (history) of our steps
    trace = np.zeros((iterations, 2))
    
    accepted = 0
    
    for i in range(iterations):
        # Propose a new state using a Gaussian random walk
        prop_mu = np.random.normal(current_mu, 0.1)
        prop_sigma = np.random.normal(current_sigma, 0.1)
        
        # Calculate log posterior of current and proposed states
        curr_log_post = log_posterior(current_mu, current_sigma, data)
        prop_log_post = log_posterior(prop_mu, prop_sigma, data)
        
        # Acceptance logic using log probabilities for numerical stability
        # log(prop) - log(curr) = log(prop/curr)
        log_accept_ratio = prop_log_post - curr_log_post
        
        # Accept if ratio > 0 (proposed is better), or randomly based on ratio
        if np.log(np.random.rand()) < log_accept_ratio:
            current_mu = prop_mu
            current_sigma = prop_sigma
            accepted += 1
            
        trace[i, 0] = current_mu
        trace[i, 1] = current_sigma
        
    print(f"Acceptance Rate: {accepted / iterations:.2%}")
    return trace

print("Running MCMC Sampler...")
trace = run_mcmc(data, iterations=15000)
# 4. Burn-in and Plotting
# Discard the first 30% of samples where the algorithm was still "searching" for the main distribution
burn_in = int(0.3 * len(trace))
trace_burned = trace[burn_in:]

estimated_mu = np.mean(trace_burned[:, 0])
estimated_sigma = np.mean(trace_burned[:, 1])

print(f"True Mu: {true_mu} | Estimated Mu: {estimated_mu:.3f}")
print(f"True Sigma: {true_sigma} | Estimated Sigma: {estimated_sigma:.3f}")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Trace plots
axes[0, 0].plot(trace[:, 0], color='blue', alpha=0.7)
axes[0, 0].axhline(true_mu, color='red', linestyle='--')
axes[0, 0].set_title(r"Trace of $\mu$")

axes[0, 1].plot(trace[:, 1], color='green', alpha=0.7)
axes[0, 1].axhline(true_sigma, color='red', linestyle='--')
axes[0, 1].set_title(r"Trace of $\sigma$")

# Posterior Histograms
axes[1, 0].hist(trace_burned[:, 0], bins=50, color='blue', density=True, alpha=0.7)
axes[1, 0].axvline(true_mu, color='red', linestyle='--', label="True $\mu$")
axes[1, 0].set_title(r"Posterior Distribution of $\mu$")
axes[1, 0].legend()

axes[1, 1].hist(trace_burned[:, 1], bins=50, color='green', density=True, alpha=0.7)
axes[1, 1].axvline(true_sigma, color='red', linestyle='--', label="True $\sigma$")
axes[1, 1].set_title(r"Posterior Distribution of $\sigma$")
axes[1, 1].legend()

plt.tight_layout()
plt.show()