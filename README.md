# Vanilla Metropolis-Hastings MCMC

A from-scratch, library-free implementation of a Markov Chain Monte Carlo (MCMC) sampler for Bayesian parameter estimation. 

This project bypasses high-level probabilistic programming frameworks (like PyMC or Stan) to explicitly implement the underlying mathematics of the Metropolis-Hastings algorithm. It is designed to demonstrate mathematical rigor, probability theory, and the mechanics of generative inference.

##  Mathematical Architecture

* **Metropolis-Hastings Algorithm:** Implements the core random-walk proposal mechanism and acceptance ratio logic to explore the parameter space.
* **Log-Space Computations:** Calculates prior, likelihood, and posterior probabilities entirely in log-space (`log_prior`, `log_likelihood`) to ensure numerical stability and prevent arithmetic underflow when working with tiny probability densities.
* **Trace & Burn-in Analysis:** Manages Markov chain history, automatically discarding the initial 30% exploration phase (burn-in) to calculate accurate posterior distributions.

##  Execution

Requires Python with `numpy` and `matplotlib`.

```bash
git clone [https://github.com/meeksha/vanilla-mcmc.git](https://github.com/meeksha/vanilla-mcmc.git)
cd vanilla-mcmc
python mcmc.py
