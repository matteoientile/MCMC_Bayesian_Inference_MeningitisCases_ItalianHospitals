# Bayesian Inference of Meningitis Mortality Rates via MCMC

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Statistics](https://img.shields.io/badge/Statistics-Bayesian_Inference-orange)
![Algorithm](https://img.shields.io/badge/Algorithm-Metropolis--Hastings-red)

## 📊 Executive Summary

This project performs a statistical analysis of mortality rates in *H. influenzae* meningitis cases using **Bayesian Inference**. 

Instead of relying solely on deterministic point estimates, this framework models the mortality rate $\theta$ as a random variable. It implements a custom **Markov Chain Monte Carlo (MCMC)** sampler using the **Metropolis-Hastings algorithm** from scratch to approximate the posterior distribution.

The custom implementation is validated against the theoretical analytical solution (Conjugate Priors) to ensure algorithmic correctness before future deployment using probabilistic programming languages (e.g., STAN).

## 🧮 Mathematical Framework

The goal is to infer the unknown mortality probability $\theta$ given observed clinical data $Y$.

### 1. The Model
* **Likelihood:** The observed deaths $y$ out of $n$ cases are modeled as a Binomial process:
    $$Y \mid \theta \sim \text{Bin}(n, \theta)$$
    $$L(\theta \mid y) \propto \theta^y (1-\theta)^{n-y}$$

* **Prior:** We assume a non-informative uniform prior (Beta distribution) to let the data drive the inference:
    $$\theta \sim \text{Beta}(\alpha=1, \beta=1)$$

* **Posterior (Target):** By conjugacy, the theoretical posterior is known:
    $$\pi(\theta \mid y) \sim \text{Beta}(\alpha + y, \beta + n - y)$$

### 2. The Algorithm (Metropolis-Hastings)
Since we cannot always calculate the normalization constant in complex models, we use MCMC to sample from the posterior.
The proposal distribution $q$ is a uniform random walk centered at the current state:
$$\theta^* \sim \mathcal{U}(\theta_{t-1} - \epsilon, \theta_{t-1} + \epsilon)$$

The acceptance ratio $\alpha$ is computed as:
$$\alpha = \min \left( 1, \frac{\pi(\theta^*)}{\pi(\theta_{t-1})} \right)$$

---

## 🖼️ Simulation Results

### Convergence Diagnostics
The algorithm was run for $10,000$ iterations. We analyzed the **Trace Plot** to assess mixing and stationarity, and the **Autocorrelation Function (ACF)** to evaluate sample independence.

![Trace Plot](LINK_TO_YOUR_TRACEPLOT_IMAGE.png)
*(Fig 1: Trace plot showing rapid convergence to the high-density region)*

### Posterior Distribution
The sampled histogram perfectly matches the theoretical Beta distribution, validating the custom sampler implementation.

![Posterior Dist](LINK_TO_YOUR_Posterior_vs_Theoretical_IMAGE.png)
*(Fig 2: MCMC Histogram vs. Analytical Posterior Density)*

**Key Findings:**
* **Posterior Mean:** $\approx 19.5\%$ mortality rate.
* **95% Credible Interval:** Calculated from the empirical quantiles of the chain.

---

## 🛠️ Code Structure

The project is built using Object-Oriented Programming (OOP) in Python to ensure modularity.

* `Homework1_357948_Ientile_File1.py`: Contains the core `MCMC_mh` class (Logic & Sampling).
    * `__init__`: Sets up priors and observed data.
    * `run()`: Executes the Metropolis-Hastings loop.
    * `plot_results()`: visualizes trace plots and histograms.
* `Homework1_Code.ipynb`: Jupyter Notebook for data ingestion, exploration, and execution of the analysis.

### Snippet: The Metropolis-Hastings Step
```python
# Core logic for the acceptance step
if self.posterior(theta_star) >= self.posterior(self.chain[i-1]):
    self.chain[i] = theta_star
    self.acceptance_rate += 1
else:
    u = np.random.uniform(0, 1)
    ratio = self.posterior(theta_star) / self.posterior(self.chain[i-1])
    if u < ratio:
        self.chain[i] = theta_star
        self.acceptance_rate += 1
    else:
        self.chain[i] = self.chain[i-1]


