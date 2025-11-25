# Bayesian Inference on Meningitis Incidence via MCMC

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Statistics](https://img.shields.io/badge/Statistics-Bayesian_Inference-orange)
![Algorithm](https://img.shields.io/badge/Algorithm-Metropolis_within_Gibbs-red)

## 📊 Executive Summary

This project implements a **Hierarchical Bayesian Model** to analyze the incidence of bacterial meningitis cases across multiple hospitals over several years.

The core of the project is a custom implementation of a **Metropolis-Hastings within Gibbs** sampler, written from scratch in Python without relying on probabilistic programming frameworks (like PyMC or Stan). The algorithm approximates the posterior distributions of the annual infection rates ($\lambda_j$) and the global hyperparameter ($\mu$).

> **Context:** This repository represents the practical output of the "Computational Statistics" course at Politecnico di Torino (M.Sc. in Mathematical Engineering).

---

## 🧮 Mathematical Framework

The problem models the count of meningitis cases $Y_{i,j}$ in hospital $i$ during year $j$ using a Poisson-Gamma hierarchical structure:

### 1. The Hierarchical Model
* **Likelihood:** Observed cases follow a Poisson distribution governed by a yearly rate $\lambda_j$:
    $$Y_{i,j} \mid \lambda_j \sim \text{Poisson}(\lambda_j)$$
* **Prior (Latent Variables):** The yearly rates flucuate around a global mean $\mu$:
    $$\lambda_j \mid \mu \sim \text{Gamma}(\mu^2, \mu) \quad \text{(parametrized by mean and variance)}$$
* **Hyperprior:** A non-informative Uniform prior is placed on the global mean:
    $$\mu \sim \text{Uniform}(0, 10)$$

### 2. Inference Strategy
Since the joint posterior is analytically intractable, we derive the full conditional distributions to implement a hybrid MCMC sampler:
* **Gibbs Step:** Used for $\lambda_j$, as the full conditional is a known Gamma distribution (due to conjugacy).
* **Metropolis-Hastings Step:** Used for $\mu$, as its full conditional does not have a standard form. An **Adaptive Gaussian Proposal** is implemented to optimize the acceptance rate during the burn-in phase.

---

## 🛠️ Code Structure & Algorithms

The codebase is structured using a functional programming approach. The key components implemented in `Meningitis.py` are:

### Core Algorithms
* `mcmc_algorithm(mu, lb_mu, ub_mu, eta, B, y)`: The main engine. It runs the hybrid sampler, updating $\mu$ via Metropolis and the vector $\boldsymbol{\lambda}$ via Gibbs sampling at each iteration. It includes an adaptive tuning mechanism for the proposal variance $\eta$.
* `burnin(...)` & `thin(...)`: Post-processing functions to remove the transient phase of the chain and reduce autocorrelation between samples.

### Statistical Analysis Tools
* `cintervals(lambda_final)`: Computes 95% Bayesian Credible Intervals for the pairwise differences between years ($\lambda_l - \lambda_h$) to detect statistically significant changes in infection rates.
* `montecarlo_approx(...)`: Estimates the **Posterior Predictive Distribution** for a new observation ($Y_{new}$) using the **Rao-Blackwell estimator** to reduce variance compared to simple histograms.

---

## 🖼️ Results Visualization

### 1. Parameter Estimation
The sampler successfully converges to the target posterior distributions.
*(Insert here your Traceplot or Posterior Density Plot)*
![Posterior Distributions](img/posterior_plot.png)

### 2. Trend Analysis (Ribbon Plot)
Using the sampled $\lambda_j$, we can visualize the trend of infection rates over the years with their associated uncertainty (95% Credible Intervals).
*(Insert here your Ribbon/Trend Plot)*
![Trend Analysis](img/ribbon_plot.png)

### 3. Predictive Modeling
The Rao-Blackwellized estimator provides a smooth probability mass function for the number of cases in a future scenario.
*(Insert here your Predictive Distribution Lollipop Plot)*
![Predictive PMF](img/predictive_plot.png)

---

## 🚀 Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/matteoientile/Meningitis-Bayesian-Inference.git](https://github.com/matteoientile/Meningitis-Bayesian-Inference.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib scipy seaborn
    ```
3.  **Run the Analysis:**
    You can import the functions in your own script or run the provided notebook. Example usage:

    ```python
    import numpy as np
    import pandas as pd
    from Homework1_357948_Ientile_File1 import mcmc_algorithm, burnin, thin

    # Load data
    y_data = pd.read_csv('data/meningitis_data.csv').to_numpy()

    # Run MCMC (22k iterations)
    lambdas, mus = mcmc_algorithm(mu=8, lb_mu=5, ub_mu=10, eta=1, B=22000, y=y_data)

    # Post-processing
    mu_clean, lambda_clean = burnin(mus, lambdas, burn=2000)
    mu_final, lambda_final = thin(mu_clean, lambda_clean, thin=10)
    ```

---

## 🔮 Future Improvements
* [ ] **Stan Implementation:** Validation of results using `CmdStanPy` for HMC sampling (Coming Soon).
* [ ] **Class Refactoring:** Encapsulate the model logic into a Python class for better state management (OOD).

## 👤 Author

**Matteo Ientile**
* M.Sc. Student in Mathematical Engineering @ Politecnico di Torino
