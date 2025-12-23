# Cardinality-Constrained Portfolio Optimisation via Dispersive Flies Optimisation and Sharpe Ratio

This project investigates the use of Dispersive Flies Optimisation (DFO) to create an optimal 10-stock, cardinality-constrained portfolio by maximising the Sharpe ratio on historical S&P500 data, for COMP1805 - Natural Computing.

> [!NOTE]
> Market conditions are simplified to highlight the optimisation capability of DFO in comparison to Particle Swarm Optimisation (PSO).

## Setup
### Steps
1. Clone or [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) this repository
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Open `portfolio_optimisation.ipynb`

## Overview
The dataset consists of daily returns for 111 S&P500 stocks (09/2015-09/2025), split into in-sample (≤09/2023) and out-of-sample (2023-2025) periods, with the Sharpe ratio as the fitness function (optimisation objective).

$$
f(\mathbf{x}) = \text{Sharpe ratio} = \frac{R_p - R_f}{\sigma_p}
$$

$$
\begin{aligned}
R_p &= \text{Return of portfolio} \\
R_f &= \text{Risk-free rate (theoretical no-risk return)} \\
\sigma_p &= \text{Standard deviation of portfolio excess returns (volatility)}
\end{aligned}
$$

DFO is compared against PSO, equal-weighted and Monte Carlo-generated portfolios, achieving higher and more consistent Sharpe ratios across multiple runs.
