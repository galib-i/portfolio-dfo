import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def enforce_constraints(weights, dimensions, cardinality):
    """Applies constraints to all found raw weights: limits to a specific amount of stocks, not negative and sum to 1 (100% of portfolio)"""
    if dimensions > cardinality:
        sorted_indices = np.argsort(weights)
        weights[sorted_indices[:-cardinality]] = 0.0

    weights = np.maximum(weights, 0.0)
    total_weight = np.sum(weights)

    if total_weight > 0.000001:  # Fixes errors where weight is included when 0.0 (rounding error)
        weights /= total_weight
    else:
        weights = np.ones(dimensions) / dimensions

    return weights


def display_portfolio(weights, dataset):
    """Maps weights to ticker names and remove ones that are worth 0%."""
    portfolio = pd.Series(weights, index=dataset.columns)
    portfolio = portfolio[portfolio > 0].sort_values(ascending=False)

    print(f"A portfolio of {len(portfolio)} assets:")
    print(portfolio.to_string(float_format="{:.2%}".format))

    return portfolio


def get_portfolio_value_over_time(weights, returns, initial_amount):
    """Given asset weights and returns, calculates the cumulative value used to draw graphs.

    Returns:
        pd.Series: Cumulative portfolio value for each day.
    """
    weights = np.array(weights)
    daily_returns = returns.dot(weights)

    return initial_amount * (1 + daily_returns).cumprod()


def plot_dfo_vs_pso(dfo_data, pso_data):
    """Plot DFO vs PSO to predict 2023-2025"""
    plt.figure(figsize=(10, 6))
    plt.plot(dfo_data, label="DFO", color="green")
    plt.plot(pso_data, label="PSO", color="orange")

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (£)")
    plt.legend()


def plot_dfo_vs_benchmark(out_of_sample, dfo_data, eq_data, mc_data):
    """Plot DFO vs Monte Carlo vs Equal Weights to predict 2023-2025"""
    plt.figure(figsize=(10, 6))
    mc_min = np.min(mc_data, axis=0)
    mc_max = np.max(mc_data, axis=0)

    plt.fill_between(
        out_of_sample.index,
        mc_min,
        mc_max,
        color="gainsboro",
        alpha=0.4,
        label="Monte Carlo",
    )
    plt.plot(dfo_data, label="DFO", color="green")
    plt.plot(eq_data, label="Equal Weight", color="cornflowerblue")

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (£)")
    plt.legend()
    plt.tight_layout()
