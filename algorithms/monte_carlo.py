import numpy as np


class MonteCarlo:
    def __init__(self, portfolio_size, simulations=1_000):
        self.portfolio_size = portfolio_size
        self.simulations = simulations

    def generate_portfolios(self, returns, get_portfolio_value_over_time):
        """Generates simulations and calculates their portfolio value over time.

        Args:
            returns (pd.DataFrame): Returns for the sample to be predicted.
            get_portfolio_value_over_time (function): Function that calculates £.

        Returns:
            list: A list of all generated values.
        """
        portfolios = []
        for _ in range(self.simulations):
            random_weights = np.random.random(self.portfolio_size)
            random_weights /= random_weights.sum()  # Normalise to sum to 1

            path = get_portfolio_value_over_time(random_weights, returns)
            portfolios.append(path)

        return portfolios
