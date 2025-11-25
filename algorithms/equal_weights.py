import numpy as np


class EqualWeights:
    def __init__(self, portfolio_size):
        self.portfolio_size = portfolio_size

    def get_weights(self):
        return np.ones(self.portfolio_size) / self.portfolio_size
