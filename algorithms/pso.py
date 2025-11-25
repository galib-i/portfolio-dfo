import numpy as np
import pyswarms as ps
from utils.helpers import enforce_constraints


class PSO:
    def __init__(
        self,
        fitness_function,
        cardinality,
        population,
        dimensions,
        max_iterations,
        lower_bound,
        upper_bound,
        options,
    ):
        self._fitness_function = fitness_function
        self._cardinality = cardinality
        self._population = population
        self._dimensions = dimensions
        self._max_iterations = max_iterations

        # Intialise bounds in all dimensions
        self._lower_bounds = np.full(dimensions, lower_bound)
        self._upper_bounds = np.full(dimensions, upper_bound)

        self._fitness = np.empty(self._population)
        # Hyperparameters: cognitive, social, inertia weight
        # e.g. default {"c1": 1.5, "c2": 1.5, "w": 0.7}
        self._options = options

    def _evaluate_fitness(self, particles):
        """Applies constraints to each particle before evaluating each fitness while keeping the position unchanged."""
        for i, particle in enumerate(particles):
            valid_portfolio = enforce_constraints(particle, self._dimensions, self._cardinality)

            value = self._fitness_function(valid_portfolio)
            self._fitness[i] = value

        return -self._fitness  # "-"" because pyswarms minimises

    def run(self):
        pso = ps.single.GlobalBestPSO(
            n_particles=self._population,
            dimensions=self._dimensions,
            bounds=(self._lower_bounds, self._upper_bounds),
            options=self._options,
        )

        # Find the best not yet constrained values, temporary evaluations (all values)
        best_cost, best_raw_position = pso.optimize(
            self._evaluate_fitness, iters=self._max_iterations
        )

        # Apply the constraints to the best result to find the 10-stock portfolio
        final_portfolio = enforce_constraints(
            best_raw_position, self._dimensions, self._cardinality
        )
        print(f"Final portfolio after enforcing constraints again: {final_portfolio}")
        return -best_cost, final_portfolio
