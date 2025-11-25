import numpy as np
from utils.helpers import enforce_constraints


class DFO:
    def __init__(
        self,
        fitness_function,
        cardinality,
        population,
        dimensions,
        disturbance,
        max_iterations,
        lower_bound,
        upper_bound,
    ):
        self._fitness_function = fitness_function
        self._cardinality = cardinality
        self._population = population
        self._dimensions = dimensions
        self._disturbance = disturbance
        self._max_iterations = max_iterations

        # Intialise bounds in all dimensions
        self._lower_bounds = np.full(dimensions, lower_bound)
        self._upper_bounds = np.full(dimensions, upper_bound)

        # Initialise flies within bounds randomly
        self._flies = np.random.uniform(
            self._lower_bounds,
            self._upper_bounds,
            (population, dimensions),
        )

        # Ensure weights sum to 1
        self._flies = self._flies / np.sum(self._flies, axis=1, keepdims=True)

        self._fitness = np.empty(self._population)

    def _evaluate_fitness(self):
        """Applies constraints to each fly before evaluating each fitness while keeping the position unchanged"""
        for i, fly in enumerate(self._flies):
            valid_portfolio = enforce_constraints(fly, self._dimensions, self._cardinality)

            value = self._fitness_function(valid_portfolio)
            self._fitness[i] = value

    def run(self):
        for i in range(self._max_iterations):
            self._evaluate_fitness()
            best_fly = np.argmax(self._fitness)

            if i % 100 == 0:
                print(
                    f"Iteration: {i}\tBest fly index: {best_fly}\tFitness value: {self._fitness[best_fly]}"
                )

            for fly_index in range(self._population):
                if fly_index == best_fly:
                    continue  # Elitist strategy

                # Find best neighbour
                left_n = (fly_index - 1) % self._population
                right_n = (fly_index + 1) % self._population
                best_n = right_n if self._fitness[right_n] > self._fitness[left_n] else left_n

                for dim in range(self._dimensions):  # Update each dimension separately
                    if np.random.rand() < self._disturbance:
                        self._flies[fly_index, dim] = np.random.uniform(
                            self._lower_bounds[dim],
                            self._upper_bounds[dim],
                        )
                        continue

                    randomness = np.random.rand()
                    self._flies[fly_index, dim] = self._flies[best_n, dim] + randomness * (
                        self._flies[best_fly, dim] - self._flies[fly_index, dim]
                    )

                    # Out of bound control (only within [0, 1])
                    if not (
                        self._lower_bounds[dim]
                        <= self._flies[fly_index, dim]
                        <= self._upper_bounds[dim]
                    ):
                        self._flies[fly_index, dim] = np.random.uniform(
                            self._lower_bounds[dim],
                            self._upper_bounds[dim],
                        )

        self._evaluate_fitness()
        best_fly_idx = np.argmax(self._fitness)
        self.constrained_fly = enforce_constraints(
            self._flies[best_fly_idx], self._dimensions, self._cardinality
        )

        return self._fitness[best_fly_idx], self.constrained_fly
