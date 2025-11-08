import numpy as np


class DFO:
    def __init__(
        self,
        fitness_function,
        population,
        dimensions,
        disturbance,
        max_iterations,
        lower_bounds,
        upper_bounds,
    ):
        self._fitness_function = fitness_function
        self._population = population
        self._dimensions = dimensions
        self._disturbance = disturbance
        self._max_iterations = max_iterations

        # Intialise bounds in all dimensions
        self._lower_bounds = np.full(dimensions, lower_bounds)
        self._upper_bounds = np.full(dimensions, upper_bounds)

        # Initialise flies within bounds randomly
        self._flies = np.random.uniform(
            self._lower_bounds,
            self._upper_bounds,
            (population, dimensions),
        )

        self._fitness = np.empty(self._population)  # Empty fitness array (population size)

    def _evaluate_fitness(self):
        """Evaluates fitness of each fly"""
        self._fitness = [self._fitness_function(fly) for fly in self._flies]

    def run(self):
        for i in range(self._max_iterations):
            self._evaluate_fitness()
            best_fly = np.argmin(self._fitness)

            if i % 100 == 0:
                print(
                    f"Iteration: {i}\tBest fly index: {best_fly}\tFitness value: {self._fitness[best_fly]}"
                )

            for fly in range(self._population):
                if fly == best_fly:
                    continue  # Elitist strategy

                # Find best neighbour
                left_n = (fly - 1) % self._population
                right_n = (fly + 1) % self._population
                best_n = right_n if self._fitness[right_n] < self._fitness[left_n] else left_n

                for dim in range(self._dimensions):  # Update each dimension separately
                    if np.random.rand() < self._disturbance:
                        self._flies[fly, dim] = np.random.uniform(
                            self._lower_bounds[dim],
                            self._upper_bounds[dim],
                        )
                        continue

                    randomness = np.random.rand()
                    self._flies[fly, dim] = self._flies[best_n, dim] + randomness * (
                        self._flies[best_fly, dim] - self._flies[fly, dim]
                    )
                    # Out of bound control
                    if not (
                        self._lower_bounds[dim] <= self._flies[fly, dim] <= self._upper_bounds[dim]
                    ):
                        self._flies[fly, dim] = np.random.uniform(
                            self._lower_bounds[dim],
                            self._upper_bounds[dim],
                        )

        self._evaluate_fitness()
        best_fly = np.argmin(self._fitness)

        return self._fitness[best_fly], self._flies[best_fly]


if __name__ == "__main__":

    def sphere(x):
        return np.sum(np.square(x))

    dfo = DFO(
        fitness_function=sphere,
        population=100,
        dimensions=30,
        disturbance=0.001,
        max_iterations=1000,
        lower_bounds=-5.12,
        upper_bounds=5.12,
    )
    best_fitness, best_position = dfo.run()
    print("\nFinal best fitness:\t", best_fitness)
    print("\nBest fly position:\n", best_position)
