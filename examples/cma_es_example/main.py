import numpy as np
from cmaes import CMA
from scipy.stats import linregress
import logging
from hpctool import Model, Task, Solver
import sys
import matplotlib.pyplot as plt


class SimpleIndividual:
    def __init__(self, bounds: np.ndarray):
        self.bounds = bounds
        self.N = len(bounds)
        self.mean = np.zeros(self.N)


class MyModel(Model):
    def results(self, x, *args, **kwargs):
        return fitness(x)


def get_slope(values):
    return abs(linregress([i for i in range(len(values))], values).slope)


def fitness(x):
    """Function which want to MINIMIZE"""
    x1, x2 = x
    return ((x1 - 1) ** 2 + 3 + x2 * 0)


def fitness_maximize_determinant(x):
    """Function which want to MINIMIZE"""
    diagonal_elements = x
    n = len(x)
    mat = np.identity(n)
    for i in range(n):
        mat[i, i] = diagonal_elements[i]

    return -np.linalg.det(mat)


def cma_es_optimizer(
        individual,
        solver: Solver,
        iterations_max=1000,
):
    """Minimizer"""
    population_size = individual.N
    optimizer = CMA(
        mean=individual.mean,
        sigma=1.3,
        bounds=individual.bounds,
        seed=42,
        population_size=population_size
    )

    num_to_slope = 10
    eps, slope = 1e-4, np.Inf
    best_values, best_value, best_individual = [], np.Inf, None
    mean_values = []
    best_generations = []

    for generation in range(iterations_max):
        solutions = []
        slope = (get_slope(mean_values[-num_to_slope:])
                 if len(mean_values) > num_to_slope
                 else np.Inf)
        if slope < eps:
            break

        tasks = []
        for _ in range(population_size):
            x = optimizer.ask()
            tasks.append(Task(x=x, tag=str(x)))

        res = solver.solve(tasks)

        tmp = []
        for value, task in zip(res, tasks):
            tmp.append(value)
            x = task.kwargs.get('x')
            if value < best_value:
                best_value = value
                best_individual = x
                best_values.append(best_value)
                best_generations.append(generation)
            solutions.append((x, value))

        logger.info("{} GEN | best value {} | slope {} |".format(generation, round(best_value, 5), np.round(slope, 5)))
        mean_values.append(np.array(tmp).mean())
        optimizer.tell(solutions)

    logger.info(f'Best value is {best_value}')
    logger.info(f'Best individual is {best_individual}')

    res = {
        'best_value': best_value,
        'best_individual': best_individual,
        'best_values': (best_generations, best_values),
        'mean_values': mean_values
    }

    return res


logger = logging.getLogger('main')
gs = logging.getLogger('hpctool.solver')
gs.addHandler(logging.StreamHandler(sys.stdout))
gs.setLevel(logging.INFO)
# gw = logging.getLogger('hpctool.worker')
# gw.addHandler(logging.StreamHandler(sys.stdout))
# gw.setLevel(logging.DEBUG)


def main(solver: Solver):
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    ind = SimpleIndividual(np.array([(-2, 2) for i in range(2)]))
    result = cma_es_optimizer(ind, solver)
    mean_values = result['mean_values']
    best_x, best_y = result['best_values']

    plt.plot(range(len(mean_values)), mean_values)
    plt.scatter(best_x, best_y)
    plt.savefig('image.png')
    logger.info('done')
