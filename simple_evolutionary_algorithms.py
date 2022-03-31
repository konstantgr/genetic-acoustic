import numpy as np
from scipy.optimize import differential_evolution
from typing import List, Dict

from individuals import CircleIndividual
from fitness_functions import high_peaks


def pretty_print_individual(ind: List):
    n = int(np.sqrt(len(ind)))
    reshaped_ind = np.array(ind).T.reshape(n, n)

    return '\n'.join([
        ' '.join(['o' if not reshaped_ind[i][j] else 'x' for j in range(n)])
        for i in range(n)
    ])


def transform_to_binary_list(x):
    return [int(x_i > 0.5) for x_i in x]


def fitness(x: List, model, info: Dict):
    x = transform_to_binary_list(x)

    ind = CircleIndividual(x, model=model)
    ind.create_model()
    ind.solve_geometry()

    res = ind.fitness(func=high_peaks)

    print('=' * 30)
    print('({}).  {:.4f} in {:.1f}s'.format(
        info['iteration'], res,
        ind.getLastComputationTime() / 1000)
    )
    print(pretty_print_individual(x))
    print('=' * 30)

    info['iteration'] += 1
    return abs(res)


def differential_evolution_circles_scipy(model, n=2):
    bounds = [(0, 1) for _ in range(n ** 2)]

    result = differential_evolution(
        fitness, bounds,
        args=(model, {'iteration': 0},),
        maxiter=0, popsize=1, seed=2
    )
    return result.x, result.fun
