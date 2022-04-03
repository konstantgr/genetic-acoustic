import numpy as np
from loguru import logger
from scipy.optimize import differential_evolution
from typing import List, Dict

from comsol.individuals import CircleIndividual, SquareIndividual
from comsol.fitness_functions import high_peaks, max_sc, peaks_contribution


solved = {}
logger.add('logs/logs.log', level='INFO')


def pretty_print_individual(ind: List):
    n = int(np.sqrt(len(ind)))
    reshaped_ind = np.array(ind).reshape(n, n)

    return '\n'.join([
        ' '.join([' ' if not reshaped_ind[j][-i-1] else 'x' for j in range(n)])
        for i in range(n)
    ])


def transform_to_binary_list(x):
    return [int(x_i > 0.5) for x_i in x]


def fitness(x: List, model, info: Dict):
    x = transform_to_binary_list(x)
    if str(x) in solved:
        info['iteration'] += 1
        return solved[str(x)]

    ind = SquareIndividual(x, model=model)
    ind.create_model()
    print('Running')
    ind.solve_geometry()

    # res = ind.fitness(func=peaks_contribution, R=0.18, c=343, multipole_n=1)
    res = ind.fitness(func=high_peaks, R=0.18, c=343)
    # res = ind.fitness(func=max_sc, )

    if res < info['best']:
        info['best'] = res

    print('=' * 30)
    print('({}).  {:.4f} in {:.1f}s [BEST: {:.4f}]'.format(
        info['iteration'], res,
        ind.getLastComputationTime() / 1000,
        info['best']))
    print(pretty_print_individual(x))
    print('=' * 30)

    logger.info(f"iteration {info['iteration']}\tindividual {str(x)}\tresult {res}")

    info['iteration'] += 1
    solved[str(x)] = res

    return res


def differential_evolution_circles_scipy(model, n=2):
    bounds = [(0, 1) for _ in range(n ** 2)]

    print('SciPy Differential Evolution started...')
    result = differential_evolution(
        fitness, bounds,
        args=(model, {'iteration': 0, 'best': np.Inf},),
        maxiter=0, popsize=1, seed=2
    )
    return result.x, result.fun
