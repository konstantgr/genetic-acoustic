import numpy as np
from loguru import logger
from scipy.optimize import differential_evolution
from typing import List, Dict

from comsol.individuals import CircleIndividual, SquareIndividual
from comsol.fitness_functions import high_peaks, max_sc, peaks_contribution


solved = {}
individuals_level = logger.level("individuals", no=38)
bests_level = logger.level("best", no=38, color="<green>")
logger.add('logs/logs_{time}.log', level='INFO')

fmt = "{time} | {level} |\t{message}"
logger.add('logs/individuals_{time}.log', format=fmt, level='individuals')


def pretty_print_individual(ind: List):
    n = int(np.sqrt(len(ind)))
    reshaped_ind = np.array(ind).reshape(n, n)

    return '\n'.join([
        ' '.join([' ' if not reshaped_ind[j][-i-1] else 'x' for j in range(n)])
        for i in range(n)
    ])


def transform_to_binary_list(x):
    return [int(x_i > 0.5) for x_i in x]


def fitness(x: List, model, IndividualType, info: Dict):
    x = transform_to_binary_list(x)
    if str(x) in solved:
        info['iteration'] += 1
        return solved[str(x)]

    ind = IndividualType(x, model=model)
    ind.create_model()
    print('Running')
    ind.solve_geometry()

    # res = ind.fitness(func=peaks_contribution, R=0.18, c=343, multipole_n=1)
    res = ind.fitness(func=high_peaks, R=0.18, c=343)
    # res = ind.fitness(func=max_sc, )
    individual_string = "".join(np.array(x).astype(str))

    if res < info['best']:
        info['best'] = res
        message = f"iteration {info['iteration']} | individual {individual_string} | result {round(res, 4)}"
        logger.log("best", message)

    print('=' * 30)
    print('({}).  {:.4f} in {:.1f}s [BEST: {:.4f}]'.format(
        info['iteration'], res,
        ind.getLastComputationTime() / 1000,
        info['best']))
    print(pretty_print_individual(x))
    print('=' * 30)

    logger.info(f"[BEST {round(info['best'], 4)}]\titeration {info['iteration']}\tindividual {individual_string}\tresult {round(res, 4)}\tcalculation_time {ind.getLastComputationTime() / 1000}")
    message = f"iteration {info['iteration']} | individual {individual_string} | result {round(res, 4)}"
    logger.log("individuals", message)

    info['iteration'] += 1
    solved[str(x)] = res

    return res


def differential_evolution_scipy(model, IndividualType, n=2):
    bounds = [(0, 1) for _ in range(n ** 2)]

    print('SciPy Differential Evolution started...')
    result = differential_evolution(
        fitness, bounds,
        args=(model, IndividualType, {'iteration': 0, 'best': np.Inf},),
        maxiter=100, popsize=10, seed=2
    )
    return result.x, result.fun
