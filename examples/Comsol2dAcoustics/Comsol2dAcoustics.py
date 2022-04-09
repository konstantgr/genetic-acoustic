import sys
sys.path.append('')

from gendev import ComsolModel, Task, Solver
import numpy as np
from utils import grid, pretty_print_individual, get_multipoles_from_res
import os
from loguru import logger
from typing import List, Dict
from scipy.optimize import differential_evolution


class SquaresModel(ComsolModel):
    def __init__(self):
        super().__init__()
        self.geometry = self / 'geometries' / 'Geometry 1'
        self.config = {
            "n": n,
            "x_limits": (-0.03, 0.03),
            "y_limits": (-0.03, 0.03),
        }

    def configure(self):
        self.geometry.java.autoRebuild('off')
        self.parameter('max_freq', '1000[Hz]')
        self.parameter('min_freq', '100[Hz]')
        self.parameter('step', '100[Hz]')

    def pre_build(self, x, *args, **kwargs):
        indices = np.nonzero(x)
        node_selections = []

        xgrid, ygrid = grid(**self.config)
        tau = abs(xgrid[1] - xgrid[0])
        width = tau

        idx = 0
        for x_i in xgrid:
            for y_j in ygrid:
                name = f"circle_xi_{x_i}, yj_{y_j}"

                if idx in list(indices[0]):
                    node, node_sel = self.add_square(name, x_i, y_j, self.geometry, width)
                    node_selections.append(node_sel)
                else:
                    node_selections.append(None)
                idx += 1

        (self/'selections'/'plastic').property(
            'input', list(np.array(node_selections)[indices])
        )

    def results(self, x, *args, **kwargs):
        evaluation = self / 'evaluations' / 'Global Evaluation 1'
        dataset = (self / 'datasets').children()[0]
        return self.global_evaluation(dataset, evaluation)

    def pre_clear(self, x, save=False, *args, **kwargs):
        if save:
            self.save(save_path)
            self.plot2d('acpr.p_s', image_path)
        self.clean_geometry(self.geometry, 'circle')


n = 3
dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname, 'empty_project.mph')
save_path = os.path.join(dirname, 'empty_project1.mph')
image_path = os.path.join(dirname, 'image.png')


def transform_to_binary_list(x):
    return [int(x_i > 0.5) for x_i in x]


def fitness(x: List, info: Dict, solver: Solver):
    x = transform_to_binary_list(x)

    data = solver.solve([Task(x=x, tag=str(x))])
    data = data[0]

    Q_multipoles = get_multipoles_from_res(data, c=343, R=0.18)
    res = -np.real(np.max(Q_multipoles[2]))

    individual_string = "".join(np.array(x).astype(str))

    if res < info['best']:
        info['best'] = res
        message = f"iteration {info['iteration']} | individual {individual_string} | result {round(res, 4)}"
        logger.log("best", message)

    print('=' * 30)
    print('({}).  {:.4f} in {:.1f}s [BEST: {:.4f}]'.format(
        info['iteration'], res,
        # ind.getLastComputationTime() / 1000,
        0,
        info['best']))
    print(pretty_print_individual(x))
    print('=' * 30)

    logger.info(
        f"[BEST {round(info['best'], 4)}]\titeration {info['iteration']}\tindividual {individual_string}\tresult {round(res, 4)}\tcalculation_time {0}")
    message = f"iteration {info['iteration']} | individual {individual_string} | result {round(res, 4)}"
    logger.log("individuals", message)

    info['iteration'] += 1

    return res


def differential_evolution_scipy(solver: Solver):
    bounds = [(0, 1) for _ in range(n ** 2)]
    print('SciPy Differential Evolution started...')
    result = differential_evolution(
        fitness, bounds,
        args=({'iteration': 0, 'best': np.Inf}, solver, ),
        maxiter=0, popsize=1, seed=2
    )
    return result.x, result.fun
