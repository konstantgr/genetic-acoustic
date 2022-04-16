from hpctool import ComsolModel, Task, Solver
import numpy as np
from utils import grid, linear_grid, pretty_print_individual, get_multipoles_from_res
import os
import sys
from loguru import logger
from typing import List, Dict
from scipy.optimize import differential_evolution
import logging


class MyModel(ComsolModel):
    def __init__(self):
        super().__init__()
        self.geometry = self / 'geometries' / 'Geometry 1'
        self.config = {
            "n": n,
            "x_limits": (-0.03, 0.03),
            "y_limits": (-0.03, 0.03),
        }

    # def configure(self):
    #     self.geometry.java.autoRebuild('off')
    #     self.parameter('max_freq', '1000[Hz]')
    #     self.parameter('min_freq', '100[Hz]')
    #     self.parameter('step', '100[Hz]')

    def pre_build(self, cylinders_length, cylinders_radii, cylinders_separations, *args, **kwargs):
        node_selections = []
        xgrid, ygrid = linear_grid(cylinders_radii, cylinders_separations)

        idx = 0
        for i, length in enumerate(cylinders_length):
            name = f"cyl_{i}"
            node, node_sel = self.add_cylinder(name, xgrid[i], ygrid[i], 0, self.geometry, length, cylinders_radii[i])
            node_selections.append(node_sel)
            idx += 1

        # node, node_sel = self.add_cylinder('cyl_xi_1 yj_1', 0, 0, 0, self.geometry, 0.00001, 0.00001)
        # (self/'selections'/'plastic').property(
        #     'input', list(np.array(node_selections)[indices])
        # )

    def results(self, x, *args, **kwargs):
        evaluation = self / 'evaluations' / 'Global Evaluation 1'
        dataset = (self / 'datasets').children()[0]
        return self.global_evaluation(dataset, evaluation)

    def pre_clear(self, x, save=False, *args, **kwargs):
        if save:
            self.save(save_path)
            self.plot2d('acpr.p_s', image_path)
        self.clean_geometry(self.geometry, 'cyl')


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
    print('({}).  {:.4f} [BEST: {:.4f}]'.format(
        info['iteration'], res,
        info['best']))
    print(pretty_print_individual(x))
    print('=' * 30)

    logger.info(
        f"[BEST {round(info['best'], 4)}]\titeration {info['iteration']}\tindividual {individual_string}\tresult {round(res, 4)}")
    message = f"iteration {info['iteration']} | individual {individual_string} | result {round(res, 4)}"
    logger.log("individuals", message)

    info['iteration'] += 1

    return res


def main(solver: Solver):
    fmt = "{time} | {level} |\t{message}"
    individuals_level = logger.level("individuals", no=38)
    bests_level = logger.level("best", no=38, color="<green>")
    logger.remove()
    logger.add(sys.stdout, level='INFO', format=fmt, enqueue=True)
    logger.add('logs/logs_{time}.log', level='INFO', format=fmt)
    logger.add('logs/individuals_{time}.log', format=fmt, level='individuals')

    # Solver logging
    _l = logging.getLogger('gendev')
    _l.setLevel(logging.DEBUG)
    _l.addHandler(logging.StreamHandler(sys.stdout))

    bounds = [(0, 1) for _ in range(n ** 2)]
    print('SciPy Differential Evolution started...')
    result = differential_evolution(
        fitness, bounds,
        args=({'iteration': 0, 'best': np.Inf}, solver, ),
        maxiter=0, popsize=1, seed=2
    )
    x = transform_to_binary_list(result.x)

    # Best individual
    solver.solve([Task(x=x, save=True, tag=str(x))])
    print(f'Project saved successfully, best result: {result.fun}')
