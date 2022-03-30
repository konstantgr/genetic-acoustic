import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import differential_evolution
import scipy.special as spc

from utils import clean, evaluate_global_ev
from geometries import circles_grid, add_circle


class Individual:
    def __init__(self, x, model):
        self.model = model
        self.n = int(np.sqrt(len(x)))

        self.air = model/'selections'/'air'
        self.plastic = model/'selections'/'plastic'
        self.selections = model/'selections'
        self.geom = model / 'geometries' / 'Geometry 1'

        self.geom.java.autoRebuild('off')

        self.clean_all()

        self.indices = self.get_indices(x)
        self.initial_params = {
            'xlim': (-0.03, 0.03),
            'ylim': (-0.03, 0.03),
            'n': self.n, 
        }

        self.dataset = None

    @staticmethod
    def get_indices(x):
        return np.nonzero(x)

    def clean_all(self):
        clean(self.geom, 'circle')
        clean(self.selections, 'circle')
        clean(self.geom, 'square')
        clean(self.selections, 'square')

        self.model.build(self.geom)
        self.model.clear()
        self.model.java.resetHist()

    def create_model(self):
        node_selections = []

        x, y = circles_grid(**self.initial_params)
        tau = abs(x[1] - x[0])
        radius = tau / 2
        alpha = 1.1
            
        idx = 0
        for x_i in tqdm(x):
            for y_j in y:
                name = f"circle_xi_{x_i}, yj_{y_j}"

                if idx in list(self.indices[0]):
                    node, node_sel = add_circle(name, x_i, y_j,
                                                self.geom, self.selections, radius, alpha)
                    node_selections.append(node_sel)
                else:
                    node_selections.append(None)

                idx += 1

        self.plastic.property('input', list(np.array(node_selections)[self.indices]))
        self.model.build(self.geom)

    def solve(self):
        self.model.mesh()
        self.model.solve()

        evaluation = self.model / 'evaluations' / 'Global Evaluation 1'
        dataset = (self.model / 'datasets').children()[0]
        self.dataset = evaluate_global_ev(dataset, evaluation)

    def getLastComputationTime(self):
        # ADD ANY STUDY SUPPORT
        studies = (self.model / 'studies').children()
        if len(studies) > 0:
            return int(studies[-1].java.getLastComputationTime())
        return -1

    def fitness(self):
        # ADD SIGMA_GEOM HERE!!!!
        results = self.dataset
        labels = [[f'{m}'] for m in range(10)]
        labels = list(results)[3:]
        k = 2*np.pi*results['Frequency']
        R = 0.18
        Q_multipoles = []

        multipoles = range(20)
        for m in multipoles:
            mu = 2 if m == 0 else 1
            bm = results[f'{m}'] / (spc.hankel1(m, R * k) * mu * np.pi * R)
            Q_multipoles.append(2 / k * mu * np.abs(bm) ** 2)

        Q_sc = results['sigma'].to_numpy()
        return np.real(np.sum(np.min((Q_sc - np.array(Q_multipoles)) / Q_sc, axis=1)))


def print_ind(a):
    n = int(np.sqrt(len(a)))
    new_a = np.array(a).T.reshape(n, n)

    s = ''
    for i in range(n):
        for j in range(n):
            s += 'o ' if not new_a[i][j] else 'x '
        s += '\n'
        
    return s


def transform_to_binary(x):
    return [int(x_i > 0.5) for x_i in x]


def fitness(x, model, info):
    x = transform_to_binary(x)
    ind = Individual(x, model=model)
    ind.create_model()
    ind.solve()
    res = ind.fitness()

    # print(res)
    # print('=====================================')

    # display information
    if info['cnt'] % 2 == 0:
        print('======================================================')
        print('({}).  {:.4f} in {:.1f}s'.format(info['cnt'], res, ind.getLastComputationTime()/1000))
        print(print_ind(x))
        print('======================================================')

    info['cnt'] += 1
    return abs(res)


def simple_genetic(model, n=2):
    bounds = [(0, 1) for _ in range(n**2)]
    # result = differential_evolution(
    #     lambda x: fitness(x, model), 
    #     bounds, seed=1, maxiter=10, popsize=10
    # )
    result = differential_evolution(
        fitness, bounds,
        args=(model, {'cnt': 0},), 
        maxiter=0, popsize=1, seed=2
    )
    return result.x, result.fun
