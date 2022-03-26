import numpy as np
from tqdm import tqdm
from scipy.optimize import differential_evolution

from utils import clean
from geometries import circles_grid, add_circle


class Individual:
    def __init__(self, x, model):
        self.indices = self.get_indices(x)
        self.model = model
        self.n = int(np.sqrt(len(x)))

        self.air = model/'selections'/'air'
        self.plastic = model/'selections'/'plastic'
        self.selections = model/'selections'
        self.amodel = model/'geometries'/'Geometry 1'

        clean(self.amodel, 'circle')
        clean(self.selections, 'circle')
        clean(self.amodel, 'square')
        clean(self.selections, 'square')

        self.initial_params = {
            'xlim': (-0.05, 0.05),
            'ylim': (-0.05, 0.05),
            'n': self.n, 
        }
        
    @staticmethod
    def get_indices(x):
        return np.nonzero(x)

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

                if idx in list(indices[0]):
                    node, node_sel = add_circle(
                        name, 
                        x_i, y_j, 
                        self.amodel, self.selections, 
                        radius, alpha
                    )        
                    node_selections.append(node_sel)

                else:
                    node_selections.append(None)

                idx += 1

        plastic.property('input', list(np.array(node_selections)[self.indices]))
        self.model.build(self.amodel)

    def solve(self):
        self.model.mesh()
        self.model.solve()

    def fitness(self):
        return 0


def transform_to_binary(x):
    return [int(x_i > 0.5) for x_i in x]


def fitness(x):
    x = transform_to_binary(x)
    ind = Individual(x, model=None)
    ind.create_model()
    ind.solve()
    return ind.fitness()


def simple_genetic(n):
    bounds = [(0, 1) for i in range(n**2)]
    result = differential_evolution(fitness, bounds, seed=1, maxiter=100)
    return result.x, result.fun