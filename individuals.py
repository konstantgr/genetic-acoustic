import numpy as np

from individual_base import Individual
from utils import ComsolModelAttributes, get_indices
from geometries import circles_grid, add_circle


class CircleIndividual(Individual):
    def __init__(self, x, model):
        super().__init__(x, model)
        self.materials = self.__init_materials()
        self.geometry = self.__init_geometry()
        self.selections = self.__init_selections()
        self.clean_geometry('circle')

        self.config = {
            "n": int(np.sqrt(len(x))),
            "x_limits": (-0.03, 0.03),
            "y_limits": (-0.03, 0.03),
        }

    def __init_materials(self):
        materials = {
            'air': self.model / f'selections' / 'air',
            'plastic': self.model / 'selections' / 'plastic'
        }
        return materials

    def __init_geometry(self):
        geometry = self.model / 'geometries' / 'Geometry 1'
        geometry.java.autoRebuild('off')
        return geometry

    def __init_selections(self):
        return self.model / 'selections'

    @staticmethod
    def get_indices(x):
        return np.nonzero(x)

    def create_model(self):
        indices = self.get_indices(self.x)
        node_selections = []

        x, y = circles_grid(**self.config)
        tau = abs(x[1] - x[0])
        radius = tau / 2
        alpha = 1.1

        idx = 0
        for x_i in x:
            for y_j in y:
                name = f"circle_xi_{x_i}, yj_{y_j}"

                if idx in list(indices[0]):
                    node, node_sel = add_circle(
                        name, x_i, y_j,
                        self.geometry, self.selections, radius, alpha
                    )
                    node_selections.append(node_sel)
                else:
                    node_selections.append(None)

                idx += 1

        self.materials['plastic'].property(
            'input', list(np.array(node_selections)[indices])
        )
        self.model.build(self.geometry)
