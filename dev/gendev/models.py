import mph
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from .utils import make_unique
from typing import Any


class Model(ABC):
    def __init__(self):
        self._x = None
        self._args = None
        self._kwargs = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        self._args = args

    @property
    def kwargs(self):
        return self._args

    @kwargs.setter
    def kwargs(self, kwargs):
        self._kwargs = kwargs

    @abstractmethod
    def results(self) -> Any:
        pass


class ComsolModel(mph.Model, Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super(Model, self).__init__()
        self.selections = self/'selections'

    def add_circle(self, name: str, x_i: float, y_j: float, geometry: mph.Node, r: float, alpha: float = 1.1):
        node = geometry.create("Circle", name=name)
        node.property("r", str(r))
        node.property("pos", [str(x_i), str(y_j)])

        node_sel = self.selections.create('Box', name=name)
        node_sel.property('entitydim', 2)

        modified_r = r * alpha
        node_sel.property('xmin', f'+{x_i}-{modified_r}')
        node_sel.property('xmax', f'+{x_i}+{modified_r}')
        node_sel.property('ymin', f'+{y_j}-{modified_r}')
        node_sel.property('ymax', f'+{y_j}+{modified_r}')
        node_sel.property('condition', 'inside')

        return node, node_sel

    def add_square(self, name: str, x_i: float, y_j: float, geometry: mph.Node, width: float, alpha: float = 1.1):
        node = geometry.create("Square", name=name)
        node.property("size", str(width))
        node.property("pos", [str(x_i), str(y_j)])

        node_sel = self.selections.create('Box', name=name)
        node_sel.property('entitydim', 2)

        modified_width = width / 2 * alpha
        node_sel.property('xmin', f'+{x_i}-{modified_width}+{width / 2}')
        node_sel.property('xmax', f'+{x_i}+{modified_width}+{width / 2}')
        node_sel.property('ymin', f'+{y_j}-{modified_width}+{width / 2}')
        node_sel.property('ymax', f'+{y_j}+{modified_width}+{width / 2}')
        node_sel.property('condition', 'inside')

        return node, node_sel

    @staticmethod
    def _clean(node: mph.Node, tag: str):
        for c in node.children():
            if tag in c.path[-1]:
                c.remove()

    def clean_geometry(self, geometry: mph.Node, tag):
        self._clean(geometry, tag)
        self._clean(self.selections, tag)
        self.build(geometry)

    @staticmethod
    def global_evaluation(dataset: mph.Node, evaluation: mph.Node) -> pd.DataFrame:
        #  https://github.com/MPh-py/MPh/blob/2b967b77352f9ce7effcd50ad4774bf5eaf731ea/mph/model.py#L425
        evaluation.property('data', dataset)
        java = evaluation.java
        real, imag = java.computeResult()
        results = np.array(real) + 1j * np.array(imag)
        return pd.DataFrame(data=results, columns=make_unique(evaluation.property('descr')))

    def clear(self):
        # super().clear()
        super().reset()

    @abstractmethod
    def results(self) -> Any:
        pass

    def pre_build(self):
        pass

    def pre_solve(self):
        pass

    def pre_clear(self):
        pass

