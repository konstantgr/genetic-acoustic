from abc import ABC
from typing import Callable, Any

from .utils import ComsolModelAttributes, evaluate_global_ev, clean


class Individual(ABC):
    def __init__(self, x, model):
        self.x = x
        self.model = model
        self.materials = None
        self.geometry = None
        self.dataset = None
        self.selections = None

    def __init_materials(self):
        raise NotImplemented

    def __init_geometry(self):
        raise NotImplemented

    def __init_selections(self):
        raise NotImplemented

    def getLastComputationTime(self):
        # TODO ADD ANY STUDY SUPPORT

        studies = (self.model / 'studies').children()
        return -1 if not len(studies) else int(studies[-1].java.getLastComputationTime())

    def clean_geometry(self, param):
        clean(self.geometry, param)
        clean(self.selections, param)

        self.model.build(self.geometry)
        self.model.clear()
        self.model.java.resetHist()

    def solve_geometry(self):
        self.model.mesh()
        self.model.solve()

        evaluation = self.model / 'evaluations' / 'Global Evaluation 1'
        dataset = (self.model / 'datasets').children()[0]
        self.dataset = evaluate_global_ev(dataset, evaluation)

    def fitness(self, func: Callable) -> Any:
        return func(self.dataset)
