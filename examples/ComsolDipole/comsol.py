from pckit import ComsolModel, MPIWorker, ComsolWorker
from pckit.workers import MultiprocessingWorker
import numpy as np
import settings
import pandas as pd


class MyWorker(ComsolWorker):
    """Класс воркера, который поддерживает смену study, в pckit 0.2.2 этого нет(("""
    def do_the_job(self, args, kwargs):
        self.model.pre_build(*args, **kwargs)
        self.model.build()
        self.model.pre_solve(*args, **kwargs)
        self.model.mesh()
        if 'study' in kwargs.keys():
            self.model.solve(kwargs['study'])
        else:
            self.model.solve()
        results = self.model.results(*args, **kwargs)
        self.model.pre_clear(*args, **kwargs)
        self.model.clear()
        return results


class MyMPIWorker(MyWorker, MPIWorker):
    """Класс, обертка воркера, который позволяет использовать MPI"""
    def start(self):
        super().start()


def linear_grid(radii, separations):
    tmp_position = 0
    num_cylinders = len(radii)
    x = []
    for i in range(num_cylinders):
        tmp_position += separations[i] + radii[i] if i > 0 else separations[i]
        x.append(tmp_position)
        tmp_position += radii[i]

    y = [0 for i, _ in enumerate(x)]

    return np.array(x), np.array(y)


class MyModel(ComsolModel):
    def __init__(self):
        super().__init__()
        self.geometry = self / 'geometries' / 'Geometry 1'

    # def configure(self):
    #     self.geometry.java.autoRebuild('off')
    #     self.parameter('max_freq', '1000[Hz]')
    #     self.parameter('min_freq', '100[Hz]')
    #     self.parameter('step', '100[Hz]')

    @staticmethod
    def global_evaluation(dataset, evaluation) -> pd.DataFrame:
        """Исправление бага pckit, связанного с тем, что данные возвращаются действительными"""
        evaluation.property('data', dataset)
        java = evaluation.java
        real, imag = java.computeResult()
        results = (np.array(real) + 1j * np.array(imag)) if imag is not None else np.array(real)
        return pd.DataFrame(data=results, columns=evaluation.property('descr'))

    def pre_build(self, cylinders_h, cylinders_r, cylinders_seps, *args, **kwargs):
        """Расстановка цилиндров"""
        node_selections = []
        xgrid, ygrid = linear_grid(cylinders_r, cylinders_seps)

        idx = 0
        for i, length in enumerate(cylinders_h):
            name = f"cyl_{i}"
            node, node_sel = self.add_cylinder(name, xgrid[i], ygrid[i], -600/2, self.geometry, length, cylinders_r[i])
            node_selections.append(node_sel)
            idx += 1

        (self / 'selections' / 'Si').property(
            'input', list(np.array(node_selections))
        )

        # расстановка диполя в первый цилиндр
        self.parameter('xi', xgrid[0])

    def results(self, *args, **kwargs):
        evaluation = self / 'evaluations' / 'Global Evaluation 1'
        dataset = (self / 'datasets').children()[0]
        return self.global_evaluation(dataset, evaluation)

    def pre_clear(self, save=False, *args, **kwargs):
        if save:
            self.save(settings.save_path)
            self.export_image(self.geometry, settings.dirname + '/u.png',
                              props={'sourcetype': 'geometry'})
            # #  geometry drawing
            self.export_image("/GeomList/geom1/GeomFeatureList/wp1/sequence2D",
                              settings.dirname + '/u1.png',
                              props={'sourcetype': 'other'})
        self.clean_geometry(self.geometry, 'cyl')
