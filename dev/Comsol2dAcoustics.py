from gendev import ComsolModel, ComsolWorker, SimpleSolver
import numpy as np
from utils import grid
import os
from loguru import logger

individuals_level = logger.level("individuals", no=38)
bests_level = logger.level("best", no=38, color="<green>")
logger.add('logs/logs_{time}.log', level='INFO')

fmt = "{time} | {level} |\t{message}"
logger.add('logs/individuals_{time}.log', format=fmt, level='individuals')


class SquaresModel(ComsolModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = self / 'geometries' / 'Geometry 1'
        self.geometry.java.autoRebuild('off')

        self.parameter('max_freq', '1000[Hz]')
        self.parameter('min_freq', '100[Hz]')
        self.parameter('step', '100[Hz]')

        self.config = {
            "n": 3,
            "x_limits": (-0.03, 0.03),
            "y_limits": (-0.03, 0.03),
        }

    def pre_build(self):
        indices = np.nonzero(self.x)
        node_selections = []

        x, y = grid(**self.config)
        tau = abs(x[1] - x[0])
        radius = tau / 2

        idx = 0
        for x_i in x:
            for y_j in y:
                name = f"circle_xi_{x_i}, yj_{y_j}"

                if idx in list(indices[0]):
                    node, node_sel = self.add_circle(name, x_i, y_j, self.geometry, radius)
                    node_selections.append(node_sel)
                else:
                    node_selections.append(None)
                idx += 1

        (self.selections/'plastic').property(
            'input', list(np.array(node_selections)[indices])
        )

    def results(self):
        evaluation = self / 'evaluations' / 'Global Evaluation 1'
        dataset = (self / 'datasets').children()[0]
        return self.global_evaluation(dataset, evaluation)

    def pre_clear(self):
        # self.save(save_path)
        self.clean_geometry(self.geometry, 'circle')


dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname, 'empty_project.mph')
save_path = os.path.join(dirname, 'empty_project1.mph')

MyWorker = ComsolWorker(SquaresModel, file_path,
                        mph_options={'classkit': True},
                        client_kwargs={'cores': 1})
# MyWorker.start()
# print(MyWorker.do_the_job([1, 0, 1, 1, 0, 1, 0, 1, 0]))

if __name__ == '__main__':
    Solver = SimpleSolver(MyWorker, workers=4)
    print(Solver.solve(np.array([[1, 0, 1, 1, 0, 1, 0, 1, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [1, 0, 1, 1, 0, 1, 0, 1, 0],
                        # [1, 0, 1, 1, 1, 1, 0, 1, 0],
                        # [1, 0, 1, 1, 0, 1, 1, 1, 0],
                        ])))

    print(Solver.solve([[1, 0, 1, 1, 0, 1, 0, 1, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 0, 1, 0, 1, 0],
                        [1, 0, 1, 1, 1, 1, 0, 1, 0],
                        [1, 0, 1, 1, 0, 1, 1, 1, 0],
                        ]))
    Solver.stop()


