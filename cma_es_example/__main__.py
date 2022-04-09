import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gendev import SimpleMPIWorker, Model, MPISolver, Task
from main import cma_es_optimizer, MyModel, SimpleIndividual
import numpy as np
import matplotlib.pyplot as plt

import logging
gs = logging.getLogger('gendev.solver')
gs.addHandler(logging.StreamHandler(sys.stdout))
gs.setLevel(logging.INFO)
# gw = logging.getLogger('gendev.worker')
# gw.addHandler(logging.StreamHandler(sys.stdout))
# gw.setLevel(logging.DEBUG)


if __name__ == '__main__':
    model = MyModel()
    worker = SimpleMPIWorker(model)
    with MPISolver(worker, buffer_size=2**20) as solver:
        logger = logging.getLogger('main')
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.INFO)

        ind = SimpleIndividual(np.array([(-2, 2) for i in range(80)]))
        result = cma_es_optimizer(ind, solver)
        mean_values = result['mean_values']
        best_x, best_y = result['best_values']

        plt.plot(range(len(mean_values)), mean_values)
        plt.scatter(best_x, best_y)
        plt.savefig('image.png')
        logger.info('done')
