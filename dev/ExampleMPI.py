import mpi4py.MPI

from gendev import MPIWorker, Model, MPISolver, Task
import time

import logging
import sys

logger = logging.getLogger('gendev')
fh = logging.StreamHandler(sys.stdout)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class MyModel(Model):
    def results(self, n, *args, **kwargs):
        res = 1
        for i in range(n):
            res *= i+1
        return res


# MPI solver

if __name__ == '__main__':
    tasks = [Task(9999) for _ in range(800)]
    model = MyModel()
    worker = MPIWorker(model)
    solver = MPISolver(worker)
    if solver.rank == 0:
        start = time.time()
        results = solver.solve(tasks)
        end = time.time()
        print(f'MPI solver used {round(end-start, 4)}s')
    solver.stop()
