import sys
sys.path.append('')

from gendev import SimpleWorker, Model, SimpleSolver, Task, MultiprocessingSolver, SimpleMultiprocessingWorker
import time


class MyModel(Model):
    def results(self, n, *args, **kwargs):
        res = 1
        for i in range(n):
            res *= i+1
        return res

import logging
l = logging.getLogger('gendev.solver')
l.addHandler(logging.StreamHandler(sys.stdout))
l.setLevel(logging.INFO)
l = logging.getLogger('gendev.worker')
l.addHandler(logging.StreamHandler(sys.stdout))
l.setLevel(logging.DEBUG)

if __name__ == '__main__':
    tasks = [Task(99999) for _ in range(5)]
    model = MyModel()

    # Simple solver
    worker = SimpleWorker(model)
    with SimpleSolver(worker) as solver:
        start = time.time()
        results = solver.solve(tasks)
        end = time.time()
    print(f'Simple solver used {round(end-start, 4)}s')

    # Multiprocessing solver
    worker = SimpleMultiprocessingWorker(model)
    with MultiprocessingSolver(worker, workers_num=3) as solver:
        start = time.time()
        results = solver.solve(tasks)
        end = time.time()
        print(f'Multiprocessing solver used {round(end - start, 2)}s')