import sys
import os
# adding gendev package folder to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gendev import SimpleWorker, Model, SimpleSolver, Task, MultiprocessingSolver, SimpleMultiprocessingWorker

# adding logging for solver and workers
# Be careful with multiprocessing and file logging at the same time
import logging
gs = logging.getLogger('gendev.solver')
gs.addHandler(logging.StreamHandler(sys.stdout))
gs.setLevel(logging.INFO)
gw = logging.getLogger('gendev.worker')
gw.addHandler(logging.StreamHandler(sys.stdout))
gw.setLevel(logging.DEBUG)


# adding own Model subclass with result method will be called for each task by solver
class MyModel(Model):
    def results(self, n, *args, **kwargs):
        res = 1
        for i in range(n):
            res *= i+1
        # res = n!
        return res


if __name__ == '__main__':
    # List of Tasks to solve in parallel mode (99999! 5 times)
    tasks = [Task(99999) for _ in range(5)]
    # init the model
    model = MyModel()

    # Simple solver
    # init the worker
    worker = SimpleWorker(model)
    with SimpleSolver(worker) as solver:
        results = solver.solve(tasks)
    print(f'Solution is ready')

    # Multiprocessing solver
    worker = SimpleMultiprocessingWorker(model)
    # init the solver. Here we can choose how many workers will me spawned
    with MultiprocessingSolver(worker, workers_num=3) as solver:
        results = solver.solve(tasks)
    print(f'Solution is ready')
