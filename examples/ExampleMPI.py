import sys
import os
# adding gendev package folder to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gendev import SimpleMPIWorker, Model, MPISolver, Task

# adding logging for solver and workers
# Do not use any FileHandlers here. Use them after "with Solver as solver"
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
    '''
    mpiexec -np 4 py -m mpi4py .\ExampleMPI.py
    '''
    # List of Tasks to solve in parallel mode (99999! 5 times)
    tasks = [Task(99999) for _ in range(5)]
    # init the model
    model = MyModel()

    # init the worker
    worker = SimpleMPIWorker(model)
    # init the solver. Here we have to set buffer_size to make sure that 99999! will fit in the buffer
    with MPISolver(worker, buffer_size=2**20) as solver:
        results = solver.solve(tasks)
