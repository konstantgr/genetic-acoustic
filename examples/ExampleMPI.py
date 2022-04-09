import sys
sys.path.append('')

from gendev import SimpleMPIWorker, Model, MPISolver, Task
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
    '''
    mpiexec -np 4 py -m mpi4py .\ExampleMPI.py
    '''
    tasks = [Task(99999) for _ in range(5)]
    model = MyModel()

    # MPI solver
    worker = SimpleMPIWorker(model)
    with MPISolver(worker, buffer_size=2**20) as solver:
        start = time.time()
        results = solver.solve(tasks)
        end = time.time()
        print(f'MPI solver used {round(end-start, 4)}s')
