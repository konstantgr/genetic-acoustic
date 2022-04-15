from hpctool import SimpleMultiprocessingWorker, MultiprocessingSolver
from main import main, MyModel

if __name__ == '__main__':
    model = MyModel()
    worker = SimpleMultiprocessingWorker(model)
    with MultiprocessingSolver(worker, workers_num=2, caching=True) as solver:
        main(solver)
