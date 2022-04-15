from hpctool import SimpleMPIWorker, MPISolver
from main import main, MyModel

if __name__ == '__main__':
    model = MyModel()
    worker = SimpleMPIWorker(model)
    with MPISolver(worker, buffer_size=2**20, caching=True) as solver:
        main(solver)
