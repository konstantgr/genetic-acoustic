from hpctool import ComsolMultiprocessingWorker, MultiprocessingSolver
from Comsol3d import *

if __name__ == '__main__':
    model = MyModel()
    myWorker = ComsolMultiprocessingWorker(model, file_path,
                                           mph_options={'classkit': True},
                                           client_kwargs={'cores': 1,
                                                          'version': '5.5'})
    with MultiprocessingSolver(myWorker, workers_num=1, caching=True) as solver:
        # Genetic Algorithm
        main(solver)
