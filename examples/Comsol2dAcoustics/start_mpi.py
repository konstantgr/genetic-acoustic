from hpctool import ComsolMPIWorker, MPISolver
from Comsol2dAcoustics import *

if __name__ == '__main__':
    model = SquaresModel()
    myWorker = ComsolMPIWorker(model, file_path,
                               mph_options={'classkit': True},
                               client_kwargs={'cores': 1,
                                              'version': '5.5'})
    with MPISolver(myWorker, caching=True) as solver:
        # Genetic Algorithm
        main(solver)
