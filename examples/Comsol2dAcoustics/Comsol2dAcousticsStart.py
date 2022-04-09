from Comsol2dAcoustics import *
import sys
import os
dirname = os.path.dirname(__file__)
sys.path.append(os.path.dirname(dirname))

from gendev import ComsolMultiprocessingWorker, ComsolWorker, MultiprocessingSolver, MultiprocessingSolver


if __name__ == '__main__':
    fmt = "{time} | {level} |\t{message}"
    individuals_level = logger.level("individuals", no=38)
    bests_level = logger.level("best", no=38, color="<green>")
    logger.remove()
    logger.add(sys.stdout, level='INFO', format=fmt, enqueue=True)

    logger.add('logs/logs_{time}.log', level='INFO', format=fmt)
    logger.add('logs/individuals_{time}.log', format=fmt, level='individuals')

    model = SquaresModel()
    myWorker = ComsolMultiprocessingWorker(model, file_path,
                                           mph_options={'classkit': True},
                                           client_kwargs={'cores': 1})
    # myWorker = ComsolWorker(model, file_path, mph_options={'classkit': True}, client_kwargs={'cores': 1})
    # or use MultiprocessingSolver(myWorker) instead of SimpleSolver
    with MultiprocessingSolver(myWorker, caching=True) as solver:
        # Genetic Algorithm
        best_x, best_res = differential_evolution_scipy(solver)
        x = transform_to_binary_list(best_x)

        # Best individual
        solver.solve([Task(x=x, save=True, tag=str(x))])

        print(f'Project saved successfully, best result: {best_res}')