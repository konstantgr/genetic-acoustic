import pckit
from comsol import MyModel, MyMPIWorker, MyWorker
import settings
from optimization import main
import logging
import sys

gs = logging.getLogger('pckit.solver')
gs.addHandler(logging.StreamHandler(sys.stdout))
gs.setLevel(logging.INFO)
# gw = logging.getLogger('pckit.worker')
# gw.addHandler(logging.StreamHandler(sys.stdout))
# gw.setLevel(logging.DEBUG)

if __name__ == '__main__':
    model = MyModel()

    mph_options = {'classkit': True}
    client_kwargs = {'cores': 1, 'version': '5.5'}

    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() == 1:
            raise RuntimeError
        myWorker = MyMPIWorker(
            model,
            settings.file_path,
            mph_options=mph_options,
            client_kwargs=client_kwargs
        )
    except:
        print('Using Simple solver instead of MPI')
        myWorker = MyWorker(
            model,
            settings.file_path,
            mph_options=mph_options,
            client_kwargs=client_kwargs
        )

    with pckit.get_solver(myWorker, workers_num=1) as solver:
        # Genetic Algorithm
        main(solver)
