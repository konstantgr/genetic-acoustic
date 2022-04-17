import multiprocessing
import socket

from .models import ComsolModel, Model
import mph
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from multiprocessing import JoinableQueue, Queue

import logging
logger = logging.getLogger(__package__ + '.worker')


class Worker(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def do_the_job(self, args: Tuple[Any], kwargs: Dict[str, Any]) -> Any:
        pass


class MultiprocessingWorker(Worker):
    @abstractmethod
    def start(self):
        pass

    def start_loop(self, jobs: JoinableQueue, results: Queue):
        logger.info(f'{multiprocessing.current_process().name} {socket.gethostname()} Starting')
        self.start()
        self._loop(jobs, results)

    def _loop(self, jobs: JoinableQueue, results: Queue):
        logger.info(f'{multiprocessing.current_process().name} {socket.gethostname()} Entering the loop')
        try:
            while True:
                (i, args, kwargs) = jobs.get()
                logger.debug(f'{multiprocessing.current_process().name} {socket.gethostname()} Starting doing the job {i}')
                res = self.do_the_job(args, kwargs)
                results.put((i, res))
                jobs.task_done()
        except Exception as e:
            results.put((-1, e))
            raise e


class MPIWorker(Worker):
    def __init__(self):
        import mpi4py.MPI
        self.comm = mpi4py.MPI.COMM_WORLD

    @abstractmethod
    def start(self):
        pass

    def start_loop(self):
        logger.info(f'Rang {self.comm.Get_rank()} {socket.gethostname()} Starting')
        self.start()
        self._loop()

    def _loop(self):
        comm = self.comm
        logger.info(f'Rang {comm.Get_rank()} {socket.gethostname()} Entering the loop')
        while True:
            req = comm.irecv(source=0)
            (i, args, kwargs) = req.wait()
            if i is None and args is None and kwargs is None:
                return
            logger.debug(f'Rang {comm.Get_rank()} {socket.gethostname()} Starting doing the job {i}')
            res = self.do_the_job(args, kwargs)
            req = comm.isend((i, res), dest=0)
            req.wait()


class SimpleWorker(Worker):
    def __init__(self, model: Model):
        self.model = model

    def start(self, *args):
        pass

    def do_the_job(self, args: Tuple[Any], kwargs: Dict[str, Any]) -> Any:
        logger.debug(f'{multiprocessing.current_process().name} {socket.gethostname()} Starting doing the job')
        return self.model.results(*args, **kwargs)


class SimpleMultiprocessingWorker(SimpleWorker, MultiprocessingWorker):
    def start(self):
        pass

    def do_the_job(self, args: Tuple[Any], kwargs: Dict[str, Any]) -> Any:
        return self.model.results(*args, **kwargs)


class SimpleMPIWorker(SimpleWorker, MPIWorker):
    def start(self):
        pass

    def do_the_job(self, args: Tuple[Any], kwargs: Dict[str, Any]) -> Any:
        return self.model.results(*args, **kwargs)


class ComsolWorker(Worker):
    def __init__(self, model: ComsolModel, filepath: str or Path,
                 mph_options: dict = None,
                 client_args=None,
                 client_kwargs=None):
        if not isinstance(model, ComsolModel):
            raise TypeError('Model has to be an object of ComsolModel class')

        self.client = None
        self.model = model

        self._mph_options = {} if mph_options is None else mph_options
        self._client_args = [] if client_args is None else client_args
        self._client_kwargs = {} if client_kwargs is None else client_kwargs
        self._filepath = filepath

    def start(self):
        for option in self._mph_options:
            mph.option(option, self._mph_options[option])
        logger.debug(f'Opening COMSOL model {self._filepath}')
        self.client = mph.start(*self._client_args, **self._client_kwargs)  # type: mph.client
        self.model.java = self.client.load(self._filepath).java
        self.model.configure()

    def do_the_job(self, args, kwargs) -> Any:
        self.model.pre_build(*args, **kwargs)
        self.model.build()
        self.model.pre_solve(*args, **kwargs)
        self.model.mesh()
        self.model.solve()
        results = self.model.results(*args, **kwargs)
        self.model.pre_clear(*args, **kwargs)
        self.model.clear()
        return results


class ComsolMultiprocessingWorker(ComsolWorker, MultiprocessingWorker):
    def start(self):
        super().start()


class ComsolMPIWorker(ComsolWorker, MPIWorker):
    def start(self):
        super().start()
