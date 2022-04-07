import numpy as np

from .models import ComsolModel
import mph
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any
from multiprocessing import Queue
from loguru import logger


class Worker(ABC):
    @abstractmethod
    def start(self, jobs: Queue, results: Queue, *client_args, **client_kwargs):
        pass

    @abstractmethod
    def do_the_job(self, x, args, kwargs) -> Any:
        pass


class ComsolWorker(Worker):
    def __init__(self, Model: type, filepath: str or Path,
                 mph_options: dict = None,
                 client_args=None,
                 client_kwargs=None):
        if not issubclass(Model, ComsolModel):
            raise TypeError('Model has to be subclass of ComsolModel')

        self.client = None
        self.model = None

        self._mph_options = {} if mph_options is None else mph_options
        self._client_args = [] if client_args is None else client_args
        self._client_kwargs = {} if client_kwargs is None else client_kwargs
        self._filepath = filepath
        self._Model = Model

    def start(self, jobs: Queue, results: Queue, *client_args, **client_kwargs):
        for option in self._mph_options:
            mph.option(option, self._mph_options[option])

        self.client = mph.start(*self._client_args, *client_args, **self._client_kwargs,
                                **client_kwargs)  # type: mph.client
        model = self.client.load(self._filepath)
        self.model = self._Model(model)  # type: ComsolModel
        self.loop(jobs, results)

    def loop(self, jobs, results):
        logger.info('Comsol started')
        while True:
            (i, p, args, kwargs) = jobs.get()
            res = self.do_the_job(p, args, kwargs)
            results.put((i, res))
            jobs.task_done()

    def do_the_job(self, x, args, kwargs) -> Any:
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()
        self.model.pre_build(x, *args, **kwargs)
        self.model.build()
        self.model.pre_solve(x, *args, **kwargs)
        self.model.mesh()
        self.model.solve()
        results = self.model.results(x, *args, **kwargs)
        self.model.pre_clear(x, *args, **kwargs)
        self.model.clear()
        return results


# class ComsolNetworkWorker(ComsolWorker):
#     def start(self, jobs: Queue, results: Queue, *client_args, **client_kwargs):
#         print('comsol init')
#         self.client = mph.start(*self._client_args, *client_args, **self._client_kwargs,
#                                 **client_kwargs)  # type: mph.client
#         print(self.client.cores)
#         model = self.client.load(self._filepath)
#         self.model = self._Model(model)  # type: ComsolModel
#         self.loop(jobs, results)
#

class TestLoopWorker(Worker):
    def start(self, jobs: Queue, results: Queue, *client_args, **client_kwargs):
        self.loop(jobs, results)

    def loop(self, jobs, results):
        while True:
            (i, p, args, kwargs) = jobs.get()
            print('loop sol')
            results.put((i, self.do_the_job(p, args, kwargs)))
            jobs.task_done()

    def do_the_job(self, x, args, kwargs) -> Any:
        for i in range(99999999):
            np.sqrt(9999999999)
        return x

