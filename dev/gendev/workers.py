from .models import ComsolModel
import mph
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any
from multiprocessing import Queue


class Worker(ABC):
    @abstractmethod
    def start(self, jobs: Queue, results: Queue, *client_args, **client_kwargs):
        pass

    @abstractmethod
    def do_the_job(self, x, config=None) -> Any:
        pass


class ComsolWorker(Worker):
    def __init__(self, Model: type, filepath: str or Path,
                 mph_options: dict = None,
                 client_args=None,
                 client_kwargs=None):
        if not issubclass(Model, ComsolModel):
            raise TypeError('Model has to be subclass of ComsolModel')

        if mph_options is not None:
            for option in mph_options:
                mph.option(option, mph_options[option])

        self.client = None
        self.model = None

        self._client_args = [] if client_args is None else client_args
        self._client_kwargs = {} if client_kwargs is None else client_kwargs
        self._filepath = filepath
        self._Model = Model

    def start(self, jobs: Queue, results: Queue, *client_args, **client_kwargs):
        print('comsol init')
        self.client = mph.start(*self._client_args, *client_args, **self._client_kwargs,
                                **client_kwargs)  # type: mph.client
        print(self.client.cores)
        model = self.client.load(self._filepath)
        self.model = self._Model(model)  # type: ComsolModel
        self.loop(jobs, results)

    def loop(self, jobs, results):
        while True:
            (i, p, config) = jobs.get()
            print('comsol sol')
            results.put((i, self.do_the_job(p, config)))

    def do_the_job(self, x, config=None) -> Any:
        self.model.x = x
        if config is not None:
            self.model.config = config

        self.model.pre_build()
        self.model.build()
        self.model.pre_solve()
        self.model.mesh()
        print('Running')
        self.model.solve()
        print('results')
        results = self.model.results()
        print('pre_clear')
        self.model.pre_clear()
        self.model.clear()
        return results


class ComsolNetworkWorker(ComsolWorker):
    def start(self, jobs: Queue, results: Queue, *client_args, **client_kwargs):
        print('comsol init')
        self.client = mph.start(*self._client_args, *client_args, **self._client_kwargs,
                                **client_kwargs)  # type: mph.client
        print(self.client.cores)
        model = self.client.load(self._filepath)
        self.model = self._Model(model)  # type: ComsolModel
        self.loop(jobs, results)
