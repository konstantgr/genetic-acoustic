import multiprocessing
from abc import ABC, abstractmethod

import numpy as np
from .models import Model
from .workers import Worker
from multiprocessing import Pool, Manager, Queue
from typing import Iterable, Any, Mapping, MutableMapping, Sequence
import queue
from .utils import x_to_solve


class Solver(ABC):
    @abstractmethod
    def solve(self, x, config=None):
        pass


class SimpleSolver(Solver):
    def __init__(self, worker: Worker, workers=1, caching=True):
        if not isinstance(worker, Worker):
            raise TypeError('Worker has to be the object of type Worker')
        self.worker = worker
        self.workers = workers
        self.caching = caching
        self.cache = {}
        self.workers_list = []

        self._jobs = Queue()
        self._results = Queue()

        for i in range(workers):
            process = multiprocessing.Process(target=self.worker.start,
                                              args=(self._jobs, self._results))
            process.start()
            self.workers_list.append(process)

    def solve(self, x: Sequence[Any], config=None):
        to_solve, cached = x_to_solve(x, self.cache)
        results = [None for _ in x]

        for i, p in enumerate(x):
            if i in to_solve:
                self._jobs.put((i, p, config))
            else:
                results[i] = self.cache[str(p)]

        for i in range(len(to_solve)):
            (i, r) = self._results.get()
            if self.caching:
                self.cache[str(x[i])] = r
            results[i] = r

        return results

    def stop(self):
        for worker in self.workers_list:
            worker.terminate()
            worker.join()
            worker.close()

