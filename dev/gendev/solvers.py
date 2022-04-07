import multiprocessing
from abc import ABC, abstractmethod

import numpy as np
from .models import Model
from .workers import Worker
from multiprocessing import Pool, Manager, Queue, JoinableQueue
from typing import Iterable, Any, Mapping, MutableMapping, Sequence, Optional, Union, List
import queue
from .utils import x_to_solve
import time


class Solver(ABC):
    @abstractmethod
    def solve(self, x, config=None):
        pass


class MultiprocessingSolver(Solver):
    def __init__(self,
                 worker: Union[Worker, Sequence[Worker]],
                 workers_num: Union[int, Sequence] = 1,
                 caching: bool = False
                 ):
        # if not isinstance(worker, Worker):
        #     raise TypeError('Worker has to be the object of type Worker')
        self.worker = worker if isinstance(worker, Sequence) else [worker]
        self.workers_num = workers_num if isinstance(workers_num, Sequence) else [workers_num]
        self.caching = caching
        self.cache = {}
        self.workers = []

        self._jobs = JoinableQueue()
        self._results = Queue()

        for worker, num in zip(self.worker, self.workers_num):
            for i in range(num):
                process = multiprocessing.Process(target=worker.start, args=(self._jobs, self._results))
                process.start()
                self.workers.append(process)

    def solve(self, x: Sequence[Any],
              args: Union[None, Sequence[Any]] = None,
              kwargs: Union[None, dict] = None,
              tags: Union[None, Sequence[Any]] = None
              ) -> List[Any]:
        if args is not None and len(x) != len(args):
            raise ValueError(f"len(x) != len(args): {len(x)} != {len(args)}")
        if kwargs is not None and len(x) != len(kwargs):
            raise ValueError(f"len(x) != len(kwargs): {len(x)} != {len(kwargs)}")
        if tags is not None and len(x) != len(tags):
            raise ValueError(f"len(x) != len(tags): {len(x)} != {len(tags)}")

        if self.caching and tags is not None:
            to_solve, cached = x_to_solve(tags, self.cache)
        else:
            to_solve, cached = range(len(x)), []

        results = [None for _ in x]
        for i, p in enumerate(x):
            if i in to_solve:
                self._jobs.put((i, p, args, kwargs))
            else:
                results[i] = self.cache[tags[i]] if tags is not None else None

        self._jobs.join()
        for i in range(len(to_solve)):
            (i, r) = self._results.get()
            if self.caching and tags is not None:
                self.cache[tags[i]] = r
            results[i] = r

        return results

    def stop(self):
        for worker in self.workers:
            worker.terminate()
            worker.join()
            worker.close()


class SimpleSolver(Solver):
    def __init__(self, worker: Worker, caching: bool = False):
        if not isinstance(worker, Worker):
            raise TypeError('Worker has to be the object of type Worker')
        self.worker = worker
        self.caching = caching
        self.cache = {}

    def solve(self, x: Sequence[Any],
              args: Union[None, Sequence[Any]] = None,
              kwargs: Union[None, dict] = None,
              tags: Union[None, Sequence[Any]] = None
              ) -> List[Any]:
        if args is not None and len(x) != len(args):
            raise ValueError(f"len(x) != len(args): {len(x)} != {len(args)}")
        if kwargs is not None and len(x) != len(kwargs):
            raise ValueError(f"len(x) != len(kwargs): {len(x)} != {len(kwargs)}")
        if tags is not None and len(x) != len(tags):
            raise ValueError(f"len(x) != len(tags): {len(x)} != {len(tags)}")

        if self.caching and tags is not None:
            to_solve, cached = x_to_solve(tags, self.cache)
        else:
            to_solve, cached = range(len(x)), []

        results = [None for _ in x]
        for i, p in enumerate(x):
            if i in to_solve:
                results[i] = self.worker.do_the_job(p, args, kwargs)
            else:
                results[i] = self.cache[tags[i]] if tags is not None else None

        return results


# class MPISolver(Solver):


