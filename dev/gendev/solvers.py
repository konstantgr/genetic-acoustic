import multiprocessing
from abc import ABC, abstractmethod
from .models import Model
from .workers import Worker
from multiprocessing import Pool, Manager, Queue
from typing import Iterable, Any, Mapping, MutableMapping
import queue


class Solver(ABC):
    @abstractmethod
    def solve(self, x, config=None):
        pass


class SimpleSolver(Solver):
    def __init__(self, worker: Worker, caching=True):
        if not isinstance(worker, Worker):
            raise TypeError('worker has to be a Worker ')
        self.worker = worker
        self.caching = caching
        self.cache = {}
        self.worker.start()

    def solve(self, x, config=None):
        results = []
        for p in x:
            if self.caching and str(p) in self.cache:
                results.append(self.cache[str(p)])
                continue
            result = self.worker.do_the_job(p, config)
            results.append(result)
            if self.caching:
                self.cache[str(p)] = result
        return results


class MultiprocessingSolver(Solver):
    def __init__(self, worker: Worker, workers=2, caching=True):
        # if not issubclass(model, Model):
        #     raise TypeError('Model has to be subclass of Model')
        # if not issubclass(worker, Worker):
        #     raise TypeError('Worker has to be subclass of Worker')
        self.worker = worker
        self.workers = workers
        self.caching = caching
        self.cache = {}

        self._jobs = Queue()
        self._results = Queue()
        # self.pool_result = self.pool.apply_async(self._worker, (worker, self._jobs, self._results))

    @staticmethod
    def _worker(worker, jobs, results, x, config):
        print('started')
        worker.start()
        while True:
            try:
                (i, p) = jobs.get(block=False)
            except queue.Empty:
                break
            print('comsol_sol')
            results.put((i, worker.do_the_job(p, config)))

    def solve(self, x, config=None):
        total_sol = 0
        for i, p in enumerate(x):
            if self.caching and str(p) in self.cache:
                print('cache used')
                continue
            self._jobs.put((i, p))
            total_sol += 1

        for _ in range(self.workers):
            process = multiprocessing.Process(target=self._worker, args=(self.worker, self._jobs, self._results, x, config))
            process.start()

        results = {}
        for _ in range(total_sol):
            (i, r) = self._results.get()
            results[i] = r

        results_list = []
        for i in range(len(x)):
            if self.caching:
                if str(x[i]) in self.cache:
                    results_list.append(self.cache[str(x[i])])
                else:
                    self.cache[str(x[i])] = results[i]
                    results_list.append(results[i])
            else:
                results_list.append(results[i])

        return results_list

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['_jobs']
        del self_dict['_results']
        return self_dict


