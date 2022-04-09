import multiprocessing
from abc import ABC, abstractmethod
from .workers import Worker, MultiprocessingWorker
from multiprocessing import Queue, JoinableQueue
from typing import Any, Sequence, Union, List, Hashable
from queue import Empty
import numpy as np
from mpi4py import MPI
import mpi4py
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor

from .utils import x_to_solve
import time

# TODO add info to return after solve()

import logging
import sys
logger = logging.getLogger(__package__)


class Task:
    def __init__(self, *args, **kwargs):
        if 'tag' in kwargs:
            tag = kwargs.pop('tag')
            if not isinstance(tag, Hashable):
                raise ValueError('Tag must be Hashable')
            self.tag = tag
        else:
            self.tag = None
        self.args = args
        self.kwargs = kwargs


class Solver(ABC):
    @abstractmethod
    def solve(self, tasks: Sequence[Task]) -> List[Any]:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MultiprocessingSolver(Solver):
    def __init__(self,
                 worker: Union[MultiprocessingWorker, Sequence[MultiprocessingWorker]],
                 workers_num: Union[int, Sequence] = 1,
                 caching: bool = False
                 ):
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

    def solve(self, tasks: Sequence[Task]) -> List[Any]:
        self.check_processes()
        if self.caching:
            to_solve, cached = x_to_solve(tasks, self.cache)
        else:
            to_solve, cached = range(len(tasks)), []

        results = [None for _ in tasks]
        for i, task in enumerate(tasks):
            if i in to_solve:
                self._jobs.put((i, task.args, task.kwargs))
            else:
                results[i] = self.cache[task.tag] if task.tag is not None else None

        # self._jobs.join()
        for i in range(len(to_solve)):
            while True:
                try:
                    (i, r) = self._results.get(block=False)
                    if self.caching and tasks[i].tag is not None:
                        self.cache[tasks[i].tag] = r
                    results[i] = r
                    break
                except Empty:
                    self.check_processes()
                    time.sleep(0.1)
        return results

    def stop(self):
        for worker in self.workers:
            worker.terminate()
            worker.join()
            worker.close()

    def check_processes(self):
        for worker in self.workers:
            if not worker.is_alive():
                raise RuntimeError('Dead worker')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class SimpleSolver(Solver):
    def __init__(self, worker: Worker, caching: bool = False):
        if not isinstance(worker, Worker):
            raise TypeError('Worker has to be the object of type Worker')
        self.worker = worker
        self.caching = caching
        self.cache = {}

        self.worker.start()

    def solve(self, tasks: Sequence[Task]) -> List[Any]:
        if self.caching:
            to_solve, cached = x_to_solve(tasks, self.cache)
        else:
            to_solve, cached = range(len(tasks)), []

        results = [None for _ in tasks]
        for i, task in enumerate(tasks):
            if i in to_solve:
                results[i] = self.worker.do_the_job(task.args, task.kwargs)
                if self.caching and tasks[i].tag is not None:
                    self.cache[task.tag] = results[i]
            else:
                results[i] = self.cache[task.tag] if task.tag is not None else None
        return results


class MPISolver(Solver):
    def __init__(self, worker: Worker, caching: bool = False):
        if not isinstance(worker, Worker):
            raise TypeError('Worker has to be the object of type Worker')
        self.worker = worker
        self.caching = caching
        self.cache = {}

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.processes = self.comm.Get_size() - 1

        # if self.rank != 0:
        #     self.worker.start()
        #     self.start_listening()
        #     exit()
        self.executor = MPICommExecutor().__enter__()
        # if self.executor is not None:
        #     self.executor.submit(self.worker.start)
            # future.cancel()

        # self.executor = MPIPoolExecutor(max_workers=3, initializer=self.worker.start)
        # future = self.executor.submit(self.worker.start)

        # future = self.executor.submit(self.worker.start)

    def solve(self, tasks: Sequence[Task]) -> List[Any]:
        if self.caching:
            to_solve, cached = x_to_solve(tasks, self.cache)
        else:
            to_solve, cached = range(len(tasks)), []

        results = [None for _ in tasks]
        requests = []
        for i, task in enumerate(tasks):
            if i in to_solve:
                requests.append(self.executor.submit(self.worker.do_the_job, task.args, task.kwargs))
            else:
                results[i] = self.cache[task.tag] if task.tag is not None else None

        for i in to_solve:
            results[i] = requests.pop().result()
            if self.caching and tasks[i].tag is not None:
                self.cache[tasks[i].tag] = results[i]
        return results

    # def start_listening(self):
    #     logger.debug('start_listening')
    #     while True:
    #         req = self.comm.irecv(source=0)
    #         (i, args, kwargs) = req.wait()
    #         logger.debug(str((i, args, kwargs)))
    #         if i is None:
    #             return 1
    #         res = self.worker.do_the_job(args, kwargs)
    #         req = self.comm.isend((i, res), dest=0)
    #         req.wait()
    #
    # def solve(self, tasks: Sequence[Task]) -> List[Any]:
    #     if self.caching:
    #         to_solve, cached = x_to_solve(tasks, self.cache)
    #     else:
    #         to_solve, cached = range(len(tasks)), []
    #
    #     results = [None for _ in tasks]
    #     requests = []
    #     for i, task in enumerate(tasks):
    #         if i in to_solve:
    #             dest = len(requests) % self.processes+1
    #             req = self.comm.isend((i, task.args, task.kwargs), dest=dest, tag=i)
    #             req.wait()
    #             requests.append(self.comm.irecv(source=dest))
    #         else:
    #             results[i] = self.cache[task.tag] if task.tag is not None else None
    #
    #     # self._jobs.join()
    #     for _ in range(len(to_solve)):
    #         while True:
    #             if len(requests) == 0:
    #                 break
    #             for p in range(len(requests)):
    #                 test, res = requests[p].test()
    #                 if test is True:
    #                     i, r = res
    #                     if self.caching and tasks[i].tag is not None:
    #                         self.cache[tasks[i].tag] = r
    #                     results[i] = r
    #                     requests.pop(p)
    #                     break
    #             time.sleep(0.1)
    #     return results
    def stop(self):
        if self.executor is not None:
            self.executor.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

