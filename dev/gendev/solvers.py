import multiprocessing
from abc import ABC, abstractmethod
from .workers import Worker, MultiprocessingWorker
from multiprocessing import Queue, JoinableQueue
from typing import Any, Sequence, Union, List, Hashable
from queue import Empty

from .utils import x_to_solve
import time

# TODO add info to return after solve()


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
            else:
                results[i] = self.cache[task.tag] if task.tag is not None else None
        return results


class MPISolver(Solver):
    pass
#     def __init__(self,
#                  worker: MultiprocessingWorker,
#                  workers_num: int = 1,
#                  caching: bool = False,
#                  ):
#         self.worker = worker
#         self.workers_num = workers_num
#         self.caching = caching
#         self.cache = {}
#         self.workers = []
#
#         self._jobs = JoinableQueue()
#         self._results = Queue()
#
#         for worker, num in zip(self.worker, self.workers_num):
#             for i in range(num):
#                 process = multiprocessing.Process(target=worker.start, args=(self._jobs, self._results))
#                 process.start()
#                 self.workers.append(process)
#
#     def solve(self, tasks: Sequence[Task]) -> List[Any]:
#         self.check_processes()
#         if self.caching:
#             to_solve, cached = x_to_solve(tasks, self.cache)
#         else:
#             to_solve, cached = range(len(tasks)), []
#
#         results = [None for _ in tasks]
#         for i, task in enumerate(tasks):
#             if i in to_solve:
#                 self._jobs.put((i, task.args, task.kwargs))
#             else:
#                 results[i] = self.cache[task.tag] if task.tag is not None else None
#
#         # self._jobs.join()
#         for i in range(len(to_solve)):
#             while True:
#                 try:
#                     (i, r) = self._results.get(block=False)
#                     if self.caching and tasks[i].tag is not None:
#                         self.cache[tasks[i].tag] = r
#                     results[i] = r
#                     break
#                 except Empty:
#                     self.check_processes()
#                     time.sleep(0.1)
#         return results
#
#     def stop(self):
#         for worker in self.workers:
#             worker.terminate()
#             worker.join()
#             worker.close()
#
#     def check_processes(self):
#         for worker in self.workers:
#             if not worker.is_alive():
#                 raise RuntimeError('Dead worker')
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.stop()
#
