import numpy as np
from typing import Tuple, List


def make_unique(labels) -> list:
    new_labels = []
    _, real_index, counts = np.unique(labels, return_counts=True, return_inverse=True)
    for index in range(len(labels)):
        for count in range(counts[real_index[index]]):
            new_label = labels[index] + (f'({count})' if count != 0 else '')
            if new_label not in new_labels:
                new_labels.append(new_label)
                break

    return new_labels


def x_to_solve(tasks, cache) -> Tuple[List[int], List[int], List[int]]:
    to_solve, cached, same = [], [], []
    to_solve_tags = []
    for i, task in enumerate(tasks):
        if task.tag is None:
            raise ValueError('If caching is True all the tasks must have a tag property')
        if task.tag in cache:
            cached.append(i)
        elif task.tag in to_solve_tags:
            same.append(i)
        else:
            to_solve.append(i)
            to_solve_tags.append(task.tag)
    return to_solve, cached, same
