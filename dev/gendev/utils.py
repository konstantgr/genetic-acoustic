import numpy as np


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


def x_to_solve(x, cache):
    to_solve, cached = [], []
    for i, p in enumerate(x):
        if str(p) in cache:
            cached.append(i)
        else:
            to_solve.append(i)
    return to_solve, cached
