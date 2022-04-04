import numpy as np
from typing import Sequence


def grid(n, x_limits, y_limits):
    x = np.linspace(*x_limits, n, endpoint=True)
    y = np.linspace(*y_limits, n, endpoint=True)
    return x, y


def pretty_print_individual(ind: Sequence):
    n = int(np.sqrt(len(ind)))
    reshaped_ind = np.array(ind).reshape(n, n)

    return '\n'.join([
        ' '.join([' ' if not reshaped_ind[j][-i-1] else 'x' for j in range(n)])
        for i in range(n)
    ])
