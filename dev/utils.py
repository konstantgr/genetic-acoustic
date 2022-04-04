import numpy as np


def grid(n, x_limits, y_limits):
    x = np.linspace(*x_limits, n, endpoint=True)
    y = np.linspace(*y_limits, n, endpoint=True)
    return x, y
