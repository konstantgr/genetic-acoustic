import numpy as np
from typing import Sequence
import pandas as pd
import scipy.special as spc

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


def get_multipoles_from_res(results: pd.DataFrame, c: float, R: float):
    # returns e0, e1, o1, e2, o2, e3, o3, e4, o4, e5, o5, e6, o6, e7, o7, e8, o8, e9, o9
    k = 2 * np.pi * results['Frequency'] / c
    Q_multipoles = []

    multipoles = range(10)
    for m in multipoles:
        mu = 2 if m == 0 else 1
        bm = results[f'e{m}'] / (spc.hankel1(m, R * k) * mu * np.pi * R)
        Q_multipoles.append(2 / k * (mu * np.abs(bm)**2))
        if m != 0:
            cm = results[f'o{m}'] / (spc.hankel1(m, R * k) * np.pi * R)
            Q_multipoles.append(2 / k * (np.abs(cm)**2))
    return Q_multipoles
