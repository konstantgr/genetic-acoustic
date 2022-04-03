import numpy as np
import pandas as pd
from comsol.utils import get_multipoles_from_res
'''
functions to be minimized
'''


def high_peaks(results: pd.DataFrame, c: float, R: float) -> float:
    Q_multipoles = get_multipoles_from_res(results, c, R)
    Q_sc = results['sigma'].to_numpy()
    return np.abs(np.real(np.sum(np.min((Q_sc - np.array(Q_multipoles)) / Q_sc, axis=1))))


def max_sc(results: pd.DataFrame) -> float:
    return -np.real(np.max(results['sigma'].to_numpy()))


def peaks_contribution(results: pd.DataFrame, c: float, R: float, multipole_n: int = None) -> float:
    Q_multipoles = get_multipoles_from_res(results, c, R)
    if multipole_n is not None:
        Q_multipoles = Q_multipoles[multipole_n]
    Q_sc = results['sigma'].to_numpy()
    return -np.sum(((Q_sc - np.array(Q_multipoles)) / Q_sc) > 0.8)/len(Q_multipoles)
