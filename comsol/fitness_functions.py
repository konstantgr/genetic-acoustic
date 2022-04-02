import numpy as np
import pandas as pd
import scipy.special as spc


def high_peaks(results: pd.DataFrame) -> float:
    k = 2 * np.pi * results['Frequency'] / 343
    R = 0.18
    Q_multipoles = []

    multipoles = range(20)
    for m in multipoles:
        mu = 2 if m == 0 else 1
        bm = results[f'{m}'] / (spc.hankel1(m, R * k) * mu * np.pi * R)
        Q_multipoles.append(2 / k * mu * np.abs(bm) ** 2)

    Q_sc = results['sigma'].to_numpy()
    return np.real(np.sum(np.min((Q_sc - np.array(Q_multipoles)) / Q_sc, axis=1)))
