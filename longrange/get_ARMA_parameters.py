"""
Functions to get ARMA parameters
"""

import numpy as np

from ACVF import ACVF

def yule_walker(X, p):
    """
    """
    acvf = ACVF(X, p + 1)
    Gamma = np.zeros((p, p))
    gamma_p = np.zeros(p)
    for i in range(0, p):
        gamma_p[i] = acvf[i + 1]
        for j in range(0, p):
            index = abs(i - j)
            Gamma[i, j] = acvf[index]
    phi = np.linalg.solve(Gamma, gamma_p)
    sigma2 = acvf[0] - np.dot(phi, acvf[1:])
    return (phi, sigma2)
