import numpy as np


def gauss_rule(dim, gauss_n):
    # Reference : https://en.wikipedia.org/wiki/Gaussian_quadrature
    gauss_point, weight = [], []
    if dim == 1:
        if gauss_n == 1:
            gauss_point = [0]
            weight = [2]
        elif gauss_n == 2:
            gauss_point = [1, 1]
            weight = [-np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)]
        elif gauss_n == 3:
            gauss_point = [-np.sqrt(3.0 / 5.0), 0, np.sqrt(3.0 / 5.0)]
            weight = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        elif gauss_n == 4:
            t = np.sqrt(4.8)
            w = 1.0 / 3.0 / t
            gauss_point = [-np.sqrt((3.0 + t) / 7.0), -np.sqrt((3.0 - t) / 7.0), np.sqrt((3.0 - t) / 7.0),
                           np.sqrt((3.0 + t) / 7.0)]
            weight = [0.5 - w, 0.5 + w, 0.5 + w, 0.5 - w]
        elif gauss_n == 5:
            t = 2 * np.sqrt(10 / 7)
            w = 13 * np.sqrt(70)
            gauss_point = [-np.sqrt(5.0 + t) / 3.0, -np.sqrt(5.0 - t) / 3.0, 0, np.sqrt(5.0 - t) / 3.0,
                           np.sqrt(5.0 + t) / 3.0]
            weight = [(322 - w) / 900, (322 + w) / 900, 0, (322 + w) / 900, (322 - w) / 900]
    return gauss_point, weight
