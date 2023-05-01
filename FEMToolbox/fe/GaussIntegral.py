from FEMToolbox.fe.function import fun, basicfun
import numpy as np


def gauss_rule(dim, n):
    # reference : https://en.wikipedia.org/wiki/Gaussian_quadrature
    if dim == 1:
        if n == 1:
            gausspoint = [0]
            weight = [2]
        elif n == 2:
            gausspoint = [1, 1]
            weight = [-np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)]
        elif n == 3:
            gausspoint = [-np.sqrt(3.0 / 5.0), 0, np.sqrt(3.0 / 5.0)]
            weight = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        elif n == 4:
            t = np.sqrt(4.8)
            w = 1.0 / 3.0 / t
            gausspoint = [-np.sqrt((3.0 + t) / 7.0), -np.sqrt((3.0 - t) / 7.0), np.sqrt((3.0 - t) / 7.0),
                          np.sqrt((3.0 + t) / 7.0)]
            weight = [0.5 - w, 0.5 + w, 0.5 + w, 0.5 - w]
    return gausspoint, weight



def GIntegral1D(n, *funlist: fun):
    val = 0.0
    if len(funlist) == 1:
        gp, w = gauss_rule(1, n)
        for i in range(len(funlist[0].basicfun_list)):
            if funlist[0].coef[i] == 0:
                continue
            bf = funlist[0].basicfun_list[i]
            for cell in bf.cell_fun_map.keys():
                a, b = cell.noderange[0][0], cell.noderange[1][0]
                for j in range(n):
                    point = ((b - a) / 2) * gp[j] + (a + b) / 2
                    if not funlist[0].return_grad:
                        funval = bf.get_value([point])
                    val += (b - a) / 2 * w[j] * funval * funlist[0].coef[i]

    elif len(funlist) == 2:
        f1,f2=funlist[0],funlist[1]
        for cell in f1.domain.cells:
            a, b = cell.noderange[0][0], cell.noderange[1][0]
            gp, w = gauss_rule(1, n)
            for j in range(n):
                point = ((b - a) / 2) * gp[j] + (a + b) / 2
                if not f1.return_grad:
                    f1_val,f2_val =f1.get_value([point]),f2.get_value([point])
                    funval = 0 if (f1_val is None or f2_val is None) else f1_val*f2_val
                    val += (b - a) / 2 * w[j] * funval

    for f in funlist:
        f.ungrad()

    return val
