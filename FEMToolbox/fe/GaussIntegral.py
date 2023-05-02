import typing

from FEMToolbox.fe.Function import Fun, udFun
import numpy as np


def gauss_rule(dim, gauss_n):
    # Reference : https://en.wikipedia.org/wiki/Gaussian_quadrature
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
    return gauss_point, weight


def gauss_intergral(n, *funlist):
    # 1d 2d
    pass


def gauss_integral_1D(*funlist:typing.Tuple[Fun,udFun],gauss_n=4):
    val = 0.0
    if len(funlist) == 1:
        gp, w = gauss_rule(1, gauss_n)
        for i in range(len(funlist[0].basicfun_list)):
            if funlist[0].coef[i] == 0:
                continue
            bf = funlist[0].basicfun_list[i]
            for cell in bf.cell_fun_map.keys():
                a, b = cell.node_range[0][0], cell.node_range[1][0]
                for j in range(gauss_n):
                    point = ((b - a) / 2) * gp[j] + (a + b) / 2
                    fun_val = bf.get_value([point])
                    val += (b - a) / 2 * w[j] * fun_val * funlist[0].coef[i]

    elif len(funlist) == 2:
        f1, f2 = funlist[0], funlist[1]
        for cell in f1.domain.cells:
            a, b = cell.node_range[0][0], cell.node_range[1][0]
            gp, w = gauss_rule(1, gauss_n)
            for j in range(gauss_n):
                point = ((b - a) / 2) * gp[j] + (a + b) / 2
                if not f1.return_grad:
                    f1_val, f2_val = f1.get_value([point]), f2.get_value([point])
                    # fun_val = 0 if (f1_val is None or f2_val is None) else f1_val * f2_val
                    fun_val = f1_val * f2_val
                    val += (b - a) / 2 * w[j] * fun_val
                else:
                    fun_grad, f1_grad, f2_grad = 0.0, f1.get_grad([point]), f2.get_grad([point])
                    # fun_grad = 0.0 if (f1_grad is None or f2_grad is None) else f1_grad.item(0) * f2_grad.item(0)
                    fun_grad = f1_grad.item(0) * f2_grad.item(0)
                    val += (b - a) / 2 * w[j] * fun_grad
        f1.ungrad()
        f2.ungrad()

    return val


def gauss_integral_2D(n, *funlist: Fun):
    pass
