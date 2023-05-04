import typing

from FEMToolbox.fe.Function import Fun, udFun
from FEMToolbox.fe.GaussRule import gauss_rule
import numpy as np

GP, W = [0], [0]
for i in range(1, 6):
    gp, w = gauss_rule(1, i)
    GP.append(gp)
    W.append(w)


def gauss_integral_1D(gauss_n=4, *funlist: typing.Tuple[Fun, udFun]):
    val = 0.0
    if len(funlist) == 1:
        for cell in funlist[0].domain.cells:
            a, b = cell.node_range[0][0], cell.node_range[1][0]
            gp, w = gauss_rule(1, gauss_n)
            for j in range(gauss_n):
                point = ((b - a) / 2) * gp[j] + (a + b) / 2
                if not funlist[0].return_grad:
                    fun_val = funlist[0].get_value([point])
                    # fun_val = 0 if (f1_val is None or f2_val is None) else f1_val * f2_val
                    val += (b - a) / 2 * w[j] * fun_val

    elif len(funlist) == 2:
        f1, f2 = funlist[0], funlist[1]
        for cell in f1.domain.cells:
            a, b = cell.node_range[0][0], cell.node_range[1][0]
            gp, w = gauss_rule(1, gauss_n)
            for j in range(gauss_n):
                point = ((b - a) / 2) * gp[j] + (a + b) / 2
                if not f1.return_grad:
                    f1_val, f2_val = f1.get_value([point]), f2.get_value([point])
                    fun_val = f1_val * f2_val
                    val += (b - a) / 2 * w[j] * fun_val
                else:
                    fun_grad, f1_grad, f2_grad = 0.0, f1.get_grad([point]), f2.get_grad([point])
                    fun_grad = f1_grad.item(0) * f2_grad.item(0)
                    val += (b - a) / 2 * w[j] * fun_grad
        f1.ungrad()
        f2.ungrad()

    return val


MEM_f = np.array([])
MEM_ff = np.array([])


def update_mem_matrix(n):
    MEM_f.resize(n, 2)
    MEM_ff.resize(n, n, 4)


def quick_gauss_integral_1D(gauss_n=4, *funlist: typing.Tuple[Fun]):
    val = 0.0
    if len(funlist) == 1:
        f: Fun = funlist[0]
        if f.funid != -1 and MEM_f[f.funid][0] == 1:
            return MEM_f[f.funid][1]

        for cell_no in f.nozero_domain:
            a, b = f.domain.cells[cell_no].node_range[0][0], f.domain.cells[cell_no].node_range[1][0]
            for bf_gval in f.CBGV[cell_no]:
                # [bf_gval[0]:no of bf which sit in domian 'cell'
                if f.coef[bf_gval[0]] == 0:
                    continue
                for e in range(gauss_n):
                    val += (b - a) / 2 * W[gauss_n][e] * f.coef[bf_gval[0]] * bf_gval[gauss_n][e]

        if f.funid != -1:
            MEM_f[f.funid][0], MEM_f[f.funid][1] = 1, val
    elif len(funlist) == 2:
        f1, f2 = funlist[0], funlist[1]
        if not f1.return_grad:
            if f1.funid != -1 and f2.funid != -1 and MEM_ff[f1.funid][f2.funid][0] == 1:
                return MEM_ff[f1.funid][f2.funid][1]

            for cell_no in f1.nozero_domain:
                a, b = f1.domain.cells[cell_no].node_range[0][0], f1.domain.cells[cell_no].node_range[1][0]
                for e in range(gauss_n):
                    f1val, f2val = 0.0, 0.0
                    for bf_gval in f1.CBGV[cell_no]:
                        f1val += f1.coef[bf_gval[0]] * bf_gval[gauss_n][e]
                        f2val += f2.coef[bf_gval[0]] * bf_gval[gauss_n][e]
                    val += (b - a) / 2 * W[gauss_n][e] * f1val * f2val

            if f1.funid != -1 and f2.funid != -1:
                MEM_ff[f1.funid][f2.funid][0], MEM_ff[f1.funid][f2.funid][1] = 1, val
                MEM_ff[f2.funid][f1.funid][0], MEM_ff[f2.funid][f1.funid][1] = 1, val
        else:
            if f1.funid != -1 and f2.funid != -1 and MEM_ff[f1.funid][f2.funid][2] == 1:
                f1.ungrad()
                f2.ungrad()
                return MEM_ff[f1.funid][f2.funid][3]

            for cell in f1.domain.cells:
                a, b = cell.node_range[0][0], cell.node_range[1][0]
                for e in range(gauss_n):
                    f1val, f2val = 0.0, 0.0
                    for bf_ggrad in f1.CBGG[cell.no]:
                        f1val += f1.coef[bf_ggrad[0]] * bf_ggrad[gauss_n][e]
                        f2val += f2.coef[bf_ggrad[0]] * bf_ggrad[gauss_n][e]
                    val += (b - a) / 2 * W[gauss_n][e] * f1val * f2val

                if f1.funid != -1 and f2.funid != -1:
                    MEM_ff[f1.funid][f2.funid][2], MEM_ff[f1.funid][f2.funid][3] = 1, val
                    MEM_ff[f2.funid][f1.funid][2], MEM_ff[f2.funid][f1.funid][3] = 1, val
            f1.ungrad()
            f2.ungrad()

    return val
