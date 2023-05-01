import numpy as np
from FEMToolbox.mesh.mesh_1d import unit_interval
from FEMToolbox.fe.function import *
from FEMToolbox.fe.GaussIntegral import GIntegral1D

n = 10
dx = 1 / n
msh = unit_interval(n)
fun_sp = functionspace(msh)
for i in range(n - 1):
    offset = 0.1 * (i + 1)
    fun_sp.append_basicfun([offset], msh.cells[i], lambda x: 1 + x / dx, msh.cells[i + 1], lambda x: 1 - x / dx)
fun_sp.append_basicfun([1], msh.cells[n - 1], lambda x: 1 + x / dx)

basfun = []
for i in range(n):
    coef = np.zeros((n))
    coef[i] = 1
    basfun.append(fun(fun_sp, coef))

for i in range(9):
    print(GIntegral1D(4,basfun[i],basfun[i+1]))


def plot():
    n = 1001
    x = np.linspace(0, 1, n)
    y = np.zeros(n)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(10):
        j = 0
        for xi in x:
            y[j] = basfun[i].get_value([xi])
            j = j + 1
        plt.plot(x, y, label='basicfun' + str(i + 1))
    plt.legend()
    plt.show()
#plot()