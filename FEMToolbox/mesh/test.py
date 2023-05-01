import numpy as np

from create_mesh_1d import create_unit_interval
from FEMToolbox.fe.function import *

n = 10
dx = 1 / n
msh = create_unit_interval(n)
fun_sp = functionspace(n)
for i in range(n - 1):
    offset = 0.1 * (i + 1)
    fun_sp.append_basicfun([offset], msh.cells[i], lambda x: 1 + x / dx, msh.cells[i + 1], lambda x: 1 - x / dx)
fun_sp.append_basicfun([1], msh.cells[n - 1], lambda x: 1 + x / dx)

basfun = []
for i in range(n):
    coef = np.zeros((n))
    coef[i] = 1
    basfun.append(fun(fun_sp, coef))

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
        v = basfun[i].get_value([xi])
        if v != 0.0:
            y[j] = v
        j = j + 1
    plt.plot(x, y, label='basicfun' + str(i + 1))

plt.legend()
plt.show()


f = fun(fun_sp, [2, 1, 4, 7, 8, 2, 3, 11, 5, 9])
print(f.get_value([0.988962]))
