import numpy as np
from FEMToolbox.mesh.Mesh1D import create_intercval_val0val0
from FEMToolbox.fe.Function import udFun, Fun, plot1D
from FEMToolbox.fe.GaussIntegral import gauss_integral_1D, update_mem_matrix, quick_gauss_integral_1D

size = 100
msh, basisfun, funspace = create_intercval_val0val0(0, 1, size)
f = udFun(msh, lambda x: -x)
#msh.plot()


n = funspace.n
A = np.zeros((n, n))
F = np.zeros(n)


for i in range(n):
    for j in range(n):
        A[i][j] = quick_gauss_integral_1D(5, basisfun[i], basisfun[j]) - quick_gauss_integral_1D(5, basisfun[i].grad(),
                                                                                                 basisfun[j].grad())
    F[i] = gauss_integral_1D(5, basisfun[i], f)

coef = np.linalg.solve(A, F)
num_fun = Fun(funspace, coef)
ana_fun = udFun(msh, lambda x: np.sin(x) / np.sin(1) - x)

l = 100
xt = np.linspace(0, 1, l)
ue, un, e = np.zeros(l), np.zeros(l), np.zeros(l)
for it in range(len(xt)):
    ue[it], un[it] = ana_fun.get_value([xt[it]]), num_fun.get_value([xt[it]])
    e[it] = (ue[it] - un[it]) ** 2
L2_error = np.sqrt(np.sum(e) / 100)
print(L2_error)

plot1D(1000, num_fun, ana_fun)

