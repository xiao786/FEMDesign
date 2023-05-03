import numpy as np

from FEMToolbox.mesh.Mesh1D import create_intercval_grad0grad0
from FEMToolbox.fe.GaussIntegral import gauss_integral_1D
from FEMToolbox.fe.Function import Fun, udFun, plot1D, plotgap

msh, basisfun, funspace = create_intercval_grad0grad0(0, 1, 10)



f = udFun(msh, lambda x: np.cos(np.pi * x))

n = funspace.n
A = np.zeros((n, n))
F = np.zeros(n)

for i in range(n):
    for j in range(n):
        A[i][j] = gauss_integral_1D(basisfun[i], basisfun[j])
    F[i] = gauss_integral_1D(basisfun[i], f)

coef = np.linalg.solve(A, F)
curfun = Fun(funspace, coef)

t0, te = 0, 0.1
step = 500
dt = (te - t0) / step
tn = t0

for k in range(step):
    tn+=dt
    for i in range(n):
        for j in range(n):
            A[i][j] = gauss_integral_1D(basisfun[j], basisfun[i])/(dt*1)+ \
                      gauss_integral_1D(basisfun[j].grad(), basisfun[i].grad())

        F[i] = gauss_integral_1D(curfun, basisfun[i])/(dt*1)+np.sin(tn)*gauss_integral_1D(basisfun[i])
        coef = np.linalg.solve(A, F)
        curfun.coef=coef

ana_fun = udFun(msh, lambda x: np.exp(-np.pi * np.pi * tn) * np.cos(np.pi * x) + (1 - np.cos(tn)))
l=100
xt=np.linspace(0,1,l)
ue,un,e=np.zeros(l),np.zeros(l),np.zeros(l)
for it in range(len(xt)):
    ue[it],un[it]=ana_fun.get_value([xt[it]]),curfun.get_value([xt[it]])
    e[it] = (ue[it]-un[it])**2
L2_error = np.sqrt(np.sum(e) / 100)
print(L2_error)

    #print(tn)

plot1D(100,curfun,ana_fun)
    #plotgap(1000, curfun,ana_fun)

