import numpy as np
from FEMToolbox.mesh.Mesh1D import create_intercval
from FEMToolbox.fe.Function import udFun,Fun
from FEMToolbox.fe.GaussIntegral import gauss_integral_1D

n=10
msh, basisfun, funspace = create_intercval(0, 1, n)
f=udFun(lambda x:-x)

A=np.zeros((n,n))
F=np.zeros(n)

for i in range(n):
    for j in range(n):
        a,b=gauss_integral_1D(basisfun[i], basisfun[j]) , gauss_integral_1D(basisfun[i].grad(), basisfun[j].grad())
        #print(a,b)
        A[i][j]=a-b

print(gauss_integral_1D(basisfun[0].grad(),basisfun[0].grad()))
for i in range(n):
    F[i]=gauss_integral_1D(basisfun[i],f)

print(A,F)

#msh.plot(basisfun,1000)
coef=np.linalg.solve(A,F)
#print(coef)

num_sol=Fun(funspace,coef)

print(num_sol.get_value([0.25]))