import numpy as np
from FEMToolbox.mesh.Mesh1D import create_intercval_00
from FEMToolbox.fe.Function import udFun,Fun,plot1D
from FEMToolbox.fe.GaussIntegral import gauss_integral_1D

n=10
msh, basisfun, funspace = create_intercval_00(0, 1, n)
f=udFun(msh,lambda x:-x)
n-=1
A=np.zeros((n,n))
F=np.zeros(n)

#msh.plot(basisfun,1000)


for i in range(n):
    for j in range(n):
        A[i][j]=gauss_integral_1D(basisfun[i], basisfun[j]) - gauss_integral_1D(basisfun[i].grad(), basisfun[j].grad())

for i in range(n):
    F[i]=gauss_integral_1D(basisfun[i],f)

coef=np.linalg.solve(A,F)

num_fun=Fun(funspace,coef)

#print(num_sol.get_value([3/4]))

ana_fun=udFun(msh,lambda x:np.sin(x)/np.sin(1)-x)
plot1D(1000,num_fun,ana_fun)