import numpy as np
from FEMToolbox.mesh.MeshCell import Cell
from FEMToolbox.fe.Function import FunctionSpace, Fun
import matplotlib.pyplot as plt


class Interval:
    def __init__(self, xmin, xmax, n):
        self.size = n
        self.cells = []
        self.xmin = xmin
        self.xmax = xmax
        self.length = xmax - xmin
        self.nodes = np.zeros(n+1)
        self.create_cells()

    def create_cells(self):
        dx = self.length / self.size
        p = self.xmin
        self.nodes[0]=0.0
        for i in range(self.size):
            coords = np.zeros((2, 1))
            coords[0][0] = p
            # self.nodes.append(p)
            p += dx
            coords[1][0] = p
            self.nodes[i+1] = p
            self.cells.append(Cell(i + 1, coords))

    def get_range(self):
        return self.xmin, self.xmax

    def print_mesh(self):
        for i in range(len(self.cells)):
            print("elem", self.cells[i].no, ":", self.cells[i].__str__())

    def plot(self, basisfun_list, t):
        x = np.linspace(self.xmin, self.xmax, t)
        y = np.zeros(t)
        plt.figure()
        plt.xlabel("x")
        plt.ylabel("y")
        for i in range(self.size):
            j = 0
            for xi in x:
                val=basisfun_list[i].get_value([xi])
                y[j] = val if val!=0 else None
                j = j + 1
            plt.plot(x, y, label='BasisFun' + str(i + 1))
        plt.legend()
        plt.show()


def create_intercval(xmin, xmax, n, order=1, method='default'):
    msh = Interval(xmin, xmax, n)
    fun_sp = FunctionSpace(msh)
    dx = msh.length / n
    if method == 'default':
        if order == 1:
            for i in range(n - 1):
                offset = msh.nodes[i+1]
                fun_sp.append_basisfun([offset], msh.cells[i], lambda x: 1 + x / dx, msh.cells[i + 1],
                                       lambda x: 1 - x / dx)
            fun_sp.append_basisfun([msh.nodes[-1]], msh.cells[n - 1], lambda x: 1 + x / dx)

    basisfun_list = []
    for i in range(n):
        coef = np.zeros(n)
        coef[i] = 1
        basisfun_list.append(Fun(fun_sp, coef))

    return msh, basisfun_list, fun_sp
