import numpy as np
from SlowFEM.mesh.MeshCell import Cell
from SlowFEM.fe.Function import FunctionSpace, Fun
import matplotlib.pyplot as plt


class Interval:
    def __init__(self, xmin, xmax, size):
        self.size = size
        self.cells = []
        self.xmin = xmin
        self.xmax = xmax
        self.length = xmax - xmin
        self.nodes = np.zeros(size + 1)
        self.create_cells()
        self.dim = 1
        self.basisfun_list=[]

    def create_cells(self):
        dx = self.length / self.size
        p = self.xmin
        self.nodes[0] = 0.0
        for i in range(self.size):
            coords = np.zeros((2, 1))
            coords[0][0] = p
            p += dx
            coords[1][0] = p
            self.nodes[i + 1] = p
            self.cells.append(Cell(i, coords))

    def get_range(self):
        return self.xmin, self.xmax

    def print_mesh(self):
        for i in range(len(self.cells)):
            print("elem", self.cells[i].no, ":", self.cells[i].__str__())

    def plot(self, t = 1000):
        x = np.linspace(self.xmin, self.xmax, t)
        y = np.zeros(t)
        plt.figure()
        plt.xlabel("x")
        plt.ylabel("y")
        for i in range(len(self.basisfun_list)):
            j = 0
            for xi in x:
                val = self.basisfun_list[i].get_value([xi])
                y[j] = val if val != 0 else None
                j = j + 1
            plt.plot(x, y, label='BasisFun' + str(i + 1))
        plt.legend()
        plt.show()


def create_intercval_val0grad0(xmin, xmax, size, order=1, method='default'):
    n = size
    msh = Interval(xmin, xmax, size)
    fun_sp = FunctionSpace(msh, n)
    dx = msh.length / size
    if method == 'default':
        if order == 1:
            for i in range(n - 1):
                offset = msh.nodes[i + 1]
                fun_sp.append_basisfun([offset], msh.cells[i], lambda x: 1 + x / dx, msh.cells[i + 1],
                                       lambda x: 1 - x / dx)
            fun_sp.append_basisfun([msh.nodes[-1]], msh.cells[n - 1], lambda x: 1 + x / dx)
    basisfun = []
    for i in range(n):
        coef = np.zeros(n)
        coef[i] = 1
        basisfun.append(Fun(fun_sp, coef, i))
    msh.basisfun_list=basisfun

    return msh, basisfun, fun_sp


def create_intercval_val0val0(xmin, xmax, size, order=1, method='default'):
    n = size - 1
    msh = Interval(xmin, xmax, size)
    fun_sp = FunctionSpace(msh, n)
    fun_sp.n = n
    dx = msh.length / size
    if method == 'default':
        if order == 1:
            for i in range(n):
                offset = msh.nodes[i + 1]
                fun_sp.append_basisfun([offset], msh.cells[i], lambda x: 1 + x / dx, msh.cells[i + 1],
                                       lambda x: 1 - x / dx)
    basisfun = []
    for i in range(n):
        coef = np.zeros(n)
        coef[i] = 1
        basisfun.append(Fun(fun_sp, coef, i))
    msh.basisfun_list = basisfun

    return msh, basisfun, fun_sp


def create_intercval_grad0grad0(xmin, xmax, size, order=1, method='default'):
    n = size + 1
    msh = Interval(xmin, xmax, size)
    fun_sp = FunctionSpace(msh, n)
    dx = msh.length / size
    if method == 'default':
        if order == 1:
            fun_sp.append_basisfun([msh.nodes[0]], msh.cells[0], lambda x: 1 - x / dx)
            for i in range(n - 2):
                offset = msh.nodes[i + 1]
                fun_sp.append_basisfun([offset], msh.cells[i], lambda x: 1 + x / dx, msh.cells[i + 1],
                                       lambda x: 1 - x / dx)
            fun_sp.append_basisfun([msh.nodes[-1]], msh.cells[n - 2], lambda x: 1 + x / dx)
    basisfun = []
    for i in range(n):
        coef = np.zeros(n)
        coef[i] = 1
        basisfun.append(Fun(fun_sp, coef, i))
    msh.basisfun_list = basisfun

    return msh, basisfun, fun_sp
