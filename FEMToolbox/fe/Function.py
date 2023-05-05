import typing
import numpy as np
import matplotlib.pyplot as plt
from FEMToolbox.fe.GaussRule import gauss_rule


class BasisFun:
    def __init__(self, offset, cellfun):
        """
        :param cellfun: domain,lambdafun,...
        """
        self.dim = cellfun[0].dim
        self.cell_fun_map = {}  # cell -> lambdafun
        self.offset = offset
        for i in range(len(cellfun)):
            if i & 1:
                self.cell_fun_map[cellfun[i - 1]] = cellfun[i]

    def get_value(self, coords):
        for cell in self.cell_fun_map.keys():
            if cell.in_cell(coords):
                if len(coords) == 1:
                    return self.cell_fun_map[cell](coords[0] - self.offset[0])

    def get_grad(self, coords):
        for cell in self.cell_fun_map.keys():
            if cell.in_cell(coords):
                if len(coords) == 1:
                    coords[0] -= self.offset[0]
                    u = self.cell_fun_map[cell]
                    dx = 0.0001
                    gradu = (u(coords[0] + dx) - u(coords[0] - dx)) / (2 * dx)
                    return np.array([gradu])


class FunctionSpace:
    def __init__(self, domain, n):
        self.n = n  # n*basisfun
        self.dim = domain.dim
        self.basisfun_list = []
        self.domain = domain
        self.CBGV = []  # [cell.no][bf(on cell)_no(0,1..][gn]->[val of 'value'](if gn=0, then->basisfun-no)
        self.CBGG = []  # [cell.no][bf(on cell)_no(0,1..][gn]->[val of 'grad'](if gn=0, then->basisfun-no)
        for i in range(self.domain.size):
            self.CBGV.append([])
            self.CBGG.append([])

    def append_basisfun(self, offset, *cellfun):
        bf = BasisFun(offset, cellfun)
        index = len(self.basisfun_list)  # index of this basisfun in the basislist
        self.basisfun_list.append(bf)
        for i in range(len(cellfun)):
            if i & 1:
                a, b = cellfun[i - 1].node_range[0][0], cellfun[i - 1].node_range[1][0]
                V, G = [index], [index]
                for j in range(1, 6):
                    gp, w = gauss_rule(self.dim, j)
                    CBGV_t, CBGG_t = [], []
                    for k in range(len(gp)):
                        point = ((b - a) / 2) * gp[k] + (a + b) / 2
                        CBGV_t.append(bf.get_value([point]))
                        CBGG_t.append(bf.get_grad([point]))
                    V.append(CBGV_t)
                    G.append(CBGG_t)
                self.CBGV[cellfun[i - 1].no].append(V)
                self.CBGG[cellfun[i - 1].no].append(G)


class Fun(FunctionSpace):
    '''
    sum(i=0:n) coef[i]*BasisFun[i].getvalue(variables)
    '''

    def __init__(self, fs, coef, funid=-1):
        super().__init__(fs.domain, fs.n)
        self.coef=[]
        self.basisfun_list = fs.basisfun_list
        self.return_grad = False
        self.CBGV = fs.CBGV
        self.CBGG = fs.CBGG
        self.nozero_domain = []
        self.update(coef)
        self.funid = funid

    def get_value(self, coords) -> float:
        val = 0.0
        for i in range(self.n):
            if self.coef[i] == 0:
                continue
            fun_val = self.basisfun_list[i].get_value(coords)
            if fun_val is None:
                fun_val = 0.0
            val += self.coef[i] * fun_val
        return val

    def get_grad(self, coords) -> np.ndarray:
        grad = 0.0
        for i in range(self.n):
            if self.coef[i] == 0:
                continue
            fun_grad = self.basisfun_list[i].get_grad(coords)
            if fun_grad is None:
                fun_grad = np.array([0.0])
            grad += self.coef[i] * fun_grad.item(0)
        return np.array([grad])

    def get_gp_value(self, gauss_n):
        pass

    def grad(self):
        self.return_grad = True
        return self

    def ungrad(self):
        self.return_grad = False

    def update(self, coef):
        self.coef = coef
        tmp_nd = []
        for cell in range(self.domain.size):
            flag = False
            for bf_v in self.CBGV[cell]:
                if self.coef[bf_v[0]] != 0:
                    flag = True
            if flag:
                tmp_nd.append(cell)
        self.nozero_domain = tmp_nd


class udFun:
    def __init__(self, domian, fun):
        self.domain = domian
        self.fun = fun
        self.return_grad = False

    def get_value(self, coords):
        if len(coords) == 1:
            return self.fun(coords[0])

    def get_grad(self, coords):
        if len(coords) == 1:
            dx = 0.0001
            gradf = (self.fun(coords[0] + dx) - self.fun(coords[0] - dx)) / (2 * dx)
            return np.array([gradf])

    def grad(self):
        self.return_grad = True
        return self

    def ungrad(self):
        self.return_grad = False


def plot1D(t, *funlist: typing.Tuple[Fun, udFun]):
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(len(funlist)):
        fun = funlist[i]
        domain = fun.domain
        x = np.linspace(domain.xmin, domain.xmax, t)
        y = np.zeros(t)
        for j in range(len(x)):
            y[j] = fun.get_value([x[j]])
        plt.plot(x, y, label='function' + str(i))
        plt.legend()
    plt.show()


def plotgap(t, *funlist: typing.Tuple[Fun, udFun]):
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    f1, f2 = funlist[0], funlist[1]
    domain = f1.domain
    x = np.linspace(domain.xmin, domain.xmax, t)
    y = np.zeros(t)
    for j in range(len(x)):
        y[j] = f1.get_value([x[j]]) - f2.get_value([x[j]])
    plt.plot(x, y)
    plt.show()
