import numpy as np


class BasisFun:
    def __init__(self, offset, cellfun):
        """
        :param cellfun: domain,lambdafun,...
        """
        # print(type(cellfun))
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
    def __init__(self, domain):
        self.n = domain.size
        self.dim = 1
        self.basisfun_list = []
        self.domain = domain
        self.global_cellfun_map = {}  # map:cell->BasisFun

    def append_basisfun(self, offset, *cellfun):
        bf = BasisFun(offset, cellfun)
        self.basisfun_list.append(bf)
        self.dim = bf.dim
        for i in range(len(cellfun)):
            if i & 1:
                self.global_cellfun_map[cellfun[i - 1]] = bf

    # TODO: define L2,H1....
    def L2_norm(self):
        pass


class Fun(FunctionSpace):
    '''
    sum(i=0:n) coef[i]*BasisFun[i].getvalue(variables)
    '''

    def __init__(self, fs, coeflist):
        super().__init__(fs.domain)
        self.n = fs.domain.size
        self.basicfun_list = fs.basisfun_list
        self.coef = coeflist
        self.return_grad = False

    def get_value(self, coords):
        val = 0.0
        for i in range(self.n):
            if self.coef[i] == 0:
                continue
            fun_val = self.basicfun_list[i].get_value(coords)
            if fun_val is None:
                fun_val = 0.0
            val += self.coef[i] * fun_val
        return val

    def get_grad(self, coords):
        grad = 0.0
        for i in range(self.n):
            if self.coef[i] == 0:
                continue
            fun_grad = self.basicfun_list[i].get_grad(coords)
            if fun_grad is None:
                fun_grad = np.array([0.0])
            grad += self.coef[i] * fun_grad.item(0)
        return np.array([grad])

    def grad(self):
        self.return_grad = True
        return self

    def ungrad(self):
        self.return_grad = False

class udFun:
    def __init__(self,fun):
        self.fun=fun
        self.return_grad=False

    def get_value(self,coords):
        if len(coords) == 1 :
            return self.fun(coords[0])

    def get_grad(self,coords):
        if len(coords) == 1:
            dx = 0.0001
            gradf = (self.fun(coords[0] + dx) - self.fun(coords[0] - dx)) / (2 * dx)
            return np.array([gradf])

    def grad(self):
        self.return_grad=True
        return self

    def ungrad(self):
        self.return_grad=False
