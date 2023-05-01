import numpy as np


class basicfun():
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


class functionspace:
    def __init__(self, domain):
        self.n = domain.n
        self.dim = 1
        self.basicfun_list = []
        self.domain = domain
        self.global_cellfun_map={} # map:cell->basicfun

    def append_basicfun(self, offset, *cellfun):
        bf = basicfun(offset, cellfun)
        self.basicfun_list.append(bf)
        self.dim = bf.dim
        for i in range(len(cellfun)):
            if i & 1:
                self.global_cellfun_map[cellfun[i - 1]] = bf

    # TODO: define L2,H1....
    def L2_norm(self):
        pass


class fun(functionspace):
    '''
    sum(i=0:n) coef[i]*basicfun[i].getvalue(variables)
    '''

    def __init__(self, fs, coeflist):
        super().__init__(fs.domain)
        self.n = fs.domain.n
        self.basicfun_list = fs.basicfun_list
        self.cell_basicfun={}
        self.coef = coeflist
        self.return_grad = False

    def get_value(self, coords):
        val = 0.0
        for i in range(self.n):
            funval = self.basicfun_list[i].get_value(coords)
            if funval is None:
                funval = 0.0
                # continue
            #print('basicfun%d: ' % (i + 1), 'coef:', self.coef[i], 'bascfunval:', funval)
            val += self.coef[i] * funval
        if val == 0.0:
            return None
        return val

    def get_grad(self, coords):
        for i in range(self.n):
            fungrad = self.basicfun_list[i].get_grad(coords)
            if fungrad is None:
                continue
            return fungrad

    def grad(self):
        self.return_grad = True

    def ungrad(self):
        self.return_grad = False

    def get_gaussvalue(self, coords):
        if self.return_grad:
            val = self.get_grad(coords)
        else:
            val = self.get_value(coords)
        if val is None:
            return 0.0
        return val
