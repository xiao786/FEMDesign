from cell import *
import numpy as np


class create_unit_interval:
    def __init__(self, nx):
        self.nx = nx
        self.cells = []
        self.size = nx
        self.create_cells(nx)

    def create_cells(self, nx):
        dx = 1 / nx
        p = 0.0
        for i in range(self.size):
            tmp = np.zeros(2)
            tmp[0] = p
            p += dx
            tmp[1] = p
            self.cells.append(Cell(tmp))

    def print(self):
        k = 1
        for i in self.cells:
            print("elem", k, ":", i.__str__())
            k = k + 1


msh = create_unit_interval(7)
msh.print()
