from cell import Cell
import numpy as np


class create_unit_interval:
    def __init__(self, nx):
        self.nx = nx
        self.cells = []
        self.size = nx
        self.nodes=np.zeros((nx))
        self.CreateCells(nx)

    def CreateCells(self, nx):
        dx = 1 / nx
        p = 0.0
        for i in range(self.size):
            coords = np.zeros((2, 1))
            coords[0][0] = p
            #self.nodes.append(p)
            p += dx
            coords[1][0] = p
            self.nodes[i]=p
            self.cells.append(Cell(i + 1, coords))

    def Print(self,):
        k = 1
        for i in self.cells:
            print("elem", i.no, ":", i.__str__())
            k = k + 1


