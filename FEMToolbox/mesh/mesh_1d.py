import numpy as np
from FEMToolbox.mesh.cell import Cell

class unit_interval:
    def __init__(self, n):
        self.n = n
        self.cells = []
        self.size = n
        self.nodes=np.zeros((n))
        self.CreateCells()

    def CreateCells(self):
        dx = 1 / self.n
        p = 0.0
        for i in range(self.size):
            coords = np.zeros((2, 1))
            coords[0][0] = p
            #self.nodes.append(p)
            p += dx
            coords[1][0] = p
            self.nodes[i]=p
            self.cells.append(Cell(i + 1, coords))

    def get_range(self):
        return 0,1

    def Print(self,):
        k = 1
        for i in self.cells:
            print("elem", i.no, ":", i.__str__())
            k = k + 1


