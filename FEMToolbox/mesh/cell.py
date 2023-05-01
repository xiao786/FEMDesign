from enum import Enum
import numpy as np

class CellType(Enum):
    interval = 1
    quadrilateral = 2

class Cell:
    def __init__(self,no,noderange):
        '''
        :param no:
        :param noderange: a [2][n] matrix,denotes that 2 (n dim points)
        '''
        self.no=no
        self.noderange=noderange
        self.dim=len(noderange[0])

    def in_cell(self,coords) -> bool:
        for i in range(self.dim):
            if coords[i] < self.noderange[0][i] or coords[i] > self.noderange[1][i]:
                return False
        return True


    def __str__(self):
        return self.noderange.__str__()