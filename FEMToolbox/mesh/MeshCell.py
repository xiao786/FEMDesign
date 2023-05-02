from enum import Enum
import numpy as np


class CellType(Enum):
    interval = 1
    quadrilateral = 2


class Cell:
    def __init__(self, no, node_range):
        '''
        :param no:
        :param node_range: a [2][n] matrix,denotes that 2 (n dim points)
        '''
        self.no = no
        self.node_range = node_range
        self.dim = len(node_range[0])

    def in_cell(self, coords) -> bool:
        for i in range(self.dim):
            if coords[i] < self.node_range[0][i] or coords[i] > self.node_range[1][i]:
                return False
        return True

    def __str__(self):
        return self.node_range.__str__()
