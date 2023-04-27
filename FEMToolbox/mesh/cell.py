from enum import Enum
import numpy as np

class CellType(Enum):
    interval = 1
    quadrilateral = 2

class Cell:
    def __init__(self,nodecoords:np.ndarray):
        self. nodecoords=nodecoords
        self.Type=nodecoords.shape[-1]

    def __str__(self):
        return self.nodecoords.__str__()
