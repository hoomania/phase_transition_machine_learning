import numpy as np


class Triangle:

    def __init__(self, lattice: np.ndarray, j_value: float = 1.):
        self.j_value = j_value
        self.lattice = lattice

        if len(lattice.shape) == 2:
            self.dim = 2
            self.len = lattice.shape[0]

        if len(lattice.shape) == 1:
            self.dim = 1
            self.len = len(lattice)
            self.len_root = np.sqrt(len(lattice))

    def nn_1d(self, index: int) -> tuple:
        cell_0 = (index + self.len_root + 1) % self.len
        cell_1 = (index + self.len_root - 1) % self.len
        cell_2 = (index - self.len_root + 1) % self.len
        cell_3 = (index - self.len_root - 1) % self.len
        cell_4 = (index + 2) % self.len
        cell_5 = (index - 2) % self.len

        return cell_0, cell_1, cell_2, cell_3, cell_4, cell_5

    def nn_2d(self, x: int, y: int) -> tuple:
        xp1 = (x + 1) % self.len
        xm1 = (x - 1) % self.len
        yp1 = (y + 1) % self.len
        ym1 = (y - 1) % self.len
        xp2 = (x + 2) % self.len
        xm2 = (x - 2) % self.len

        return xp1, xm1, xp2, xm2, yp1, ym1

    def node_energy(self) -> float:
        if self.dim == 1:
            index = np.random.randint(self.len)
            nn = self.nn_1d(index)

            return -self.j_value * self.lattice[index] * (
                    self.lattice[nn[0]] +
                    self.lattice[nn[1]] +
                    self.lattice[nn[2]] +
                    self.lattice[nn[3]] +
                    self.lattice[nn[4]] +
                    self.lattice[nn[5]])

        if self.dim == 2:
            x = np.random.randint(self.len)
            y = np.random.randint(self.len)
            nn = self.nn_2d(x, y)

            return -self.j_value * self.lattice[x, y] * (
                    self.lattice[nn[0], nn[5]] +
                    self.lattice[nn[2], y] +
                    self.lattice[nn[0], nn[4]] +
                    self.lattice[nn[1], nn[4]] +
                    self.lattice[nn[3], y] +
                    self.lattice[nn[1], nn[5]])
