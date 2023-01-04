import numpy as np
import model.lattice as ltc


class Simple:

    def __init__(self, lattice_length: int, lattice_dim: int = 2, j_value: float = 1.):
        self.j_value = j_value
        self.lattice_length = lattice_length
        self.lattice_dim = lattice_dim
        lattice = ltc.Lattice(lattice_length, lattice_dim)
        # if lattice_dim == 1:
        #     self.valid_cell_1d = lattice.valid_cell_r45_1d()
        # if lattice_dim == 2:
        #     self.valid_cell_2d = lattice.valid_cell_r45_2d()

    # def generate_lattice_r45(self) -> np.ndarray:
    #     return ltc.Lattice(self.lattice_length, self.lattice_dim).generate_r45()

    def generate_lattice(self) -> np.ndarray:
        return ltc.Lattice(self.lattice_length, self.lattice_dim).generate()

    def nn_1d(self, index: int) -> tuple:
        cell_0 = (index + self.lattice_length + 1) % self.lattice_length * self.lattice_length
        cell_1 = (index + self.lattice_length - 1) % self.lattice_length * self.lattice_length
        cell_2 = (index - self.lattice_length + 1) % self.lattice_length * self.lattice_length
        cell_3 = (index - self.lattice_length - 1) % self.lattice_length * self.lattice_length

        return cell_0, cell_1, cell_2, cell_3

    def nn_2d(self, x: int, y: int) -> tuple:
        xp1 = (x + 1) % self.lattice_length
        xm1 = (x - 1) % self.lattice_length
        yp1 = (y + 1) % self.lattice_length
        ym1 = (y - 1) % self.lattice_length

        return xp1, xm1, yp1, ym1

    def random_cell_energy_1d(self, lattice: np.ndarray) -> tuple:
        index = np.random.randint(len(self.valid_cell_1d))
        index = self.valid_cell_1d[index]
        nn = self.nn_1d(index)

        return index, -self.j_value * lattice[index] * (
                    lattice[nn[0]] +
                    lattice[nn[1]] +
                    lattice[nn[2]] +
                    lattice[nn[3]])

    def random_cell_energy_2d(self, lattice: np.ndarray) -> tuple:
        index = np.random.randint(len(self.valid_cell_2d))
        x = self.valid_cell_2d[index][0]
        y = self.valid_cell_2d[index][1]
        nn = self.nn_2d(x, y)

        return x, y, -self.j_value * lattice[x, y] * (
                lattice[nn[0], y] +
                lattice[nn[1], y] +
                lattice[x, nn[2]] +
                lattice[x, nn[3]])
