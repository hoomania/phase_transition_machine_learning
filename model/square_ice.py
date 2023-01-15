import numpy as np
import model.lattice as ltc


class SquareIce:

    def __init__(self, lattice_length: int, lattice_dim: int = 1, j_value: float = 1.):
        self.j_value = j_value
        self.lattice_dim = lattice_dim
        self.lattice_length = lattice_length
        self.vector_length = lattice_length * lattice_length

    def generate_lattice(self) -> np.ndarray:
        return ltc.Lattice(self.lattice_length, self.lattice_dim).generate()

    def nn_1d(self, index: int) -> tuple:
        cell_0 = (index - self.lattice_length) % self.vector_length
        cell_1 = (index - self.lattice_length + 1) % self.vector_length
        cell_2 = (index + 1) % self.vector_length
        cell_3 = index

        return cell_0, cell_1, cell_2, cell_3

    def nn_2d(self, x: int, y: int) -> tuple:
        xm1 = (x - 1) % self.lattice_length
        yp1 = (y + 1) % self.lattice_length

        return [xm1, y], [xm1, yp1], [x, yp1], [x, y]

    def random_cell_energy_1d(self, lattice: np.ndarray) -> tuple:
        index = np.random.randint(self.vector_length)
        nn = self.nn_1d(index)

        s = 0
        for i in nn:
            s += lattice[i]
        return index, self.j_value * s * s

    def random_cell_energy_2d(self, lattice: np.ndarray) -> tuple:
        x = np.random.randint(self.lattice_length)
        y = np.random.randint(self.lattice_length)
        nn = self.nn_2d(x, y)

        s = 0
        for i in nn:
            s += lattice[i[0], i[1]]

        return x, y, self.j_value * s * s

    def random_cell_diff_energy_flip_2d(self, lattice: np.ndarray) -> tuple:
        x = np.random.randint(self.lattice_length)
        y = np.random.randint(self.lattice_length)
        nn = self.nn_2d(x, y)

        s = 0
        for i in range(3):
            s += lattice[nn[i][0], nn[i][1]]

        return x, y, -4 * s