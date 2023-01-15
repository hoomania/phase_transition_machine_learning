import numpy as np
import model.lattice as ltc


class Square:

    def __init__(self, lattice_length: int, lattice_dim: int = 1, j_value: float = 1.):
        self.j_value = j_value
        self.lattice_dim = lattice_dim
        self.lattice_length = lattice_length
        self.lattice_length_r45 = lattice_length * 2
        self.vector_length = lattice_length * lattice_length
        self.lattice = ltc.Lattice(self.lattice_length, self.lattice_dim)

    def generate_lattice(self) -> np.ndarray:
        return self.lattice.generate()
        # return ltc.Lattice(self.lattice_length, self.lattice_dim).generate()

    def nn_1d(self, index: int) -> tuple:
        cell_0 = (index + 1) % self.vector_length
        cell_1 = (index - 1) % self.vector_length
        cell_2 = (index + self.lattice_length) % self.vector_length
        cell_3 = (index - self.lattice_length) % self.vector_length

        return cell_0, cell_1, cell_2, cell_3

    def nn_2d(self, x: int, y: int) -> tuple:
        xp1 = (x + 1) % self.lattice_length
        xm1 = (x - 1) % self.lattice_length
        yp1 = (y + 1) % self.lattice_length
        ym1 = (y - 1) % self.lattice_length

        return [xp1, y], [xm1, y], [x, yp1], [x, ym1]

    def random_cell_energy_1d(self, lattice: np.ndarray) -> tuple:
        index = np.random.randint(self.vector_length)
        nn = self.nn_1d(index)

        s = 0
        for i in nn:
            s += lattice[i]
        return index, -self.j_value * lattice[index] * s

    def random_cell_energy_2d(self, lattice: np.ndarray) -> tuple:
        x = np.random.randint(self.lattice_length)
        y = np.random.randint(self.lattice_length)
        nn = self.nn_2d(x, y)

        s = 0
        for i in nn:
            s += lattice[i[0], i[1]]

        return x, y, -self.j_value * lattice[x, y] * s

    def random_cell_diff_energy_flip_2d(self, lattice: np.ndarray) -> tuple:
        cell_info = self.random_cell_diff_energy_flip_2d(lattice)
        return cell_info[0], cell_info[1], -2 * cell_info[2]

    def generate_lattice_r45(self) -> np.ndarray:
        return self.lattice.generate_r45()

    def nn_r45_2d(self, x: int, y: int) -> tuple:
        xp1 = (x + 1) % self.lattice_length_r45
        xm1 = (x - 1) % self.lattice_length_r45
        yp1 = (y + 1) % self.lattice_length_r45
        ym1 = (y - 1) % self.lattice_length_r45

        return [xm1, ym1], [xm1, yp1], [xp1, yp1], [xp1, ym1]

    def random_cell_energy_r45_2d(self, lattice: np.ndarray) -> tuple:
        valid_cell = self.lattice.valid_cell_r45_2d()
        random_cell = np.random.randint(len(valid_cell))
        random_cell = valid_cell[random_cell]

        x = random_cell[0]
        y = random_cell[1]
        nn = self.nn_r45_2d(x, y)

        s = 0
        for i in nn:
            s += lattice[i[0], i[1]]

        return x, y, -self.j_value * lattice[x, y] * s

    def random_cell_diff_energy_flip_r45_2d(self, lattice: np.ndarray) -> tuple:
        cell_info = self.random_cell_energy_r45_2d(lattice)
        return cell_info[0], cell_info[1], -2 * cell_info[2]
