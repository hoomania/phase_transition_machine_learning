import numpy as np


class Lattice:

    def __init__(self, length: int, dim: int = 2):
        self.length = length
        self.length_r45 = length * 2
        self.dim = dim

    def generate(self) -> np.ndarray:
        lattice = np.random.randint(2, size=(self.length, self.length))
        lattice = 2 * lattice - 1

        if self.dim == 1:
            return lattice.reshape(self.length * self.length)

        return lattice

    def generate_r45(self) -> np.ndarray:
        lattice_len = self.length * 2
        spin = 2 * np.random.randint(2, size=(lattice_len, self.length)) - 1
        lattice = np.zeros((lattice_len, lattice_len), dtype=int)
        for i in range(lattice_len):
            for j in range(lattice_len):
                if i % 2 == 0 and j % 2 == 0:
                    lattice[i, j] = spin[i, int(j / 2)]
                if i % 2 != 0 and j % 2 != 0:
                    lattice[i, j] = spin[i, int((j - 1) / 2)]

        if self.dim == 1:
            return lattice.reshape(lattice_len * lattice_len)

        return lattice

    def valid_cell_r45_1d(self) -> list:
        lattice_len = self.length * 2
        lattice = np.zeros((lattice_len, lattice_len), dtype=int)
        for i in range(lattice_len):
            for j in range(lattice_len):
                if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
                    lattice[i, j] = 1

        valid_cell = [0 for i in range(int(lattice_len * lattice_len / 2))]
        lattice = lattice.reshape(lattice_len * lattice_len)
        counter = 0
        for i in range(len(lattice)):
            if lattice[i] == 1:
                valid_cell[counter] = i
                counter += 1

        return valid_cell

    def valid_cell_r45_2d(self) -> list:
        lattice_len = self.length * 2
        valid_cell = [[] for i in range(int(lattice_len * lattice_len / 2))]
        counter = 0
        for i in range(lattice_len):
            for j in range(lattice_len):
                if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
                    valid_cell[counter] = [i, j]
                    counter += 1

        return valid_cell
