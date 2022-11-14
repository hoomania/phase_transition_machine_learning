import numpy
import numpy as np
import scipy.constants as spc
import matplotlib.pyplot as plt


class MonteCarlo:

    def __init__(self, lattice_length: int, sample_count: int, temperature: float):
        self.length = lattice_length
        self.sample_count = sample_count
        self.temp = temperature

    def convert_to_toric(self, matrix: numpy.ndarray) -> numpy.ndarray:
        length = self.length
        toric = np.zeros([length, length * 2])
        rows_index = [np.arange(0, length, 2), np.arange(1, length, 2)]

        for i in range(2):
            for row in rows_index[i]:
                for col_index in range(length):
                    toric[row][(2 * col_index) + i] = matrix[row][col_index]

        return toric

    def generate_lattice(self) -> numpy.ndarray:
        lattice = np.random.randint(0, 2, (self.length, self.length))
        lattice[lattice == 0] = - 1
        return lattice

    def node_energy(self, lattice: np.ndarray, x: int, y: int, coupling_value: float):
        return coupling_value * lattice[x, y] * (
                lattice[self.boundary_condition(x + 1), y] +
                lattice[self.boundary_condition(x - 1), y] +
                lattice[x, self.boundary_condition(y + 1)] +
                lattice[x, self.boundary_condition(y - 1)])

    def lattice_energy(self, lattice: np.ndarray) -> float:
        energy = 0
        for i in range(self.length):
            for j in range(self.length):
                energy = + self.node_energy(lattice, i, j, -1)

        return energy

    def boundary_condition(self, index: int) -> int:
        max_index = self.length - 1
        if index > max_index - 1:
            return 0
        if index < 0:
            return max_index - 1
        else:
            return index

    def sampling(self) -> list:
        mag_init = 1
        mag_min = mag_init / np.e
        tensor = []
        lattice = self.generate_lattice()
        mag = np.abs(np.mean(lattice.flatten()))
        if mag <= mag_min:
            tensor.append(lattice)
        energy_init = self.lattice_energy(tensor[0])

        while len(tensor) < self.sample_count:
            lattice = self.generate_lattice()
            mag = np.abs(np.mean(lattice.flatten()))
            if mag <= mag_min:
                current_lattice_energy = self.lattice_energy(lattice)
                if current_lattice_energy < energy_init:
                    tensor.append(lattice)
                elif np.random.random(1) < np.exp((current_lattice_energy - energy_init) / self.temp * spc.Boltzmann):
                    tensor.append(lattice)

        return tensor


samples = MonteCarlo(32, 1, 0.00001).sampling()

plt.imshow(samples[0], cmap=plt.cm.bwr)
plt.title("High Temperature State")
plt.show()
