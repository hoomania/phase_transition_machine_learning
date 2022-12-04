import time

import numpy
import numpy as np
import scipy.constants as spc
import matplotlib.pyplot as plt
import pickle
import datetime


class MonteCarlo:

    def __init__(self, lattice_side_length: int):
        self.length = lattice_side_length
        first_sample_param = self.relaxation_step_length(100)
        self.step_length = first_sample_param[0]
        self.first_sample = first_sample_param[1]

    def convert_to_toric(self, matrix: numpy.ndarray) -> numpy.ndarray:
        length = self.length
        toric = np.zeros([length, length * 2])
        rows_index = [np.arange(0, length, 2), np.arange(1, length, 2)]

        for i in range(2):
            for row in rows_index[i]:
                for col_index in range(length):
                    toric[row][(2 * col_index) + i] = matrix[row][col_index]

        return toric

    def generate_lattice(self, length: int) -> numpy.ndarray:
        lattice = np.random.randint(0, 2, (length, length))
        lattice[lattice == 0] = -1
        return lattice

    def node_energy(self, lattice: np.ndarray, x: int, y: int, coupling_value: float) -> int:
        return coupling_value * lattice[x, y] * (
                lattice[self.boundary_condition(x + 1), y] +
                lattice[self.boundary_condition(x - 1), y] +
                lattice[x, self.boundary_condition(y + 1)] +
                lattice[x, self.boundary_condition(y - 1)])

    def node_energy_random(self, lattice: np.ndarray) -> int:
        i = np.random.randint(0, self.length - 1)
        j = np.random.randint(0, self.length - 1)
        return self.node_energy(lattice, i, j, -1)

    def lattice_energy(self, lattice: np.ndarray) -> int:
        energy = 0
        for i in range(self.length):
            for j in range(self.length):
                energy += self.node_energy(lattice, i, j, -1)

        return energy

    def boundary_condition(self, index: int) -> int:
        max_index = self.length - 1
        if index > max_index - 1:
            return 0
        if index < 0:
            return max_index - 1
        else:
            return index

    def sampling(self, sample_count: int, beta: float, temp_input: bool = False) -> list:
        tensor_list = []
        energy_list = []
        mag_list = []
        first_sample = self.first_sample
        energy_prior = self.node_energy_random(first_sample)

        if temp_input:
            beta = 1/beta

        while len(tensor_list) < sample_count:
            tensor = np.ones([self.length, self.length])
            for i in range(self.step_length):
                x_rand = np.random.randint(0, self.length - 1)
                y_rand = np.random.randint(0, self.length - 1)
                tensor[x_rand][y_rand] = self.flip(tensor[x_rand][y_rand])

            energy_current = self.node_energy_random(tensor)
            if energy_current < energy_prior:
                energy_prior = energy_current
                tensor_list.append(tensor)
                energy_list.append(energy_current)
                mag_list.append(np.mean(tensor.flatten()))
            elif np.random.random(1) < np.exp(
                    (energy_current - energy_prior) * beta * -1):
                tensor_list.append(tensor)
                energy_list.append(energy_current)
                mag_list.append(np.mean(tensor.flatten()))

        return [tensor_list, energy_list, mag_list, beta, 1/beta]

    def sample_without_relaxation(self, sample_count: int, beta: float) -> list:
        tensor_list = []
        energy_list = []
        mag_list = []
        first_sample = self.first_sample
        energy_prior = self.node_energy_random(first_sample)

        while len(tensor_list) < sample_count:
            tensor = self.generate_lattice(self.length)

            energy_current = self.node_energy_random(tensor)
            if energy_current < energy_prior:
                energy_prior = energy_current
                tensor_list.append(tensor)
                energy_list.append(energy_current)
                mag_list.append(np.mean(tensor.flatten()))
            elif np.random.random(1) < np.exp(
                    (energy_current - energy_prior) * beta * -1):
                tensor_list.append(tensor)
                energy_list.append(energy_current)
                mag_list.append(np.mean(tensor.flatten()))

        return [tensor_list, energy_list, mag_list, beta, 1 / beta]

    def flip(self, value: int) -> int:
        if value == 1:
            return -1
        else:
            return 1

    def relaxation_step_length(self, sample_count: int) -> list:
        step_list = []
        tensor_list = []
        for _ in range(sample_count):
            steps = 0
            tensor = np.ones([self.length, self.length])
            mag_mean = np.mean(tensor)
            while mag_mean > np.e ** -1:
                x_rand = np.random.randint(0, self.length - 1)
                y_rand = np.random.randint(0, self.length - 1)
                tensor[x_rand][y_rand] = self.flip(tensor[x_rand][y_rand])
                mag_mean = np.mean(tensor.flatten())
                steps += 1

            step_list.append(steps)
            tensor_list.append(tensor)

        first_sample = tensor_list[np.random.randint(0, len(step_list) - 1)]
        return [
            np.round(np.mean(step_list)).astype(int),
            first_sample
        ]

    def relaxation_graph(self, samples_size: int):
        mean_mag_list = []
        tensor = np.ones([self.length, self.length])
        count = self.relaxation_step_length(samples_size)[0] * 10
        for __ in range(count):
            x_rand = np.random.randint(0, self.length - 1)
            y_rand = np.random.randint(0, self.length - 1)
            tensor[x_rand][y_rand] = self.flip(tensor[x_rand][y_rand])
            mean_mag_list.append(np.mean(tensor.flatten()))

        plt.scatter([i for i in range(count)], mean_mag_list, s=0.7)
        x_e, y_e = [0, count], [np.e ** -1, np.e ** -1]
        x_0, y_0 = [0, count], [0, 0]
        plt.plot(x_e, y_e, x_0, y_0)
        plt.title(f"Square Lattice, {self.length ** 2} Particles | Relaxation Step: {count/10}")
        plt.xlabel('Steps')
        plt.ylabel('M (magnetization)')
        plt.show()

    def energy_mean_graph(self, samples: list):
        mean_energy_list = []
        for item in samples:
            frac_up = 0
            partition_function = 0

            for e in item[1]:
                power_value = np.e ** (-1 * item[3] * e)
                frac_up += e * power_value
                partition_function += power_value

            mean_energy_list.append(frac_up / partition_function)

        plt.scatter([item[3] for item in samples], mean_energy_list, s=0.7)
        plt.title(f"Mean Energy")
        plt.xlabel("beta")
        plt.ylabel("<E>")
        plt.show()

        plt.scatter([item[4] for item in samples], mean_energy_list, s=0.7)
        plt.title(f"Mean Energy")
        plt.xlabel("T")
        plt.ylabel("<E>")
        plt.show()


# arr = [0, 1, 2]
# pickle.dump(arr, open('test.pkl', 'wb'))
# load_file = pickle.load(open('./test.pkl', 'rb'))
# print(load_file)



# samples = MonteCarlo(1, 4, 1.0).sampling()
# print(samples)
# plt.imshow(samples[0], cmap=plt.cm.bwr)
# plt.title("High Temperature State")
# plt.show()
