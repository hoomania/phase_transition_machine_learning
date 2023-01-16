import numpy as np
import matplotlib.pyplot as plt
from phyml.model import lattice as ltc, square, square_ice, triangle, honeycomb


class MonteCarlo:

    def __init__(self, lattice_length: int, lattice_dim: int = 1, lattice_profile: str = 'square', r45: bool = False):
        self.length = lattice_length
        self.lattice_dim = lattice_dim
        self.time_relaxation = self.relaxation_step_length(sample_count=100)

        lattice = ltc.Lattice(lattice_length, lattice_dim)
        if not r45:
            self.tensor = lattice.generate()
        else:
            self.tensor = lattice.generate_r45()

        if lattice_profile == 'square':
            self.lattice_model = square.Square(lattice_length, lattice_dim)
        if lattice_profile == 'square_ice':
            self.lattice_model = square_ice.SquareIce(lattice_length, lattice_dim)
        if lattice_profile == 'triangle':
            self.lattice_model = triangle.Triangle(lattice_length, lattice_dim)
        if lattice_profile == 'honeycomb':
            self.lattice_model = honeycomb.Honeycomb(lattice_length, lattice_dim)

    def check_condition_1d(self, beta: float):
        rnd_cell = self.lattice_model.random_cell_energy_1d(self.tensor)
        diff_energy = -2 * rnd_cell[1]
        if diff_energy < 0 or np.random.random() < np.exp(-1 * diff_energy * beta):
            self.tensor[rnd_cell[0]] = -1 * self.tensor[rnd_cell[0]]

    def check_condition_2d(self, beta: float):
        rnd_cell = self.lattice_model.random_cell_diff_energy_flip_2d(self.tensor)
        # diff_energy = -2 * rnd_cell[2]
        if rnd_cell[2] < 0 or np.random.random() < np.exp(-1 * rnd_cell[2] * beta):
            self.tensor[rnd_cell[0], rnd_cell[1]] = -1 * self.tensor[rnd_cell[0], rnd_cell[1]]

    def check_condition_r45_2d(self, beta: float):
        rnd_cell = self.lattice_model.random_cell_diff_energy_flip_r45_2d(self.tensor)
        # diff_energy = -2 * rnd_cell[2]
        if rnd_cell[2] < 0 or np.random.random() < np.exp(-1 * rnd_cell[2] * beta):
            self.tensor[rnd_cell[0], rnd_cell[1]] = -1 * self.tensor[rnd_cell[0], rnd_cell[1]]

    def sampling(self, sample_count: int, beta: float, beta_inverse: bool = False) -> list:
        if beta_inverse:
            beta = 1 / beta

        check_func = f"check_condition_{self.lattice_dim}d"

        for i in range(10 * self.length ** 3):
            getattr(self, check_func)(beta)

        tensor_list = []
        while len(tensor_list) < sample_count:
            for i in range(self.time_relaxation):
                getattr(self, check_func)(beta)

            if self.lattice_dim == 2:
                tensor_list.append(self.tensor.flatten())
            else:
                tensor_list.append(self.tensor)

        return tensor_list

    def sampling_r45(self, sample_count: int, beta: float, beta_inverse: bool = False) -> list:
        if beta_inverse:
            beta = 1 / beta

        check_func = f"check_condition_r45_{self.lattice_dim}d"

        for i in range(10 * self.length ** 3):
            getattr(self, check_func)(beta)

        tensor_list = []
        while len(tensor_list) < sample_count:
            for i in range(self.time_relaxation):
                getattr(self, check_func)(beta)

            if self.lattice_dim == 2:
                tensor_list.append(self.tensor.flatten())
            else:
                tensor_list.append(self.tensor)

        return tensor_list

    def relaxation_step_length(self, sample_count: int) -> int:
        step_list = []
        for _ in range(sample_count):
            steps = 0
            tensor = np.ones([self.length, self.length])
            mag_mean = np.mean(tensor)
            while mag_mean > 1 / np.e:
                x_rand = np.random.randint(self.length)
                y_rand = np.random.randint(self.length)
                tensor[x_rand, y_rand] = -1 * tensor[x_rand, y_rand]
                mag_mean = np.mean(tensor.flatten())
                steps += 1

            step_list.append(steps)

        return int(np.round(np.mean(step_list)))

    def relaxation_graph(self, samples_size: int):
        mean_mag_list = []
        tensor = np.ones([self.length, self.length])
        count = self.relaxation_step_length(samples_size) * 10
        for __ in range(count):
            x_rand = np.random.randint(0, self.length - 1)
            y_rand = np.random.randint(0, self.length - 1)
            tensor[x_rand, y_rand] = -1 * tensor[x_rand, y_rand]  # self.flip(tensor[x_rand][y_rand])
            mean_mag_list.append(np.mean(tensor.flatten()))

        plt.scatter([i for i in range(count)], mean_mag_list, s=0.7)
        x_e, y_e = [0, count], [np.e ** -1, np.e ** -1]
        x_0, y_0 = [0, count], [0, 0]
        plt.plot(x_e, y_e, x_0, y_0)
        plt.title(f"Square Lattice, {self.length ** 2} Particles | Relaxation Step: {count / 10}")
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
