import numpy
import numpy as np
import matplotlib.pyplot as plt


class MonteCarlo:

    def __init__(self, lattice_side_length: int):
        self.length = lattice_side_length
        first_sample_param = self.relaxation_step_length(100)
        self.step_length = first_sample_param
        self.lattice = self.generate_lattice()
        # self.first_sample = first_sample_param[1]

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
        lattice = np.random.randint(2, size=(self.length, self.length))
        return 2 * lattice - 1

    def node_energy(self, lattice: np.ndarray, x: int, y: int, coupling_value: float) -> int:
        xm, xp, ym, yp = (x - 1) % self.length, (x + 1) % self.length, (y - 1) % self.length, (y + 1) % self.length
        return -1 * coupling_value * lattice[x, y] * (
                lattice[xp, y] +
                lattice[xm, y] +
                lattice[x, yp] +
                lattice[x, ym])

    def random_node_energy(self, lattice: np.ndarray) -> tuple:
        i = np.random.randint(self.length)
        j = np.random.randint(self.length)
        return self.node_energy(lattice, i, j, 1), i, j

    def lattice_energy(self, lattice: np.ndarray) -> int:
        energy = 0
        for i in range(self.length):
            for j in range(self.length):
                energy += self.node_energy(lattice, i, j, 1)

        return energy

    def sampling(self, sample_count: int, beta: float, beta_inverse: bool = False) -> list:
        if beta_inverse:
            beta = 1 / beta

        tensor_list = []
        tensor = self.lattice
        for i in range(10 * self.length ** 3):
            rnd_node = self.random_node_energy(tensor)
            diff_energy = -2 * rnd_node[0]
            if diff_energy < 0 or np.random.random() < np.exp(-1 * diff_energy * beta):
                tensor[rnd_node[1], rnd_node[2]] = -1 * tensor[rnd_node[1], rnd_node[2]]

        while len(tensor_list) < sample_count:
            for i in range(self.step_length):
                rnd_node = self.random_node_energy(tensor)
                diff_energy = -2 * rnd_node[0]
                if diff_energy < 0 or np.random.random() < np.exp(-1 * diff_energy * beta):
                    tensor[rnd_node[1], rnd_node[2]] = -1 * tensor[rnd_node[1], rnd_node[2]]

            if sample_count == len(tensor_list) + 1:
                if beta_inverse:
                    print(f'temp: {1/beta:.2f}, #: {sample_count}, completed!')
                else:
                    print(f'beta: {beta:.2f}, #: {sample_count}%, completed!')

            tensor_list.append(tensor.flatten())

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
