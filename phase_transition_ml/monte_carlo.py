import numpy
import numpy as np
import scipy.constants as spc
import matplotlib.pyplot as plt


def convert_to_toric(matrix: numpy.ndarray, length: int) -> numpy.ndarray:
    toric = np.zeros([length, length * 2])
    rows_index = [np.arange(0, length, 2), np.arange(1, length, 2)]

    for i in range(2):
        for row in rows_index[i]:
            for col_index in range(length):
                toric[row][(2 * col_index) + i] = matrix[row][col_index]

    return toric


def generate_lattice(length: int) -> numpy.ndarray:
    lattice = np.random.randint(0, 2, (length, length))
    lattice[lattice == 0] = - 1
    return lattice


def pauli_matrix(name: str) -> numpy.ndarray:
    if name == 'x':
        return np.reshape([0, 1, 1, 0], [2, 2])

    if name == 'y':
        return np.reshape([0, -1j, 1j, 0], [2, 2])

    if name == 'z':
        return np.reshape([1, 0, 0, -1], [2, 2])


def magnetization(lattice: np.ndarray):
    lattice_shape = np.shape(lattice)
    length = 1
    for i in range(len(lattice_shape)):
        length = lattice_shape[i] * length

    vector = np.reshape(lattice, [length])
    return np.sum(vector) / len(vector)


def node_energy(lattice: np.ndarray, x: int, y: int, coupling_value: float, length: int):
    return coupling_value * lattice[x, y] * (
            lattice[boundary_condition(x + 1, length), y] +
            lattice[boundary_condition(x - 1, length), y] +
            lattice[x, boundary_condition(y + 1, length)] +
            lattice[x, boundary_condition(y - 1, length)])


def lattice_energy(lattice: np.ndarray, length: int) -> float:
    energy = 0
    for i in range(length):
        for j in range(length):
            energy = + node_energy(lattice, i, j, -1, length)

    return energy


def boundary_condition(index: int, max_index: int) -> int:
    if index > max_index - 1:
        return 0
    if index < 0:
        return max_index - 1
    else:
        return index


def sampling(count: int, length: int, temperature: float) -> list:
    mag_init = 1
    mag_min = mag_init / np.e
    tensor = []
    lattice = generate_lattice(length)
    mag = np.abs(magnetization(lattice))
    if mag <= mag_min:
        tensor.append(lattice)
    energy_init = lattice_energy(tensor[0], length)

    while len(tensor) < count:
        lattice = generate_lattice(length)
        mag = np.abs(magnetization(lattice))
        if mag <= mag_min:
            current_lattice_energy = lattice_energy(lattice, length)
            if current_lattice_energy < energy_init:
                tensor.append(lattice)
            elif np.random.random(1) < np.exp((current_lattice_energy - energy_init) / temperature * spc.Boltzmann):
                tensor.append(lattice)

    return tensor


lattice = sampling(4, 32, 3.5)
plt.imshow(lattice[0], cmap=plt.cm.bwr)
plt.title("High Temperature State")
plt.show()

plt.imshow(convert_to_toric(lattice[0], 32), cmap=plt.cm.bwr)
plt.title("High Temperature State")
plt.show()
