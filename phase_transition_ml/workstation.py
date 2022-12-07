import monte_carlo as mc
import pickle
import datetime
import numpy as np
import mysql.connector
import json
import matplotlib.pyplot as plt
import pandas as pd


def connect_to_sql():
    db = mysql.connector.connect(user='admin', passwd='password', host='localhost', database='monte_carlo')
    my_cursor = db.cursor()

    query = "INSERT INTO samples(tensor, energy, mag, beta, temp) VALUES (%s,%s,%s,%s,%s)"
    json_test = json.dumps([1, -1, 1, 1])
    stds = [(json_test, 2, 4, 6, 1 / 6)]

    my_cursor.executemany(query, stds)
    db.commit()
    print(my_cursor.rowcount, "records are inserted.")
    query = "select * from samples"
    my_cursor.execute(query)
    for tab in my_cursor:
        print(tab)


def mc_sampling():
    mc_object = mc.MonteCarlo(32)

    sample_0 = []
    sample_1 = []
    print('Start: ', datetime.datetime.now())
    for b in np.arange(0.28, 10.08, 0.05):
        sample_0.append(mc_object.sample_without_relaxation(1000, b))
        sample_1.append(mc_object.sampling(1000, b))

    print('  End: ', datetime.datetime.now())
    mc_object.energy_mean_graph(sample_0)
    mc_object.energy_mean_graph(sample_1)

    pickle.dump(sample_0, open('../data/samples_random_generator.pkl', 'wb'))
    pickle.dump(sample_1, open('../data/samples_mc.pkl', 'wb'))
    # load_file = pickle.load(open('./test.pkl', 'rb'))
    # print(load_file)


def read_samples():
    db = mysql.connector.connect(user='admin', passwd='password', host='localhost', database='monte_carlo')
    my_cursor = db.cursor()
    query = "INSERT INTO samples(tensor, energy, mag, beta, temp) VALUES (%s,%s,%s,%s,%s)"

    for i in range(166, 196):
        load_file = pickle.load(open(f'../data/samples_mc_{i}.pkl', 'rb'))
        values = []

        for index in range(0, len(load_file[0])):
            values.append((json.dumps(load_file[0][index].tolist()), float(load_file[1][index]),
                           float(load_file[2][index]), float(load_file[3]), float(load_file[4])))

        my_cursor.executemany(query, values)
        db.commit()
        print(f'{i}: ', my_cursor.rowcount, "records are inserted.")


# ising_data = pd.read_csv('../data/ising_model.csv')
# query = [i for i in ising_data.get('Temp').drop_duplicates()]
#
# class_left_train = []
# class_right_train = []
# class_left_test = []
# class_right_test = []
# for i in [2.4000000000000004, 2.5, 2.6, 2.7, 4.7, 4.800000000000001, 4.9]:
#     query.remove(i)
#
# # print(len(ising_data.query(f'Temp > 2.5')))
# # print(len(ising_data.query(f'Temp < 2.5')))
# count_left = 0
# count_right = 0
# for i in query:
#     if i < 2.5:
#         target = [0, 1]
#     else:
#         target = [1, 0]
#     train = ising_data.query(f'Temp == {i}').head(80).drop(['Temp', 'Unnamed: 0'], axis=1)
#     test = ising_data.query(f'Temp == {i}').tail(20).drop(['Temp', 'Unnamed: 0'], axis=1)
#     for j in range(len(train.values) - 1):
#         class_left_train.append([train.values[j], target])
#     for q in range(len(test.values) - 1):
#         class_left_test.append([test.values[q], target])
# #
# # print(train[0])
# print(len(class_left_train[0][0]))
# print(query)
# head = ising_data.head()
#
# tensor = ising_data.drop(['Temp', 'Unnamed: 0'], axis=1)

# import torch
# # import numpy as np
# from torch.utils.data import TensorDataset, DataLoader
#
# my_x = [np.array([[1.0, 2], [3, 4]]), np.array([[5., 6], [7, 8]])]  # a list of numpy arrays
# my_y = [np.array([4.]), np.array([2.])]  # another list of numpy arrays (targets)
#
# tensor_x = torch.Tensor(my_x)  # transform to torch tensor
# tensor_y = torch.Tensor(my_y)
#
# my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
# my_dataloader = DataLoader(my_dataset)  # create your dataloader
# print(my_dataset)
# print(my_dataloader)
