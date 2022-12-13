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
    mc_object = mc.MonteCarlo(20)

    sample_list = []
    print(f'Start:  {datetime.datetime.now()}')
    for temp in np.arange(0.5, 4.0, 0.1):
        sample_list.append(mc_object.sampling(100, temp, temp_input=True))

    print(f'  \nEnd:  {datetime.datetime.now()}')
    mc_object.energy_mean_graph(sample_list)

    pickle.dump(sample_list, open('../data/samples_mc.pkl', 'wb'))


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


def sample(length: int):
    print(f'Start:  {datetime.datetime.now()}')
    temp_low = 0.5
    temp_high = 4.9
    temp_step = 0.1
    temperatures = np.linspace(temp_high, temp_low, int((temp_high - temp_low) / temp_step) + 1)
    row = []
    temp_value = []
    for i in range(10):
        print(f'configure {i}:')
        mc_object = mc.MonteCarlo(length)
        for temp in temperatures:
            samples = mc_object.sampling(10, temp, beta_inverse=True)
            # [tensor_list, energy_list, mag_list, beta, 1/beta]
            for sample in samples:
                array = [i for i in sample]
                row.append(array)
                temp_value.append(temp)

    dataframe = pd.DataFrame(row)
    dataframe['Temp'] = temp_value
    dataframe.to_csv(fr"../data/data_L{length}.csv")
    print(f'\nEnd:  {datetime.datetime.now()}')


length_array = [40, 50, 60]
for len in length_array:
    sample(len)
