import monte_carlo as mc
import pickle
import datetime
import numpy as np
import mysql.connector
import json


def connect_to_sql():
    db = mysql.connector.connect(user='admin', passwd='password', host='localhost', database='monte_carlo')
    my_cursor = db.cursor()

    query = "INSERT INTO samples(tensor, energy, mag, beta, temp) VALUES (%s,%s,%s,%s,%s)"
    json_test = json.dumps([1, -1, 1, 1])
    stds = [(json_test, 2, 4, 6, 1/6)]

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


read_samples()
