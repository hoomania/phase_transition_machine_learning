import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import monte_carlo as mc
import matplotlib.pyplot as plt
import mysql.connector


class FCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(FCN, self).__init__()
        hidden_layer = 100
        self.fc1 = nn.Linear(input_dim, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        x = nnf.relu(self.fc1(x))
        return torch.relu(self.fc2(x))


def make_train_test_set():
    db = mysql.connector.connect(user='admin', passwd='password', host='localhost', database='monte_carlo')
    my_cursor = db.cursor()

    samples_per_side = 500
    samples_test = 100
    list_right = random.sample(range(3000), samples_per_side)
    query_in_right = ''
    for index in range(len(list_right) - 1):
        query_in_right += str(list_right[index]) + ', '
    query_in_right += str(list_right[len(list_right) - 1])

    query = f'select tensor from samples where id in ({query_in_right})'
    my_cursor.execute(query)
    result = my_cursor.fetchall()

    train_set = []
    for i in result:
        train_set.append([np.array(json.loads(i[0])).flatten(), [0, 1]])

    list_left = random.sample(range(3001, 13000), samples_per_side)
    query_in_left = ''
    for index in range(len(list_left) - 1):
        query_in_left += str(list_left[index]) + ', '
    query_in_left += str(list_left[len(list_left) - 1])

    query = f'select tensor from samples where id in ({query_in_left})'
    my_cursor.execute(query)
    result = my_cursor.fetchall()

    for i in result:
        train_set.append([np.array(json.loads(i[0])).flatten(), [1, 0]])

    random.shuffle(train_set)

    query_not_in = query_in_left + ', ' + query_in_right
    query = f'select tensor, beta from samples where id >= 1 and id <= 13000 and id not in ({query_not_in})'
    my_cursor.execute(query)
    result = my_cursor.fetchall()

    test_set_random = random.sample(range(1, 12000), samples_test)
    test_set = []
    for i in test_set_random:
        if 0.28 <= result[i][1] <= 0.4:
            target = [0, 1]
        else:
            target = [1, 0]
        test_set.append([np.array(json.loads(result[i][0])).flatten(), target])

    return [train_set, test_set]


def train_network():
    data_set = make_train_test_set()
    train_set = data_set[0]
    test_set = data_set[1]
    X_list = []
    Y_list = []
    for i in range(len(train_set)):
        X_list.append(train_set[i][0])
        Y_list.append(train_set[i][1])

    print(f'samples are ready!')

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    n_samples, n_features = X.shape
    print(f'Samples: {n_samples}, \nFeatures: {n_features}')

    # X_test = torch.tensor(test_set[0][0], dtype=torch.float32)

    input_size = n_features
    output_size = 2

    model = FCN(input_size, output_size)

    # print(f'prediction before training:  {model(X_test)}')

    learning_rate = 0.1
    iterations = 3000

    # loss = nn.CrossEntropyLoss()
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(iterations):
        # forward
        y_pred = model(X)

        # loss
        lss = loss(Y, y_pred)

        # gradient
        lss.backward()

        # update weights
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 25 == 0:
            print(f'epoch {epoch}: loss = {lss:.8f}')

    for t in range(0, len(train_set) - 1):
        X_test = torch.tensor(train_set[t][0], dtype=torch.float32)
        # print(f'\nprediction after training: {model(X_test)} :: target is {test_set[t][1]}')
        print(f'\nprediction after training: {model(X_test)} :: target is {train_set[t][1][0]}')
    # print(f'prediction after training:  {model(X_test)} :: target is {test_set[0][1]}')


train_network()
