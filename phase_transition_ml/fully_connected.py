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
    def __init__(self, input_dim: int, output_dim: int, hidden_layer: int = 100):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)

    # def forward(self, x):
    #     x = torch.relu(self.fc1(x))
    #     return torch.sigmoid(self.fc2(x))


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

    # train_set = data_set[0]
    # test_set = data_set[1]
    data_set = []
    for d in [train_set, test_set]:
        x_list = []
        y_list = []
        for i in range(len(d) - 1):
            x_list.append(d[i][0])
            y_list.append(d[i][1])

        # print(f'Samples are ready!')
        data_set.append([torch.tensor(np.array(x_list), dtype=torch.float32),
                         torch.tensor(np.array(y_list), dtype=torch.float32)])
        # X = torch.tensor(np.array(X_list), dtype=torch.float32)
        # Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    return data_set
    # return [train_set, test_set]


def train_network(data_set, iterations):
    # data_set = make_train_test_set()
    # train_set = data_set[0]
    # test_set = data_set[1]
    # X_list = []
    # Y_list = []
    # for i in range(len(train_set)):
    #     X_list.append(train_set[i][0])
    #     Y_list.append(train_set[i][1])
    #
    # print(f'Samples are ready!')

    # X = torch.tensor(np.array(X_list), dtype=torch.float32)
    # Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    feature = data_set[0][0]
    target = data_set[0][1]

    n_samples, n_features = feature.shape
    print(f'Samples: {n_samples}, \nFeatures: {n_features}')

    # X_test = torch.tensor(test_set[0][0], dtype=torch.float32)

    input_size = n_features
    output_size = 2

    model = FCN(input_size, output_size, hidden_layer=100)

    # print(f'prediction before training:  {model(X_test)}')

    learning_rate = 0.01

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(iterations):
        # forward
        y_pred = model(feature)

        # loss
        lss = loss(y_pred, target)

        # gradient
        lss.backward()

        # update weights
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 200 == 0:
            print(f'epoch {epoch}: loss = {lss:.8f}')

    acc = 0
    test_set = data_set[1]
    for t in range(0, len(test_set[0]) - 1):
        # x_test = torch.tensor(test_set[0][t], dtype=torch.float32)
        with torch.no_grad():
            predict = model(test_set[0][t]).tolist()
            acc += np.inner(predict, test_set[1][t])
            # print(f'\nprediction for test sample {t}: {predict} :: target is {test_set[1][t]}')
    print(f'Mean Accuracy: {acc / len(test_set[0])}')


def train_network_online_sample(data_set, iterations):
    # data_set = make_train_test_set()
    train_set = data_set[0]
    test_set = data_set[1]
    X_list = []
    Y_list = []
    for i in range(len(train_set)):
        X_list.append(train_set[i][0])
        Y_list.append(train_set[i][1])

    print(f'Samples are ready!')

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    n_samples, n_features = X.shape
    print(f'Samples: {n_samples}, \nFeatures: {n_features}')

    # X_test = torch.tensor(test_set[0][0], dtype=torch.float32)

    input_size = n_features
    output_size = 2

    # f1 = nn.Flatten()
    l1 = nn.Linear(32 * 32, 20)
    r1 = nn.ReLU()
    # l2 = nn.Linear(20, 100)
    # r2 = nn.ReLU()
    l3 = nn.Linear(20, 2)
    s3 = nn.Sigmoid()
    # s3 = nn.Softmax()
    model = nn.Sequential(
        # f1,
        l1,
        r1,
        # l2,
        # r2,
        l3,
        s3,
    )

    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # iterations = 5000

    for epoch in range(iterations):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(X)

        # Compute and print loss
        loss = criterion(y_pred, Y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()

        if epoch % 200 == 0:
            print(f'epoch {epoch}: loss = {loss:.8f}')

    acc = 0
    for t in range(0, len(test_set) - 1):
        X_test = torch.tensor(test_set[t][0], dtype=torch.float32)
        with torch.no_grad():
            predict = model(X_test).tolist()
            acc += np.inner(predict, test_set[t][1])
            print(f'\nprediction for test sample {t}: {predict} :: target is {test_set[t][1]}')
    print(f'Mean Accuracy: {acc / len(test_set)}')


data_set_ = make_train_test_set()
train_network(data_set_, iterations=500)
print('\n----------------\n')
# train_network_online_sample(data_set_, 2000)
