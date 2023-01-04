import json
import random
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as nnf
# import monte_carlo as mc
import matplotlib.pyplot as plt
import mysql.connector
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sbrn
import pandas as pd


class FCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layer: int = 20):
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


def dataset_maker():
    ising_data = pd.read_csv('../data/data_L20.csv')
    query = [i for i in ising_data.get('Temp').drop_duplicates()]

    class_left_train = []
    class_right_train = []
    class_left_test = []
    class_right_test = []
    # for i in [2.4000000000000004, 2.5, 2.6, 2.7, 4.7, 4.800000000000001, 4.9]:
    #     query.remove(i)

    for i in query:
        train = ising_data.query(f'Temp == {i}').head(2).drop(['Temp', 'Unnamed: 0'], axis=1)
        test = ising_data.query(f'Temp == {i}').tail(2).drop(['Temp', 'Unnamed: 0'], axis=1)

        if i < 2.5:
            target = [1, 0]
            for j in range(len(train.values) - 1):
                class_left_train.append([train.values[j], target])

            for q in range(len(test.values) - 1):
                class_left_test.append([test.values[q], target])
        else:
            target = [0, 1]
            for j in range(len(train.values) - 1):
                class_right_train.append([train.values[j], target])

            for q in range(len(test.values) - 1):
                class_right_test.append([test.values[q], target])

    data_train = class_left_train + class_right_train
    print(type(data_train))
    random.shuffle(data_train)
    data_test = class_left_test + class_right_test

    return data_train, data_test


def train_network(data_set, iterations):
    feature = data_set[0][0][0]
    train_set = data_set[0]
    test_set = data_set[1]

    n_samples = len(train_set)
    n_features = feature.shape[0]
    print(f'Samples: {n_samples} \nFeatures: {n_features}')

    hidden_layer_len = 100
    model = nn.Sequential(
        # nn.Flatten(),
        nn.Linear(n_features, hidden_layer_len),
        nn.ReLU(),
        nn.Linear(hidden_layer_len, 2),
        nn.Sigmoid(),
        # nn.Softmax(dim=0)
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    batches = np.array_split(train_set, 10)

    loss_values = []
    acc_values = []
    for epoch in range(iterations):
        for batch in batches:
            feature = torch.tensor(np.array([child[0] for child in batch]), dtype=torch.float32)
            target = torch.tensor(np.array([child[1] for child in batch]), dtype=torch.float32)

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(feature)

            # Compute and print loss
            loss = criterion(y_pred, target)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # if epoch % epoch_step_print == 0:
            #     print(f'epoch {epoch}: loss = {loss:.8f}')
        loss_values.append(loss.item())
        acc = check_test_set(model, test_set)
        acc_values.append(acc)
        print(f'Epoch {epoch + 1}/{iterations}\n loss: {loss.item():.5f}, acc: {acc:.5f}')

    # test_set = data_set[0]
    # as_train_set = []
    # for item in test_set:
    #     as_train_set.append([
    #         torch.tensor(item[0], dtype=torch.float32),
    #         torch.tensor(item[1], dtype=torch.float32)
    #     ])
    # test_set = as_train_set

    # test_set = test_set[1]
    cf_target = []
    cf_predict = []
    for t in range(0, len(test_set) - 1):
        with torch.no_grad():
            cf_predict.append(torch.IntTensor.item(np.round(model(test_set[t][0])[0])))
            cf_target.append(torch.IntTensor.item(test_set[t][1][0]))
            # print(f'model: {model(test_set[t][0]).tolist()}, target: {np.round(model(test_set[t][0]).tolist())}')

    confusion_graph(confusion_matrix(cf_target, cf_predict))
    loss_graph(loss_values)
    acc_graph(acc_values)
    print(classification_report(cf_target, cf_predict, zero_division=True))


def loss_graph(values):
    # plt.scatter([i for i in range(len(values))], values, s=0.7)
    plt.plot([i for i in range(len(values))], values)
    plt.title("Loss Diagram")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.show()


def acc_graph(values):
    # plt.scatter([i for i in range(len(values))], values, s=0.7)
    plt.plot([i for i in range(len(values))], values)
    plt.title("Accuracy Diagram")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Value")
    plt.show()


def confusion_graph(cf_matrix):
    fx = sbrn.heatmap(cf_matrix, annot=True, cmap='copper')

    # labels the title and x, y axis of plot
    fx.set_title('\nConfusion Matrix\n')
    fx.set_xlabel('Predicted Values')
    fx.set_ylabel('Actual Values ')

    # labels the boxes
    fx.xaxis.set_ticklabels(['0', '1'])
    fx.yaxis.set_ticklabels(['0', '1'])

    plt.show()


def check_test_set(model, test_set):
    cf_target = []
    cf_predict = []
    for t in range(0, len(test_set) - 1):
        with torch.no_grad():
            cf_predict.append(torch.IntTensor.item(np.round(model(test_set[t][0])[0])))
            cf_target.append(torch.IntTensor.item(test_set[t][1][0]))

    return accuracy_score(cf_target, cf_predict)


def train_network_develop(data_set, iterations):
    train_set = data_set[0]
    test_set = data_set[1]

    n_samples = len(train_set)
    n_features = train_set[0][0].shape[0]
    print(f'Samples: {n_samples} \nFeatures: {n_features}')

    hidden_layer_len = 100
    model = nn.Sequential(
        # nn.Flatten(),
        nn.Linear(n_features, hidden_layer_len),
        nn.ReLU(),
        nn.Linear(hidden_layer_len, 2),
        nn.Sigmoid(),
        # nn.Softmax(dim=0)
    )

    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    split_data = np.array_split(train_set, 10)
    batches = []
    for bunch in split_data:
        for item in bunch:
            # print(item[1])
            x = torch.tensor(item[0], dtype=torch.float32)
            y = torch.tensor(item[1], dtype=torch.float32)
            batches.append([x, y])
    # print(batches)
    loss_values = []
    acc_values = []
    for epoch in range(iterations):
        for batch in batches:
            feature = batch[0]
            target = batch[1]

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(feature)

            # Compute and print loss
            loss = criterion(y_pred, target)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # if epoch % epoch_step_print == 0:
            #     print(f'epoch {epoch}: loss = {loss:.8f}')
        # loss_values.append(loss.item())
        # acc = check_test_set(model, data_set[1])
        # acc_values.append(acc)
        acc = 0
        print(f'Epoch {epoch + 1}/{iterations}\n loss: {loss.item():.5f}, acc: {acc:.5f}')

    # test_set = train_set
    # as_train_set = []
    # for item in test_set:
    #     as_train_set.append([
    #         torch.tensor(item[0], dtype=torch.float32),
    #         torch.tensor(item[1], dtype=torch.float32)
    #     ])
    # test_set = as_train_set

    feature = []
    target = []
    for item in test_set:
        feature.append(torch.tensor(item[0], dtype=torch.float32))
        target.append(torch.tensor(item[1], dtype=torch.float32))

    cf_target = []
    cf_predict = []
    for t in range(0, len(test_set) - 1):
        with torch.no_grad():
            cf_predict.append(torch.IntTensor.item(np.round(model(feature[t])[0])))
            cf_target.append(torch.IntTensor.item(target[t][0]))

    confusion_graph(confusion_matrix(cf_target, cf_predict))
    loss_graph(loss_values)
    acc_graph(acc_values)
    print(classification_report(cf_target, cf_predict, zero_division=True))


data_set_ = dataset_maker()
# print(data_set_[0][0][0])
# print(type(data_set_[0][0][0]))
# train_network_develop(data_set_, iterations=10)

# test = [[1, 2], [2], [3], [4]]
# print(type(test[0]))
#
# X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
# Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
# print(X)
# print(Y)
#
# test_x = []
# test_y = []
# for i in range(1, 5):
#     test_x.append([i])
#     test_y.append([i*2])
# print(type(test_x))
# print(test_y)

# X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
# Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

