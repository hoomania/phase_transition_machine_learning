import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import regularizers
from keras.utils import to_categorical


class MachineLearning:

    def __init__(self, profile: dict, fec: bool = False):
        self.data_train = pd.read_csv(profile['path']['train'])
        self.data_train.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)

        self.data_test = pd.read_csv(profile['path']['test'])
        self.data_test.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)

        self.length = int(np.sqrt(self.data_train.shape[1]))
        self.n_sites = self.length ** 2
        if fec:
            self.n_sites = int((self.length ** 2) / 2)

        self.configs = profile['configs']  # 10
        self.sample_per_temp = profile['samples_per_temp']  # 10
        self.temp_high = profile['temp_high']  # 4.9
        self.temp_low = profile['temp_low']  # 0.5
        self.temp_step = profile['temp_step']  # 0.1
        self.n_temp = int((self.temp_high - self.temp_low) / self.temp_step) + 1
        self.temperatures = np.linspace(self.temp_high, self.temp_low, self.n_temp)
        self.chi_config = np.zeros((self.n_temp, self.configs))

    def mag_chi_temp_dataframe(self, data_set) -> pd.DataFrame:
        data = data_set.drop(['Temp', 'Index'], axis=1)
        m = data.sum(axis=1) / self.n_sites
        m2 = ((data * data).sum(axis=1)) / self.n_sites
        temp = data_set['Temp']
        chi = (m2 - m ** 2) / temp

        return pd.DataFrame({'m': m,
                             'chi': chi,
                             'Temp': temp})

    def calc_critical_temp(self, data_set) -> tuple:
        mag_chi_temp = self.mag_chi_temp_dataframe(data_set)

        for j in range(self.configs):
            for i in range(self.n_temp):
                self.chi_config[i, j] = np.average(
                    mag_chi_temp['chi'][
                    j * self.sample_per_temp * self.n_temp + i * self.sample_per_temp:j * self.sample_per_temp * self.n_temp + (
                            i + 1) * self.sample_per_temp - 1])

        tcs = self.temperatures[np.argmax(self.chi_config, axis=0)]
        tc = np.average(tcs)
        return tcs, tc

    def compile_model(self, dense: int = 100, learning_rate: float = 0.01, print_diagram: bool = False):
        tc = self.calc_critical_temp(self.data_train)[1]
        self.data_train['Phase'] = (self.data_train['Temp'] < tc).astype(int)
        x = self.data_train.drop(['Index', 'Temp', 'Phase'], axis=1)
        y = self.data_train['Phase']

        model = Sequential()
        model.add(Dense(dense, activation='relu', kernel_regularizer=regularizers.L1(learning_rate),
                        input_dim=self.length * self.length))
        model.add(Dense(2, activation='sigmoid'))

        # Complie the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        x_train_fit = model.fit(
            x,
            to_categorical(y),
            epochs=10,
        )

        # Plot the training performance
        if print_diagram:
            plt.figure(figsize=(12, 4))
            plt.suptitle('Training Performance')

            plt.subplot(121)
            plt.plot(x_train_fit.epoch, x_train_fit.history['accuracy'])
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')

            plt.subplot(122)
            plt.plot(x_train_fit.epoch, x_train_fit.history['loss'])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            plt.show()

        return model

    def evaluate_model(self, model):
        temp = self.data_test['Temp']
        mag = self.mag_chi_temp_dataframe(self.data_test)
        tc = self.calc_critical_temp(self.data_test)[1]

        x_test = self.data_test.drop(['Index', 'Temp'], axis=1)
        y_test = (self.data_test['Temp'] < tc).astype(int)
        print(x_test)
        print(y_test)

        # Evaluate the model
        model.evaluate(
            x_test,
            to_categorical(y_test)
        )

        y_pred = model.predict(x_test)
        plt.figure(figsize=(16, 6))

        plt.suptitle('Scatter plot of output weights')

        plt.subplot(121)
        plt.scatter(temp, y_pred[:, 0], label='W_0', s=0.7)
        plt.scatter(temp, y_pred[:, 1], label='W_1', s=0.7)
        plt.xlabel('Temp.')
        plt.legend()

        plt.subplot(122)
        plt.scatter(mag['m'], y_pred[:, 0], label='W0', s=0.7)
        plt.scatter(mag['m'], y_pred[:, 1], label='W1', s=0.7)
        plt.xlabel('Magnetization')
        plt.legend()
        plt.show()

        fig, ax1 = plt.subplots()

        ax1.scatter(temp, np.abs(mag['m']), color='m', label='|m|', s=0.7)
        plt.legend()

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.scatter(temp, y_pred[:, 0], color='b', label='W0', s=0.7)
        ax2.scatter(temp, y_pred[:, 1], color='g', label='W1', s=0.7)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()
        plt.xlabel('Temp.')
        plt.show()

    def phase_transition_diagram(self, model):
        # Sort test data w.r.t. temperature (With Respect To)
        self.data_test.sort_values(by='Temp', ascending=False, inplace=True)
        tc = self.calc_critical_temp(self.data_test)[1]
        x_test = self.data_test.drop(['Temp', 'Index'], axis=1)
        y_pred = model.predict(x_test)

        n_values = self.configs * self.sample_per_temp

        y_pred_comb = np.zeros((self.n_temp, 2))
        for i in range(self.n_temp):
            y_pred_comb[i, 0] = np.average(y_pred[i * n_values:(i + 1) * n_values - 1, 0])
            y_pred_comb[i, 1] = np.average(y_pred[i * n_values:(i + 1) * n_values - 1, 1])

        plt.figure(figsize=(18, 6))
        plt.suptitle('Average Output weights vs Temperature')

        plt.subplot(121)
        plt.plot(self.temperatures, y_pred_comb[:, 0], '.-', label='W0')
        plt.plot(self.temperatures, y_pred_comb[:, 1], '.-', label='W1')
        plt.vlines(x=2.26, ymin=0, ymax=1, label='T=2.26', colors='green')
        plt.vlines(x=tc, ymin=0, ymax=1, label=f'T={tc:.4}', colors='orange')
        plt.xlabel('Temp.')
        plt.legend()

        plt.subplot(122)
        plt.xlim((tc - 0.5, tc + 0.5))
        plt.plot(self.temperatures, y_pred_comb[:, 0], '.-', label='<W0>')
        plt.plot(self.temperatures, y_pred_comb[:, 1], '.-', label='<W1>')
        plt.vlines(x=2.26, ymin=0, ymax=1, label='T=2.26', colors='green')
        plt.vlines(x=tc, ymin=0, ymax=1, label=f'T={tc:.4}', colors='orange')
        plt.xlabel('Temp.')
        plt.legend()
        plt.show()
