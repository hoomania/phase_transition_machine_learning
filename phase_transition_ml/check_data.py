import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


class CheckData:

    def __init__(self, path: str, profile: dict, fec: bool = False):
        self.data = pd.read_csv(path)
        self.data.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)

        self.length = int(np.sqrt(self.data.shape[1]))
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

    def mag_chi_temp_dataframe(self) -> pd.DataFrame:
        data = self.data.drop(['Temp', 'Index'], axis=1)
        m = data.sum(axis=1) / self.n_sites
        m2 = ((data * data).sum(axis=1)) / self.n_sites
        temp = self.data['Temp']
        chi = (m2 - m ** 2) / temp

        return pd.DataFrame({'m': m,
                             'chi': chi,
                             'Temp': temp})

    def calc_critical_temp(self) -> tuple:
        mag_chi_temp = self.mag_chi_temp_dataframe()

        for j in range(self.configs):
            for i in range(self.n_temp):
                self.chi_config[i, j] = np.average(
                    mag_chi_temp['chi'][
                    j * self.sample_per_temp * self.n_temp + i * self.sample_per_temp:j * self.sample_per_temp * self.n_temp + (
                            i + 1) * self.sample_per_temp - 1])

        tcs = self.temperatures[np.argmax(self.chi_config, axis=0)]
        tc = np.average(tcs)
        return tcs, tc

    def magnetic_susceptibility(self):
        data = self.data.drop(['Temp', 'Index'], axis=1)
        m = data.sum(axis=1) / self.n_sites
        m2 = ((data * data).sum(axis=1)) / self.n_sites
        temp = self.data['Temp']
        chi = (m2 - m ** 2) / temp

        plt.figure(figsize=(12, 6))
        plt.title('Magnetic Susceptibility')
        plt.xlabel('Temp')
        plt.xlabel('Susceptibility')
        plt.scatter(temp, chi, s=0.7)
        plt.show()

    def magnetization(self):
        data = self.data.drop(['Temp', 'Index'], axis=1)
        m = data.sum(axis=1) / self.n_sites
        temp = self.data['Temp']

        plt.figure(figsize=(12, 6))
        plt.title('Magnetization')
        plt.xlabel('Temp')
        plt.ylabel('M')
        plt.scatter(temp, m, s=0.7)
        plt.show()

    def critical_temperature(self):
        mag_chi_temp = self.mag_chi_temp_dataframe()
        for j in range(self.configs):
            for i in range(self.n_temp):
                self.chi_config[i, j] = np.average(
                    mag_chi_temp['chi'][
                    j * self.sample_per_temp * self.n_temp + i * self.sample_per_temp:j * self.sample_per_temp * self.n_temp + (
                            i + 1) * self.sample_per_temp - 1])

        tcs = self.temperatures[np.argmax(self.chi_config, axis=0)]
        tc = np.average(tcs)

        plt.figure()
        plt.title(f'Average Critical Temperature <Tc> = {tc:.4}')
        plt.ylabel('Susceptibility')
        plt.xlabel('Temp')

        for i in range(self.configs):
            plt.scatter(self.temperatures, self.chi_config[:, i], s=0.8)

        plt.show()

    def avg_mag_sus(self):
        mag_chi_temp = self.mag_chi_temp_dataframe()

        mag_chi_temp.sort_values(by='Temp', ascending=False, inplace=True)

        n_values = self.configs * self.sample_per_temp
        chi_avg = np.zeros(self.n_temp)
        m_avg = np.zeros(self.n_temp)
        for i in range(self.n_temp):
            chi_avg[i] = np.average(mag_chi_temp['chi'][i * n_values:(i + 1) * n_values - 1])
            m_avg[i] = np.average(abs(mag_chi_temp['m'][i * n_values:(i + 1) * n_values - 1]))

        fig, ax1 = plt.subplots()

        plt.title('Avg. Magnetization and Susceptibility')
        plt.scatter(self.temperatures, m_avg, color='b', s=0.8)
        plt.ylabel('<|m|>')
        plt.xlabel('Temperature')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        plt.ylabel('<chi>')
        plt.scatter(self.temperatures, chi_avg, color='r', s=0.8)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    def lattice_config(self):
        tc = self.calc_critical_temp()

        diff = [0 for i in range(len(tc[0]))]
        for i in range(len(tc[0])):
            diff[i] = np.absolute(tc[1] - tc[0][i])

        tc = tc[0][np.argmin(diff)]

        lattice_t_gt_tc = self.data.loc[self.data['Temp'] == 4.0].sample().drop(['Temp', 'Index'],
                                                                                axis=1).values.reshape(
            (self.length, self.length))
        lattice_t_eq_tc = self.data.loc[self.data['Temp'] == tc].sample().drop(['Temp', 'Index'],
                                                                               axis=1).values.reshape(
            (self.length, self.length))
        lattice_t_lt_tc = self.data.loc[self.data['Temp'] == 1.0].sample().drop(['Temp', 'Index'],
                                                                                axis=1).values.reshape(
            (self.length, self.length))

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        c = mpl.colors.ListedColormap(['#cecece', 'white', 'black'])
        n = mpl.colors.Normalize(vmin=-1, vmax=1)

        axes[0].matshow(lattice_t_gt_tc, cmap=c, norm=n)
        axes[0].set_title("T = 4.0 > Tc")
        axes[1].matshow(lattice_t_eq_tc, cmap=c, norm=n)
        axes[1].set_title(f"T = Tc ~ {tc}")
        axes[2].matshow(lattice_t_lt_tc, cmap=c, norm=n)
        axes[2].set_title("T = 1.0 < Tc")
        for i in range(3):
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        plt.show()

    def configs_brief_view(self):
        config_len = self.n_temp * self.sample_per_temp  # 450

        if self.configs % 5 != 0:
            cnt = self.configs + 5 - (self.configs % 5)
        else:
            cnt = self.configs

        configs = [0 for _ in range(cnt)]
        for i in range(cnt):
            arr = self.data.query(f'Index >= {i * config_len} and Index < {(i + 1) * config_len}').drop(
                ['Temp', 'Index'], axis=1)
            if len(arr) != 0:
                arr = np.array(arr)
                arr = arr[arr != 0]
                configs[i] = arr.reshape(config_len, self.n_sites)
            else:
                configs[i] = self.data.query(f'Index >= {i * config_len} and Index < {(i + 1) * config_len}').drop(
                    ['Temp', 'Index'], axis=1)

        fig, axes = plt.subplots(nrows=int(cnt / 5), ncols=5, figsize=(12, 4))
        fig.tight_layout(pad=2.0)
        for i in range(int(cnt / 5)):
            for j in range(5):
                if int(cnt / 5) != 1:
                    axes[i, j].matshow(configs[(i * 5) + j])
                    axes[i, j].set_title(f"\nConfig: {(i * 5) + j}")
                else:
                    axes[j].matshow(configs[(i * 5) + j])
                    axes[j].set_title(f"\nConfig: {(i * 5) + j}")

        for i in range(int(cnt / 5)):
            for j in range(5):
                if int(cnt / 5) != 1:
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                else:
                    axes[j].set_xticks([])
                    axes[j].set_yticks([])

        plt.show()
