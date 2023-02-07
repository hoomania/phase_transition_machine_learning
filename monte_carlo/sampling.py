from phyml.monte_carlo import monte_carlo as mc
import datetime
import numpy as np
import pandas as pd
from os.path import exists
from colorama import Fore


class Sampling:

    def __init__(self, profile: dict):
        self.lattice_length = profile['lattice_length']
        self.lattice_dim = profile['lattice_dim']
        self.lattice_profile = profile['lattice_profile']
        self.configs = profile['configs']
        self.temp_low = profile['temp_low']
        self.temp_high = profile['temp_high']
        self.temp_step = profile['temp_step']
        self.temp_step_auto = profile['temp_step_auto']
        self.temp_list = profile['temp_list']
        self.samples_per_temp = profile['samples_per_temp']
        self.beta_inverse = profile['beta_inverse']
        self.path_output = profile['path_output']

    def take_sample(self):
        ts_sampling = datetime.datetime.now()
        print(f'\n---- Sampling Start ----')

        if self.temp_step_auto:
            temperatures = np.linspace(self.temp_high, self.temp_low,
                                       int((self.temp_high - self.temp_low) / self.temp_step) + 1)
        else:
            temperatures = self.temp_list
        # temperatures = [0. for _ in range(int(np.ceil((self.temp_high - self.temp_low) / self.temp_step)) + 1)]
        # temp_high = self.temp_high
        # for i in range(len(temperatures)):
        #     temperatures[i] = temp_high
        #     temp_high -= self.temp_step

        row = []
        temp_value = []
        for i in range(self.configs):
            print(f'\n---- Configure {i} Started ----')
            ts_configs = datetime.datetime.now()
            mc_object = mc.MonteCarlo(self.lattice_length, self.lattice_dim, self.lattice_profile)
            for temp in temperatures:
                # ts_temp = datetime.datetime.now()
                samples = mc_object.sampling(self.samples_per_temp, temp, self.beta_inverse)
                # te_temp = datetime.datetime.now()
                # dt_temp = te_temp - ts_temp
                # if self.beta_inverse:
                #     print(f'config: {i}, temp: {temp:.2f}, #: {self.samples_per_temp}, time duration: {dt_temp}')
                # else:
                #     print(f'config: {i}, beta: {1 / temp:.2f}, #: {self.samples_per_temp}, time duration: {dt_temp}')

                for sample in samples:
                    array = [j for j in sample]
                    row.append(array)
                    temp_value.append(temp)

            te_configs = datetime.datetime.now()
            print(Fore.MAGENTA + f'\n#### Configure {i} ####')
            print(f'   Time Start:  {ts_configs}')
            print(f'     Time End:  {te_configs}')
            print(f'Time Duration:  {te_configs - ts_configs}')
            print(f'---------------------')
            print(Fore.RESET)

        print(f'\n#### Saving Data Is In Progress ####')
        dataframe = pd.DataFrame(row)
        dataframe['Temp'] = temp_value
        if exists(f"{self.path_output}/data_{self.lattice_profile}_L{self.lattice_length}.csv"):
            stamp = int(datetime.datetime.now().timestamp())
            file_name = f'data_{self.lattice_profile}_L{self.lattice_length}_{stamp}.csv'
        else:
            file_name = f'data_{self.lattice_profile}_L{self.lattice_length}.csv'

        dataframe.to_csv(fr"{self.path_output}/{file_name}")
        print(f'\n#### Saving Data Finished ####')

        te_sampling = datetime.datetime.now()
        print(Fore.GREEN + f'\n#### Sampling Profile ####')
        print(f'             Time Start: {ts_sampling}')
        print(f'               Time End: {te_sampling}')
        print(f'          Time Duration: {te_sampling - ts_sampling}')
        print(f'\n        Lattice Profile: {self.lattice_profile}')
        print(f'         Lattice Length: {self.lattice_length}')
        print(f'      Lattice Dimension: {self.lattice_dim}')
        print(f'                Configs: {self.configs}')
        print(f'        Temperature Low: {self.temp_low}')
        print(f'       Temperature High: {self.temp_high}')
        print(f'      Temperature Steps: {self.temp_step}')
        print(f'Samples Per Temperature: {self.samples_per_temp}')
        print(f'           Beta Inverse: {self.beta_inverse}')
        print(f'              File Name: {file_name}')
        print(f'--------------------------')

        print(Fore.RESET)

        print(f'\n---- Sampling End ----')

    def take_sample_r45(self):
        ts_sampling = datetime.datetime.now()
        print(f'\n---- Sampling Start ----')

        if self.temp_step_auto:
            temperatures = np.linspace(self.temp_high, self.temp_low,
                                       int((self.temp_high - self.temp_low) / self.temp_step) + 1)
        else:
            temperatures = self.temp_list
        # temperatures = [0. for _ in range(int(np.ceil((self.temp_high - self.temp_low) / self.temp_step)) + 1)]
        # temp_high = self.temp_high
        # for i in range(len(temperatures)):
        #     temperatures[i] = temp_high
        #     temp_high -= self.temp_step

        row = []
        temp_value = []
        for i in range(self.configs):
            print(f'\n---- Configure {i} Start ----')
            ts_configs = datetime.datetime.now()
            mc_object = mc.MonteCarlo(self.lattice_length, self.lattice_dim, self.lattice_profile, r45=True)
            for temp in temperatures:
                # ts_temp = datetime.datetime.now()
                samples = mc_object.sampling_r45(self.samples_per_temp, temp, self.beta_inverse)
                # te_temp = datetime.datetime.now()
                # dt_temp = te_temp - ts_temp
                # if self.beta_inverse:
                #     print(f'config: {i}, temp: {temp:.2f}, #: {self.samples_per_temp}, time duration: {dt_temp}')
                # else:
                #     print(f'config: {i}, beta: {1 / temp:.2f}, #: {self.samples_per_temp}, time duration: {dt_temp}')

                for sample in samples:
                    array = [j for j in sample]
                    row.append(array)
                    temp_value.append(temp)

            te_configs = datetime.datetime.now()
            print(Fore.MAGENTA + f'\n#### Configure {i} ####')
            print(f'   Time Start:  {ts_configs}')
            print(f'     Time End:  {te_configs}')
            print(f'Time Duration:  {te_configs - ts_configs}')
            print(f'---------------------')
            print(Fore.RESET)

        print(f'\n#### Saving Data Started ####')
        dataframe = pd.DataFrame(row)
        dataframe['Temp'] = temp_value
        if exists(f"../data/data_{self.lattice_profile}_L{self.lattice_length}_r45.csv"):
            stamp = int(datetime.datetime.now().timestamp())
            file_name = f'data_{self.lattice_profile}_L{self.lattice_length}_r45_{stamp}.csv'
        else:
            file_name = f'data_{self.lattice_profile}_L{self.lattice_length}_r45.csv'

        dataframe.to_csv(fr"../data/{file_name}")
        print(f'\n#### Saving Data Finished ####')

        te_sampling = datetime.datetime.now()
        print(Fore.GREEN + f'\n#### Sampling Profile ####')
        print(f'             Time Start: {ts_sampling}')
        print(f'               Time End: {te_sampling}')
        print(f'          Time Duration: {te_sampling - ts_sampling}')
        print(f'\n        Lattice Profile: {self.lattice_profile}')
        print(f'         Lattice Length: {self.lattice_length}')
        print(f'      Lattice Dimension: {self.lattice_dim}')
        print(f'                Configs: {self.configs}')
        print(f'        Temperature Low: {self.temp_low}')
        print(f'       Temperature High: {self.temp_high}')
        print(f'      Temperature Steps: {self.temp_step}')
        print(f'Samples Per Temperature: {self.samples_per_temp}')
        print(f'           Beta Inverse: {self.beta_inverse}')
        print(f'              File Name: {file_name}')
        print(f'--------------------------')

        print(Fore.RESET)

        print(f'\n---- Sampling End ----')
