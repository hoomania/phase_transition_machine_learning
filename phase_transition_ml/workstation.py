import sampling as smpl
import numpy as np

# lattice_profiles = ['square', 'square_ice', 'triangle', 'honeycomb']
# lattice_lengths = [10, 20, 30, 40, 50]

lattice_profiles = ['square']
lattice_lengths = [20]

for prf in lattice_profiles:
    for ln in lattice_lengths:
        sampling_profile = {
            'lattice_length': ln,
            'lattice_dim': 1,
            'lattice_profile': prf,
            'configs': 10,
            'temp_low': 0.4,
            'temp_high': 4.9,
            'temp_step': 0.1,
            'samples_per_temp': 10,
            'beta_inverse': True
        }
        smpl.Sampling(sampling_profile).take_sample()
