import numpy as np

from plotting_functions import plot_success_rates


paths = ["data/success_rates/collision_agnostic.csv",
         "data/success_rates/acc_based.csv",
         "data/success_rates/ours.csv"]
raw_data = []

for path in paths:
    raw_data += [np.loadtxt(path, delimiter=',')]

raw_data = np.array(raw_data)

velocities = raw_data[0, 0, :]

rates = 100 * np.mean(raw_data[:, 1:, :], axis=1)
sigma = 100 * np.std(raw_data[:, 1:, :], axis=1)

plot_success_rates(velocities, rates, sigma, N=3)