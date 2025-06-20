import numpy as np

from plotting_functions import plot_success_rates

velocities = np.arange(1.0, 6.5, 0.5)
rates = np.array([np.zeros_like(velocities), 
                  np.zeros_like(velocities),
                  100 * np.ones_like(velocities)])

rates[0, 0] = 100
rates[1, :6] = 100

sigma = np.array([10 / velocities, 
                  10 / velocities, 
                  10 / velocities])

plot_success_rates(velocities, rates, sigma)