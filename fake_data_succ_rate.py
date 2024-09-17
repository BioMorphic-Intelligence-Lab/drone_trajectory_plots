import numpy as np

from plotting_functions import plot_success_rates

velocities = np.arange(1.0, 6.5, 0.5)
rates = np.array([10 * 1.0 / velocities, 
                  50 * 1.0 / velocities, 
                  100 * 1.0 / velocities])
sigma = np.array([10 / velocities, 
                  10 / velocities, 
                  10 / velocities])

plot_success_rates(velocities, rates, sigma)