import matplotlib.pyplot as plt
import numpy as np

from plotting_functions import plot_time_to_recovery


trials = 10
speeds = np.arange(0.5, 8.5, step=0.5)
paths = ["data/success_rates/baseline1_logs/",
         #"data/success_rates/baseline2_logs/",
         "data/success_rates/our_logs/"]
means = np.zeros([len(paths), len(speeds)])
stds = np.zeros([len(paths), len(speeds)])

threshold = 0.5

for j in range(len(paths)):
    for k in range(len(speeds)):
        time_to_perch = []
        for i in range(trials):
            data = np.genfromtxt(paths[j] + f"Simulation_{speeds[k]:.1f}_{i:03}.csv", delimiter=',')
            t = data[:, 0]
            velocity = data[:, 4:7]
            collisions = data[:, -13:-1].any(axis=1)
            vel_norm = np.linalg.norm(velocity, axis=1)
            z = data[:, 3]

            # Find the index of the first collision
            collisions_index = (np.where(collisions)[0])
            
            # Find the index after the first collision where the velocity norm is below the threshold
            index = np.where(vel_norm[collisions_index[0]:] < threshold)[0]
            
            if z[-1] > 0:
                time_to_perch += [t[collisions_index[0] + index[0]] - t[collisions_index[0]]]
            else:
                time_to_perch += [-1]
    
        means[j, k] = np.mean(time_to_perch)
        stds[j, k] = np.std(time_to_perch)

plot_time_to_recovery(speeds, means, stds, N=2)