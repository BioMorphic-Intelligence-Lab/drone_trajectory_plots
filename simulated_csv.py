import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from plotting_functions import plot_trajectory, animate_trajectory

#path = "./colliding_drone_plots/Simulation.csv"
path = "data/Simulation000.csv"
data_raw = np.genfromtxt(path, delimiter=',')

t = data_raw[:, 0]
p_gt = data_raw[:, 1:4].T
v_gt = data_raw[:, 4:7].T
r_gt = np.array([Rotation.from_euler(seq="xyz", angles=data_raw[i, 7:10]) for i in range(len(data_raw))])
omega_gt = np.rad2deg(data_raw[:, 10:13].T)
f_gt = data_raw[:, 13:17].T

p_est = data_raw[:, 17:20].T
v_est = data_raw[:, 20:23].T
r_est = np.array([Rotation.from_euler(seq="xyz", angles=data_raw[i, 23:26]) for i in range(len(data_raw))])
omega_est = np.rad2deg(data_raw[:, 26:29].T)

p_des = data_raw[:, 29:32].T
panic = data_raw[:, 32]
radioCmd = data_raw[:, 33:37].T

contacts = data_raw[:, 37:49].T

fig = plot_trajectory(t, p_gt, v_gt, r_gt, omega_gt, contacts,
                np.array([0, 4.25 + 0.5 * 0.22, 2]), np.array([0.5, 0.5, 0.5]), Rotation.identity(),
                p_des)
fig.savefig("plot.png")

"""ani = animate_trajectory(t, p_gt, v_gt, r_gt, omega_gt, contacts,
                np.array([0, 4.25 + 0.5 * 0.22, 2]), np.array([0.5, 0.5, 0.5]), Rotation.identity(),
                p_des)
ani.save(filename="animation.mp4", writer="ffmpeg",
         dpi=200, fps=30, codec="libx264", extra_args=["-preset", "fast"])"""
