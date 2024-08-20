import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from plotting_functions import plot_trajectory, animate_trajectory

paths = [f"data/Simulation{i:03d}.csv" for i in range(100)]

t = [[] for i in range(len(paths))]
p_gt = [[] for i in range(len(paths))]
v_gt = [[] for i in range(len(paths))]
r_gt = [[] for i in range(len(paths))]
omega_gt = [[] for i in range(len(paths))]
f_gt = [[] for i in range(len(paths))]

p_est = [[] for i in range(len(paths))]
v_est = [[] for i in range(len(paths))]
r_est = [[] for i in range(len(paths))]
omega_est = [[] for i in range(len(paths))]

p_des = [[] for i in range(len(paths))]
panic = [[] for i in range(len(paths))]
radioCmd = [[] for i in range(len(paths))]

contacts = [[] for i in range(len(paths))]

for i in range(len(paths)):
    data_raw = np.genfromtxt(paths[i], delimiter=',')

    t[i] = data_raw[:, 0]
    p_gt[i] = data_raw[:, 1:4].T
    v_gt[i] = data_raw[:, 4:7].T
    r_gt[i] = np.array([Rotation.from_euler(seq="xyz", angles=data_raw[i, 7:10]) for i in range(len(data_raw))])
    omega_gt[i] = np.rad2deg(data_raw[:, 10:13].T)
    f_gt[i] = data_raw[:, 13:17].T

    p_est[i] = data_raw[:, 17:20].T
    v_est[i] = data_raw[:, 20:23].T
    r_est[i] = np.array([Rotation.from_euler(seq="xyz", angles=data_raw[i, 23:26]) for i in range(len(data_raw))])
    omega_est[i] = np.rad2deg(data_raw[:, 26:29].T)

    p_des[i] = data_raw[:, 29:32].T
    panic[i] = data_raw[:, 32]
    radioCmd[i] = data_raw[:, 33:37].T

    contacts[i] = data_raw[:, 37:49].T

t = np.array(t)
p_gt = np.array(p_gt)
v_gt = np.array(v_gt)
r_gt = np.array(r_gt)
omega_gt = np.array(omega_gt)
f_gt = np.array(f_gt)

p_est = np.array(p_est)
v_est = np.array(v_est)
r_est = np.array(r_est)
omega_est = np.array(omega_est)

p_des = np.array(p_des)
panic = np.array(panic)
radioCmd = np.array(radioCmd)

contacts = np.array(contacts)

fig = plot_trajectory(t, p_gt, v_gt, r_gt, omega_gt, contacts,
                np.array([0, 4.25 + 0.5 * 0.22, 2]), np.array([0.5, 0.5, 0.5]), Rotation.identity(),
                None)
fig.savefig("plot.png")

"""ani = animate_trajectory(t, p_gt, v_gt, r_gt, omega_gt, contacts,
                np.array([0, 4.25 + 0.5 * 0.22, 2]), np.array([0.5, 0.5, 0.5]), Rotation.identity(),
                p_des)
ani.save(filename="animation.mp4", writer="ffmpeg",
         dpi=200, fps=30, codec="libx264", extra_args=["-preset", "fast"])"""
