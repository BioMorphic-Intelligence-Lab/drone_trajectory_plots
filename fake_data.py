import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from plotting_functions import animate_trajectory, plot_trajectory

# Define Constants
n = 1000
a = 1.0
speed = 1.0/10.0
t_end = 10.0
rec_pos = np.array([0, 1, 1.25])
rec_sl = np.array([0.5, 0.5, 0.5])
rec_rot = Rotation.identity()

# Compute the trajectory
t = np.linspace(0, t_end, n)
p = np.array([a * np.sin(2 * speed * t * 2 * np.pi),
              a * np.cos(    speed * t * 2 * np.pi),
              np.ones_like(t) * 1.25])
p_dot = np.array([ 4 * speed * np.pi * a * np.cos(2 * speed * t * 2 * np.pi),
                  -2 * speed * np.pi * a * np.sin(    speed * t * 2 * np.pi),
                  0 * t])

p_ddot = np.array([-16 * speed**2 * np.pi**2 * a * np.sin(2 * speed * t * 2 * np.pi),
                   - 4 * speed**2 * np.pi**2 * a * np.cos(    speed * t * 2 * np.pi),
                   0 * t])

angles = np.arctan2(p_dot[1], p_dot[0])
r = np.array([Rotation.from_rotvec(angles[:, np.newaxis] * np.array([0, 0, 1]))]).flatten()
omega = np.array([0 * t,
                  0 * t,
                  np.rad2deg(p_dot[0,:] * p_ddot[1, :] - p_dot[1, :] * p_ddot[0, :]) / (p_dot[0,:]**2 + p_dot[1,:]**2)])


vertices = 0.22 * (np.array([
                        [0.00, 0.50, 0.25], 
                        [1.00, 0.50, 0.25],
                        [0.00, 0.50, 0.75], 
                        [1.00, 0.50, 0.75], 
                        [0.25, 0.00, 0.50], 
                        [0.25, 1.00, 0.50], 
                        [0.75, 0.00, 0.50], 
                        [0.75, 1.00, 0.50], 
                        [0.50, 0.25, 0.00], 
                        [0.50, 0.25, 1.00], 
                        [0.50, 0.75, 0.00], 
                        [0.50, 0.75, 1.00]
                    ]) - 0.5)
contacts = np.zeros([12, len(t)])

for i in range(12):
    contacts[i, :] = (np.abs(p + vertices[i, :].reshape(3,1) - np.array([0, 1, 1.25]).reshape(3,1)) < 0.5 * np.ones((3,1))).all(axis=0)

fig = plot_trajectory(t,
                      p, p_dot,
                      r, omega,
                      contacts,
                      rec_pos, rec_sl, rec_rot)
fig.savefig("plot.png")
ani = animate_trajectory(t,
                        p, p_dot,
                        r, omega,
                        contacts,
                        rec_pos, rec_sl, rec_rot)
ani.save(filename="animation.mp4", writer="ffmpeg")
#plt.show()