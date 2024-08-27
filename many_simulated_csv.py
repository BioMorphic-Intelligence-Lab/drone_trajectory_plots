import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from plotting_functions import plot_trajectory, animate_trajectory, anim_3d_plot
from tqdm import tqdm

#paths = [f"./colliding_drone_plots/data/oneMS/Simulation{i:03d}.csv" for i in range(2)]
paths = [f"data/ellipse/Simulation{i:03d}.csv" for i in range(100)]

t = []
p_gt = []
v_gt = []
r_gt = []
omega_gt = []
f_gt = []

p_est = []
v_est = []
r_est = []
omega_est = []

p_des = []
panic = []
radioCmd = []

contacts = []

continued = 0

for i in range(len(paths)):
    data_raw = np.genfromtxt(paths[i], delimiter=',')

    # Check four outliers, i.e. crashed
    if min(data_raw[100:, 3]) <= 0.0:
        continued += 1
        print(f"Continued {continued}")
        continue

    t += [data_raw[:, 0]]
    p_gt += [data_raw[:, 1:4].T]
    v_gt += [data_raw[:, 4:7].T]
    r_gt += [np.array([Rotation.from_euler(seq="zyx", angles=data_raw[i, 7:10]) for i in range(len(data_raw))])]
    omega_gt += [np.rad2deg(data_raw[:, 10:13].T)]
    f_gt += [data_raw[:, 13:17].T]

    p_est += [data_raw[:, 17:20].T]
    v_est += [data_raw[:, 20:23].T]
    r_est += [np.array([Rotation.from_euler(seq="zyx", angles=data_raw[i, 23:26]) for i in range(len(data_raw))])]
    omega_est += [np.rad2deg(data_raw[:, 26:29].T)]

    p_des += [data_raw[:, 29:32].T]
    panic += [data_raw[:, 32]]
    radioCmd += [data_raw[:, 33:37].T]

    contacts += [data_raw[:, 37:49].T]

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

#orig_traj = lambda t: np.array([1 * np.sin(2*np.pi * t), 2 * np.cos(2*np.pi * t), 1.75 * np.ones_like(t)])

fig = plot_trajectory(t, p_gt, v_gt, r_gt, omega_gt, contacts,
                np.array([-1, 0.3, 1.75]), np.array([0.6, 0.6, 0.6]), Rotation.identity(),
                None,
                None)
plt.show()
fig.savefig("plot.png")


#ani = anim_3d_plot(t, p_gt, r_gt,
#                np.array([0, 3.625, 2]), np.array([0.5, 0.5, 0.5]), Rotation.identity())
#ani.save(filename="3d_anim.mp4", writer="ffmpeg", dpi=250)


"""ani = animate_trajectory(t, p_gt, v_gt, r_gt, omega_gt, contacts,
                np.array([0, 3.625, 2]), np.array([0.25, 0.25, 0.25]), Rotation.identity(),
                None)
# Manually iterate through each frame
for i, frame_data in enumerate(tqdm(ani.new_frame_seq(), total=t.shape[1])):
    # Call the update function for the current frame
    ani._draw_next_frame(frame_data, blit=False)
    
    # Save the current frame as a PNG file
    filename = f'frames/frame_{i:05d}.png'
    plt.savefig(filename)"""
