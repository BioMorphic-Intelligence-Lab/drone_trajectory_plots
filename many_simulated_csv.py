import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from plotting_functions import (plot_trajectory, animate_trajectory,
                                anim_3d_plot, plot_3d, plot_timeplot,
                                plot_topview)
from tqdm import tqdm


def main(argv):
    if len(argv) > 1:
        if argv[1] == "replan":
            paths = [f"data/ellipse/Simulation{i:03d}.csv" for i in range(10)]
            cubeP = np.array([
                #[-1, 0.3, 1.75],
                #[-1, 0.6, 1.75]
            ])
            cubeDp = np.array([
                #[0.6, 0.6, 0.6],
                #[0.6, 0.6, 0.6]
            ])
            cubeR = [
               # Rotation.identity(),
               # Rotation.identity()
            ]

            cylP = np.array([
                [-0.8, 0.3, 1.0],
                [ 0.0, 2.3, 1.0]
            ])
            cylDim = np.array([
                [0.2, 1.0],
                [0.2, 1.0]
            ])
            cylR = [
                Rotation.identity(),
                Rotation.identity()
            ]

            orig_traj = lambda t: np.array([0.75 * np.sin(2*np.pi * t),
                                            2.25 * np.cos(2*np.pi * t),
                                            1.75 * np.ones_like(t)])

            orig_traj = lambda t: np.array([0.75 * np.sin(2*np.pi * t),
                                            2.25 * np.cos(2*np.pi * t),
                                            1.75 * np.ones_like(t)])
        elif argv[1] == "hover":
            paths = [f"data/collisions/Simulation_6.0_{i:03d}.csv" for i in range(10)]
            cubeP = np.array([
                [0.0, 12.0, 1.75]
            ])
            cubeDp = np.array([
                [4.0, 4, 4.0]
            ])
            cubeR = [
                Rotation.identity()
            ]
            orig_traj = None
    else:
        paths =  [f"data/ellipse/Simulation{i:03d}.csv" for i in range(10)]
        cubeP = np.array([
            [-1, 0.3, 1.75]
        ])
        cubeDp = np.array([
            [0.6, 0.6, 0.6]
        ])
        cubeR = [
            Rotation.identity()
        ]

        orig_traj = lambda t: np.array([0.75 * np.sin(2*np.pi * t),
                                        2.25 * np.cos(2*np.pi * t),
                                        1.75 * np.ones_like(t)])

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
            #continue

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


    #fig = plot_trajectory(t, p_gt, v_gt, r_gt, omega_gt, contacts,
    #               cubeP, cubeDp, cubeR,
    #               None,
    #               orig_traj)
    
    fig = plot_3d(t, p_gt, r_gt,
                  cubeP,cubeDp, cubeR,
                  cylP, cylDim, cylR,
                  orig_traj=orig_traj)
    
    #fig = plot_timeplot(t, p_gt, v_gt, r_gt,
    #                    np.array([-0.75, 0.3-0.11, 1.75]), np.array([0.6, 0.6, 0.6]), Rotation.identity(),
    #                    des_pos=None,
    #                    orig_traj=orig_traj
    #                    )
    #
    fig.set_size_inches((45, 30))
    fig.savefig("plot.pdf", dpi=100, bbox_inches="tight", transparent=True)
    #plt.show()
    #ani = anim_3d_plot(t, p_gt, r_gt,
    #                np.array([-1, 0.3, 1.75]), np.array([0.6, 0.6, 0.6]), Rotation.identity())
    #ani.save(filename="3d_anim.mp4", writer="ffmpeg", dpi=250)


    step_size = 4
    j = 0
    #ani = animate_trajectory(t, p_gt, v_gt, r_gt, omega_gt, contacts,
    #                cubeP, cubeDp, cubeR,
    #                orig_traj,
    #                None)
    
    # Manually iterate through each frame
    #for i, frame_data in enumerate(tqdm(ani.new_frame_seq(), total=t.shape[1])):
    #    if i % step_size == 0:
    #        # Call the update function for the current frame
    #        ani._draw_next_frame(frame_data, blit=False)
    #        
    #        # Save the current frame as a PNG file
    #        filename = f'frames/frame_{j:05d}.png'
    #        j = j+1
    #        plt.savefig(filename)

if __name__=="__main__":
    main(sys.argv)