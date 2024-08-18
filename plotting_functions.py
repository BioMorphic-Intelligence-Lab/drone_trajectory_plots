import numpy as np
import matplotlib.pyplot  as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib.animation import FuncAnimation

# Define Colors & Linestyles
delft_blue = "#00A6D6"
color_x = "#F70035"
color_y = "#54F100"
color_z = "#FF8100"
color_contact="#C06100"
linestyle0 = "-"
linestyle1 = "--"

# Enable LaTeX text rendering
plt.rcParams['text.usetex'] = True
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 21}

matplotlib.rc('font', **font)


def plot_rectoid(ax, center, sidelength, rot, **kwargs):

    step_front_bottom_left = 0.5 * np.array([-sidelength[0],
                                       -sidelength[1],
                                       -sidelength[2]])
    step_front_bottom_right = 0.5 * np.array([ sidelength[0],
                                        -sidelength[1],
                                        -sidelength[2]])
    step_front_top_right = 0.5 * np.array([ sidelength[0],
                                     -sidelength[1],
                                      sidelength[2]])
    step_front_top_left = 0.5 * np.array([ -sidelength[0],
                                     -sidelength[1],
                                      sidelength[2]])
    step_back_bottom_left = 0.5 * np.array([-sidelength[0],
                                       sidelength[1],
                                       -sidelength[2]])
    step_back_bottom_right = 0.5 * np.array([ sidelength[0],
                                        sidelength[1],
                                        -sidelength[2]])
    step_back_top_right = 0.5 * np.array([ sidelength[0],
                                     sidelength[1],
                                      sidelength[2]])
    step_back_top_left = 0.5 * np.array([ -sidelength[0],
                                     sidelength[1],
                                      sidelength[2]])

    faces = []
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    
    # Front Face
    faces[0][0,:]  = (np.array(center + rot.apply(step_front_bottom_left)))
    faces[0][1,:]  = (np.array(center + rot.apply(step_front_bottom_right)))
    faces[0][2,:]  = (np.array(center + rot.apply(step_front_top_right)))
    faces[0][3,:]  = (np.array(center + rot.apply(step_front_top_left)))

    # Left Face
    faces[1][0,:]  = (np.array(center + rot.apply(step_front_bottom_left)))
    faces[1][1,:]  = (np.array(center + rot.apply(step_front_top_left)))
    faces[1][2,:]  = (np.array(center + rot.apply(step_back_top_left)))
    faces[1][3,:]  = (np.array(center + rot.apply(step_back_bottom_left)))

    # Back Face
    faces[2][0,:]  = (np.array(center + rot.apply(step_back_bottom_left)))
    faces[2][1,:]  = (np.array(center + rot.apply(step_back_top_left)))
    faces[2][2,:]  = (np.array(center + rot.apply(step_back_top_right)))
    faces[2][3,:]  = (np.array(center + rot.apply(step_back_bottom_right)))

    # Right Face
    faces[3][0,:]  = (np.array(center + rot.apply(step_front_bottom_right)))
    faces[3][1,:]  = (np.array(center + rot.apply(step_front_top_right)))
    faces[3][2,:]  = (np.array(center + rot.apply(step_back_top_right)))
    faces[3][3,:]  = (np.array(center + rot.apply(step_back_bottom_right)))

    # Top Face
    faces[4][0,:]  = (np.array(center + rot.apply(step_front_top_right)))
    faces[4][1,:]  = (np.array(center + rot.apply(step_front_top_left)))
    faces[4][2,:]  = (np.array(center + rot.apply(step_back_top_left)))
    faces[4][3,:]  = (np.array(center + rot.apply(step_back_top_right)))

    # Bottom Face
    faces[5][0,:]  = (np.array(center + rot.apply(step_front_bottom_right)))
    faces[5][1,:]  = (np.array(center + rot.apply(step_front_bottom_left)))
    faces[5][2,:]  = (np.array(center + rot.apply(step_back_bottom_left)))
    faces[5][3,:]  = (np.array(center + rot.apply(step_back_bottom_right)))
    
    
    ax.add_collection3d(Poly3DCollection(faces, **kwargs))

def init_plot(t,
              p, p_dot,
              r, omega,
              contacts,
              rec_pos, rec_sl, rec_rot):
    pos_range = (min(p.flatten()) - 0.1, max(p.flatten()) + 0.1)
    vel_range = (min(p_dot.flatten()) - 0.1, max(p_dot.flatten()) + 0.1)
    att_range = (-180.0, 180.0)
    omega_range = (min(omega.flatten()) - 0.1, max(omega.flatten()) + 0.1)
    range_3d = max(np.abs(pos_range))
    # Set up a figure with aspect ration 3:2
    fig = plt.figure(constrained_layout=True, figsize=7.5 * np.array([4, 2]))
    gs = GridSpec(12, 2, figure=fig)

    # 3D Plot
    ax3d = fig.add_subplot(gs[:-2, 0], projection='3d')
    line3d, = ax3d.plot([], [], [], color=delft_blue)
    att3d = ax3d.quiver(p[0,0],p[1,0],p[2,0],1,0,0, length=0.3, color="black")
    ax3d.set_xlabel(r"$x$ [m]")
    ax3d.set_ylabel(r"$y$ [m]")
    ax3d.set_zlabel(r"$z$ [m]")
    ax3d.set_xlim((-range_3d, range_3d))
    ax3d.set_ylim((-range_3d, range_3d))
    ax3d.set_zlim((0, 2*range_3d))
    plot_rectoid(ax3d, rec_pos, rec_sl, rec_rot,
                 facecolors='xkcd:grey', edgecolor="black", alpha=0.5)
    ax3d.view_init(elev=30, azim=65)

    # Position Plot
    axPos = fig.add_subplot(gs[0:3, 1])
    linePosX, = axPos.plot([], [], label=r"$x$", color=color_x, linestyle=linestyle0)
    linePosY, = axPos.plot([], [], label=r"$y$", color=color_y, linestyle=linestyle0)
    linePosZ, = axPos.plot([], [], label=r"$z$", color=color_z, linestyle=linestyle0)
    axPos.set_xlabel("")
    axPos.set_ylim(pos_range)
    axPos.tick_params(labelbottom=False)
    axPos.set_ylabel(r"Position [m]")
    axPos.legend()

    # Velocity Plot
    axVel = fig.add_subplot(gs[3:6, 1], sharex=axPos)
    lineVelX, = axVel.plot([], [], label=r"$\dot{x}$", color=color_x, linestyle=linestyle1)
    lineVelY, = axVel.plot([], [], label=r"$\dot{y}$", color=color_y, linestyle=linestyle1)
    lineVelZ, = axVel.plot([], [], label=r"$\dot{z}$", color=color_z, linestyle=linestyle1)
    axVel.set_ylim(vel_range)
    axVel.set_ylabel(r"Velocity [m / s]")
    axVel.tick_params(labelbottom=False)
    axVel.legend()

    # Attitude Plot
    axAtt = fig.add_subplot(gs[6:9, 1], sharex=axPos)
    lineAttX, = axAtt.plot([], [], label=r"$\varphi$", color=color_x, linestyle=linestyle0)
    lineAttY, = axAtt.plot([], [], label=r"$\theta$", color=color_y, linestyle=linestyle0)
    lineAttZ, = axAtt.plot([], [], label=r"$\psi$", color=color_z, linestyle=linestyle0)
    axAtt.set_ylim(att_range)
    axAtt.set_ylabel(r"Attitude [$^\circ$]")
    axAtt.tick_params(labelbottom=False)
    axAtt.legend()

    # Rates Plot
    axRates = fig.add_subplot(gs[9:12, 1], sharex=axPos)
    lineRatesX, = axRates.plot([], [], label=r"$\omega_x$", color=color_x, linestyle=linestyle1)
    lineRatesY, = axRates.plot([], [], label=r"$\omega_y$", color=color_y, linestyle=linestyle1)
    lineRatesZ, = axRates.plot([], [], label=r"$\omega_z$", color=color_z, linestyle=linestyle1)
    axRates.set_ylim(omega_range)
    axRates.set_xlim((t[0], t[-1]))
    axRates.set_ylabel(r"Angular Rates[$^\circ / s$]")
    axRates.set_xlabel(r"Time [s]")
    axRates.legend()

    # Contact Plot
    axContact = fig.add_subplot(gs[-2:, 0])
    contactLines = []
    for i in range(12):
        line,  = axContact.plot([], [],
                                linestyle="", marker="o", color=color_contact,
                                markersize=2)
        contactLines += [line]
    axContact.set_xlim((t[0], t[-1]))
    axContact.set_xlabel(r"Time [s]")
    axContact.set_yticks([i for i in range(14)])
    axContact.set_yticklabels([rf"$v_{{{i}}}$" for i in range(14)])
    axContact.set_ylim((0.5, 12.5))
    axContact.grid(axis='y')

    return (fig, ax3d, line3d, att3d,
            linePosX, linePosY, linePosZ,
            lineVelX, lineVelY, lineVelZ,
            lineAttX, lineAttY, lineAttZ,
            lineRatesX, lineRatesY, lineRatesZ,
            contactLines)

def plot_trajectory(t, p, p_dot, 
                    r, omega, contacts,
                    rec_pos, rec_sl, rec_rot):
    
    (fig, ax3d, line3d, att3d,
     linePosX, linePosY, linePosZ,
     lineVelX, lineVelY, lineVelZ,
     lineAttX, lineAttY, lineAttZ,
     lineRatesX, lineRatesY, lineRatesZ,
     contactLines) = init_plot(
        t, p, p_dot, r, omega, contacts,
        rec_pos, rec_sl, rec_rot)
    
    att_pry = np.array([rot.as_euler(seq="xyz", degrees=True).T for rot in r]).T
    contactsPlotData = (np.arange(1, 13).reshape(12, 1) * contacts).T
    
    line3d.set_data(p[0, :], p[1, :])
    line3d.set_3d_properties(p[2, :])
    linePosX.set_data(t[:], p[0, :])
    linePosY.set_data(t[:], p[1, :])
    linePosZ.set_data(t[:], p[2, :])
    lineVelX.set_data(t[:], p_dot[0, :])
    lineVelY.set_data(t[:], p_dot[1, :])
    lineVelZ.set_data(t[:], p_dot[2, :])
    lineAttX.set_data(t[:], att_pry[0,:])
    lineAttY.set_data(t[:], att_pry[1,:])
    lineAttZ.set_data(t[:], att_pry[2,:])
    lineRatesX.set_data(t[:], omega[0,:])
    lineRatesY.set_data(t[:], omega[1,:])
    lineRatesZ.set_data(t[:], omega[2,:])
    for i in range(12):
        contactLines[i].set_data(t[:], contactsPlotData[:, i])
    
    return fig


def animate_trajectory(t, p, p_dot, 
                       r, omega, contacts,
                       rec_pos, rec_sl, rec_rot):

    (fig, ax3d, line3d, att3d, 
     linePosX, linePosY, linePosZ,
     lineVelX, lineVelY, lineVelZ,
     lineAttX, lineAttY, lineAttZ,
     lineRatesX, lineRatesY, lineRatesZ,
     contactLines) = init_plot(
        t, p, p_dot, r, omega, contacts,
        rec_pos, rec_sl, rec_rot)
    
    att_pry = np.array([rot.as_euler(seq="xyz", degrees=True).T for rot in r]).T
    contactsPlotData = (np.arange(1, 13).reshape(12, 1) * contacts).T
    dir_vec = np.array([r[i].as_matrix() @ np.array([1, 0, 0]) for i in range(len(r))]).T
    
    # Update function for animation
    def update(frame):
        line3d.set_data(p[0, :frame], p[1, :frame])
        line3d.set_3d_properties(p[2, :frame])
        nonlocal att3d
        att3d.remove()    
        att3d = ax3d.quiver(
            p[0, max(0, frame-1)], p[1, max(0, frame-1)], p[2, max(0, frame-1)],
            dir_vec[0, max(0, frame-1)], dir_vec[1, max(0, frame-1)], dir_vec[2, max(0, frame-1)],
            length=0.3, color="black"
        )
        linePosX.set_data(t[:frame], p[0, :frame])
        linePosY.set_data(t[:frame], p[1, :frame])
        linePosZ.set_data(t[:frame], p[2, :frame])
        lineVelX.set_data(t[:frame], p_dot[0, :frame])
        lineVelY.set_data(t[:frame], p_dot[1, :frame])
        lineVelZ.set_data(t[:frame], p_dot[2, :frame])
        lineAttX.set_data(t[:frame], att_pry[0,:frame])
        lineAttY.set_data(t[:frame], att_pry[1,:frame])
        lineAttZ.set_data(t[:frame], att_pry[2,:frame])
        lineRatesX.set_data(t[:frame], omega[0,:frame])
        lineRatesY.set_data(t[:frame], omega[1,:frame])
        lineRatesZ.set_data(t[:frame], omega[2,:frame])
        for i in range(12):
            contactLines[i].set_data(t[:frame], contactsPlotData[:frame, i])
        return (line3d, att3d,
                linePosX, linePosY, linePosZ,
                lineVelX, lineVelY, lineVelZ,
                *contactLines)


    # Run Animation
    return FuncAnimation(fig, update, frames=len(t), interval=t[-1] / len(t) * 1e3, blit=True)

