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
linestyle2 = ":"

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

def getNumTrials(t):
    # Number of trials
    n = 0
    if t.ndim == 1:
        n = 1
    elif t.ndim == 2:
        n = np.shape(t)[0]
    else:
        print("Incorrect number of dimensions in time vector!")

    return n


def init_plot(t,
              p, p_dot,
              r, omega,
              contacts,
              rec_pos, rec_sl, rec_rot,
              des_pos=None):
    
    n = getNumTrials(t)

    # Find the ranges
    pos_range = (min(p.flatten()) - 0.1, max(p.flatten()) + 0.1)
    vel_range = (min(p_dot.flatten()) - 0.1, max(p_dot.flatten()) + 0.1)
    att_range = (-180.0, 180.0)
    omega_range = (min(omega.flatten()) - 0.1, max(omega.flatten()) + 0.1)
    range_3d = max(np.abs(pos_range))
    if t.ndim == 1:
        t_range = (t[0], t[-1])
    else:
        t_range = (min(t[:, 0]), max(t[:, -1]))
    
    # Set up a figure with aspect ration 3:2
    fig = plt.Figure(constrained_layout=True, figsize=7.5 * np.array([4, 2]))
    gs = GridSpec(12, 2, figure=fig)

    # 3D Plot
    ax3d = fig.add_subplot(gs[:-2, 0], projection='3d')
    ax3d.set_xlabel(r"$x$ [m]")
    ax3d.set_ylabel(r"$y$ [m]")
    ax3d.set_zlabel(r"$z$ [m]")
    ax3d.set_xlim((-range_3d / 2, range_3d / 2))
    ax3d.set_ylim((0.0, range_3d))
    ax3d.set_zlim((0, range_3d))
    plot_rectoid(ax3d, rec_pos, rec_sl, rec_rot,
                 facecolors='xkcd:grey', edgecolor="black", alpha=0.5)
    ax3d.view_init(elev=30, azim=65)

    # Position Plot
    axPos = fig.add_subplot(gs[0:3, 1])
    axPos.set_xlabel("")
    axPos.set_ylim(pos_range)
    axPos.tick_params(labelbottom=False)
    axPos.set_ylabel(r"Position [m]")

    # Velocity Plot
    axVel = fig.add_subplot(gs[3:6, 1], sharex=axPos)
    axVel.set_ylim(vel_range)
    axVel.set_ylabel(r"Velocity [m / s]")
    axVel.tick_params(labelbottom=False)

    # Attitude Plot
    axAtt = fig.add_subplot(gs[6:9, 1], sharex=axPos)
    axAtt.set_ylim(att_range)
    axAtt.set_ylabel(r"Attitude [$^\circ$]")
    axAtt.tick_params(labelbottom=False)

    # Rates Plot
    axRates = fig.add_subplot(gs[9:12, 1], sharex=axPos)
    axRates.set_ylim(omega_range)
    axRates.set_xlim(t_range)
    axRates.set_ylabel(r"Angular Rates[$^\circ / s$]")
    axRates.set_xlabel(r"Time [s]")

    # Contact Plot
    axContact = fig.add_subplot(gs[-2:, 0])
    axContact.set_xlim(t_range)
    axContact.set_xlabel(r"Time [s]")
    axContact.set_yticks([i for i in range(14)])
    axContact.set_yticklabels([rf"$v_{{{i}}}$" for i in range(14)])
    axContact.set_ylim((0.5, 12.5))
    axContact.grid(axis='y')

    lines3d = []
    atts3d = []
    linesPosX = []
    linesPosY = []
    linesPosZ = []
    linesDesPosX = []
    linesDesPosY = []
    linesDesPosZ = []
    linesVelX = []
    linesVelY = []
    linesVelZ = []
    linesAttX = []
    linesAttY = []
    linesAttZ = []
    linesRatesX = []
    linesRatesY = []
    linesRatesZ = []
    linesContacts = []

    for i in range(n):
        line3d, = ax3d.plot([], [], [], color=delft_blue)
        if t.ndim == 1:
            att3d = ax3d.quiver(p[0, 0], p[1, 0], p[2, 0], 1, 0, 0,
                            length=0.3, color="black")
        else:
            att3d = ax3d.quiver(p[i, 0, 0], p[i, 1, 0], p[i, 2, 0], 1, 0, 0,
                            length=0.3, color="black")
        linePosX, = axPos.plot([], [], label=r"$x$", color=color_x, linestyle=linestyle0)
        linePosY, = axPos.plot([], [], label=r"$y$", color=color_y, linestyle=linestyle0)
        linePosZ, = axPos.plot([], [], label=r"$z$", color=color_z, linestyle=linestyle0)
        lineDesPosX, = axPos.plot([], [], color="black", linestyle=linestyle2)
        lineDesPosY, = axPos.plot([], [], color="black", linestyle=linestyle2)
        lineDesPosZ, = axPos.plot([], [], color="black", linestyle=linestyle2)
        lineVelX, = axVel.plot([], [], label=r"$\dot{x}$", color=color_x, linestyle=linestyle1)
        lineVelY, = axVel.plot([], [], label=r"$\dot{y}$", color=color_y, linestyle=linestyle1)
        lineVelZ, = axVel.plot([], [], label=r"$\dot{z}$", color=color_z, linestyle=linestyle1)
        lineAttX, = axAtt.plot([], [], label=r"$\varphi$", color=color_x, linestyle=linestyle0)
        lineAttY, = axAtt.plot([], [], label=r"$\theta$", color=color_y, linestyle=linestyle0)
        lineAttZ, = axAtt.plot([], [], label=r"$\psi$", color=color_z, linestyle=linestyle0)
        lineRatesX, = axRates.plot([], [], label=r"$\omega_x$", color=color_x, linestyle=linestyle1)
        lineRatesY, = axRates.plot([], [], label=r"$\omega_y$", color=color_y, linestyle=linestyle1)
        lineRatesZ, = axRates.plot([], [], label=r"$\omega_z$", color=color_z, linestyle=linestyle1)
        contactLines = []
        for i in range(12):
            line,  = axContact.plot([], [],
                                    linestyle="", marker="o", color=color_contact,
                                    markersize=4)
            contactLines += [line]
        
        lines3d += [line3d]
        atts3d += [att3d]
        linesPosX += [linePosX]
        linesPosY += [linePosY]
        linesPosZ += [linePosZ]
        linesDesPosX += [lineDesPosX]
        linesDesPosY += [lineDesPosY]
        linesDesPosZ += [lineDesPosZ]
        linesVelX += [lineVelX]
        linesVelY += [lineVelY]
        linesVelZ += [lineVelZ]
        linesAttX += [lineAttX]
        linesAttY += [lineAttY]
        linesAttZ += [lineAttZ]
        linesRatesX += [lineRatesX]
        linesRatesY += [lineRatesY]
        linesRatesZ += [lineRatesZ]
        linesContacts += [contactLines]

    if n == 1:
        axPos.legend(loc='upper left')
        axVel.legend(loc='upper left')
        axAtt.legend(loc='upper left')
        axRates.legend(loc='upper left')
    else:
        axPos.legend(labels=[r"$x$", r"$y$", r"$z$"], loc='upper left')
        axVel.legend(labels=[r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"], loc='upper left')
        axAtt.legend(labels=[r"$\varphi$", r"$\theta$", r"$\psi$"], loc='upper left')
        axRates.legend(labels=[r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"], loc='upper left')

    return (fig, ax3d, lines3d, atts3d,
            linesPosX, linesPosY, linesPosZ,
            linesDesPosX, linesDesPosY, linesDesPosZ,
            linesVelX, linesVelY, linesVelZ,
            linesAttX, linesAttY, linesAttZ,
            linesRatesX, linesRatesY, linesRatesZ,
            linesContacts)

def plot_trajectory(t, p, p_dot, 
                    r, omega, contacts,
                    rec_pos, rec_sl, rec_rot,
                    des_pos=None):
    
    (fig, ax3d, lines3d, atts3d,
    linesPosX, linesPosY, linesPosZ,
    linesDesPosX, linesDesPosY, linesDesPosZ,
    linesVelX, linesVelY, linesVelZ,
    linesAttX, linesAttY, linesAttZ,
    linesRatesX, linesRatesY, linesRatesZ,
    linesContacts) = init_plot(
        t, p, p_dot, r, omega, contacts,
        rec_pos, rec_sl, rec_rot)
    
    n = getNumTrials(t)

    if n == 1:
        t = t.reshape([1, len(t)])
        p = p.reshape([1, p.shape[0], p.shape[1]])
        p_dot = p_dot.reshape([1, p_dot.shape[0], p_dot.shape[1]])
        r = r.reshape([1, len(r)])
        omega = omega.reshape([1, omega.shape[0], omega.shape[1]])
        contacts = contacts.reshape([1, contacts.shape[0], contacts.shape[1]])
        if des_pos is not None:
            des_pos = des_pos.reshape([1, des_pos.shape[0], des_pos.shape[1]])
    for i in range(n):
    
        att_pry = np.array([rot.as_euler(seq="xyz", degrees=True).T for rot in r[i]]).T
        contactsPlotData = (np.arange(1, 13).reshape(12, 1) * contacts[i]).T
        
        lines3d[i].set_data(p[i, 0, :], p[i, 1, :])
        lines3d[i].set_3d_properties(p[i, 2, :])
        atts3d[i].remove()
        linesPosX[i].set_data(t[i, :], p[i, 0, :])
        linesPosY[i].set_data(t[i, :], p[i, 1, :])
        linesPosZ[i].set_data(t[i, :], p[i, 2, :])
        if des_pos is not None:
            linesDesPosX[i].set_data(t[i, :], des_pos[i, 0,:])
            linesDesPosY[i].set_data(t[i, :], des_pos[i, 1,:])
            linesDesPosZ[i].set_data(t[i, :], des_pos[i, 2,:])
        linesVelX[i].set_data(t[i, :], p_dot[i, 0, :])
        linesVelY[i].set_data(t[i, :], p_dot[i, 1, :])
        linesVelZ[i].set_data(t[i, :], p_dot[i, 2, :])
        linesAttX[i].set_data(t[i, :], att_pry[0,:])
        linesAttY[i].set_data(t[i, :], att_pry[1,:])
        linesAttZ[i].set_data(t[i, :], att_pry[2,:])
        linesRatesX[i].set_data(t[i, :], omega[i, 0,:])
        linesRatesY[i].set_data(t[i, :], omega[i, 1,:])
        linesRatesZ[i].set_data(t[i, :], omega[i, 2,:])
        for j in range(12):
            linesContacts[i][j].set_data(t[i, :], contactsPlotData[:, j])
    
    return fig


def animate_trajectory(t, p, p_dot, 
                       r, omega, contacts,
                       rec_pos, rec_sl, rec_rot,
                       des_pos=None):
    
    (fig, ax3d, lines3d, atts3d,
    linesPosX, linesPosY, linesPosZ,
    linesDesPosX, linesDesPosY, linesDesPosZ,
    linesVelX, linesVelY, linesVelZ,
    linesAttX, linesAttY, linesAttZ,
    linesRatesX, linesRatesY, linesRatesZ,
    linesContacts) = init_plot(
        t, p, p_dot, r, omega, contacts,
        rec_pos, rec_sl, rec_rot)
    
    n = getNumTrials(t)

    if n == 1:
        t = t.reshape([1, len(t)])
        p = p.reshape([1, p.shape[0], p.shape[1]])
        p_dot = p_dot.reshape([1, p_dot.shape[0], p_dot.shape[1]])
        r = r.reshape([1, len(r)])
        omega = omega.reshape([1, omega.shape[0], omega.shape[1]])
        contacts = contacts.reshape([1, contacts.shape[0], contacts.shape[1]])
        if des_pos is not None:
            des_pos = des_pos.reshape([1, des_pos.shape[0], des_pos.shape[1]])
    
    
    # Update function for animation
    def update(frame):
        nonlocal atts3d
        for i in range(n): 
            att_pry = np.array([rot.as_euler(seq="xyz", degrees=True).T for rot in r[i]]).T
            contactsPlotData = (np.arange(1, 13).reshape(12, 1) * contacts[i]).T
            dir_vec = np.array([r[i][j].as_matrix() @ np.array([1, 0, 0]) for j in range(len(r[i]))]).T
            
            lines3d[i].set_data(p[i, 0, :frame], p[i, 1, :frame])
            lines3d[i].set_3d_properties(p[i, 2, :frame])

            atts3d[i].remove()    
            atts3d[i] = ax3d.quiver(
                p[i, 0, max(0, frame-1)], p[i, 1, max(0, frame-1)], p[i, 2, max(0, frame-1)],
                dir_vec[0, max(0, frame-1)], dir_vec[1, max(0, frame-1)], dir_vec[2, max(0, frame-1)],
                length=0.3, color="black"
            )
            linesPosX[i].set_data(t[i, :frame], p[i, 0, :frame])
            linesPosY[i].set_data(t[i, :frame], p[i, 1, :frame])
            linesPosZ[i].set_data(t[i, :frame], p[i, 2, :frame])
            if des_pos is not None:
                linesDesPosX[i].set_data(t[i, :frame], des_pos[i, 0,:frame])
                linesDesPosY[i].set_data(t[i, :frame], des_pos[i, 1,:frame])
                linesDesPosZ[i].set_data(t[i, :frame], des_pos[i, 2,:frame])
            linesVelX[i].set_data(t[i, :frame], p_dot[i, 0, :frame])
            linesVelY[i].set_data(t[i, :frame], p_dot[i, 1, :frame])
            linesVelZ[i].set_data(t[i, :frame], p_dot[i, 2, :frame])
            linesAttX[i].set_data(t[i, :frame], att_pry[0, :frame])
            linesAttY[i].set_data(t[i, :frame], att_pry[1, :frame])
            linesAttZ[i].set_data(t[i, :frame], att_pry[2, :frame])
            linesRatesX[i].set_data(t[i, :frame], omega[i, 0,:frame])
            linesRatesY[i].set_data(t[i, :frame], omega[i, 1,:frame])
            linesRatesZ[i].set_data(t[i, :frame], omega[i, 2,:frame])
            for j in range(12):
                linesContacts[i][j].set_data(t[i, :frame], contactsPlotData[:frame, j])
        return (lines3d, atts3d,
                linesPosX, linesPosY, linesPosZ,
                linesDesPosX, linesDesPosY, linesDesPosZ,
                linesVelX, linesVelY, linesVelZ,
                *linesContacts)

    # Run Animation
    return FuncAnimation(fig, update, frames=len(t[0]), interval=max(t[:, -1]) / len(t[0]) * 1e3, blit=True)

