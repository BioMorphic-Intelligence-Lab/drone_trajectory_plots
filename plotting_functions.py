import numpy as np
import matplotlib.pyplot  as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib.animation import FuncAnimation

# Define Colors & Linestyles
delft_blue = "#00A6D6"
delft_blue_darker = "#258EFC"
delft_blue_ddarker = "#002FDC"
color_x = "#008E2B"
color_y = "#001A83"
color_z = "#FF8100"
bright_grey = "#a1a095ff"
dark_grey = "#2e2e2e"
color_contact="red"
linestyle0 = "-"
linestyle1 = "--"
linestyle2 = ":"
colors = [color_x, color_y, delft_blue]

# Enable LaTeX text rendering
plt.rcParams['text.usetex'] = True
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 95}

matplotlib.rc('font', **font)

def plot_success_rates(velocities, rates, sigma, N=3):

    assert(len(velocities) == rates.shape[1])
    assert(rates.shape == sigma.shape)

    legend= ["Collision Agnostic", "Accelerometer Based", "Ours"]
    bar_width = 0.6 / (N + 2)

    fig = plt.figure(figsize=(45, 30))
    #fig.set_facecolor(bright_grey)
    ax = fig.add_subplot()

    ax.set_xlabel(r"Collision Velocity [$m / s$]", labelpad=35)
    ax.set_ylabel("Cumulative Recovery Success Rate [\\%]", labelpad=35)
    ax.set_xticks(velocities)
    ax.set_yticks(np.arange(0, 110, 50))
    ax.tick_params(axis='both', which='major', width=5, length=30,pad=35)
    #ax.set_xlim((velocities[0], velocities[-1]))
    ax.set_ylim((-5, 102))
    #ax.set_facecolor(bright_grey)
    radius = 1.0

    for i in range(rates.shape[0]):
        xloc = velocities + (-(bar_width + 0.025) + i * (bar_width + 0.025))
        ax.bar(xloc, rates[i, :] + 5.0 - radius,
               color=colors[i], width=bar_width, label=legend[i],
               bottom=-5, alpha=0.75)
        for j in range(rates.shape[1]):
            ellipse = Ellipse((xloc[j], rates[i, j] - radius),
                              bar_width, 2 * radius, facecolor=colors[i], 
                              linewidth=None, alpha=0.75)
            ax.add_patch(ellipse)

    ax.legend(loc="upper right")
    fig.savefig('success_rate.png', bbox_inches="tight", dpi=300, transparent=True)
        #ax.errorbar(xloc, rates[i, :], sigma[i], fmt='.', color='Black',
        #            elinewidth=5,capthick=5,errorevery=1, alpha=0.9, 
        #            ms=4, capsize =10)


    # Customize each bar to have rounded corners

    
def plot_time_to_recovery(velocities, times, sigma, N=3):
    
    assert(len(velocities) == times.shape[1])
    assert(times.shape == sigma.shape)

    legend= ["Collision Agnostic", "Accelerometer Based", "Ours"]
    bar_width = 0.6 / (N + 2)

    fig = plt.figure(figsize=(45, 30))
    #fig.set_facecolor(bright_grey)
    ax = fig.add_subplot()

    ax.set_xlabel(r"Collision Velocity [$m / s$]", labelpad=35)
    ax.set_ylabel("Time to Recovery [s]", labelpad=35)
    ax.set_xticks(velocities)
    ax.tick_params(axis='both', which='major', width=5, length=30,pad=35)
    #ax.set_xlim((velocities[0], velocities[-1]))
    #ax.set_facecolor(bright_grey)
    radius = 1.0

    for i in range(times.shape[0]):
        xloc = velocities + (-(bar_width + 0.025) + i * (bar_width + 0.025))
        ax.bar(xloc, times[i, :],
               color=colors[i], width=bar_width, label=legend[i], alpha=0.75)
        #for j in range(times.shape[1]):
        #    ellipse = Ellipse((xloc[j], times[i, j] - radius),
        #                      bar_width, 2 * radius, facecolor=colors[i], 
        #                      linewidth=None, alpha=0.75)
        #    ax.add_patch(ellipse)

        ax.errorbar(xloc, times[i, :], sigma[i], fmt='.', color='Black',
                    elinewidth=5,capthick=5,errorevery=1, alpha=0.9, 
                    ms=4, capsize =10)


    ax.legend(loc="upper right")
    fig.savefig('time_to_recovery.png', bbox_inches="tight", dpi=100, transparent=False)

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

def plot_cylinder(ax, center, radius, height, rot, n_segments=20, **kwargs):
    """
    Plot a 3D cylinder with top and bottom faces.
    
    Parameters:
    ax: matplotlib 3D axis
    center: array-like, center point of the cylinder [x, y, z]
    radius: float, radius of the cylinder
    height: float, height of the cylinder
    rot: scipy.spatial.transform.Rotation object for orientation
    n_segments: int, number of segments to approximate the cylinder (default: 20)
    **kwargs: additional arguments passed to Poly3DCollection
    """
    
    # Generate angles for circular cross-section
    theta = np.linspace(0, 2*np.pi, n_segments + 1)
    
    # Create faces list
    faces = []
    
    # Generate points for top and bottom circles
    top_circle = np.zeros((n_segments + 1, 3))
    bottom_circle = np.zeros((n_segments + 1, 3))
    
    for i, angle in enumerate(theta):
        # Bottom circle points (local coordinates)
        bottom_local = np.array([radius * np.cos(angle), 
                                radius * np.sin(angle), 
                                -height/2])
        # Top circle points (local coordinates)
        top_local = np.array([radius * np.cos(angle), 
                             radius * np.sin(angle), 
                             height/2])
        
        # Apply rotation and translation
        bottom_circle[i] = center + rot.apply(bottom_local)
        top_circle[i] = center + rot.apply(top_local)
    
    # Create side faces (rectangular panels between top and bottom)
    for i in range(n_segments):
        face = np.zeros((4, 3))
        # Bottom edge of current segment
        face[0] = bottom_circle[i]
        face[1] = bottom_circle[i + 1]
        # Top edge of current segment
        face[2] = top_circle[i + 1]
        face[3] = top_circle[i]
        faces.append(face)
    
    # Create bottom face (circle)
    if n_segments > 2:  # Need at least 3 points for a face
        bottom_face = bottom_circle[:-1]  # Remove duplicate point
        faces.append(bottom_face)
    
    # Create top face (circle) - reverse order for proper normal direction
    if n_segments > 2:
        top_face = top_circle[:-1][::-1]  # Remove duplicate and reverse
        faces.append(top_face)
    
    # Add collection to axes
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

def init_topview(t, p,
                 rec_pos, rec_sl, rec_rot,
                 orig_traj=None):
    
    n = getNumTrials(t)

    # Find the ranges
    if n == 1:
        pos_range_x = (min(p[0, :].flatten()) - 0.25, max(p[0, :].flatten()) + 0.25)
        pos_range_y = (min(p[1, :].flatten()) - 0.25, max(p[1, :].flatten()) + 0.25)
    else:
        pos_range_x = (min(p[:, 0, :].flatten()) - 0.25, max(p[:, 0, :].flatten()) + 0.25)
        pos_range_y = (min(p[:, 1, :].flatten()) - 0.25, max(p[:, 1, :].flatten()) + 0.25)

    fig = plt.figure(constrained_layout=True,
                     facecolor=dark_grey)
    fig.set_size_inches((15,55))
    
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    ax.set_xlim(pos_range_x)
    ax.set_ylim(pos_range_y)
    ax.set_xlabel(r"x [m]")
    ax.set_ylabel(r"y [m]")
    ax.set_facecolor(dark_grey)
    ax.tick_params(axis='both', which='major', width=5, length=30,pad=35)

    ax.add_patch(Rectangle(rec_pos[0:2] - 0.5*rec_sl[0:2],
                           width=rec_sl[0], height=rec_sl[1],
                           angle=rec_rot.as_euler(seq="xyz")[2],
                           facecolor='xkcd:grey', linewidth=10,
                           edgecolor="black", alpha=0.5))
    
    # change all spines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(10)

    # increase tick width
    ax.tick_params(width=10)
    
    if orig_traj is not None:
        vals = orig_traj(np.linspace(0, 1, 100))
        ax.plot(vals[0, :], vals[1, :], color="black", linestyle=linestyle1, linewidth=10)
    
    pre_collision_GTs = []
    recoveries = []
    post_collision_GTs = []

    for _ in range(n):
        
        pre_collision_GT, = ax.plot([], [], label=r"pre-collision", color=delft_blue_ddarker, linestyle=linestyle0, linewidth=10)
        recovery, = ax.plot([], [], label=r"recover", color=delft_blue_darker, linestyle=linestyle0, linewidth=10)
        post_collision_GT, = ax.plot([], [], label=r"post-collison", color=delft_blue, linestyle=linestyle0, linewidth=10)
        
        pre_collision_GTs += [pre_collision_GT]
        recoveries += [recovery]
        post_collision_GTs += [post_collision_GT]

    #ax.legend(loc='upper center', fancybox=True)

    return (fig, pre_collision_GTs, recoveries, post_collision_GTs)
    
def plot_topview(t, p, idx,
                 rec_pos, rec_sl, rec_rot,
                 orig_traj=None):
    (fig, pre_collision_GTs,
     recoveries, post_collision_GTs) = init_topview(t, p,
                                 rec_pos, rec_sl, rec_rot,
                                 orig_traj)
    
    n = getNumTrials(t)

    if n == 1:
        t = t.reshape([1, len(t)])
        p = p.reshape([1, p.shape[0], p.shape[1]])

   
    for i in range(n):        
        pre_collision_GTs[i].set_data(p[i, 0, 0:idx[0]], p[i, 1, 0:idx[0]])
        recoveries[i].set_data(p[i, 0, idx[0]-1:idx[1]], p[i, 1, idx[0]-1:idx[1]])
        post_collision_GTs[i].set_data(p[i, 0, idx[1]-1:idx[2]], p[i, 1, idx[1]-1:idx[2]])

    return fig

def anim_topview(t, p, idx,
                 rec_pos, rec_sl, rec_rot,
                 orig_traj=None):
    (fig, pre_collision_GTs,
     recoveries, post_collision_GTs) = init_topview(t, p,
                                 rec_pos, rec_sl, rec_rot,
                                 orig_traj)
    fig.set_size_inches((35, 25))
    
    n = getNumTrials(t)

    if n == 1:
        t = t.reshape([1, len(t)])
        p = p.reshape([1, p.shape[0], p.shape[1]])
   
    # Update function for animation
    def update(frame):
        for i in range(n): 
            if frame < idx[0]:
                pre_collision_GTs[i].set_data(p[i, 0, 0:frame], p[i, 1, 0:frame])
            elif frame < idx[1]:
                pre_collision_GTs[i].set_data(p[i, 0, 0:idx[0]], p[i, 1, 0:idx[0]])
                recoveries[i].set_data(p[i, 0, idx[0]-1:frame], p[i, 1, idx[0]-1:frame])
            elif frame > idx[1]:
                pre_collision_GTs[i].set_data(p[i, 0, 0:idx[0]], p[i, 1, 0:idx[0]])
                recoveries[i].set_data(p[i, 0, idx[0]-1:idx[1]], p[i, 1, idx[0]-1:idx[1]])
                post_collision_GTs[i].set_data(p[i, 0, idx[1]-1:frame], p[i, 1, idx[1]-1:frame])
            
        return (*pre_collision_GTs, *recoveries, *post_collision_GTs)

    # Run Animation
    return FuncAnimation(fig, update, frames=t.shape[1], interval=(max(t[:, -1]) - min(t[:, 0])) / t.shape[1] * 1e3, blit=True)

def init_pos_vel_plot(t, axis,
                p, p_dot,
                rec_pos, rec_sl, rec_rot):
    n = getNumTrials(t)

    # Find the ranges
    pos_range = (min(p.flatten()) - 1.5, max(p.flatten()) + 0.25)
    vel_range = (min(p_dot.flatten()) - 0.75, max(p_dot.flatten()) + 0.1)
    time_range = (t[0], t[-1])
    
    # Set up a figure with aspect ration 3:2
    fig = plt.figure(constrained_layout=True,
                     facecolor=dark_grey)
    fig.set_size_inches((35, 25))
    gs = GridSpec(2, 1, figure=fig)

    # Position Plot
    axPos = fig.add_subplot(gs[0, 0])
    axPos.set_xlabel("")
    axPos.set_ylim(pos_range)
    axPos.tick_params(labelbottom=False)
    axPos.plot(t, np.ones_like(t) * (rec_pos[axis] - 0.5*rec_sl[axis]), linewidth=10, color= "black", linestyle=linestyle2)
    axPos.set_ylabel(r"Position [m]", labelpad=25)
    axPos.set_facecolor(dark_grey)

    axPos.tick_params(axis='both', which='major', width=5, length=30,pad=35)

    # Velocity Plot
    axVel = fig.add_subplot(gs[1, 0], sharex=axPos)
    axVel.set_ylim(vel_range)
    axVel.set_yticks([-2, 0, 2, 4])
    axVel.set_ylabel(r"Velocity [m / s]", labelpad=25)
    axVel.set_xlabel(r"t [s]", labelpad=15)
    axVel.set_xlim(time_range)
    axVel.set_facecolor(dark_grey)
    axVel.tick_params(axis='both', which='major', width=5, length=30,pad=35)


    # change all spines
    for axis in ['top','bottom','left','right']:
        axPos.spines[axis].set_linewidth(10)
        axVel.spines[axis].set_linewidth(10)

    # increase tick width
    axPos.tick_params(width=10)
    axVel.tick_params(width=10)
    
    linesPos = []
    linesVel = []
    for _ in range(n):
        
        linePos, = axPos.plot([], [], label=r"Position", linewidth = 10, color=delft_blue, linestyle=linestyle0, alpha=1)
        lineVel, = axVel.plot([], [], label=r"Velocity", linewidth = 10, color=delft_blue, linestyle=linestyle1, alpha=1)
       
        linesPos += [linePos]
        linesVel += [lineVel]

    return (fig,
            linesPos, linesVel)

def plot_pos_vel(t, axis,
                p, p_dot,
                rec_pos, rec_sl, rec_rot):
    (fig,
    linesPos, linesVel) = init_pos_vel_plot(t, axis,
                                            p, p_dot,
                                            rec_pos, rec_sl, rec_rot)
    
    n = getNumTrials(t)

    if n == 1:
        t = t.reshape([1, len(t)])
        p = p.reshape([1, p.shape[0], p.shape[1]])
        p_dot = p_dot.reshape([1, p_dot.shape[0], p_dot.shape[1]])
    
    for i in range(n):
        
        linesPos[i].set_data(t[i, :], p[i, axis, :])
        linesVel[i].set_data(t[i, :], p_dot[i, axis, :])
    return fig

def anim_pos_vel_plot(t, axis,
                p, p_dot,
                rec_pos, rec_sl, rec_rot):
    
    (fig, linesPos, linesVel) = init_pos_vel_plot(t, axis,
                                            p, p_dot,
                                            rec_pos, rec_sl, rec_rot)
    
    n = getNumTrials(t)

    if n == 1:
        t = t.reshape([1, len(t)])
        p = p.reshape([1, p.shape[0], p.shape[1]])
        p_dot = p_dot.reshape([1, p_dot.shape[0], p_dot.shape[1]])
    
    # Update function for animation
    def update(frame):
        for i in range(n): 
            linesPos[i].set_data(t[i, :frame], p[i, axis, :frame])
            linesVel[i].set_data(t[i, :frame], p_dot[i, axis, :frame])
            
        return (*linesPos, *linesVel)

    # Run Animation
    return FuncAnimation(fig, update, frames=t.shape[1], interval=(max(t[:, -1]) - min(t[:, 0])) / t.shape[1] * 1e3, blit=True)


def add_pos_line(ax, t, y, alpha):
    ax.plot(t, y, alpha=alpha, color=delft_blue, linewidth=10, linestyle = linestyle0)

def add_vel_line(ax, t, y, alpha):
    ax.plot(t, y, alpha=alpha, color=delft_blue, linewidth=10, linestyle = linestyle1)

def init_timeplot(t, 
                p, p_dot, r,
                rec_pos, rec_sl, rec_rot,
                des_pos=None,
                orig_traj=None):
    n = getNumTrials(t)

    # Find the ranges
    pos_range = (min(p.flatten()) - 0.1, max(p.flatten()) + 0.1)
    vel_range = (min(p_dot.flatten()) - 0.1, max(p_dot.flatten()) + 0.1)
    att_range = (-180.0, 180.0)
    
    # Set up a figure with aspect ration 3:2
    fig = plt.figure(constrained_layout=True, figsize=7.5 * np.array([4, 2]))
    gs = GridSpec(3, 1, figure=fig)

    # Position Plot
    axPos = fig.add_subplot(gs[0, 0])
    axPos.set_xlabel("")
    axPos.set_ylim(pos_range)
    axPos.tick_params(labelbottom=False)
    axPos.set_ylabel(r"Position [m]")

    # Velocity Plot
    axVel = fig.add_subplot(gs[1, 0], sharex=axPos)
    axVel.set_ylim(vel_range)
    axVel.set_ylabel(r"Velocity [m / s]")
    axVel.tick_params(labelbottom=False)

    # Attitude Plot
    axAtt = fig.add_subplot(gs[2, 0], sharex=axPos)
    axAtt.set_ylim(att_range)
    axAtt.tick_params(axis='both', which='major', pad=15)
    axAtt.set_yticks([-180, 0, 180])
    axAtt.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    axAtt.set_ylabel(r"Attitude [$^\circ$]")
    axAtt.set_xlabel(r"t [s]")
    axAtt.set_xlim((min(t[0, :]), max(t[-1, :])))

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

    for i in range(n):
        
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

    if n == 1:
        axPos.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                      ncol=3, fancybox=True)
    else:
        axPos.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                      ncol=3, fancybox=True,
                      labels=[r"$x$", r"$y$", r"$z$"])

    return (fig, atts3d,
            linesPosX, linesPosY, linesPosZ,
            linesDesPosX, linesDesPosY, linesDesPosZ,
            linesVelX, linesVelY, linesVelZ,
            linesAttX, linesAttY, linesAttZ)

def plot_timeplot(t, 
                p, p_dot, r,
                rec_pos, rec_sl, rec_rot,
                des_pos=None,
                orig_traj=None):
    (fig, atts3d,
    linesPosX, linesPosY, linesPosZ,
    linesDesPosX, linesDesPosY, linesDesPosZ,
    linesVelX, linesVelY, linesVelZ,
    linesAttX, linesAttY, linesAttZ) = init_timeplot(t, 
                                                    p, p_dot, r,
                                                    rec_pos, rec_sl, rec_rot,
                                                    des_pos=None,
                                                    orig_traj=None)
    
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
    
    return fig
    

def init_3dplot(t, p,
                rec_pos, rec_sl, rec_rot,
                cyl_pos, cyl_dim, cyl_rot,
                orig_traj=None):
    n = getNumTrials(t)
    
    # Find the ranges
    pos_range = (min(p.flatten()), max(p.flatten()))
    range_3d = max(np.abs(pos_range))

     # Set up a figure with aspect ration 3:2
    fig = plt.figure(constrained_layout=True, figsize=7.5 * np.array([4, 2]))
    gs = GridSpec(12, 2, figure=fig)

    # 3D Plot
    ax3d = fig.add_subplot(gs[:, :], projection='3d')
    ax3d.tick_params(axis='both', which='major', pad=25) 
    ax3d.set_xlabel(r"$x$ [m]", labelpad=95)
    ax3d.set_ylabel(r"$y$ [m]", labelpad=95)
    ax3d.set_zlabel(r"$z$ [m]", labelpad=95)
    ax3d.set_xlim((-2, 2)) #(-range_3d , range_3d))
    ax3d.set_ylim((-2, 2)) #(-range_3d, range_3d))
    ax3d.set_zlim((-1, 3)) #(-0.5 * range_3d , 1.5 * range_3d))
    
    # Plot Rectoids
    for i in range(len(rec_pos)):
        plot_rectoid(ax3d, rec_pos[i], rec_sl[i], rec_rot[i],
                    facecolors='xkcd:grey', edgecolor="black", alpha=0.5)
        
    # Plot Cylinders
    for i in range(len(cyl_pos)):
        plot_cylinder(ax3d, cyl_pos[i], cyl_dim[i][0], cyl_dim[i][1],
                      cyl_rot[i],
                      facecolors='xkcd:grey', edgecolor="black", alpha=0.5)
    

    if orig_traj is not None:
        vals = orig_traj(np.linspace(0, 1, 100))
        ax3d.plot(vals[0, :], vals[1, :], vals[2, :],
                  color="black", linestyle=linestyle1, linewidth=5.0)
        
    ax3d.view_init(azim=-110, elev=30)



    lines3d = []
    atts3d = []
    for i in range(n):
        line3d, = ax3d.plot([], [], [],
                            linestyle=linestyle1, linewidth=1.5, color=delft_blue)
        if t.ndim == 1:
            att3d = ax3d.quiver(p[0, 0], p[1, 0], p[2, 0], 1, 0, 0,
                            length=0.3, color="black")
        else:
            att3d = ax3d.quiver(p[i, 0, 0], p[i, 1, 0], p[i, 2, 0], 1, 0, 0,
                            length=0.3, color="black")
            
        lines3d += [line3d]
        atts3d += [att3d]

    return (fig, ax3d, lines3d, atts3d)

def plot_3d(t, p, r,
            rec_pos=[], rec_sl=[], rec_rot=[],
            cyl_pos=[], cyl_dim=[], cyl_rot=[],
            orig_traj=None):
    (fig, ax3d, lines3d, atts3d) = init_3dplot(t, p,
                                               rec_pos, rec_sl, rec_rot,
                                               cyl_pos, cyl_dim, cyl_rot,
                                               orig_traj=orig_traj)
    
    n = getNumTrials(t)

    if n == 1:
        t = t.reshape([1, len(t)])
        p = p.reshape([1, p.shape[0], p.shape[1]])
        r = r.reshape([1, len(r)])
    

    for i in range(n): 
        dir_vec = np.array([r[i][j].as_matrix() @ np.array([0, 1, 0]) for j in range(len(r[i]))]).T
        
        lines3d[i].set_data(p[i, 0, :], p[i, 1, :])
        lines3d[i].set_3d_properties(p[i, 2, :])

        atts3d[i].remove()    

    ax3d.set_xticks([-1, 0, 1])
    ax3d.set_yticks([-1, 0, 1])
    ax3d.set_zticks([0, 1, 2])
    ax3d.set_xlim([-1.1, 1.1])
    ax3d.set_ylim([-1.1, 1.1])
    ax3d.set_zlim([0.3, 2.5])
    # Run Animation
    return fig

def anim_3d_plot(t,
                p,
                r,
                rec_pos, rec_sl, rec_rot):
    
    (fig, ax3d, lines3d, atts3d) = init_3dplot(t, p,
                                                rec_pos, rec_sl, rec_rot)
    
    n = getNumTrials(t)

    if n == 1:
        t = t.reshape([1, len(t)])
        p = p.reshape([1, p.shape[0], p.shape[1]])
        r = r.reshape([1, len(r)])
    
    # Update function for animation
    def update(frame):
        nonlocal atts3d
        for i in range(n): 
            dir_vec = np.array([r[i][j].as_matrix() @ np.array([0, 1, 0]) for j in range(len(r[i]))]).T
            
            lines3d[i].set_data(p[i, 0, :frame], p[i, 1, :frame])
            lines3d[i].set_3d_properties(p[i, 2, :frame])

            atts3d[i].remove()    
            atts3d[i] = ax3d.quiver(
                p[i, 0, max(0, frame-1)], p[i, 1, max(0, frame-1)], p[i, 2, max(0, frame-1)],
                dir_vec[0, max(0, frame-1)], dir_vec[1, max(0, frame-1)], dir_vec[2, max(0, frame-1)],
                length=0.3, color="black"
            )
            
        return (*lines3d, *atts3d)

    # Run Animation
    return FuncAnimation(fig, update, frames=t.shape[1], interval=(max(t[:, -1]) - min(t[:, 0])) / t.shape[1] * 1e3, blit=True)

def init_plot(t,
              p, p_dot,
              r, omega,
              contacts,
              rec_pos, rec_sl, rec_rot,
              des_pos=None,
              orig_traj=None):
    
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
    fig = plt.figure(constrained_layout=True, figsize=7.5 * np.array([4, 2]))
    gs = GridSpec(12, 2, figure=fig)

    # 3D Plot+ np.random.uniform(-sigma, sigma)
    ax3d = fig.add_subplot(gs[:-2, 0], projection='3d')
    ax3d.set_xlabel(r"$x$ [m]")
    ax3d.set_ylabel(r"$y$ [m]")
    ax3d.set_zlabel(r"$z$ [m]")
    ax3d.set_xlim((-range_3d, range_3d))
    ax3d.set_ylim((-range_3d, range_3d))
    ax3d.set_zlim((-0.5 * range_3d, 1.5 * range_3d))
    for i in range(len(rec_pos)):
        plot_rectoid(ax3d, rec_pos[i], rec_sl[i], rec_rot[i],
                     facecolors='xkcd:grey', edgecolor="black", alpha=0.5)
    if orig_traj is not None:
        vals = orig_traj(np.linspace(0, 1, 100))
        ax3d.plot(vals[0, :], vals[1, :], vals[2, :], color="black", linestyle=linestyle1)
        lim = max(np.abs([min(vals.flatten()), max(vals.flatten())])) + 0.5
        ax3d.set_xlim([-lim,lim])
        ax3d.set_ylim([-lim,lim])
        ax3d.set_zlim([-0.5 * lim,1.5 * lim])
        
    ax3d.view_init(elev=30, azim=55)

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
                    des_pos=None,
                    orig_traj=None):
    
    (fig, ax3d, lines3d, atts3d,
    linesPosX, linesPosY, linesPosZ,
    linesDesPosX, linesDesPosY, linesDesPosZ,
    linesVelX, linesVelY, linesVelZ,
    linesAttX, linesAttY, linesAttZ,
    linesRatesX, linesRatesY, linesRatesZ,
    linesContacts) = init_plot(
        t, p, p_dot, r, omega, contacts,
        rec_pos, rec_sl, rec_rot,
        des_pos, orig_traj)
    
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


def init_anim(t,
              p, p_dot,
              r, omega,
              contacts,
              rec_pos, rec_sl, rec_rot,
              orig_traj=None,
              des_pos=None):
    
    n = getNumTrials(t)

    # Find the ranges
    pos_range = (min(p.flatten()) - 0.1, max(p.flatten()) + 0.1)
    vel_range = (min(p_dot.flatten()) - 0.1, max(p_dot.flatten()) + 0.1)
    att_range = (-180.0, 180.0)
    range_3d = max(np.abs(pos_range))
    if t.ndim == 1:
        t_range = (t[0], t[-1])
    else:
        t_range = (min(t[:, 0]), max(t[:, -1]))
    
    # Set up a figure with aspect ration 3:2
    fig = plt.figure(constrained_layout=True,
                     facecolor=dark_grey)
    fig.set_size_inches((85, 40))

    gs = GridSpec(12, 2, figure=fig)

    # 3D Plot
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax3d.set_xlabel(r"$x$ [m]", labelpad=150)
    ax3d.set_ylabel(r"$y$ [m]", labelpad=150)
    ax3d.set_zlabel(r"$z$ [m]", labelpad=150)
    ax3d.set_xlim((-range_3d / 2, range_3d / 2))
    ax3d.set_ylim((-range_3d / 2, range_3d / 2))
    ax3d.set_zlim((-0.1 * range_3d, 0.9 * range_3d))
    plot_rectoid(ax3d, rec_pos, rec_sl, rec_rot,
                 facecolors='xkcd:grey', edgecolor="black", alpha=0.5)
    ax3d.view_init(elev=30, azim=65)
    ax3d.tick_params(axis='both', which='major', width=5, length=30, pad=50)
    ax3d.set_facecolor(dark_grey)

    if orig_traj is not None:
        vals = orig_traj(np.linspace(0, 1, 100))
        ax3d.plot(vals[0, :], vals[1, :], vals[2, :], color="black", linestyle=linestyle1, linewidth=10, zorder=100)

    # Position Plot
    axPos = fig.add_subplot(gs[0:3, 1])
    axPos.set_xlabel("")
    axPos.set_ylim(pos_range)
    axPos.tick_params(labelbottom=False)
    axPos.set_ylabel(r"Position [m]", labelpad=50)
    axPos.set_facecolor(dark_grey)
    axPos.tick_params(axis='both', which='major', width=5, length=30, pad=50)

    # Velocity Plot
    axVel = fig.add_subplot(gs[3:6, 1], sharex=axPos)
    axVel.set_ylim(vel_range)
    axVel.set_ylabel(r"Velocity [m / s]", labelpad=50)
    axVel.tick_params(labelbottom=False)
    axVel.set_facecolor(dark_grey)
    axVel.tick_params(axis='both', which='major', width=5, length=30, pad=50)

    # Attitude Plot
    axAtt = fig.add_subplot(gs[6:9, 1], sharex=axPos)
    axAtt.set_ylim(att_range)
    axAtt.set_yticks([-180, 0, 180])
    axAtt.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    axAtt.set_ylabel(r"Attitude [$^\circ$]", labelpad=50)
    axAtt.tick_params(labelbottom=False)
    axAtt.set_facecolor(dark_grey)
    axAtt.tick_params(axis='both', which='major', width=5, length=30, pad=50)

    # Contact Plot
    axContact = fig.add_subplot(gs[9:12, 1], sharex=axPos)
    axContact.set_xlim(t_range)
    axContact.set_xlabel(r"Time [s]", labelpad=50)
    axContact.set_ylabel("Contact", labelpad=50)
    axContact.set_yticks([i for i in range(14)])
    axContact.set_yticklabels([rf"$v_{{{i}}}$" for i in range(14)])
    axContact.set_ylim((0.5, 12.5))
    axContact.grid(axis='y')
    axContact.set_facecolor(dark_grey)
    axContact.tick_params(axis='both', which='major', width=5, length=30, pad=50)

    # change all spines
    for axis in ['top','bottom','left','right']:
        axPos.spines[axis].set_linewidth(10)
        axVel.spines[axis].set_linewidth(10)
        axAtt.spines[axis].set_linewidth(10)
        axContact.spines[axis].set_linewidth(10)

    # increase tick width
    axPos.tick_params(width=10)
    axVel.tick_params(width=10)
    axAtt.tick_params(width=10)
    axContact.tick_params(width=10)

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
    linesContacts = []

    for i in range(n):
        line3d, = ax3d.plot([], [], [], color=delft_blue, linewidth=10)
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
        contactLines = []
        for i in range(12):
            line,  = axContact.plot([], [],
                                    linestyle="", marker="o", color=color_contact,
                                    markersize=40)
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
        linesContacts += [contactLines]

    if n == 1:
        axPos.legend(loc='lower left', ncol=3)
        for legobj in axPos.get_legend().legend_handles:
            legobj.set_linewidth(10.0)

        axVel.legend(loc='lower left', ncol=3)
        for legobj in axVel.get_legend().legend_handles:
            legobj.set_linewidth(10.0)
        axAtt.legend(loc='lower left', ncol=3)
        for legobj in axAtt.get_legend().legend_handles:
            legobj.set_linewidth(10.0)
    else:
        axPos.legend(labels=[r"$x$", r"$y$", r"$z$"], loc='lower left', ncol=3)
        for legobj in axPos.get_legend().legend_handles:
            legobj.set_linewidth(10.0)
        axVel.legend(labels=[r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"], loc='lower left', ncol=3)
        for legobj in axVel.get_legend().legend_handles:
            legobj.set_linewidth(10.0)
        axAtt.legend(labels=[r"$\varphi$", r"$\theta$", r"$\psi$"], loc='lower left', ncol=3)
        for legobj in axAtt.get_legend().legend_handles:
            legobj.set_linewidth(10.0)

    return (fig, ax3d, lines3d, atts3d,
            linesPosX, linesPosY, linesPosZ,
            linesDesPosX, linesDesPosY, linesDesPosZ,
            linesVelX, linesVelY, linesVelZ,
            linesAttX, linesAttY, linesAttZ,
            linesContacts)

def animate_trajectory(t, p, p_dot, 
                       r, omega, contacts,
                       rec_pos, rec_sl, rec_rot,
                       orig_traj=None,
                       des_pos=None):
    
    (fig, ax3d, lines3d, atts3d,
    linesPosX, linesPosY, linesPosZ,
    linesDesPosX, linesDesPosY, linesDesPosZ,
    linesVelX, linesVelY, linesVelZ,
    linesAttX, linesAttY, linesAttZ,
    linesContacts) = init_anim(
        t, p, p_dot, r, omega, contacts,
        rec_pos, rec_sl, rec_rot,
        orig_traj=orig_traj)
    
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
            dir_vec = np.array([r[i][j].as_matrix() @ np.array([0, 1, 0]) for j in range(len(r[i]))]).T
            
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
            for j in range(12):
                linesContacts[i][j].set_data(t[i, :frame], contactsPlotData[:frame, j])
        return (lines3d[0], atts3d[0],
                linesPosX[0], linesPosY[0], linesPosZ[0],
                linesDesPosX[0], linesDesPosY[0], linesDesPosZ[0],
                linesVelX[0], linesVelY[0], linesVelZ[0],
                linesAttX[0], linesAttY[0], linesAttZ[0], 
                *linesContacts[0])

    # Run Animation
    return FuncAnimation(fig, update, frames=len(t[0]), interval=max(t[:, -1]) / len(t[0]) * 1e3, blit=True)

