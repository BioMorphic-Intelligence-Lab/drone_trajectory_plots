import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from plotting_functions import plot_topview, plot_pos_vel, add_pos_line, add_vel_line

from pathlib import Path
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from rosbags.rosbag1 import Reader


def guess_msgtype(path: Path) -> str:
    """Guess message type name from path."""
    name = path.relative_to(path.parents[2]).with_suffix('')
    if 'msg' not in name.parts:
        name = name.parent / 'msg' / name.name
    return str(name)


def rosbag2data(path: str):

    ############## Register non-standard msg types ##############
    typestore = get_typestore(Stores.ROS1_NOETIC)
    add_types = {}

    for pathstr in [
        '/home/antbre/projects/colliding-drone/tensegrity_flightcontroller/GeneralCode/ROS/hiperlab_hardware/msg/PoseEulerStamped.msg',
        '/home/antbre/projects/colliding-drone/tensegrity_flightcontroller/GeneralCode/ROS/hiperlab_rostools/msg/estimator_output.msg',
        '/home/antbre/projects/colliding-drone/tensegrity_flightcontroller/GeneralCode/ROS/hiperlab_rostools/msg/estimator_output.msg',
        '/home/antbre/projects/colliding-drone/tensegrity_flightcontroller/GeneralCode/ROS/hiperlab_rostools/msg/mocap_output.msg',
        '/home/antbre/projects/colliding-drone/tensegrity_flightcontroller/GeneralCode/ROS/hiperlab_rostools/msg/radio_command.msg',
        '/home/antbre/projects/colliding-drone/tensegrity_flightcontroller/GeneralCode/ROS/hiperlab_rostools/msg/simulator_truth.msg',
        '/home/antbre/projects/colliding-drone/tensegrity_flightcontroller/GeneralCode/ROS/hiperlab_rostools/msg/telemetry.msg',
    ]:
        msgpath = Path(pathstr)
        msgdef = msgpath.read_text(encoding='utf-8')
        add_types.update(get_types_from_msg(msgdef, guess_msgtype(msgpath)))

    typestore.register(add_types)

    ##############################################################
    ############## Load all the data #############################
    ##############################################################

    t_estimator = []
    estimator_p = []
    estimator_pdot = []
    estimator_att = []

    t_contact = []
    contact = []

    t_motorForces = []
    motorForces = []

    # Create reader instance and open for reading.
    with Reader(path) as reader:
        # Iterate over messages.
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/estimator25': #'/mocap_output25':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                t_estimator += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                estimator_p += [[msg.posx, msg.posy, msg.posz]]
                estimator_pdot += [[msg.velx, msg.vely, msg.velz]]
                estimator_att += [Rotation.from_quat([
                    msg.attq0, msg.attq1, msg.attq2, msg.attq3 
                ], scalar_first=True)]

            if connection.topic == '/telemetry25':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                t_contact += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]

                c = [bool((msg.customPacket1[0] >> i) & 0b1) if i < 8
                             else  bool((msg.customPacket1[1] >> (i-8)) & 0b1)
                             for i in range(12)]
                
                contact += [c]  

                t_motorForces += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                motorForces += [[msg.motorForces[0], msg.motorForces[1],
                                 msg.motorForces[2], msg.motorForces[3]]]          
    
    t_start = min(np.concatenate((t_estimator, t_contact, t_motorForces)))
    t_estimator = np.array(t_estimator) - t_start
    t_contact = np.array(t_contact) - t_start
    t_motorForces = np.array(t_motorForces) - t_start

    estimator_p = np.array(estimator_p)
    estimator_pdot = np.array(estimator_pdot)
    estimator_att = np.array(estimator_att)
    contact = np.array(contact)
    motorForces = np.array(motorForces)

    cutoff = 50

    return (t_estimator[cutoff: -cutoff], estimator_p[cutoff:-cutoff, :], estimator_pdot[cutoff:-cutoff, :], estimator_att[cutoff:-cutoff],
            t_contact[cutoff:-cutoff], contact[cutoff:-cutoff, :],
            t_motorForces[cutoff:-cutoff], motorForces[cutoff:-cutoff, :]) 


paths = [
    #"/home/antbre/projects/colliding-drone/rosbags/real_collisions/29-08-24/2024-08-29-21-46-03-replanMegaSuccess.bag",
    "/home/antbre/projects/colliding-drone/rosbags/real_collisions/29-08-24/2024-08-29-22-13-37-2msCollisionSuccess.bag",
    "/home/antbre/projects/colliding-drone/rosbags/real_collisions/29-08-24/2024-08-29-22-24-00-2p5msCollisionSuccess.bag",
    "/home/antbre/projects/colliding-drone/rosbags/real_collisions/29-08-24/2024-08-29-22-44-12-2p75msCollisionSuccess.bag",
    "/home/antbre/projects/colliding-drone/rosbags/real_collisions/30-08-24/2024-08-30-17-33-30-3ms-collisionSuccess.bag",
    "/home/antbre/projects/colliding-drone/rosbags/real_collisions/30-08-24/2024-08-30-18-41-16-3p25msCollisionSuccess.bag"
    ]

(t_estimator, estimator_p, estimator_pdot, estimator_att,
    t_contact, contacts,
    t_motorForces, motorForces)  = rosbag2data(paths[0])

t_estimator = t_estimator - t_estimator[0]
fig = plot_pos_vel(t_estimator, 1, 
                   estimator_p.T, estimator_pdot.T,
                   np.array([0.0, 3.4, 1.75]), np.array([0.6, 0.6, 0.6]), Rotation.identity())
fig.set_size_inches((30, 20))

for idx, path in enumerate(paths[1:]):
    (t_estimator, estimator_p, estimator_pdot, estimator_att,
    t_contact, contacts,
    t_motorForces, motorForces)  = rosbag2data(path)

    t_estimator = t_estimator-t_estimator[0]
    
    if idx == 3:
        add_pos_line(fig.axes[0], t_estimator, estimator_p[:,1], alpha=1.0)
        add_vel_line(fig.axes[1], t_estimator, estimator_pdot[:,1], alpha=1.0)
    else: 
        add_pos_line(fig.axes[0], t_estimator, estimator_p[:,1], alpha=0.5)
        add_vel_line(fig.axes[1], t_estimator, estimator_pdot[:,1], alpha=0.5)
    
# Find first contact index
cIdx = -1 #np.argmin(np.any(contacts, axis=0))
for i in range(contacts.shape[0]):
    if contacts[i, :].any():
        cIdx = i
        break

# Find the corresponding index in mocap data
tIdx = np.argmin((t_estimator - t_contact[cIdx])**2) 
print(t_contact[cIdx])
print(tIdx)

tIdx = 220

orig_traj = lambda t: np.array([0.75 * np.sin(2*np.pi * t),
                                            2.0* np.cos(2*np.pi * t),
                                            1.75 * np.ones_like(t)])


fig.savefig("collisionRecovery.pdf", dpi=300, bbox_inches="tight", transparent=True)

#fig = plot_topview(t_estimator[::5], estimator_p[::5, :].T, [tIdx, tIdx + 30, -1],
#                   np.array([-0.7, -0.25, 1.75]), np.array([0.6, 0.6, 0.6]), Rotation.identity(),
#                   orig_traj=orig_traj)
#fig.savefig("topViewReal.svg", dpi=300, bbox_inches="tight", transparent=True)
