import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import numpy as np


def plot_pendulum(positions):
    positions = np.array(positions)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    #ax.plot(positions_decreased[:,0], positions_decreased[:,1], positions_decreased[:,2], linewidth=0.6)
    ax.plot(positions[:,0], positions[:,1], positions[:,2], color='red', linewidth=0.6)
    plt.show()