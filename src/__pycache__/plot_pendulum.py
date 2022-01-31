import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import utils 
import numpy as np


def plot_pendulum():
    simulation = utils.read_from_csv('simulation.csv')

    if simulation.empty is False:
        position_x = np.array(simulation["position_x"])
        position_y = np.array(simulation["position_y"])
        position_z = np.array(simulation["position_z"])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        #ax.plot(positions_decreased[:,0], positions_decreased[:,1], positions_decreased[:,2], linewidth=0.6)
        ax.plot(position_x, position_y, position_z, color='red', linewidth=0.6)
        plt.show()
    else:
        print("No Simulation File found!")


if __name__ == "__main__":
    plot_pendulum()