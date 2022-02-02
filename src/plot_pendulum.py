import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import utils 
import numpy as np

import sys


def plot_pendulum(path):
    print(type(path))
    if (type(path) == str): 
        print("Single file")
        simulation = utils.read_from_csv(path+'.csv')

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
    else: 
        data1 = utils.read_from_csv(path[0]+'.csv')
        data2 = utils.read_from_csv(path[1]+'.csv')
        data3 = utils.read_from_csv(path[2]+'.csv')

        if data1.empty is False and data2.empty is False:
            m_position_x = np.array(data1["position_x"])
            m_position_y = np.array(data1["position_y"])
            m_position_z = np.array(data1["position_z"])
            
            k_position_x = np.array(data2["position_x"])
            k_position_y = np.array(data2["position_y"])
            k_position_z = np.array(data2["position_z"])

            s_position_x = np.array(data3["position_x"])
            s_position_y = np.array(data3["position_y"])
            s_position_z = np.array(data3["position_z"])
            

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            ax.plot(m_position_x, m_position_y, m_position_z, color='green', linewidth=1.0, label=path[0])
            ax.plot(k_position_x, k_position_y, k_position_z, color='red', linewidth=0.8, label=path[1])
            ax.plot(s_position_x, s_position_y, s_position_z, color='blue', linewidth=0.3, label=path[2])
            plt.legend()
            plt.show()
        else:
            print("No Simulation File found!")




if __name__ == "__main__":
    print(len(sys.argv))
    if (len(sys.argv) == 2):
        plot_pendulum(str(sys.argv[1]))
    elif (len(sys.argv) == 4): 
        datapath1 = sys.argv[1]
        datapath2 = sys.argv[2]
        datapath3 = sys.argv[3]
        plot_pendulum([str(datapath1), str(datapath2), str(datapath3)])
    else: 
        print('Type name of data file in following order')
        print('messy_data kalman_data')