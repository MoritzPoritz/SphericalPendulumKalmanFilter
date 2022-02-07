import matplotlib
from sympy import S
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import utils 
import numpy as np

import sys


def f(t):
    return np.cos(2*np.pi*t) * np.exp(-t)


def plot_results(path, occStart, occEnd):
# Set up a figure twice as tall as it is wide
    fig = plt.figure(figsize=plt.figaspect(1.5))
    fig.suptitle('A tale of 2 subplots')


    data1 = utils.read_from_csv(path[0]+'.csv')
    data2 = utils.read_from_csv(path[1]+'.csv')
    if data1.empty is False and data2.empty is False:
        rx = data1["position_x"] - data2["position_x"]
        ry = data1["position_y"] - data2["position_y"]
        rz = data1["position_z"] - data2["position_z"]
        mse = (rx**2 + ry**2 + rz**2).mean()
        mse_db = np.log10(mse)*10

        ax = fig.add_subplot(2, 1, 2)
        ax.plot(rx + ry +rz, label="Sum of Errors")
        ax.set_title("Residual Plot (Error %.2fdB)" %mse_db)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Residual [m]")
        ax.grid(True)



    # Second subplot
    ax = fig.add_subplot(2,1,1, projection='3d')

    data1 = utils.read_from_csv(path[0]+'.csv')
    data2 = utils.read_from_csv(path[1]+'.csv')
    data3 = utils.read_from_csv(path[2]+'.csv')

    if data1.empty is False and data2.empty is False:
        s_position_x = np.array(data1["position_x"])
        s_position_y = np.array(data1["position_y"])
        s_position_z = np.array(data1["position_z"])
        
        k_position_x = np.array(data2["position_x"])
        k_position_y = np.array(data2["position_y"])
        k_position_z = np.array(data2["position_z"])

        m_position_x = np.array(data3["position_x"])
        m_position_y = np.array(data3["position_y"])
        m_position_z = np.array(data3["position_z"])

        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.text(s_position_x[0], s_position_y[0], s_position_z[0], "P0", color="black", fontsize=8)
        ax.text(k_position_x[occStart], k_position_y[occStart], k_position_z[occStart], "OccStart", color="black", fontsize=8)

        ax.plot(s_position_x, s_position_y, s_position_z, color='green', linewidth=1, alpha=0.8, label=path[0])

        ax.plot(k_position_x[0:occStart], k_position_y[0:occStart], k_position_z[0:occStart], 
            color='blue', linewidth=0.6, alpha=0.6, label=path[1] + "with Data")

        ax.plot(k_position_x[occStart:occEnd], k_position_y[occStart:occEnd], k_position_z[occStart:occEnd],
            color='red', linewidth=0.9, label=path[1] + "Prediction only")

        ax.plot(k_position_x[occEnd:], k_position_y[occEnd:], k_position_z[occEnd:], 
            color='blue', linewidth=0.6, alpha=0.6, label=path[1] + "with Data")

        ax.plot(m_position_x, m_position_y, m_position_z, color='black', linewidth=0.3, alpha=0.6, label=path[2])
    plt.legend()
    plt.show()






if __name__ == "__main__":
    if (len(sys.argv) == 5): 
        datapath1 = sys.argv[1]
        datapath2 = sys.argv[2]
        datapath3 = sys.argv[3]
        s = int(sys.argv[4])
        e = int(sys.argv[5])
        plot_results([str(datapath1), str(datapath2), str(datapath3)], s, e)
    else: 
        print('Type name of data file in following order')
        print('messy_data kalman_data')