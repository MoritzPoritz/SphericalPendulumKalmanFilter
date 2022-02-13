import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import utils 
import numpy as np

import sys


def plot_residual(path):
    data1 = utils.read_from_csv(path[0]+'.csv') # 
    data2 = utils.read_from_csv(path[1]+'.csv')
    if data1.empty is False and data2.empty is False:
        rx = data1["position_x"] - data2["position_x"]
        ry = data1["position_y"] - data2["position_y"]
        rz = data1["position_z"] - data2["position_z"]
        mse = (rx**2 + ry**2 + rz**2).mean()
        mse_db = np.log10(mse)*10


        #plt.plot(rx, label='Position x')
        #plt.plot(ry, label='Position y')
        #plt.plot(rz, label='Position z')

        plt.plot(rx + ry +rz, label="Sum of Errors")


        plt.legend()
        plt.title("Residual Plot (Error %.2fdB)"%mse_db)
        plt.xlabel("Time [s]")
        plt.ylabel("Residual [m]")
        
        plt.show()
    else:
        print("No Simulation File found!")




if __name__ == "__main__":
    print(len(sys.argv))
    if (len(sys.argv) == 3): 
        dataset1 = sys.argv[1]
        dataset2 = sys.argv[2]
        plot_residual([str(dataset1), str(dataset2)])
    else: 
        print('Type name of data file in following order')
        print('messy_data kalman_data')