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

def plot_results(path, occStart, occEnd, kalman_variances):
# Set up a figure twice as tall as it is wide
    fig = plt.figure(figsize=plt.figaspect(1.5))
    fig.suptitle('A tale of 2 subplots')

    data1 = utils.read_from_csv(path[0]+'.csv') # grounn truth
    data2 = utils.read_from_csv(path[1]+'.csv') # kalman
    if data1.empty is False and data2.empty is False:
        var_theta = data2['var_theta']
        var_phi = data2['var_phi']
        var_dtheta = data2['var_dtheta']
        var_dphi = data2['var_dphi']
        kalman_mse = (var_theta + var_phi + var_dtheta + var_dphi).mean()

        meaned_filter_std = []
        
        #This only works for the measurement data filter gives variance for the state x = [theta, phi, dtheta, dphi]
        rx = data1["position_x"] - data2["position_x"]
        ry = data1["position_y"] - data2["position_y"]
        rz = data1["position_z"] - data2["position_z"]
        mse = (rx**2 + ry**2 + rz**2).mean()
        mse_db = np.log10(mse)*10
        
        rtheta = data1["theta"] - data2["theta"]
        rphi = data1["phi"] - data2["phi"]
        rdtheta = data1["dtheta"] - data2["dtheta"]
        rdphi = data1["dphi"] - data2["dphi"]
        # This is not necessary as the optitrack gives us also cartesian coordinates
        #state_mse = (rtheta**2 + rphi**2 +rdtheta**2 + rdphi**2).mean()
        #state_mse_db = np.log10(state_mse)*10
        meaned_real_std = []
        for i in range(len(data1['timestep'])): 
            meaned_filter_std.append((np.sqrt(var_theta[i]) + np.sqrt(var_phi[i]) + np.sqrt(var_dtheta[i]) + np.sqrt(var_dphi[i]))/4)
            meaned_real_std.append(rx[i] + ry[i] + rz[i])
        
        meaned_filter_std = np.array(meaned_filter_std)
        print(meaned_filter_std.shape)

        meaned_filter_std_minus = meaned_filter_std*-1
        
        in_std_count = 0
        
        for i in range(len(meaned_filter_std)): 
            if (meaned_filter_std_minus[i] < meaned_real_std[i] < meaned_filter_std[i]): 
                in_std_count+=1
        print("STD of values is in range: " + str(in_std_count/len(meaned_filter_std)*100) + " of the time")        
        # Hier: Anteil der Samples, bei denen das Reale Rauschen in den Grenzen des Filter liegt (vgl. real state error < kalman state error)

    
        #kalman_mse = (kalman_variances[0] + kalman_variances[1] + kalman_variances[2] + kalman_variances[3])
        #kalman_mse_db = np.log10(kalman_mse)*10
        print("Kalman MSE: Error %.20f " %kalman_mse)
        print("Measurement MSE: Error %.20f " %mse)
   
        
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(meaned_filter_std, color="orange", label="Std")
        ax.plot(meaned_filter_std_minus, color="orange",label="Std")
        
        ax.plot(rx + ry + rz, label="Sum of Errors")
        #ax.plot((rx + ry + rz)/3, label="Sum of Errors")
        
        #ax.plot(var_theta, label="Variance Theta")
        #ax.plot(var_phi, label="Variance Phi")
        #ax.plot(var_dtheta, label="Variance DTheta")
        #ax.plot(var_dphi, label="Variance DPhi")
        ax.legend()
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
        print(occStart, occEnd)
        ax.plot(k_position_x[occStart:occEnd], k_position_y[occStart:occEnd], k_position_z[occStart:occEnd],
            color='red', linewidth=0.9, label=path[1] + "Prediction only")

        ax.plot(k_position_x[occEnd:], k_position_y[occEnd:], k_position_z[occEnd:], 
            color='blue', linewidth=0.6, alpha=0.6, label=path[1] + "with Data")
        ax.plot(m_position_x, m_position_y, m_position_z, color='black', linewidth=0.3, alpha=0.6, label=path[2])
    plt.legend()
    plt.show()
    






if __name__ == "__main__":
    if (len(sys.argv) == 6): 
        datapath1 = sys.argv[1]
        datapath2 = sys.argv[2]
        datapath3 = sys.argv[3]
        s = int(sys.argv[4])
        e = int(sys.argv[5])
        plot_results([str(datapath1), str(datapath2), str(datapath3)], s, e)
    else: 
        print('Type name of data file in following order')
        print('simulation_data messy_data kalman_data')