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


def plot_simulation_and_kalman(simulation, kalman, messy, occStart, occEnd):
    # Set up a figure twice as tall as it is wide
    fig = plt.figure(figsize=plt.figaspect(1.5))
    fig.suptitle('A tale of 2 subplots')

    if simulation.empty is False and kalman.empty is False:
        # these values are given from the filter for each timestep
        var_theta = kalman['var_theta']
        var_phi = kalman['var_phi']
        var_dtheta = kalman['var_dtheta']
        var_dphi = kalman['var_dphi']
        # mse of variance from kalman filter
        kalman_mse = (var_theta + var_phi + var_dtheta + var_dphi).mean()

        #This only works for the measurement data filter gives variance for the state x = [theta, phi, dtheta, dphi]
        rx = simulation["position_x"] - kalman["position_x"]
        ry = simulation["position_y"] - kalman["position_y"]
        rz = simulation["position_z"] - kalman["position_z"]
        mse = (rx**2 + ry**2 + rz**2).mean()
        mse_db = np.log10(mse)*10

        # Store angles of pendulum based on tracking data
        theta = []
        phi = []
                
        meaned_filter_std = []
        meaned_real_std = []
 
        for i in range(len(kalman['timestep'])): 
            # Get mean std and error values for state (given by filter) and measurement data (given by filter)
            meaned_filter_std.append((np.sqrt(var_theta[i]) + np.sqrt(var_phi[i]) + np.sqrt(var_dtheta[i]) + np.sqrt(var_dphi[i]))/4)
            meaned_real_std.append(rx[i] + ry[i] + rz[i])
            # calculate angles from tracking data for plotting
            t,p = utils.cartesian_to_polar(simulation["position_x"][i], simulation['position_y'][i], simulation['position_z'][i])
            theta.append(t)
            phi.append(p)

        # convert angles in numpy arrays
        theta = np.array(theta)
        phi = np.array(phi)
        # prepare std for plotting
        meaned_filter_std = np.array(meaned_filter_std)
        meaned_filter_std_minus = meaned_filter_std*-1
        
        in_std_count = 0

        # Evaluate how often the filter error is inside the std of the filter        
        for i in range(len(meaned_filter_std)): 
            if (meaned_filter_std_minus[i] < meaned_real_std[i] < meaned_filter_std[i]): 
                in_std_count+=1
        print("STD of values is in range: " + str(in_std_count/len(meaned_filter_std)*100) + " of the time")        

    
        #kalman_mse = (kalman_variances[0] + kalman_variances[1] + kalman_variances[2] + kalman_variances[3])
        #kalman_mse_db = np.log10(kalman_mse)*10
        #print("Kalman MSE: Error %.20f " %kalman_mse)
        print("Measurement MSE: Error %.20f " %mse)

        ax = fig.add_subplot(2, 1, 2)
        # Plotting needs to be limited by the time frame, filter works (thats why on x axis kalman['timestep'] (filter result) is used)
        ax.plot(kalman['timestep'],meaned_filter_std, color="orange", label="Std")
        ax.plot(kalman['timestep'],meaned_filter_std_minus, color="orange",label="Std")
        #plot sum of error on tracking data to filter data
        ax.plot(kalman['timestep'],(rx + ry + rz)[0:len(kalman['timestep'])], label="Sum of Errors")
        #ax.plot((rx + ry + rz)/3, label="Sum of Errors")
        
        # Additionally plot angles theta and phi
        ax2 = ax.twinx()
        ax2.set_label("Theta")
        ax2.plot(kalman['timestep'],theta, color="red", label="theta")
        ax2.plot(kalman['timestep'],phi, color="purple", label="phi")
        ax2.legend()
        ax2.tick_params(axis="y")
        
        # plot residual
        ax.legend()
        ax.set_title("Residual Plot (Error %.2fdB)" %mse_db)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Residual [m]")
        ax.set_ylim(-.25,.25)
        ax.grid(True)
    # Second subplot
    ax = fig.add_subplot(2,1,1, projection='3d')

    simulation = utils.read_from_csv(path[0]+'.csv')
    kalman = utils.read_from_csv(path[1]+'.csv')
    messy = utils.read_from_csv(path[2]+'.csv')

    if simulation.empty is False and kalman.empty is False:
        s_position_x = np.array(simulation["position_x"])
        s_position_y = np.array(simulation["position_y"])
        s_position_z = np.array(simulation["position_z"])
        
        k_position_x = np.array(kalman["position_x"])
        k_position_y = np.array(kalman["position_y"])
        k_position_z = np.array(kalman["position_z"])
       
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.text(s_position_x[0], s_position_y[0], s_position_z[0], "P0", color="black", fontsize=8)
        ax.text(k_position_x[occStart], k_position_y[occStart], k_position_z[occStart], "OccStart", color="black", fontsize=8)

        ax.plot(s_position_x[0:len(k_position_x)-1], s_position_y[0:len(k_position_y)-1], s_position_z[0:len(k_position_z)-1], color='green', linewidth=1, alpha=0.8, label=path[0])

        ax.plot(k_position_x[0:occStart], k_position_y[0:occStart], k_position_z[0:occStart], 
            color='blue', linewidth=0.6, alpha=0.6, label=path[1] + "with Data")
        print(occStart, occEnd)
        ax.plot(k_position_x[occStart:occEnd], k_position_y[occStart:occEnd], k_position_z[occStart:occEnd],
            color='red', linewidth=0.9, label=path[1] + "Prediction only")

        ax.plot(k_position_x[occEnd:], k_position_y[occEnd:], k_position_z[occEnd:], 
            color='blue', linewidth=0.6, alpha=0.6, label=path[1] + "with Data")

        # only plot 3d dataset if given
        if (messy.empty is False):            
            m_position_x = np.array(messy["position_x"])
            m_position_y = np.array(messy["position_y"])
            m_position_z = np.array(messy["position_z"])
            ax.plot(m_position_x, m_position_y, m_position_z, color='black', linewidth=0.3, alpha=0.6, label=path[2])

    plt.legend()
    plt.show()


def plot_tracking_and_kalman(tracking, kalman, occStart, occEnd): 
    '''
    First Subplot
    Show Errors, STD and the State angles
    '''
    data_length = kalman['timestep'].shape[0]

    fig = plt.figure(figsize=plt.figaspect(1.5))
    fig.suptitle('A tale of 2 subplots')

    if tracking.empty is False and kalman.empty is False:
        # these values are given from the filter for each timestep
        var_theta = kalman['var_theta']
        var_phi = kalman['var_phi']
        var_dtheta = kalman['var_dtheta']
        var_dphi = kalman['var_dphi']
        # mse of variance from kalman filter
        kalman_mse = (var_theta + var_phi + var_dtheta + var_dphi).mean()

        #This only works for the measurement data filter gives variance for the state x = [theta, phi, dtheta, dphi]
        rx = tracking["position_x"][:data_length] - kalman["position_x"]
        ry = tracking["position_y"][:data_length] - kalman["position_y"]
        rz = tracking["position_z"][:data_length] - kalman["position_z"]
        mse = (rx**2 + ry**2 + rz**2).mean()
        mse_db = np.log10(mse)*10

        # Store angles of pendulum based on tracking data
        theta = []
        phi = []
                
        meaned_filter_std = []
        meaned_real_std = []
 
        for i in range(len(kalman['timestep'])): 
            # Get mean std and error values for state (given by filter) and measurement data (given by filter)
            meaned_filter_std.append((np.sqrt(var_theta[i]) + np.sqrt(var_phi[i]) + np.sqrt(var_dtheta[i]) + np.sqrt(var_dphi[i]))/4)
            meaned_real_std.append(rx[i] + ry[i] + rz[i])
            # calculate angles from tracking data for plotting
            t,p = utils.cartesian_to_polar(tracking["position_x"][i], tracking['position_y'][i], tracking['position_z'][i])
            theta.append(t)
            phi.append(p)

        # convert angles in numpy arrays
        theta = np.array(theta)
        phi = np.array(phi)
        # prepare std for plotting
        meaned_filter_std = np.array(meaned_filter_std)
        meaned_filter_std_minus = meaned_filter_std*-1
        
        in_std_count = 0

        # Evaluate how often the filter error is inside the std of the filter        
        for i in range(len(meaned_filter_std)): 
            if (meaned_filter_std_minus[i] < meaned_real_std[i] < meaned_filter_std[i]): 
                in_std_count+=1
        print("STD of values is in range: " + str(in_std_count/len(meaned_filter_std)*100) + " of the time")        

        # Print kalman mse and data mse    
        #kalman_mse_db = np.log10(kalman_mse)*10
        print("Kalman MSE: Error %.20f " %kalman_mse)
        print("Measurement MSE: Error %.20f " %mse)

        ax = fig.add_subplot(2, 1, 2)
        # Plotting needs to be limited by the time frame, filter works (thats why on x axis data2['timestep'] (filter result) is used)
        ax.plot(kalman['timestep'],meaned_filter_std, color="orange", label="Std")
        ax.plot(kalman['timestep'],meaned_filter_std_minus, color="orange",label="Std")
        #plot sum of error on tracking data to filter data
        ax.plot(kalman['timestep'],(rx + ry + rz), label="Sum of Errors")
        #ax.plot((rx + ry + rz)/3, label="Sum of Errors")
        
        # Additionally plot angles theta and phi
        #ax2 = ax.twinx()
        #ax2.set_label("Theta")
        #ax2.plot(kalman['timestep'],theta, color="red", label="theta")
        #ax2.plot(kalman['timestep'],phi, color="purple", label="phi")
        #ax2.legend()
        #ax2.tick_params(axis="y")
        
        # plot residual
        ax.legend()
        ax.set_title("Residual Plot (Error %.2fdB)" %mse_db)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Residual [m]")
        ax.set_ylim(-.25,.25)
        ax.grid(True)
    
    '''
    Second Subplot: 
    Showing the path of the pendulum
    '''
    ax = fig.add_subplot(2,1,1, projection='3d')

    if tracking.empty is False and kalman.empty is False:
        t_position_x = np.array(tracking["position_x"])
        t_position_y = np.array(tracking["position_y"])
        t_position_z = np.array(tracking["position_z"])
        
        k_position_x = np.array(kalman["position_x"])
        k_position_y = np.array(kalman["position_y"])
        k_position_z = np.array(kalman["position_z"])
       
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.text(t_position_x[0], t_position_y[0], t_position_z[0], "P0", color="black", fontsize=8)
        ax.text(k_position_x[occStart], k_position_y[occStart], k_position_z[occStart], "OccStart", color="black", fontsize=8)

        ax.plot(t_position_x[0:len(k_position_x)-1], t_position_y[0:len(k_position_y)-1], t_position_z[0:len(k_position_z)-1], color='green', linewidth=1, alpha=0.8, label='Tracking')

        ax.plot(k_position_x[0:occStart], k_position_y[0:occStart], k_position_z[0:occStart], 
            color='blue', linewidth=0.6, alpha=0.6, label="Kalman Filter with data before occlusion")
        print(occStart, occEnd)
        ax.plot(k_position_x[occStart:occEnd], k_position_y[occStart:occEnd], k_position_z[occStart:occEnd],
            color='red', linewidth=0.9, label="Kalman Prediction only")

        ax.plot(k_position_x[occEnd:], k_position_y[occEnd:], k_position_z[occEnd:], 
            color='blue', linewidth=0.6, alpha=0.6, label="Kalman Filter with data after occlusion")

    plt.legend()
    plt.show()





def plot_results(path, occStart, occEnd):
    if (len(path) == 2): 
        print("Plotting Tracking Data and Kalman Result")
        ground_truth = utils.read_from_csv(path[0]+'.csv') # grounn truth
        kalman = utils.read_from_csv(path[1]+'.csv') # kalman
        plot_tracking_and_kalman(ground_truth, kalman, occStart, occEnd)
    elif (len(path) == 3): 
        print("Plotting Tracking Data and Kalman Result")
        ground_truth = utils.read_from_csv(path[0]+'.csv') # grounn truth
        kalman = utils.read_from_csv(path[1]+'.csv') # kalman
        messy = utils.read_from_csv(path[2]+'.csv')
        plot_simulation_and_kalman(ground_truth, kalman, messy, occStart, occEnd)


if __name__ == "__main__":
    if (len(sys.argv) == 6): 
        datapath1 = sys.argv[1]
        datapath2 = sys.argv[2]
        datapath3 = sys.argv[3]
        occStart = int(sys.argv[4])
        occEnd = int(sys.argv[5])
        plot_results([str(datapath1), str(datapath2), str(datapath3)], occStart, occEnd)
    elif(len(sys.argv) == 5): 
        datapath1 = sys.argv[1]
        datapath2 = sys.argv[2]
        occStart = int(sys.argv[3])
        occEnd = int(sys.argv[4])
        plot_results([str(datapath1), str(datapath2)], occStart, occEnd)
    