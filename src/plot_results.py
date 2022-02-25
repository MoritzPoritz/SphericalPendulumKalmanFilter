import matplotlib
from sympy import S
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import utils 
import numpy as np

import sys


def difference_between_angles(correct, incorrect): 
    correct += np.pi
    incorrect += np.pi
    diff = (correct - incorrect) -np.pi
    return diff
        
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



    if tracking.empty is False and kalman.empty is False:
        # these values are given from the filter for each timestep
        var_theta = kalman['var_theta']
        var_phi = kalman['var_phi']
        var_dtheta = kalman['var_dtheta']
        var_dphi = kalman['var_dphi']
        # mse of variance from kalman filter
        kalman_mse = (var_theta + var_phi + var_dtheta + var_dphi).mean()

        #Error in cartisian
        rx = tracking["position_x"][:data_length] - kalman["position_x"]
        ry = tracking["position_y"][:data_length] - kalman["position_y"]
        rz = tracking["position_z"][:data_length] - kalman["position_z"]
        mse = (rx**2 + ry**2 + rz**2).mean()
        mse_db = np.log10(mse)*10

        # Error on angles
        rtheta = [difference_between_angles(a,b) for a,b in zip(tracking["theta"], kalman['theta'])]
        rphi = tracking["phi"] - kalman['phi']
       
    
        print(tracking["theta"][100],kalman['theta'][100],rtheta[100])


        filter_std_theta = np.array([np.sqrt(var_t) for var_t in var_theta])
        filter_std_phi = np.array([np.sqrt(var_p) for var_p in var_phi])
        minus_filter_std_theta = filter_std_theta*-1
        minus_filter_std_phi = filter_std_phi*-1
    
        # Evaluate how often the filter error is inside the std of the filter   
        std_count_theta = 0
        std_count_phi = 0

        meaned_filter_std = []
       
        
        for i in range(len(rphi)): 
            meaned_filter_std.append((np.sqrt(var_theta[i]) + np.sqrt(var_phi[i]) + np.sqrt(var_dtheta[i]) + np.sqrt(var_dphi[i]))/4)
            if (minus_filter_std_theta[i] < rtheta[i] < filter_std_theta[i]): 
                std_count_theta += 1
            if (minus_filter_std_phi[i] < rphi[i] < filter_std_phi[i]): 
                std_count_phi += 1    

          
        rphi_in_filter_std = round(std_count_phi/len(rphi)*100,2)
        rtheta_in_filter_std = round(std_count_theta/len(rtheta)*100,2)
        meaned_filter_std = np.array(meaned_filter_std)
        meaned_filter_std_minus = meaned_filter_std*-1
        
        print("MSE DB: ", mse_db)
        print("Phi in Filter STD: " + str(rphi_in_filter_std) + " % of the time")        
        print("Theta in Filter STD: " + str(rtheta_in_filter_std) + " % of the time")        
       

      
    # Plotting needs to be limited by the time frame, filter works (thats why on x axis data2['timestep'] (filter result) is used)
        plt.plot(kalman['timestep'],filter_std_theta, color="orange", label="Theta Std")
        plt.plot(kalman['timestep'],minus_filter_std_theta, color="orange",label="Theta Std")
        #plot sum of error on tracking data to filter data
        plt.plot(kalman['timestep'],rtheta, label="Summe der Fehler")
        #ax.plot((rx + ry + rz)/3, label="Sum of Errors")
        plt.legend()
        plt.show()


        # Plotting needs to be limited by the time frame, filter works (thats why on x axis data2['timestep'] (filter result) is used)
        plt.plot(kalman['timestep'],filter_std_phi, color="orange", label="Phi Std")
        plt.plot(kalman['timestep'],minus_filter_std_phi, color="orange",label="Phi Std")
        #plot sum of error on tracking data to filter data
        plt.plot(kalman['timestep'],rphi, label="Summe der Fehler")
        #ax.plot((rx + ry + rz)/3, label="Sum of Errors")
        plt.legend()
        plt.show()
      
        # Plotting needs to be limited by the time frame, filter works (thats why on x axis data2['timestep'] (filter result) is used)
        plt.plot(kalman['timestep'],meaned_filter_std, color="orange", label="Phi Std")
        plt.plot(kalman['timestep'],meaned_filter_std_minus, color="orange",label="Phi Std")
        #plot sum of error on tracking data to filter data
        plt.plot(kalman['timestep'],(rx + ry + rz)/3, label="Summe der Fehler")
        #ax.plot((rx + ry + rz)/3, label="Sum of Errors")
        plt.legend()
        plt.show()

         # Plotting needs to be limited by the time frame, filter works (thats why on x axis data2['timestep'] (filter result) is used)
        plt.plot(kalman['timestep'],tracking["position_x"],  label="Ground Truth X")
        plt.plot(kalman['timestep'],kalman["position_x"], label="Preditction X")
     
        plt.legend()
        plt.show()
         # Plotting needs to be limited by the time frame, filter works (thats why on x axis data2['timestep'] (filter result) is used)
        plt.plot(kalman['timestep'],tracking["position_y"],  label="Ground Truth Y")
        plt.plot(kalman['timestep'],kalman["position_y"], label="Preditction Y")
        plt.legend()
        plt.show()
         # Plotting needs to be limited by the time frame, filter works (thats why on x axis data2['timestep'] (filter result) is used)
        plt.plot(kalman['timestep'],tracking["position_z"],  label="Ground Truth Z")
        plt.plot(kalman['timestep'],kalman["position_z"], label="Preditction Z")
     
        plt.legend()
        plt.show()



       
    
    
    
    ax = plt.subplot(1,1,1, projection='3d')

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
            color='blue', linewidth=0.6, alpha=0.6, label="Kalman-Filter mit Daten vor Verdeckung")
        print(occStart, occEnd)
        ax.plot(k_position_x[occStart:occEnd], k_position_y[occStart:occEnd], k_position_z[occStart:occEnd],
            color='red', linewidth=0.9, label="Nur Kalman Vorhersage")

        ax.plot(k_position_x[occEnd:], k_position_y[occEnd:], k_position_z[occEnd:], 
            color='blue', linewidth=0.6, alpha=0.6, label="Kalman-Filter mit Daten nach Verdeckung")

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
    