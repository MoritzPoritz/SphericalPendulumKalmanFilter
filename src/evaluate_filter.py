import matplotlib
from sympy import S
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import utils 
import numpy as np


def evaluate_filter(ground_truth, kalman, crashed, crashed_at): 
    ground_truth_data = utils.read_from_csv(ground_truth+'.csv') # grounn truth
    kalman_data = utils.read_from_csv(kalman+'.csv') # kalman
    if ground_truth_data.empty is False and kalman_data.empty is False:
        #This only works for the measurement data filter gives variance for the state x = [theta, phi, dtheta, dphi]
        rx = ground_truth_data["position_x"] - kalman_data["position_x"]
        ry = ground_truth_data["position_y"] - kalman_data["position_y"]
        rz = ground_truth_data["position_z"] - kalman_data["position_z"]
        mse = (rx**2 + ry**2 + rz**2).mean()
        mse_db = np.log10(mse)*10

        var_theta = kalman_data['var_theta']
        var_phi = kalman_data['var_phi']
        var_dtheta = kalman_data['var_dtheta']
        var_dphi = kalman_data['var_dphi']
        meaned_variance = (var_theta + var_phi + var_dtheta + var_dphi).mean()

        meaned_real_std = []
        meaned_filter_std = []
        for i in range(len(ground_truth_data['timestep'])): 
            meaned_filter_std.append((np.sqrt(var_theta[i]) + np.sqrt(var_phi[i]) + np.sqrt(var_dtheta[i]) + np.sqrt(var_dphi[i]))/4)
            meaned_real_std.append(rx[i] + ry[i] + rz[i])
        
        meaned_filter_std = np.array(meaned_filter_std)
        meaned_filter_std_minus = meaned_filter_std*-1
        
        in_std_count = 0
        
        for i in range(len(meaned_filter_std)): 
            if (meaned_filter_std_minus[i] < meaned_real_std[i] < meaned_filter_std[i]): 
                in_std_count+=1
        error_in_place = in_std_count/len(meaned_filter_std)*100
        print("STD of values is in range: " + str(error_in_place) + " percent of the time")  

        return mse, mse_db, error_in_place, meaned_variance, crashed, crashed_at