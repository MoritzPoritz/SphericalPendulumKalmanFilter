import kalman
import plot_results
import plot_residual
import spherical_pendulum_sim
import mess_sim_data
import os
from evaluate_filter import evaluate_filter
import utils
import numpy as np

# Standardabweichung des auf die Simdaten aufgebrachten Rauschens
std_x, std_y, std_z = 0.05, 0.05, 0.05

# An den Filter übergebene Standardabweichung. Sollte im Realfall die Sensorstandardabweichung sein
kstd_x, kstd_y, kstd_z = 0.02, 0.02, 0.02

occStart = 1000
occEnd = 10000
occStep = 500

iteration_count = int((occEnd - occStart) / occStep) 


Q_var = 0.5
csv_name = "CleanedOptitrack_3_perfect_flower"

def run_project(shall_plot, occStart, occEnd, csv_name, Q_var):

    #print("Starting Sim!")
    #spherical_pendulum_sim.runSim()
    #print("Generating Noisy Sensor Data")
    #mess_sim_data.mess_sim_data(std_x, std_y, std_z)
    #print("Filtering Data")
    kalman_filter, filename = kalman.runFilter(kstd_x, kstd_y, kstd_z, occStart, occEnd, csv_name, Q_var)
    #print("Filtered. Plotting:")
    if (shall_plot):
        results = plot_results.plot_results(["simulation", "kalman", "messy"], occStart, occEnd, kalman_filter.variances)
    return evaluate_filter(csv_name, filename)

#spherical_pendulum_sim.runSim()
simulation_results = []
for i in range(iteration_count): 
    mse, mse_db, error_in_place, meaned_kalman_var = run_project(False, occStart, occStart + occStep*i, csv_name, Q_var)
    #mse, mse_db, error_in_place = run_project(True, occStart, occStart + occEnd)
    simulation_results.append([i, mse, mse_db, error_in_place, occStart, occStart + occStep*i,(occStart + occStep*i) - occStart, Q_var, meaned_kalman_var])
    #simulation_results.append([i, mse, mse_db, error_in_place, occStart, occEnd,500])
simulation_results = np.array(simulation_results)
utils.write_evaluation_to_csv(simulation_results[:,0], simulation_results[:,1], simulation_results[:,2], simulation_results[:,3], simulation_results[:,4], simulation_results[:,5], simulation_results[:,6], simulation_results[:,7],simulation_results[:,8])
