import kalman
import plot_results
import plot_residual
import spherical_pendulum_sim
import mess_sim_data
import os
from evaluate_filter import evaluate_filter
import utils
import numpy as np
import sys
# Standardabweichung des auf die Simdaten aufgebrachten Rauschens
std_x, std_y, std_z = 0.05, 0.05, 0.05

# An den Filter übergebene Standardabweichung. Sollte im Realfall die Sensorstandardabweichung sein
# First Evaluation: kstd_x, kstd_y, kstd_z = 0.02, 0.02, 0.02
#Second Evaluation
kstd_x, kstd_y, kstd_z = 0.0002, 0.0002, 0.0002
occStart = 1000
occEnd = 10000
occStep = 300

iteration_count = int((occEnd - occStart) / occStep) 

Q_var = 0.05
csv_name = "CleanedOptitrack_3_perfect_flower"
#csv_name = "CleanedOptitrack_1_Shorter"
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
    return evaluate_filter(csv_name, filename, kalman_filter)


def run_mse_evaluation():
    #spherical_pendulum_sim.runSim()
    simulation_results = []
    for i in range(iteration_count): 
        mse, mse_db, error_in_place, meaned_kalman_var, crashed, crashed_at = run_project(False, occStart, occStart + occStep*i, csv_name, Q_var)
        #mse, mse_db, error_in_place = run_project(True, occStart, occStart + occEnd)
        simulation_results.append([i, mse, mse_db, error_in_place, occStart, occStart + occStep*i,(occStart + occStep*i) - occStart, Q_var, meaned_kalman_var, crashed, crashed_at])
        #simulation_results.append([i, mse, mse_db, error_in_place, occStart, occEnd,500])
    simulation_results = np.array(simulation_results)
    utils.write_evaluation_to_csv(simulation_results[:,0], simulation_results[:,1], simulation_results[:,2], simulation_results[:,3], simulation_results[:,4], simulation_results[:,5], simulation_results[:,6], simulation_results[:,7],simulation_results[:,8],simulation_results[:,9], simulation_results[:,10])

def run_Q_evaluation():
    qvar_start = 0
    qvar_end = 2
    qvar_step = 0.1
    q_eval_iter = int((qvar_end - qvar_start)/qvar_step)
    for i in range(q_eval_iter):
        print("Evaluation for Q: " + str(qvar_start + (qvar_step * i)))
        kalman_filter, filename = kalman.runFilter(kstd_x, kstd_y, kstd_z, occStart, occEnd, "CleanedOptitrack_3_perfect_flower", qvar_start + (qvar_step * i))
        evaluate_filter("CleanedOptitrack_3_perfect_flower", filename, kalman_filter)

if __name__ == "__main__":
    if (len(sys.argv) == 2): 
        if (sys.argv[1] == "evaluation"): 
            run_mse_evaluation()
        elif(sys.argv[1] == "processnoise"): 
            run_Q_evaluation()