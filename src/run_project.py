import kalman
import plot_results
import plot_residual
import spherical_pendulum_sim
import mess_sim_data
import os

# Standardabweichung des auf die Simdaten aufgebrachten Rauschens
std_x, std_y, std_z = 0.05, 0.05, 0.05


# An den Filter übergebene Standardabweichung. Sollte im Realfall die Sensorstandardabweichung sein
kstd_x, kstd_y, kstd_z = 0.05, 0.05, 0.05

occStart = 0
occEnd = 0
print("Starting Sim!")
spherical_pendulum_sim.runSim()
print("Generating Noisy Sensor Data")
mess_sim_data.mess_sim_data(std_x, std_y, std_z)
print("Filtering Data")
kalman.runFilter(kstd_x, kstd_y, kstd_z, occStart, occEnd)
print("Filtered. Plotting:")
plot_results.plot_results(["simulation", "kalman", "messy"], occStart, occEnd)

#os.system("python plot_residual.py kalman simulation")


#spherical_pendulum_sim.runSim()
#os.system("python plot_pendulum.py simulation simulation simulation")

