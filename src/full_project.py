import kalman
import plot_pendulum
import plot_residual
import spherical_pendulum_sim
import mess_sim_data
import os

# Standardabweichung des auf die Simdaten aufgebrachten Rauschens
std_x, std_y, std_z = 0.01, 0.01, 0.01


# An den Filter übergebene Standardabweichung. Sollte im Realfall die Sensorstandardabweichung sein
kstd_x, kstd_y, kstd_z = 0.01, 0.01, 0.01


print("Starting Sim!")
spherical_pendulum_sim.runSim()
print("Generating Noisy Sensor Data")
mess_sim_data.mess_sim_data(std_x, std_y, std_z)
print("Filtering Data")
kalman.runFilter(kstd_x, kstd_y, kstd_z)
print("Filtered. Plotting:")
os.system("python plot_pendulum.py kalman simulation messy")
os.system("python plot_residual.py kalman simulation")

