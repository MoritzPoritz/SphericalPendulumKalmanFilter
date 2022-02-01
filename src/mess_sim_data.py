from turtle import position
import utils
import numpy as np 

std_x, std_y, std_z = .01,.01,.01


def mess_sim_data(): 
    sim_data = utils.read_from_csv('simulation.csv')
    states, positions, ts = utils.simulation_data_to_array(sim_data)
    positions = [[p[0]+std_x*np.random.randn(), p[1]+std_y*np.random.randn(), p[2]+std_z*np.random.randn()] for p in positions]

    utils.write_to_csv(states, positions, ts, "messy")





if __name__ == "__main__":
    mess_sim_data()
