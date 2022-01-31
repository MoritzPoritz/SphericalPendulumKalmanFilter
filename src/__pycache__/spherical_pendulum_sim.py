from turtle import position
import numpy as np
import pandas as pd
from sympy import false
import utils 
import sys
from plot_pendulum import plot_pendulum

DEG_TO_RAD = np.pi/180

class SphericalPendulumSimulation(): 
    def __init__(self, x, pendulum_length, dt, N):
        print("Beginning Simulation!")
        self.x = [x]
        self.g = 9.81
        self.l = pendulum_length
        self.positions = []
        self.Ts = [0]
        self.N = N
        self.dt = dt

    def state_first_deriv(self, x):
        dq = x[2:]
        dq2 = self.second_deriv_theta_phi(x)
        
        return np.concatenate([dq, dq2])

    def second_deriv_theta_phi(self, x):
        theta, phi, d_theta, d_phi = x
        c, s, t = np.cos(theta), np.sin(theta), np.tan(theta)

        d2_theta = (d_phi**2 * c - self.g / self.l) * s
        d2_phi = -2 * d_theta * d_phi / t
        if (t == 0):
            print ('t is 0')

        return np.array([d2_theta, d2_phi])
    
    def f(self, x, dt):
        n = 100
        x_new = x
        for i in range(n):
            x_new = x_new + self.state_first_deriv(x_new) * dt/n
        return x_new

    def run(self):
        #calculate states
        for i in range(self.N):            
            self.x.append(self.f(self.x[i],self.dt))
            self.Ts.append(i*self.dt)
        #calculate positions
        for state in self.x:
            self.positions.append(utils.polar_to_kartesian(self.l, state[0], state[1]))


    def return_sim_values(self):
        return [self.x, self.positions, self.Ts]
    
    def write_sim_data(self):
        utils.write_to_csv(self.x, self.positions, self.Ts)



def main(config):
    if config.empty is False:
        theta = float(config['theta']) * DEG_TO_RAD
        phi = float(config['phi']) * DEG_TO_RAD
        dtheta = float(config['dtheta']) * DEG_TO_RAD
        dphi = float(config['dphi']) * DEG_TO_RAD
        pendulum_length = float(config['pendulum__length'])
        dt = float(config['timestep'])
        T = config['sim_time']
        N = int(T / dt)
        print("Total Samples: {}".format(N))
        
        pendulum = SphericalPendulumSimulation([theta, phi, dtheta, dphi], pendulum_length, dt, N)
        pendulum.run()
        pendulum.write_sim_data()
        print("Simulation finished, wrote Data!")
        plot_pendulum()
    else:
        print("No Configuration found!")

if __name__ == "__main__":
    config = utils.read_from_csv('config.csv')
    main(config)


