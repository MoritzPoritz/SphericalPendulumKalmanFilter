from re import X
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
        self.imu_speed_data = []
        self.imu_acceleration_data = [[0,0,0]]
        self.Ts = [0]
        self.N = N
        self.dt = dt

    def imu_speed(self, x):
        dx = self.l *  np.cos(x[0]) * np.cos(x[1]) * x[2] - self.l * np.sin(x[0]) * np.sin(x[1]) * x[3]
        dy = self.l *  np.cos(x[0]) * np.sin(x[1]) * x[2] - self.l * np.sin(x[0]) * np.cos(x[1]) * x[3]
        dz = self.l * np.sin(x[0]) * x[2]
        return [dx,dy,dz]
        #p = position pb = position_before
    def imu_acceleration(self, dp, dpb, p, pb):
        ddx = (dp[0] - dpb[0]) / p[0] - pb[0]
        ddy = (dp[1] - dpb[1]) / p[1] - pb[1]
        ddz = (dp[2] - dpb[2]) / p[2] - pb[2]
        
        return [ddx, ddy, ddz]

    def state_first_deriv(self, x):
        dq = x[2:]
        dq2 = self.second_deriv_theta_phi(x)
        
        return np.concatenate([dq, dq2])


    def second_deriv_theta(self, x):
        theta, phi, d_theta, d_phi = x

        d2_theta = d_phi**2 * np.sin(theta) * np.cos(theta) - (self.g / self.l) * np.sin(theta)

        return d2_theta

    def second_deriv_phi(self, x):
        theta, phi, d_theta, d_phi = x
        t = np.tan(theta)

        d2_phi = -2 * d_phi * d_theta * (np.cos(theta) / np.sin(theta))
        #d2_phi = -2 * d_theta * d_phi / t
        if (t == 0):
            print ('t is 0')
            
        return d2_phi
        

    def second_deriv_theta_phi(self, x):
        theta, phi, d_theta, d_phi = x
        if (theta == 0): 
            print('theta is 0')
        c, s, t = np.cos(theta), np.sin(theta), np.tan(theta)

        d2_theta = (d_phi**2 * c - self.g / self.l) * s
        d2_phi = -2 * d_theta * d_phi / t
        if (t == 0):
            print ('t is 0')

        return np.array([d2_theta, d2_phi])
    
    # state is defined as x = [theta phi dtheta dphi]
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
        for i,state in enumerate(self.x):
            self.imu_speed_data.append(self.imu_speed(self.x[i]))
            self.positions.append(utils.polar_to_kartesian(self.l, state[0], state[1]))
            if(i > 0):
                self.imu_acceleration_data.append(self.imu_acceleration(self.imu_speed_data[i], self.imu_speed_data[i-1], self.positions[i], self.positions[i-1]))
            
    def return_sim_values(self):
        return [self.x, self.positions, self.Ts]
    
    def write_sim_data(self):
        utils.write_to_csv(self.x, self.positions, self.imu_acceleration_data, self.Ts,"simulation")



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
        #plot_pendulum()
    else:
        print("No Configuration found!")

def runSim():
    if (len(sys.argv) == 2): 
        config = utils.read_from_csv(str(sys.argv[1])+'.csv')
    else:
        config = utils.read_from_csv('config_flower.csv')

    main(config)


if __name__ == "__main__":
    runSim()



