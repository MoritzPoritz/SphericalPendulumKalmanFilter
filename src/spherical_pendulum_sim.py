from turtle import position
import numpy as np
import pandas as pd
import utils 
import sys

DEG_TO_RAD = np.pi/180

class SphericalPendulumSimulation(): 
    def __init__(self, x, pendulum_length, dt, N):
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
        x_new = x + self.state_first_deriv(x) * dt
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



def main(argv): 
    if argv[1] == 'help':
        print("Please type: " + 'theta(degrees), phi(degrees), theta\'(degrees), phi\'(degrees), pendulum__length, timestep, number_of_steps')
        if argv[2] == 'values':
            print('Good starting values are:')
            print('72, 0, 0, 42.8')
            print('45, 0, 0, 30')
    else: 
        theta = float(argv[1]) * DEG_TO_RAD
        phi = float(argv[2]) * DEG_TO_RAD
        dtheta = float(argv[3]) * DEG_TO_RAD
        dphi = float(argv[4]) * DEG_TO_RAD
        pendulum_length = float(argv[5])
        dt = float(argv[6])
        N = int(argv[7])
        pendulum = SphericalPendulumSimulation([theta, phi, dtheta, dphi], pendulum_length, dt, N)
        pendulum.run()
        pendulum.write_sim_data()

if __name__ == "__main__":
    main(sys.argv)

