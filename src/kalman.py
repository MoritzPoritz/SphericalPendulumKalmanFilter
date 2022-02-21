from ctypes import util
from turtle import pos
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from numpy.random import randn
import numpy as np
import sympy as sp
import math
import utils
import matplotlib.pyplot as plt
import sys
import time
from datetime import datetime


DEG_TO_RAD = math.pi/180
RAD_TO_DEG = 180/math.pi
gravitation = 9.81
l = 0.643974

def state_first_deriv(x):
    dq = x[2:]
    dq2 = second_deriv_theta_phi(x)
    
    return np.concatenate([dq, dq2])

def second_deriv_theta_phi(x):
    theta, phi, d_theta, d_phi = x
    c, s, t = np.cos(theta), np.sin(theta), np.tan(theta)
    #print(x)
    d2_theta = (d_phi**2 * c - gravitation / l) * s
    d2_phi = -2 * d_theta * d_phi / t
    if (t == 0):
        print ('t is 0')
    return np.array([d2_theta, d2_phi])
    

def h(x):
    y = utils.polar_to_kartesian(l, x[0], x[1]) 
    return y

    
def f(x, dt):
    n = 150
    x_new = x
    for i in range(n):
        x_new = x_new + state_first_deriv(x_new) * dt/n
    
    return x_new
    #x_new = x + state_first_deriv(x) * dt
    #return x_new


class PendulumUKF(): 
    def __init__(self, x, positions, dt, stdx, stdy, stdz, occStart, occEnd,Q_var):
        self.sigmas = MerweScaledSigmaPoints(4, alpha=.001, beta=2, kappa=-1)
       
        self.std_x, self.std_y, self.std_z = stdx, stdy, stdz
        self.positions = positions
        self.occStart = occStart
        self.occEnd = occEnd
        self.ukf = UKF(dim_x=4, dim_z=3,fx=f, hx=h, dt=dt, points=self.sigmas)
        self.init_theta, self.init_phi = utils.cartesian_to_polar(self.positions[0][0], self.positions[0][1],self.positions[0][2])
        #self.ukf.x = [self.init_theta, self.init_phi,0,0]
        #self.ukf.P *= 2
        self.ukf.x = x[0]
        self.ukf.R = np.diag([self.std_x**2, self.std_y**2, self.std_z**2])
        self.ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2,dt=dt, var=Q_var)
        self.ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2,dt=dt, var=Q_var)

    
    def run(self):
        uxs = []
        ps = []
        last_time = time.time()
        for idx, p in enumerate(self.positions):
            if (idx % 100 == 0): 
                print(idx, self.ukf.x)


            try:
                self.ukf.predict()
                #print(idx)
                if idx < self.occStart or idx > self.occEnd:
                    self.ukf.update(p)
                uxs.append(self.ukf.x.copy())
                #print("id: ", idx, "x: ", self.ukf.x)
                ps.append(np.diag(self.ukf.P))
            except: 
                #print(self.ukf.x)
                uxs.append([0,0,0,0])
                ps.append([0,0,0,0])

        print("Filtering took %.2f Minutes", str(time.time() - last_time))
                
        uxs = np.array(uxs)
        ps = np.array(ps)
        ux_positions = []
        for ux in uxs:
            ux_positions.append(utils.polar_to_kartesian(l, ux[0], ux[1]))

        ux_positions = np.array(ux_positions)
        
        return ux_positions, uxs, ps


def runFilter(std_x, std_y, std_z, occStart, occEnd, csv_name,Q_var):
    simulation = utils.read_from_csv(csv_name + '.csv')
    
    x, positions,ts = utils.simulation_data_to_array(simulation) 
    kalman = PendulumUKF(x, positions, 1/120, std_x, std_y, std_z, occStart, occEnd, Q_var)
    kalman_positions,kalman_states, vars = kalman.run()
    filename = "kalman_new"+str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    utils.write_to_csv(kalman_states,kalman_positions,ts, filename, vars)
    
    return kalman, filename

if __name__ == "__main__":
    # Good Datasets are: CleanedOptitrack_1_slightly_shorter, CleanedOptitrack_3_perfect_flower
    if (len(sys.argv) == 7): 
        dataset = sys.argv[1]
        pendulum_length = sys.argv[2]
        std = sys.argv[3]
        occ_start = sys.argv[4]
        occ_end = sys.argv[5]
        Q_var = sys.argv[6]
        runFilter(float(std), float(std), float(std), float(occ_start), float(occ_end), dataset, float(Q_var))
    else:
        runFilter(0.002, 0.002, 0.002, 0,0,"CleanedOptitrack")


