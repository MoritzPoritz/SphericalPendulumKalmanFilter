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

DEG_TO_RAD = math.pi/180
RAD_TO_DEG = 180/math.pi
gravitation = 9.81
l = 0.5

def state_first_deriv(x):
    dq = x[2:]
    dq2 = second_deriv_theta_phi(x)
    
    return np.concatenate([dq, dq2])


def second_deriv_theta_phi(x):
    theta, phi, d_theta, d_phi = x
    c, s, t = np.cos(theta), np.sin(theta), np.tan(theta)

    d2_theta = (d_phi**2 * c - gravitation / l) * s
    d2_phi = -2 * d_theta * d_phi / t
    if (t == 0):
        print ('t is 0')

    return np.array([d2_theta, d2_phi])
    

def h(x):
    y = utils.polar_to_kartesian(l, x[0], x[1]) 
    return y

def f(x, u, dt):
    n = 150
    x_new = x
    for i in range(n):
        x_new = x_new + state_first_deriv(x_new) * dt/n
    return x_new
    #x_new = x + state_first_deriv(x) * dt
    #return x_new


class PendulumUKF(): 
    def __init__(self, x, positions, dt, stdx, stdy, stdz):
        self.sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2, kappa=-1)
       
        self.std_x, self.std_y, self.std_z = stdx, stdy, stdz
        self.positions = positions

        self.ukf = UKF(dim_x=4, dim_z=3,fx=f, hx=h, dt=dt, points=self.sigmas)
        self.ukf.x = x[0]
        self.ukf.R = np.diag([self.std_x**2, self.std_y**2, self.std_z**2])
        self.ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2,dt=dt, var=.02)
        self.ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2,dt=dt, var=.02)

    def run(self):
        uxs = []
        for p in self.positions:
            self.ukf.predict()
            self.ukf.update(p)
            uxs.append(self.ukf.x.copy())

        uxs = np.array(uxs)
        ux_positions = []
        for ux in uxs:
            ux_positions.append(utils.polar_to_kartesian(l, ux[0], ux[1]))

        ux_positions = np.array(ux_positions)
        return ux_positions


def runFilter(std_x, std_y, std_z):
    simulation = utils.read_from_csv('messy.csv')
    
    x, positions,ts = utils.simulation_data_to_array(simulation) 
    kalman = PendulumUKF(x, positions, 0.01, std_x, std_y, std_z)
    kalman_positions = np.array(kalman.run())
    utils.write_to_csv(x,kalman_positions,ts, "kalman")
    print('STD UKF', np.std(kalman_positions - positions))



if __name__ == "__main__":
    runFilter()


